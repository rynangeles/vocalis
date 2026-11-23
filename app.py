"""Vocalis application entry point."""

from __future__ import annotations

import atexit
import argparse
import logging
import os
import sys
import threading
import time

from vocalis.core.config import Settings, load_settings
from vocalis.core.exceptions import (
    ConfigurationError,
    InferenceError,
    PersistenceError,
    QueueUnavailableError,
    VocalisError,
)
from vocalis.core.logging import configure_logging
from vocalis.core.types import ModelProvider
from vocalis.engine.model_manager import ModelManager
from vocalis.services.history import HistoryService
from vocalis.services.queue import GenerationQueue

# Gradio phones home unless this is set before it is imported.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# The running process owns one manager and one generation queue.
# A stop waits for the current generation. CPU inference can outlast the
# queue's short default, so the process uses its own limit.
_SHUTDOWN_TIMEOUT_SECONDS = 300
_runtime_lock = threading.Lock()
_manager: ModelManager | None = None
_queue: GenerationQueue | None = None
_history: HistoryService | None = None
_runtime_shutdown_registered = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vocalis — AI Voice Studio")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration, prepare runtime directories, and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_settings()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    logger = configure_logging(settings)
    logger.info(
        "Configuration loaded env=%s provider=%s host=%s port=%s",
        settings.app_env.value,
        settings.model_provider.value,
        settings.app_host,
        settings.app_port,
    )
    logger.info(
        "Runtime directories data=%s output=%s uploads=%s logs=%s database=%s",
        settings.data_dir,
        settings.output_dir,
        settings.upload_dir,
        settings.log_dir,
        settings.database_path,
    )
    if settings.model_provider is ModelProvider.MOCK:
        logger.info("Mock provider selected. VoxCPM2 will not be imported.")
    elif settings.voxcpm_model_path is not None:
        logger.info("VoxCPM model path=%s", settings.voxcpm_model_path)

    if not _initialize_engine(logger, settings, retain=not args.check):
        return 1

    if args.check:
        logger.info("Configuration check completed.")
        _flush(logger)
        return 0

    from vocalis.ui.runtime import StudioServices
    from vocalis.ui.studio import launch_app

    launch_app(
        settings,
        StudioServices(
            settings=settings,
            queue=get_generation_queue(),
            history=get_history_service(),
            manager=get_model_manager(),
            started_at=time.monotonic(),
        ),
    )
    return 0


def get_model_manager() -> ModelManager:
    """Return the manager retained by a running server."""
    if _manager is None:
        raise InferenceError("The model manager is not initialized.")
    return _manager


def get_generation_queue() -> GenerationQueue:
    """Return the queue retained by a running server."""
    if _queue is None:
        raise QueueUnavailableError("The generation queue is not initialized.")
    return _queue


def get_history_service() -> HistoryService:
    """Return history queries for the running server."""
    if _history is None:
        raise PersistenceError("Generation history is not initialized.")
    return _history


def shutdown_runtime() -> None:
    """Stop the queue, then release the model. Safe to call more than once."""
    global _manager, _queue, _history
    with _runtime_lock:
        queue, _queue = _queue, None
        manager, _manager = _manager, None
        _history = None
    if queue is not None:
        queue.shutdown(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    if manager is not None:
        manager.shutdown()


def _initialize_engine(logger: logging.Logger, settings: Settings, *, retain: bool) -> bool:
    """Initialize the selected engine and its single generation worker.

    ``--check`` starts both and shuts them down. A normal start keeps them.
    """
    global _manager, _queue, _history
    from vocalis.engine.factory import create_engine
    from vocalis.persistence.database import Database
    from vocalis.persistence.repositories.generations import GenerationRepository

    manager: ModelManager | None = None
    queue: GenerationQueue | None = None
    try:
        database = Database(settings.database_path)
        database.migrate()
        repository = GenerationRepository(database)
        repository.interrupt_incomplete()
        manager = ModelManager(create_engine(settings))
        manager.initialize()
        queue = GenerationQueue(
            manager,
            max_pending=settings.max_queue_size,
            store=repository,
        )
        queue.start()
        history = HistoryService(repository)
        if retain:
            _ensure_runtime_shutdown()
            with _runtime_lock:
                _manager = manager
                _queue = queue
                _history = history
        else:
            queue.shutdown()
            manager.shutdown()
    except VocalisError as exc:
        if queue is not None:
            queue.shutdown()
        label = "Database error" if isinstance(exc, PersistenceError) else "Engine error"
        logger.error("%s: %s", label, exc)
        print(f"{label}: {exc}", file=sys.stderr)
        return False
    return True


def _ensure_runtime_shutdown() -> None:
    global _runtime_shutdown_registered
    if _runtime_shutdown_registered:
        return
    atexit.register(shutdown_runtime)
    _runtime_shutdown_registered = True


def _flush(logger: logging.Logger) -> None:
    handlers = getattr(logger, "handlers", ())
    for handler in handlers:
        handler.flush()


if __name__ == "__main__":
    raise SystemExit(main())
