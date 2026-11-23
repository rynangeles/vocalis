"""Logging setup."""

from pathlib import Path

from vocalis.core.config import load_settings
from vocalis.core.logging import configure_logging


def test_configure_logging_writes_a_rotating_log(tmp_path: Path) -> None:
    settings = load_settings(environ={}, project_root=tmp_path)
    logger = configure_logging(settings)
    logger.info("startup check")
    for handler in logger.handlers:
        handler.flush()

    log_path = settings.log_dir / "vocalis.log"
    text = log_path.read_text(encoding="utf-8")
    assert "startup check" in text
    assert "T" in text and text.endswith("\n")


def test_configure_logging_replaces_handlers(tmp_path: Path) -> None:
    settings = load_settings(environ={}, project_root=tmp_path)
    configure_logging(settings)
    logger = configure_logging(settings)

    assert len(logger.handlers) == 2
    logger.info("once")
    for handler in logger.handlers:
        handler.flush()
    text = (settings.log_dir / "vocalis.log").read_text(encoding="utf-8")
    assert text.count("once") == 1
