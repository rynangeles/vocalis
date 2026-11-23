"""Synchronous model lifecycle.

The generation queue is the only component that should call ``generate`` for
user requests. This manager stays single-threaded and does not start a worker.
"""

from __future__ import annotations

import logging
import time

from vocalis.core.exceptions import InferenceError, VocalisError
from vocalis.engine.base import InferenceEngine
from vocalis.engine.types import (
    EngineCapabilities,
    GenerationRequest,
    GenerationResult,
    ModelState,
    ModelStatus,
)

logger = logging.getLogger("vocalis.engine")


class ModelManager:
    """Own initialization state for one inference engine.

    Calls are synchronous. This manager does not queue work and does not start
    a background thread.
    """

    def __init__(self, engine: InferenceEngine) -> None:
        self._engine = engine
        self._state = ModelState.NOT_INITIALIZED
        self._error: str | None = None
        self._capabilities = engine.get_capabilities()

    @property
    def state(self) -> ModelState:
        return self._state

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    def get_status(self) -> ModelStatus:
        return ModelStatus(
            state=self._state,
            provider=self._capabilities.provider,
            ready=self._state is ModelState.READY,
            message=self._message(),
            error=self._error,
        )

    def initialize(self) -> ModelStatus:
        """Move to READY, or to ERROR when preparation fails.

        A second call while READY does not initialize the engine again.
        A call while ERROR tries again.
        """
        if self._state is ModelState.READY:
            return self.get_status()
        if self._state in {ModelState.INITIALIZING, ModelState.SHUTTING_DOWN}:
            raise InferenceError(f"The model manager cannot initialize while {self._state.value}.")

        self._state = ModelState.INITIALIZING
        self._error = None
        started = time.perf_counter()
        try:
            self._engine.initialize()
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._state = ModelState.ERROR
            self._error = str(exc)
            logger.error(
                "Model initialization failed provider=%s duration_seconds=%.3f error=%s",
                self._capabilities.provider.value,
                elapsed,
                exc,
            )
            if isinstance(exc, VocalisError):
                raise
            raise InferenceError("Model initialization failed.") from exc

        elapsed = time.perf_counter() - started
        self._capabilities = self._engine.get_capabilities()
        self._state = ModelState.READY
        logger.info(
            "Model initialization completed provider=%s state=%s duration_seconds=%.3f "
            "voice_design=%s voice_cloning=%s real_inference=%s",
            self._capabilities.provider.value,
            self._state.value,
            elapsed,
            self._capabilities.voice_design.value,
            self._capabilities.voice_cloning.value,
            self._capabilities.real_inference,
        )
        return self.get_status()

    def shutdown(self) -> ModelStatus:
        """Release the engine and return to NOT_INITIALIZED."""
        if self._state is ModelState.NOT_INITIALIZED:
            return self.get_status()
        if self._state is ModelState.SHUTTING_DOWN:
            raise InferenceError("The model manager is already shutting down.")

        self._state = ModelState.SHUTTING_DOWN
        try:
            self._engine.shutdown()
        except Exception as exc:
            self._state = ModelState.ERROR
            self._error = str(exc)
            logger.error(
                "Model shutdown failed provider=%s error=%s",
                self._capabilities.provider.value,
                exc,
            )
            if isinstance(exc, VocalisError):
                raise
            raise InferenceError("Model shutdown failed.") from exc

        self._error = None
        self._state = ModelState.NOT_INITIALIZED
        logger.info(
            "Model shutdown completed provider=%s state=%s",
            self._capabilities.provider.value,
            self._state.value,
        )
        return self.get_status()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate only while the lifecycle state is READY."""
        if self._state is not ModelState.READY:
            raise InferenceError(f"The model is not ready (state={self._state.value}).")
        return self._engine.generate(request)

    def _message(self) -> str:
        if self._state is ModelState.READY:
            return self._engine.get_status().detail
        if self._state is ModelState.ERROR:
            return self._error or "The model is in an error state."
        if self._state is ModelState.INITIALIZING:
            return "The model is initializing."
        if self._state is ModelState.SHUTTING_DOWN:
            return "The model is shutting down."
        return "The model has not been initialized."
