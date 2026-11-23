"""The queue's view of history. The repository implements this protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from vocalis.engine.types import GenerationResult
from vocalis.persistence.models import NewGeneration


class GenerationStore(Protocol):
    """Record job transitions. Implementations must not run inference."""

    def create(self, generation: NewGeneration) -> None:
        """Insert a queued generation. Raise if the row was not stored."""

    def mark_running(self, job_id: str, started_at: datetime) -> None:
        """Record that a queued job has started."""

    def mark_succeeded(self, job_id: str, completed_at: datetime, result: GenerationResult) -> None:
        """Record a finished generation and its audio metadata."""

    def mark_failed(self, job_id: str, completed_at: datetime, error_message: str) -> None:
        """Record a failure for a job that was queued or running."""

    def mark_cancelled(
        self,
        job_id: str,
        completed_at: datetime,
        error_message: str | None,
    ) -> None:
        """Record cancellation of a job that has not started."""
