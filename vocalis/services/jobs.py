"""Typed generation jobs returned to callers. The worker stays private."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from vocalis.engine.types import GenerationRequest, GenerationResult


class JobState(StrEnum):
    """Lifecycle of one queued generation."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    """A read-only view of the worker. It does not expose the thread."""

    accepting: bool
    pending_count: int
    running: bool
    max_pending: int


@dataclass(frozen=True, slots=True)
class GenerationJob:
    """A snapshot of one job. Callers cannot change the queue through it."""

    id: str
    state: JobState
    request: GenerationRequest
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    result: GenerationResult | None
    error_message: str | None
