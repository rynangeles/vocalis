"""Read and delete generation history. This service does not run jobs."""

from __future__ import annotations

from vocalis.core.types import ModelProvider
from vocalis.engine.types import GenerationType
from vocalis.persistence.models import GenerationRecord
from vocalis.persistence.repositories.generations import (
    DEFAULT_HISTORY_PAGE_SIZE,
    GenerationRepository,
)
from vocalis.services.jobs import JobState


class HistoryService:
    """Application queries over persisted generations."""

    def __init__(self, repository: GenerationRepository) -> None:
        self._repository = repository

    def get(self, job_id: str) -> GenerationRecord | None:
        return self._repository.get(job_id)

    def list_recent(
        self,
        *,
        limit: int = DEFAULT_HISTORY_PAGE_SIZE,
        offset: int = 0,
        generation_type: GenerationType | None = None,
        status: JobState | None = None,
        provider: ModelProvider | None = None,
    ) -> tuple[GenerationRecord, ...]:
        """Return newest generations first."""
        return self._repository.list_recent(
            limit=limit,
            offset=offset,
            generation_type=generation_type,
            status=status,
            provider=provider,
        )

    def count(
        self,
        *,
        generation_type: GenerationType | None = None,
        status: JobState | None = None,
        provider: ModelProvider | None = None,
    ) -> int:
        return self._repository.count(
            generation_type=generation_type,
            status=status,
            provider=provider,
        )

    def delete(self, job_id: str) -> bool:
        """Delete one history row. Generated and reference audio files stay on disk."""
        return self._repository.delete(job_id)
