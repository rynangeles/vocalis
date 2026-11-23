"""Typed generation records stored in SQLite. Audio bytes stay on disk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vocalis.core.types import ModelProvider
from vocalis.engine.types import GenerationType, InferenceParameters
from vocalis.services.jobs import JobState


@dataclass(frozen=True, slots=True)
class NewGeneration:
    """The fields known when a job is accepted into the queue."""

    id: str
    generation_type: GenerationType
    provider: ModelProvider
    input_text: str
    voice_description: str | None
    reference_audio_path: str | None
    reference_transcript: str | None
    parameters: InferenceParameters
    created_at: datetime


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One history row. Paths point at files; they are not file contents."""

    id: str
    generation_type: GenerationType
    provider: ModelProvider
    input_text: str
    voice_description: str | None
    reference_audio_path: str | None
    reference_transcript: str | None
    parameters: InferenceParameters
    status: JobState
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    output_audio_path: str | None
    sample_rate: int | None
    channels: int | None
    sample_width_bytes: int | None
    frame_count: int | None
    duration_seconds: float | None
    audio_format: str | None
    simulated: bool | None
    audio_disclaimer: str | None
    generation_seconds: float | None
    error_message: str | None
