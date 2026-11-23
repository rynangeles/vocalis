"""Typed inference requests, results, and capability reports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from vocalis.core.types import ModelProvider


class GenerationType(StrEnum):
    """The two speech workflows Vocalis exposes."""

    VOICE_DESIGN = "voice_design"
    VOICE_CLONING = "voice_cloning"


class CapabilitySupport(StrEnum):
    """Whether a feature is real, simulated, or absent."""

    AVAILABLE = "available"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"


class ModelState(StrEnum):
    """Lifecycle of the in-process model manager."""

    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass(frozen=True, slots=True)
class InferenceParameters:
    """Request controls for a future VoxCPM2 call.

    These stay on the generation request. They are not application settings.
    The mock engine stores them and does not apply them to a voice model.
    ``seed`` is omitted until the installed VoxCPM2 API confirms it.
    """

    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize: bool = False
    denoise: bool = False


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One voice-design or voice-cloning request."""

    generation_type: GenerationType
    text: str
    voice_description: str | None = None
    reference_audio_path: Path | None = None
    reference_transcript: str | None = None
    parameters: InferenceParameters | None = None

    def resolved_parameters(self) -> InferenceParameters:
        return self.parameters if self.parameters is not None else InferenceParameters()


@dataclass(frozen=True, slots=True)
class AudioMetadata:
    """Measured properties of a written audio file."""

    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_seconds: float
    audio_format: str
    simulated: bool
    disclaimer: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The outcome of one engine call."""

    provider: ModelProvider
    generation_type: GenerationType
    output_path: Path
    audio: AudioMetadata
    simulated: bool
    generation_seconds: float
    parameters: InferenceParameters
    text: str
    voice_description: str | None
    reference_audio_path: Path | None
    reference_transcript: str | None
    message: str


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """What the selected engine can actually do in this build."""

    provider: ModelProvider
    voice_design: CapabilitySupport
    voice_cloning: CapabilitySupport
    real_inference: bool
    progress_reporting: bool
    cancellation: bool
    sample_rate: int | None
    accepted_parameters: tuple[str, ...]
    applied_parameters: tuple[str, ...]
    notice: str


@dataclass(frozen=True, slots=True)
class EngineStatus:
    """The engine's own readiness, separate from the manager lifecycle."""

    provider: ModelProvider
    ready: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Lifecycle status reported by the model manager."""

    state: ModelState
    provider: ModelProvider
    ready: bool
    message: str
    error: str | None
