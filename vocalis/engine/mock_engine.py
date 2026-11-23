"""Simulated inference that writes a tone and never claims to speak or clone."""

from __future__ import annotations

import logging
import math
import os
import struct
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from vocalis.core.exceptions import InferenceError
from vocalis.core.types import ModelProvider
from vocalis.engine.types import (
    AudioMetadata,
    CapabilitySupport,
    EngineCapabilities,
    EngineStatus,
    GenerationRequest,
    GenerationResult,
    GenerationType,
)
from vocalis.engine.validation import prepare_request
from vocalis.utils.audio import write_pcm_wav

logger = logging.getLogger("vocalis.engine")

SAMPLE_RATE = 16_000
SIMULATED_DISCLAIMER = "Vocalis simulated audio. Not VoxCPM2 speech or a cloned voice."
_DESIGN_MESSAGE = "Simulated tone. This is not VoxCPM2 speech."
_CLONE_MESSAGE = "Simulated tone. The reference audio was not used to clone a voice."
_ACCEPTED_PARAMETERS = (
    "cfg_value",
    "inference_timesteps",
    "normalize",
    "denoise",
)

MOCK_CAPABILITIES = EngineCapabilities(
    provider=ModelProvider.MOCK,
    voice_design=CapabilitySupport.SIMULATED,
    voice_cloning=CapabilitySupport.SIMULATED,
    real_inference=False,
    progress_reporting=False,
    cancellation=False,
    sample_rate=SAMPLE_RATE,
    accepted_parameters=_ACCEPTED_PARAMETERS,
    applied_parameters=(),
    notice=(
        "The mock engine writes a tone for workflow testing. "
        "It does not synthesize speech and it does not clone a voice."
    ),
)


class MockInferenceEngine:
    """Local stand-in for VoxCPM2. Output is always marked simulated."""

    def __init__(
        self,
        design_output_dir: Path,
        clone_output_dir: Path,
        *,
        name_factory: Callable[[], str] | None = None,
    ) -> None:
        self._design_output_dir = Path(design_output_dir)
        self._clone_output_dir = Path(clone_output_dir)
        self._name_factory = name_factory or (lambda: uuid.uuid4().hex)
        self._ready = False

    def initialize(self) -> None:
        self._design_output_dir.mkdir(parents=True, exist_ok=True)
        self._clone_output_dir.mkdir(parents=True, exist_ok=True)
        self._ready = True

    def shutdown(self) -> None:
        self._ready = False

    def get_status(self) -> EngineStatus:
        if self._ready:
            detail = "Ready. Output is a simulated tone, not VoxCPM2 speech."
        else:
            detail = "Not initialized."
        return EngineStatus(provider=ModelProvider.MOCK, ready=self._ready, detail=detail)

    def get_capabilities(self) -> EngineCapabilities:
        return MOCK_CAPABILITIES

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._ready:
            raise InferenceError("The mock engine is not initialized.")
        prepared = prepare_request(request)
        started = time.perf_counter()
        directory, prefix, frequency, message = _route(prepared.generation_type)
        output_path = _allocate_output_path(
            self._design_output_dir if directory == "design" else self._clone_output_dir,
            prefix,
            self._name_factory,
        )
        try:
            audio = _write_tone(output_path, prepared.text, frequency)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        elapsed = time.perf_counter() - started
        result = GenerationResult(
            provider=ModelProvider.MOCK,
            generation_type=prepared.generation_type,
            output_path=output_path,
            audio=audio,
            simulated=True,
            generation_seconds=elapsed,
            parameters=prepared.resolved_parameters(),
            text=prepared.text,
            voice_description=prepared.voice_description,
            reference_audio_path=prepared.reference_audio_path,
            reference_transcript=prepared.reference_transcript,
            message=message,
        )
        logger.info(
            "Simulated generation completed type=%s file=%s audio_seconds=%.3f elapsed_seconds=%.3f",
            result.generation_type.value,
            result.output_path.name,
            result.audio.duration_seconds,
            result.generation_seconds,
        )
        return result


def _route(generation_type: GenerationType) -> tuple[str, str, float, str]:
    if generation_type is GenerationType.VOICE_DESIGN:
        return "design", "mock-design", 523.25, _DESIGN_MESSAGE
    if generation_type is GenerationType.VOICE_CLONING:
        return "clone", "mock-clone", 392.0, _CLONE_MESSAGE
    raise InferenceError(f"Unsupported generation type: {generation_type}.")


def _allocate_output_path(
    directory: Path,
    prefix: str,
    name_factory: Callable[[], str],
) -> Path:
    root = directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        token = name_factory()
        if not token or token in {".", ".."} or "/" in token or "\\" in token:
            raise InferenceError("Output name is not a single file name.")
        candidate = (root / f"{prefix}-{token}.wav").resolve()
        if not _within_directory(candidate, root):
            raise InferenceError("Refusing to write outside the configured output directory.")
        try:
            descriptor = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(descriptor)
        return candidate
    raise InferenceError("Could not allocate a unique output filename.")


def _write_tone(path: Path, text: str, frequency: float) -> AudioMetadata:
    duration = min(5.0, 0.25 + (0.045 * len(text)))
    pcm = _render_tone(frequency=frequency, duration=duration, sample_rate=SAMPLE_RATE)
    return write_pcm_wav(
        path,
        pcm,
        sample_rate=SAMPLE_RATE,
        simulated=True,
        disclaimer=SIMULATED_DISCLAIMER,
    )


def _render_tone(*, frequency: float, duration: float, sample_rate: int) -> bytes:
    frame_count = max(1, int(round(duration * sample_rate)))
    fade = min(int(0.01 * sample_rate), frame_count // 2)
    frames = bytearray()
    for index in range(frame_count):
        envelope = 1.0
        if fade and index < fade:
            envelope = index / fade
        elif fade and index >= frame_count - fade:
            envelope = (frame_count - index) / fade
        sample = math.sin((2 * math.pi * frequency * index) / sample_rate)
        amplitude = int(0.25 * 32767 * sample * envelope)
        frames += struct.pack("<h", amplitude)
    return bytes(frames)


def _within_directory(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
