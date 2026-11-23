"""VoxCPM2 adapter.

The call shape comes from the published ``voxcpm==2.0.3`` package
(``src/voxcpm/core.py`` at tag ``2.0.3``). This module does not import
``voxcpm`` until a model load is requested, and it does not download weights.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from vocalis.core.exceptions import EngineUnavailableError, InferenceError
from vocalis.core.types import ModelProvider
from vocalis.engine.types import (
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

_ACCEPTED_PARAMETERS = ("cfg_value", "inference_timesteps", "normalize", "denoise")
_APPLIED_PARAMETERS = ("cfg_value", "inference_timesteps", "normalize")
_UNAVAILABLE_CAPABILITIES = EngineCapabilities(
    provider=ModelProvider.VOXCPM,
    voice_design=CapabilitySupport.UNAVAILABLE,
    voice_cloning=CapabilitySupport.UNAVAILABLE,
    real_inference=False,
    progress_reporting=False,
    cancellation=False,
    sample_rate=None,
    accepted_parameters=(),
    applied_parameters=(),
    notice="",
)


class VoxCPMInferenceEngine:
    """One loaded VoxCPM2 instance. A second initialize does not load it again."""

    def __init__(
        self,
        model_path: Path | None,
        design_output_dir: Path | None = None,
        clone_output_dir: Path | None = None,
        *,
        loader: Callable[[Path], object] | None = None,
        name_factory: Callable[[], str] | None = None,
    ) -> None:
        self.model_path = None if model_path is None else Path(model_path)
        self._design_output_dir = None if design_output_dir is None else Path(design_output_dir)
        self._clone_output_dir = None if clone_output_dir is None else Path(clone_output_dir)
        self._loader = loader or load_installed_model
        self._name_factory = name_factory or (lambda: uuid.uuid4().hex)
        self._model: object | None = None
        self._sample_rate: int | None = None
        self._ready = False

    def initialize(self) -> None:
        if self._ready:
            return
        if self.model_path is None:
            raise EngineUnavailableError(unavailable_reason())
        try:
            model = self._loader(self.model_path)
            sample_rate = _sample_rate(model)
        except EngineUnavailableError:
            raise
        except InferenceError:
            raise
        except Exception as exc:
            logger.error("VoxCPM2 load failed error_type=%s", type(exc).__name__)
            raise InferenceError("The VoxCPM2 model could not be loaded.") from exc
        self._model = model
        self._sample_rate = sample_rate
        self._ready = True

    def shutdown(self) -> None:
        self._model = None
        self._sample_rate = None
        self._ready = False

    def get_status(self) -> EngineStatus:
        if self._ready and self._sample_rate is not None:
            detail = f"Ready. VoxCPM2 is loaded at {self._sample_rate} Hz."
        else:
            detail = "Not initialized."
        return EngineStatus(provider=ModelProvider.VOXCPM, ready=self._ready, detail=detail)

    def get_capabilities(self) -> EngineCapabilities:
        if not self._ready or self._sample_rate is None:
            return _UNAVAILABLE_CAPABILITIES
        return EngineCapabilities(
            provider=ModelProvider.VOXCPM,
            voice_design=CapabilitySupport.AVAILABLE,
            voice_cloning=CapabilitySupport.AVAILABLE,
            real_inference=True,
            progress_reporting=False,
            cancellation=False,
            sample_rate=self._sample_rate,
            accepted_parameters=_ACCEPTED_PARAMETERS,
            applied_parameters=_APPLIED_PARAMETERS,
            notice=(
                "VoxCPM2 speech. Progress and cancellation are not available. "
                "The denoiser is not loaded, so denoise does not change the audio."
            ),
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._ready or self._model is None or self._sample_rate is None:
            raise InferenceError("The VoxCPM2 engine is not initialized.")
        prepared = prepare_request(request)
        started = time.perf_counter()
        directory, prefix, message = _route(prepared)
        output_dir = self._design_output_dir if directory == "design" else self._clone_output_dir
        if output_dir is None:
            raise InferenceError("The VoxCPM2 engine has no output directory.")
        output_path = _allocate_output_path(output_dir, prefix, self._name_factory)
        try:
            waveform = _call_generate(self._model, prepared)
            audio = write_pcm_wav(
                output_path,
                _pcm_from_waveform(waveform),
                sample_rate=self._sample_rate,
                simulated=False,
                disclaimer="",
            )
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        elapsed = time.perf_counter() - started
        if prepared.parameters is not None and prepared.parameters.denoise:
            message = f"{message} Denoise was requested, and the denoiser is not loaded."
        logger.info(
            "VoxCPM2 generation completed type=%s file=%s audio_seconds=%.3f elapsed_seconds=%.3f",
            prepared.generation_type.value,
            output_path.name,
            audio.duration_seconds,
            elapsed,
        )
        return GenerationResult(
            provider=ModelProvider.VOXCPM,
            generation_type=prepared.generation_type,
            output_path=output_path,
            audio=audio,
            simulated=False,
            generation_seconds=elapsed,
            parameters=prepared.resolved_parameters(),
            text=prepared.text,
            voice_description=prepared.voice_description,
            reference_audio_path=prepared.reference_audio_path,
            reference_transcript=prepared.reference_transcript,
            message=message,
        )


def load_installed_model(model_path: Path) -> object:
    """Load the local model directory. This does not download weights."""
    try:
        from voxcpm import VoxCPM
    except ModuleNotFoundError as exc:
        raise EngineUnavailableError(unavailable_reason()) from exc
    return VoxCPM.from_pretrained(
        hf_model_id=str(model_path),
        load_denoiser=False,
        local_files_only=True,
        optimize=False,
        device="cpu",
    )


def unavailable_reason() -> str:
    """Explain why the real engine cannot run, without importing it."""
    return "The VoxCPM2 engine is unavailable. The voxcpm package is not installed."


def _call_generate(model: object, request: GenerationRequest) -> object:
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise InferenceError("The VoxCPM2 model cannot generate audio.")
    parameters = request.resolved_parameters()
    kwargs: dict[str, object] = {
        "text": _spoken_text(request),
        "cfg_value": parameters.cfg_value,
        "inference_timesteps": parameters.inference_timesteps,
        "normalize": parameters.normalize,
        "denoise": parameters.denoise,
    }
    if request.generation_type is GenerationType.VOICE_CLONING:
        reference = str(request.reference_audio_path)
        kwargs["reference_wav_path"] = reference
        if request.reference_transcript:
            kwargs["prompt_wav_path"] = reference
            kwargs["prompt_text"] = request.reference_transcript
    try:
        return generate(**kwargs)
    except InferenceError:
        raise
    except Exception as exc:
        logger.error("VoxCPM2 generation failed error_type=%s", type(exc).__name__)
        raise InferenceError("VoxCPM2 could not generate audio.") from exc


def _spoken_text(request: GenerationRequest) -> str:
    if request.voice_description:
        return f"({request.voice_description}){request.text}"
    return request.text


def _route(request: GenerationRequest) -> tuple[str, str, str]:
    if request.generation_type is GenerationType.VOICE_DESIGN:
        return "design", "voxcpm-design", "Generated with VoxCPM2."
    if request.generation_type is GenerationType.VOICE_CLONING:
        if request.reference_transcript:
            return (
                "clone",
                "voxcpm-clone",
                "Generated with VoxCPM2. The reference audio and transcript were used for voice cloning.",
            )
        return (
            "clone",
            "voxcpm-clone",
            "Generated with VoxCPM2. The reference audio was used for voice cloning.",
        )
    raise InferenceError(f"Unsupported generation type: {request.generation_type}.")


def _sample_rate(model: object) -> int:
    tts_model = getattr(model, "tts_model", None)
    rate = getattr(tts_model, "sample_rate", None)
    if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1:
        raise InferenceError("The VoxCPM2 model did not report a sample rate.")
    return rate


def _pcm_from_waveform(waveform: object) -> bytes:
    values = _as_floats(waveform)
    frames = bytearray()
    for value in values:
        if not math.isfinite(value):
            raise InferenceError("VoxCPM2 returned audio that cannot be saved.")
        clipped = max(-1.0, min(1.0, value))
        frames += struct.pack("<h", int(round(clipped * 32767)))
    return bytes(frames)


def _as_floats(waveform: object) -> list[float]:
    if hasattr(waveform, "tolist"):
        waveform = waveform.tolist()
    if isinstance(waveform, (bytes, str)) or not isinstance(waveform, (list, tuple)):
        raise InferenceError("VoxCPM2 returned audio that cannot be saved.")
    values = list(waveform)
    if values and isinstance(values[0], (list, tuple)):
        flat: list[object] = []
        for item in values:
            if not isinstance(item, (list, tuple)) or len(item) != 1:
                raise InferenceError("VoxCPM2 returned audio that cannot be saved.")
            flat.append(item[0])
        values = flat
    if not values:
        raise InferenceError("VoxCPM2 returned empty audio.")
    try:
        return [float(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise InferenceError("VoxCPM2 returned audio that cannot be saved.") from exc


def _allocate_output_path(directory: Path, prefix: str, name_factory: Callable[[], str]) -> Path:
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


def _within_directory(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
