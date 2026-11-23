"""Shared request checks used by every engine."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from vocalis.core.exceptions import InferenceError
from vocalis.engine.types import GenerationRequest, GenerationType, InferenceParameters

_MAX_TEXT_LENGTH = 20_000


def prepare_request(request: GenerationRequest) -> GenerationRequest:
    """Return a stripped request or raise InferenceError.

    Voice design rejects reference audio. Voice cloning requires a non-empty
    reference file and does not read its contents here.
    """
    text = request.text.strip()
    if not text:
        raise InferenceError("Text to generate must not be empty.")
    if len(text) > _MAX_TEXT_LENGTH:
        raise InferenceError(f"Text to generate must be {_MAX_TEXT_LENGTH} characters or fewer.")

    description = _blank_to_none(request.voice_description)
    transcript = _blank_to_none(request.reference_transcript)
    parameters = _validate_parameters(request.resolved_parameters())
    reference = request.reference_audio_path

    if request.generation_type is GenerationType.VOICE_DESIGN:
        if reference is not None:
            raise InferenceError("Voice design does not accept reference audio.")
        if transcript is not None:
            raise InferenceError("A reference transcript is only accepted for voice cloning.")
        reference_path = None
    elif request.generation_type is GenerationType.VOICE_CLONING:
        reference_path = _require_reference(reference)
    else:
        raise InferenceError(f"Unsupported generation type: {request.generation_type}.")

    return replace(
        request,
        text=text,
        voice_description=description,
        reference_transcript=transcript,
        reference_audio_path=reference_path,
        parameters=parameters,
    )


def _validate_parameters(parameters: InferenceParameters) -> InferenceParameters:
    cfg_value = parameters.cfg_value
    timesteps = parameters.inference_timesteps
    if isinstance(cfg_value, bool) or not isinstance(cfg_value, (int, float)):
        raise InferenceError("cfg_value must be a number.")
    if not math.isfinite(float(cfg_value)) or float(cfg_value) <= 0:
        raise InferenceError("cfg_value must be a positive finite number.")
    if isinstance(timesteps, bool) or not isinstance(timesteps, int):
        raise InferenceError("inference_timesteps must be an integer.")
    if timesteps < 1:
        raise InferenceError("inference_timesteps must be at least 1.")
    if not isinstance(parameters.normalize, bool) or not isinstance(parameters.denoise, bool):
        raise InferenceError("normalize and denoise must be true or false.")
    return InferenceParameters(
        cfg_value=float(cfg_value),
        inference_timesteps=timesteps,
        normalize=parameters.normalize,
        denoise=parameters.denoise,
    )


def _require_reference(reference: Path | None) -> Path:
    if reference is None:
        raise InferenceError("Voice cloning requires a reference audio file.")
    path = Path(reference)
    if not path.exists():
        raise InferenceError("Voice cloning requires a reference audio file.")
    if not path.is_file():
        raise InferenceError("Reference audio must be a file.")
    if path.stat().st_size <= 0:
        raise InferenceError("Reference audio is empty.")
    return path.resolve()


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
