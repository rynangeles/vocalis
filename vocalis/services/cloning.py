"""Submit voice-cloning work without waiting for the worker."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from vocalis.core.exceptions import MissingInputError
from vocalis.engine.types import GenerationRequest, GenerationType, InferenceParameters
from vocalis.engine.validation import prepare_request
from vocalis.services.audio import AudioService
from vocalis.services.queue import GenerationQueue


class VoiceCloningService:
    """Validate a reference file, then queue a cloning request."""

    def __init__(self, queue: GenerationQueue, audio_service: AudioService) -> None:
        self._queue = queue
        self._audio = audio_service

    def submit(
        self,
        text: str,
        reference_audio: Path | None,
        reference_transcript: str | None = None,
        parameters: InferenceParameters | None = None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise MissingInputError("Text to generate must not be empty.")
        if reference_audio is None:
            raise MissingInputError("Voice cloning requires a reference audio file.")
        inspection = self._audio.inspect(Path(reference_audio))
        prepared = prepare_request(
            GenerationRequest(
                generation_type=GenerationType.VOICE_CLONING,
                text=text,
                reference_audio_path=inspection.path,
                reference_transcript=reference_transcript,
                parameters=parameters,
            )
        )
        stored = self._audio.retain_reference(inspection.path)
        return self._queue.submit(replace(prepared, reference_audio_path=stored))
