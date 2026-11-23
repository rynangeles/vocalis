"""Submit voice-design work without waiting for the worker."""

from __future__ import annotations

from vocalis.core.exceptions import MissingInputError
from vocalis.engine.types import GenerationRequest, GenerationType, InferenceParameters
from vocalis.engine.validation import prepare_request
from vocalis.services.queue import GenerationQueue


class VoiceDesignService:
    """Build a voice-design request and hand it to the generation queue."""

    def __init__(self, queue: GenerationQueue) -> None:
        self._queue = queue

    def submit(
        self,
        text: str,
        voice_description: str | None = None,
        parameters: InferenceParameters | None = None,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise MissingInputError("Text to generate must not be empty.")
        request = prepare_request(
            GenerationRequest(
                generation_type=GenerationType.VOICE_DESIGN,
                text=text,
                voice_description=voice_description,
                parameters=parameters,
            )
        )
        return self._queue.submit(request)
