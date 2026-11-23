"""Validate reference audio and keep a private copy of an accepted upload."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from vocalis.core.exceptions import InvalidAudioError, UnsupportedAudioFormatError
from vocalis.utils.audio_inspect import parse_audio, sniff_audio_format

logger = logging.getLogger("vocalis.services")

_FORMAT_BY_EXTENSION = {
    ".wav": "wav",
    ".flac": "flac",
    ".mp3": "mp3",
}


@dataclass(frozen=True, slots=True)
class AudioInspection:
    """Measured properties of a reference file. Duration is omitted when unknown."""

    path: Path
    audio_format: str
    size_bytes: int
    sample_rate: int | None
    channels: int | None
    duration_seconds: float | None
    duration_measured: bool
    note: str


class AudioService:
    """Check uploads by their bytes. The original file is only read."""

    def __init__(self, *, max_upload_bytes: int, reference_dir: Path | None = None) -> None:
        if max_upload_bytes < 1:
            raise InvalidAudioError("The upload size limit must be at least 1 byte.")
        self._max_upload_bytes = max_upload_bytes
        self._reference_dir = None if reference_dir is None else Path(reference_dir)

    def inspect(self, path: Path) -> AudioInspection:
        """Validate one reference file and return its metadata."""
        source = Path(path)
        file_name = source.name or "audio"
        if not source.exists() or not source.is_file():
            raise InvalidAudioError(f"{file_name} is not an audio file.")
        try:
            before = source.read_bytes()
        except OSError as exc:
            logger.error("Could not read audio file name=%s", file_name)
            raise InvalidAudioError(f"{file_name} could not be read.") from exc
        size_bytes = len(before)
        if size_bytes < 1:
            raise InvalidAudioError(f"{file_name} is empty.")
        if size_bytes > self._max_upload_bytes:
            raise InvalidAudioError(
                f"{file_name} is larger than the {self._max_upload_bytes} byte upload limit."
            )
        extension = source.suffix.lower()
        expected = _FORMAT_BY_EXTENSION.get(extension)
        if expected is None:
            raise UnsupportedAudioFormatError(
                f"{file_name} must be a WAV, FLAC, or MP3 file."
            )
        detected = sniff_audio_format(before)
        if detected is None:
            raise InvalidAudioError(f"{file_name} is not a valid {expected.upper()} file.")
        if detected != expected:
            raise InvalidAudioError(
                f"{file_name} content does not match its .{expected} extension."
            )
        parsed = parse_audio(before, expected, file_name)
        if source.read_bytes() != before:
            raise InvalidAudioError(f"{file_name} changed while it was being checked.")
        return AudioInspection(
            path=source.resolve(),
            audio_format=parsed.audio_format,
            size_bytes=size_bytes,
            sample_rate=parsed.sample_rate,
            channels=parsed.channels,
            duration_seconds=parsed.duration_seconds,
            duration_measured=parsed.duration_measured,
            note=parsed.note,
        )

    def retain_reference(self, source: Path) -> Path:
        """Copy a validated reference into persistent storage.

        The source is only read. The copy uses a new name, so a later upload
        cannot replace the file that history records.
        """
        if self._reference_dir is None:
            raise InvalidAudioError("Reference storage is not configured.")
        inspection = self.inspect(source)
        try:
            payload = inspection.path.read_bytes()
        except OSError as exc:
            logger.error("Could not read audio file name=%s", inspection.path.name)
            raise InvalidAudioError(f"{inspection.path.name} could not be read.") from exc
        if (
            len(payload) != inspection.size_bytes
            or sniff_audio_format(payload) != inspection.audio_format
        ):
            raise InvalidAudioError(f"{inspection.path.name} changed while it was being stored.")

        self._reference_dir.mkdir(parents=True, exist_ok=True)
        suffix = inspection.path.suffix.lower()
        for _attempt in range(8):
            destination = self._reference_dir / f"reference-{uuid.uuid4().hex}{suffix}"
            try:
                descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                continue
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
            except OSError as exc:
                destination.unlink(missing_ok=True)
                logger.error("Could not store reference name=%s", destination.name)
                raise InvalidAudioError("The reference audio could not be stored.") from exc
            try:
                unchanged = inspection.path.read_bytes() == payload
            except OSError as exc:
                destination.unlink(missing_ok=True)
                raise InvalidAudioError(f"{inspection.path.name} could not be read.") from exc
            if not unchanged:
                destination.unlink(missing_ok=True)
                raise InvalidAudioError(f"{inspection.path.name} changed while it was being stored.")
            logger.info("Reference stored name=%s bytes=%s", destination.name, len(payload))
            return destination.resolve()
        raise InvalidAudioError("The reference audio could not be stored.")
