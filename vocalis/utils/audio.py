"""WAV writing and inspection using the Python standard library."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

from vocalis.core.exceptions import InferenceError
from vocalis.engine.types import AudioMetadata

_PCM_FORMAT = 1
_CHANNELS = 1
_SAMPLE_WIDTH = 2


def write_pcm_wav(
    path: Path,
    pcm: bytes,
    *,
    sample_rate: int,
    simulated: bool,
    disclaimer: str,
) -> AudioMetadata:
    """Write a 16-bit mono PCM WAV, including a comment when the file is simulated."""
    if sample_rate < 1:
        raise InferenceError("Sample rate must be a positive integer.")
    if len(pcm) % _SAMPLE_WIDTH != 0:
        raise InferenceError("PCM payload is not aligned to the sample width.")
    frame_count = len(pcm) // _SAMPLE_WIDTH
    if frame_count < 1:
        raise InferenceError("PCM payload does not contain any frames.")

    fmt_payload = struct.pack(
        "<HHIIHH",
        _PCM_FORMAT,
        _CHANNELS,
        sample_rate,
        sample_rate * _CHANNELS * _SAMPLE_WIDTH,
        _CHANNELS * _SAMPLE_WIDTH,
        _SAMPLE_WIDTH * 8,
    )
    chunks = [_chunk(b"fmt ", fmt_payload), _chunk(b"data", pcm)]
    if simulated:
        comment = disclaimer.encode("ascii")
        chunks.insert(1, _chunk(b"LIST", b"INFO" + _chunk(b"ICMT", comment)))
    body = b"WAVE" + b"".join(chunks)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)

    inspected = inspect_wav(path)
    if (
        inspected.channels != _CHANNELS
        or inspected.sample_width_bytes != _SAMPLE_WIDTH
        or inspected.sample_rate != sample_rate
        or inspected.frame_count != frame_count
    ):
        raise InferenceError("Wrote a WAV file that could not be read back.")
    return AudioMetadata(
        sample_rate=sample_rate,
        channels=_CHANNELS,
        sample_width_bytes=_SAMPLE_WIDTH,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
        audio_format="wav",
        simulated=simulated,
        disclaimer=disclaimer,
    )


def inspect_wav(path: Path) -> AudioMetadata:
    """Read WAV header fields. Duration comes from the frame count."""
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
    except (wave.Error, EOFError) as exc:
        raise InferenceError(f"Audio file is not a readable WAV: {path.name}") from exc
    if sample_rate < 1 or frame_count < 1:
        raise InferenceError(f"Audio file does not contain playable frames: {path.name}")
    return AudioMetadata(
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
        audio_format="wav",
        simulated=False,
        disclaimer="",
    )


def _chunk(chunk_id: bytes, payload: bytes) -> bytes:
    if len(chunk_id) != 4:
        raise InferenceError("WAV chunk ids must be 4 bytes.")
    padding = b"\x00" if len(payload) % 2 else b""
    return chunk_id + struct.pack("<I", len(payload)) + payload + padding
