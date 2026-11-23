"""Identify WAV, FLAC, and MP3 files from their contents.

Duration is reported only when the file structure contains it. This module
does not decode compressed audio samples and does not invent a duration when
the structure has none.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from vocalis.core.exceptions import InvalidAudioError

_STREAMINFO_BYTES = 34


@dataclass(frozen=True, slots=True)
class ParsedAudio:
    """Metadata read from a file's own headers or frames."""

    audio_format: str
    sample_rate: int | None
    channels: int | None
    duration_seconds: float | None
    duration_measured: bool
    note: str


def sniff_audio_format(data: bytes) -> str | None:
    """Return wav, flac, or mp3 when the leading bytes identify that format."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"fLaC"):
        return "flac"
    if data.startswith(b"ID3") or _find_mp3_sync(data, 0) is not None:
        return "mp3"
    return None


def parse_audio(data: bytes, audio_format: str, file_name: str) -> ParsedAudio:
    """Parse a file whose format was already identified from its contents."""
    if audio_format == "wav":
        return _parse_wav(data, file_name)
    if audio_format == "flac":
        return _parse_flac(data, file_name)
    if audio_format == "mp3":
        return _parse_mp3(data, file_name)
    raise InvalidAudioError(f"{file_name} is not a supported audio file.")


def _parse_wav(data: bytes, file_name: str) -> ParsedAudio:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            payload = handle.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise InvalidAudioError(f"{file_name} is not a valid WAV file.") from exc
    expected_bytes = frame_count * sample_width * channels
    if channels < 1 or sample_rate < 1 or frame_count < 1 or len(payload) != expected_bytes:
        raise InvalidAudioError(f"{file_name} is not a valid WAV file.")
    return ParsedAudio(
        audio_format="wav",
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=frame_count / sample_rate,
        duration_measured=True,
        note="Duration is calculated from the WAV frame count.",
    )


def _parse_flac(data: bytes, file_name: str) -> ParsedAudio:
    if not data.startswith(b"fLaC"):
        raise InvalidAudioError(f"{file_name} is not a valid FLAC file.")
    offset = 4
    streaminfo: bytes | None = None
    saw_last = False
    first = True
    while offset + 4 <= len(data):
        header = data[offset]
        block_type = header & 0x7F
        length = int.from_bytes(data[offset + 1 : offset + 4], "big")
        offset += 4
        if offset + length > len(data):
            raise InvalidAudioError(f"{file_name} is not a valid FLAC file.")
        block = data[offset : offset + length]
        offset += length
        if first:
            if block_type != 0 or length != _STREAMINFO_BYTES:
                raise InvalidAudioError(f"{file_name} is not a valid FLAC file.")
            streaminfo = block
            first = False
        if header & 0x80:
            saw_last = True
            break
    if streaminfo is None or not saw_last:
        raise InvalidAudioError(f"{file_name} is not a valid FLAC file.")
    sample_rate, channels, total_samples = _streaminfo_fields(streaminfo)
    if sample_rate < 1 or channels < 1:
        raise InvalidAudioError(f"{file_name} is not a valid FLAC file.")
    if total_samples < 1:
        return ParsedAudio(
            audio_format="flac",
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=None,
            duration_measured=False,
            note=(
                "The FLAC STREAMINFO block does not contain a sample count, "
                "so duration is unavailable. Audio frames were not decoded."
            ),
        )
    return ParsedAudio(
        audio_format="flac",
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=total_samples / sample_rate,
        duration_measured=True,
        note=(
            "Duration is calculated from the FLAC STREAMINFO sample count. "
            "Audio frames were not decoded."
        ),
    )


def _streaminfo_fields(block: bytes) -> tuple[int, int, int]:
    word = int.from_bytes(block[10:18], "big")
    sample_rate = word >> 44
    channels = ((word >> 41) & 0x7) + 1
    total_samples = word & 0xFFFFFFFFF
    return sample_rate, channels, total_samples


def _parse_mp3(data: bytes, file_name: str) -> ParsedAudio:
    offset = _skip_id3v2(data, file_name)
    sample_rate: int | None = None
    channels: int | None = None
    total_samples = 0
    frames = 0
    while offset < len(data):
        if data[offset : offset + 3] == b"TAG" and len(data) - offset == 128:
            break
        sync = _find_mp3_sync(data, offset)
        if sync is None or (frames == 0 and sync - offset > 64):
            break
        if frames > 0 and sync != offset:
            raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
        header = _frame_header(data, sync, file_name)
        frame_end = sync + header.length
        if frame_end > len(data):
            raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
        sample_rate = header.sample_rate
        channels = header.channels
        total_samples += header.samples_per_frame
        frames += 1
        offset = frame_end
        if frames > len(data):
            raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
    trailing_tag = data[offset : offset + 3] == b"TAG" and len(data) - offset == 128
    if frames < 1 or sample_rate is None or channels is None or (offset < len(data) and not trailing_tag):
        raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
    return ParsedAudio(
        audio_format="mp3",
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=total_samples / sample_rate,
        duration_measured=True,
        note="Duration is calculated from MPEG frame headers. Samples were not decoded.",
    )


def _skip_id3v2(data: bytes, file_name: str) -> int:
    if not data.startswith(b"ID3") or len(data) < 10:
        return 0
    size = (
        ((data[6] & 0x7F) << 21)
        | ((data[7] & 0x7F) << 14)
        | ((data[8] & 0x7F) << 7)
        | (data[9] & 0x7F)
    )
    offset = 10 + size
    if data[5] & 0x10:
        offset += 10
    if offset > len(data):
        raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
    return offset


def _find_mp3_sync(data: bytes, start: int) -> int | None:
    limit = len(data) - 4
    for index in range(start, limit + 1):
        if data[index] == 0xFF and data[index + 1] & 0xE0 == 0xE0:
            return index
    return None


@dataclass(frozen=True, slots=True)
class _FrameHeader:
    length: int
    sample_rate: int
    channels: int
    samples_per_frame: int


def _frame_header(data: bytes, offset: int, file_name: str) -> _FrameHeader:
    if offset + 4 > len(data):
        raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
    first, second, third, fourth = data[offset : offset + 4]
    if first != 0xFF or second & 0xE0 != 0xE0:
        raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
    version = (second >> 3) & 0x3
    layer = (second >> 1) & 0x3
    bitrate_index = (third >> 4) & 0xF
    sample_index = (third >> 2) & 0x3
    padding = (third >> 1) & 0x1
    if version == 1 or layer == 0 or bitrate_index in {0, 15} or sample_index == 3:
        raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
    bitrate = _BITRATES[(version, layer)][bitrate_index] * 1000
    sample_rate = _SAMPLE_RATES[version][sample_index]
    if version == 3:
        samples = 384 if layer == 3 else 1152
    elif layer == 3:
        samples = 384
    elif layer == 1:
        samples = 576
    else:
        samples = 1152
    if layer == 3:
        length = ((12 * bitrate) // sample_rate + padding) * 4
    elif version != 3 and layer == 1:
        length = (72 * bitrate) // sample_rate + padding
    else:
        length = (144 * bitrate) // sample_rate + padding
    if length < 4:
        raise InvalidAudioError(f"{file_name} is not a valid MP3 file.")
    channels = 1 if (fourth >> 6) & 0x3 == 3 else 2
    return _FrameHeader(length, sample_rate, channels, samples)


# Indexes match the MPEG bitrate field. Index 0 and 15 are rejected earlier.
_MPEG1_LAYER1 = (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0)
_MPEG1_LAYER2 = (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0)
_MPEG1_LAYER3 = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0)
_MPEG2_LAYER1 = (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0)
_MPEG2_LAYER23 = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)
_BITRATES = {
    (3, 3): _MPEG1_LAYER1,
    (3, 2): _MPEG1_LAYER2,
    (3, 1): _MPEG1_LAYER3,
    (2, 3): _MPEG2_LAYER1,
    (2, 2): _MPEG2_LAYER23,
    (2, 1): _MPEG2_LAYER23,
    (0, 3): _MPEG2_LAYER1,
    (0, 2): _MPEG2_LAYER23,
    (0, 1): _MPEG2_LAYER23,
}
_SAMPLE_RATES = {
    3: (44100, 48000, 32000),
    2: (22050, 24000, 16000),
    0: (11025, 12000, 8000),
}
