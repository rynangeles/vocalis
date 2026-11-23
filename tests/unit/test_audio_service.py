"""Reference audio is accepted only when its bytes match a supported format."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from vocalis.core.exceptions import InvalidAudioError, UnsupportedAudioFormatError
from vocalis.services.audio import AudioService
from vocalis.utils.audio import write_pcm_wav


def _service(limit: int = 1024 * 1024) -> AudioService:
    return AudioService(max_upload_bytes=limit)


def _wav(path: Path, frames: int = 160, sample_rate: int = 16_000) -> bytes:
    write_pcm_wav(
        path,
        b"\x00\x01" * frames,
        sample_rate=sample_rate,
        simulated=False,
        disclaimer="",
    )
    return path.read_bytes()


def _flac(sample_rate: int = 44_100, channels: int = 2, bits: int = 16, total_samples: int = 44_100) -> bytes:
    word = (sample_rate << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | total_samples
    streaminfo = struct.pack(">HH", 4096, 4096)
    streaminfo += (0).to_bytes(3, "big")
    streaminfo += (0).to_bytes(3, "big")
    streaminfo += word.to_bytes(8, "big")
    streaminfo += b"\x00" * 16
    assert len(streaminfo) == 34
    block = bytes([0x80]) + len(streaminfo).to_bytes(3, "big") + streaminfo
    return b"fLaC" + block


def _mp3_frame() -> bytes:
    # MPEG-1 Layer III, 128 kbps, 44100 Hz, no padding: 417-byte frame, 1152 samples.
    return b"\xff\xfb\x90\x00" + (b"\x00" * 413)


def test_valid_wav_exposes_duration_and_leaves_the_file_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "clip.WAV"
    original = _wav(path, frames=1600, sample_rate=16_000)
    inspected = _service().inspect(path)

    assert inspected.audio_format == "wav"
    assert inspected.sample_rate == 16_000
    assert inspected.channels == 1
    assert inspected.duration_seconds == pytest.approx(0.1)
    assert inspected.duration_measured is True
    assert inspected.size_bytes == len(original)
    assert path.read_bytes() == original


def test_valid_flac_duration_comes_from_streaminfo(tmp_path: Path) -> None:
    path = tmp_path / "clip.flac"
    payload = _flac(total_samples=22_050, sample_rate=44_100, channels=1)
    path.write_bytes(payload)
    inspected = _service().inspect(path)

    assert inspected.audio_format == "flac"
    assert inspected.sample_rate == 44_100
    assert inspected.channels == 1
    assert inspected.duration_seconds == pytest.approx(0.5)
    assert inspected.duration_measured is True
    assert "STREAMINFO" in inspected.note
    assert path.read_bytes() == payload


def test_flac_without_a_sample_count_does_not_invent_duration(tmp_path: Path) -> None:
    path = tmp_path / "unknown.flac"
    path.write_bytes(_flac(total_samples=0))
    inspected = _service().inspect(path)

    assert inspected.duration_seconds is None
    assert inspected.duration_measured is False
    assert inspected.sample_rate == 44_100
    assert "unavailable" in inspected.note


def test_valid_mp3_duration_comes_from_frame_headers(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp3"
    payload = _mp3_frame() + _mp3_frame()
    path.write_bytes(payload)
    inspected = _service().inspect(path)

    assert inspected.audio_format == "mp3"
    assert inspected.sample_rate == 44_100
    assert inspected.channels == 2
    assert inspected.duration_seconds == pytest.approx((1152 * 2) / 44_100)
    assert inspected.duration_measured is True
    assert "not decoded" in inspected.note
    assert path.read_bytes() == payload


def test_mp3_with_id3_tags_is_measured_from_the_frames(tmp_path: Path) -> None:
    path = tmp_path / "tagged.mp3"
    id3v2 = b"ID3\x03\x00\x00" + b"\x00\x00\x00\x00"
    id3v1 = b"TAG" + (b" " * 125)
    payload = id3v2 + _mp3_frame() + id3v1
    path.write_bytes(payload)
    inspected = _service().inspect(path)

    assert inspected.duration_seconds == pytest.approx(1152 / 44_100)
    assert path.read_bytes() == payload


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(InvalidAudioError, match="empty"):
        _service().inspect(path)
    assert path.read_bytes() == b""


def test_corrupt_wav_flac_and_mp3_are_rejected(tmp_path: Path) -> None:
    wav = tmp_path / "broken.wav"
    wav.write_bytes(b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE")
    flac = tmp_path / "broken.flac"
    flac.write_bytes(b"fLaC" + b"\x00\x00")
    mp3 = tmp_path / "broken.mp3"
    mp3.write_bytes(b"not audio at all")
    truncated = tmp_path / "short.mp3"
    truncated.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 20)

    for path in (wav, flac, mp3, truncated):
        original = path.read_bytes()
        with pytest.raises(InvalidAudioError, match="not a valid"):
            _service().inspect(path)
        assert path.read_bytes() == original


def test_extension_must_match_the_sniffed_contents(tmp_path: Path) -> None:
    wav_as_mp3 = tmp_path / "clip.mp3"
    _wav(wav_as_mp3)
    flac_as_wav = tmp_path / "clip.wav"
    flac_as_wav.write_bytes(_flac())

    with pytest.raises(InvalidAudioError, match="does not match"):
        _service().inspect(wav_as_mp3)
    with pytest.raises(InvalidAudioError, match="does not match"):
        _service().inspect(flac_as_wav)


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "clip.aiff"
    path.write_bytes(b"FORM" + b"\x00" * 16)
    with pytest.raises(UnsupportedAudioFormatError, match="WAV, FLAC, or MP3"):
        _service().inspect(path)
    assert path.read_bytes().startswith(b"FORM")


def test_upload_limit_uses_the_configured_size(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    original = _wav(path, frames=800)
    with pytest.raises(InvalidAudioError, match="800 byte upload limit"):
        _service(limit=800).inspect(path)
    assert path.read_bytes() == original
    assert len(original) > 800


def test_missing_path_and_directory_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidAudioError, match="not an audio file"):
        _service().inspect(tmp_path / "missing.wav")
    with pytest.raises(InvalidAudioError, match="not an audio file"):
        _service().inspect(tmp_path)


def test_a_validated_reference_is_copied_without_changing_the_original(tmp_path: Path) -> None:
    source = tmp_path / "upload.wav"
    original = _wav(source)
    destination_dir = tmp_path / "references"
    service = AudioService(max_upload_bytes=1024 * 1024, reference_dir=destination_dir)

    stored = service.retain_reference(source)
    again = service.retain_reference(source)

    assert source.read_bytes() == original
    assert stored != source.resolve()
    assert stored.parent == destination_dir.resolve()
    assert stored.name.startswith("reference-")
    assert stored.suffix == ".wav"
    assert stored.read_bytes() == original
    assert again != stored
    assert again.read_bytes() == original


def test_reference_storage_must_be_configured(tmp_path: Path) -> None:
    source = tmp_path / "upload.wav"
    _wav(source)
    with pytest.raises(InvalidAudioError, match="not configured"):
        _service().retain_reference(source)
    assert list(tmp_path.iterdir()) == [source]
