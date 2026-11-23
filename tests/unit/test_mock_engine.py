"""Mock engine behavior, WAV output, and input validation."""

import math
import sys
import wave
from pathlib import Path

import pytest

from vocalis.core.exceptions import InferenceError
from vocalis.core.types import ModelProvider
from vocalis.engine.mock_engine import SIMULATED_DISCLAIMER, MockInferenceEngine
from vocalis.engine.types import GenerationRequest, GenerationType, InferenceParameters


def _engine(tmp_path: Path, **kwargs) -> MockInferenceEngine:
    engine = MockInferenceEngine(tmp_path / "design", tmp_path / "clone", **kwargs)
    engine.initialize()
    return engine


def _design(text: str = "Hello from Vocalis", **kwargs) -> GenerationRequest:
    return GenerationRequest(generation_type=GenerationType.VOICE_DESIGN, text=text, **kwargs)


def _clone(reference: Path, text: str = "Hello from Vocalis", **kwargs) -> GenerationRequest:
    return GenerationRequest(
        generation_type=GenerationType.VOICE_CLONING,
        text=text,
        reference_audio_path=reference,
        **kwargs,
    )


def _frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        return handle.readframes(handle.getnframes())


def test_voice_design_writes_a_playable_simulated_wav(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    result = engine.generate(
        _design(voice_description="A warm conversational voice")
    )

    frames = _frames(result.output_path)
    assert result.provider is ModelProvider.MOCK
    assert result.generation_type is GenerationType.VOICE_DESIGN
    assert result.simulated is True
    assert result.audio.simulated is True
    assert result.audio.audio_format == "wav"
    assert result.audio.duration_seconds == pytest.approx(len(frames) / 2 / 16_000)
    assert result.audio.duration_seconds > 0
    assert result.generation_seconds >= 0
    assert result.voice_description == "A warm conversational voice"
    assert result.reference_audio_path is None
    assert result.message == "Simulated tone. This is not VoxCPM2 speech."
    assert result.output_path.name.startswith("mock-design-")
    assert result.output_path.parent == (tmp_path / "design").resolve()
    assert any(sample != 0 for sample in frames)
    assert SIMULATED_DISCLAIMER.encode("ascii") in result.output_path.read_bytes()
    assert result.output_path.read_bytes().startswith(b"RIFF")


def test_voice_cloning_keeps_the_reference_and_does_not_claim_a_clone(tmp_path: Path) -> None:
    reference = tmp_path / "speaker.wav"
    reference.write_bytes(b"reference-bytes-not-decoded")
    before = reference.read_bytes()
    engine = _engine(tmp_path)

    result = engine.generate(
        _clone(reference, reference_transcript="  exact transcript  ")
    )

    assert reference.read_bytes() == before
    assert result.generation_type is GenerationType.VOICE_CLONING
    assert result.simulated is True
    assert result.reference_audio_path == reference.resolve()
    assert result.reference_transcript == "exact transcript"
    assert result.message == "Simulated tone. The reference audio was not used to clone a voice."
    assert result.output_path.name.startswith("mock-clone-")
    assert result.output_path.parent == (tmp_path / "clone").resolve()
    assert result.output_path != reference.resolve()
    _frames(result.output_path)


def test_inference_parameters_are_stored_and_do_not_change_the_tone(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    quiet = engine.generate(_design(parameters=InferenceParameters(cfg_value=1.5, inference_timesteps=4)))
    strong = engine.generate(
        _design(parameters=InferenceParameters(cfg_value=3.0, inference_timesteps=30, normalize=True))
    )

    assert quiet.parameters.cfg_value == 1.5
    assert strong.parameters.normalize is True
    assert _frames(quiet.output_path) == _frames(strong.output_path)
    assert quiet.output_path != strong.output_path


def test_filenames_stay_unique_when_a_name_collides(tmp_path: Path) -> None:
    tokens = iter(["taken", "taken", "fresh"])
    engine = _engine(tmp_path, name_factory=lambda: next(tokens))

    first = engine.generate(_design("one"))
    original = first.output_path.read_bytes()
    second = engine.generate(_design("two"))

    assert first.output_path.name == "mock-design-taken.wav"
    assert second.output_path.name == "mock-design-fresh.wav"
    assert first.output_path.read_bytes() == original
    assert first.output_path != second.output_path


def test_repeated_name_collisions_do_not_overwrite(tmp_path: Path) -> None:
    engine = _engine(tmp_path, name_factory=lambda: "same")
    first = engine.generate(_design("keep"))
    original = first.output_path.read_bytes()

    with pytest.raises(InferenceError, match="unique output filename"):
        engine.generate(_design("again"))

    assert first.output_path.read_bytes() == original


def test_output_names_cannot_escape_the_configured_directory(tmp_path: Path) -> None:
    engine = _engine(tmp_path, name_factory=lambda: "../escape")

    with pytest.raises(InferenceError, match="single file name"):
        engine.generate(_design())

    assert list((tmp_path / "design").iterdir()) == []
    assert not (tmp_path / "mock-design-..").exists()


def test_generate_requires_initialization(tmp_path: Path) -> None:
    engine = MockInferenceEngine(tmp_path / "design", tmp_path / "clone")

    with pytest.raises(InferenceError, match="not initialized"):
        engine.generate(_design())


def test_invalid_requests_are_rejected(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reference = tmp_path / "speaker.bin"
    reference.write_bytes(b"audio")
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    folder = tmp_path / "folder"
    folder.mkdir()

    cases = [
        _design("   "),
        _design(""),
        _clone(tmp_path / "missing.wav"),
        _clone(empty),
        _clone(folder),
        GenerationRequest(generation_type=GenerationType.VOICE_CLONING, text="Hello"),
        _design(reference_audio_path=reference),
        _design(reference_transcript="transcript"),
        _design(parameters=InferenceParameters(cfg_value=float("nan"))),
        _design(parameters=InferenceParameters(cfg_value=-1)),
        _design(parameters=InferenceParameters(cfg_value=True)),  # type: ignore[arg-type]
        _design(parameters=InferenceParameters(inference_timesteps=0)),
        _design(parameters=InferenceParameters(inference_timesteps=True)),  # type: ignore[arg-type]
    ]
    for request in cases:
        with pytest.raises(InferenceError):
            engine.generate(request)

    assert list((tmp_path / "design").glob("*.wav")) == []
    assert list((tmp_path / "clone").glob("*.wav")) == []


def test_whitespace_around_text_and_description_is_removed(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    result = engine.generate(_design("  Hello  ", voice_description="  warm voice  "))

    assert result.text == "Hello"
    assert result.voice_description == "warm voice"
    assert math.isfinite(result.audio.duration_seconds)


def test_mock_generation_does_not_import_model_packages(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reference = tmp_path / "speaker.bin"
    reference.write_bytes(b"audio")
    engine.generate(_design())
    engine.generate(_clone(reference))

    assert "voxcpm" not in sys.modules
    assert "torch" not in sys.modules
    assert "torchaudio" not in sys.modules
