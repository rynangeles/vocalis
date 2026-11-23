"""The VoxCPM2 adapter calls an injected model and does not import voxcpm."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vocalis.core.exceptions import EngineUnavailableError, InferenceError
from vocalis.core.types import ModelProvider
from vocalis.engine.model_manager import ModelManager
from vocalis.engine.types import CapabilitySupport, GenerationRequest, GenerationType, InferenceParameters
from vocalis.engine.voxcpm_engine import VoxCPMInferenceEngine
from vocalis.utils.audio import inspect_wav, write_pcm_wav


class _Voice:
    sample_rate = 24_000


class _Model:
    def __init__(self) -> None:
        self.tts_model = _Voice()
        self.calls: list[dict[str, object]] = []
        self.waveform: object = [0.0, 0.5, -0.5, 1.0]

    def generate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.waveform


def _engine(tmp_path: Path, model: _Model | None = None, *, fail: BaseException | None = None):
    selected = model or _Model()
    loads: list[Path] = []

    def loader(path: Path) -> _Model:
        loads.append(path)
        if fail is not None:
            raise fail
        return selected

    names = iter(f"file{index}" for index in range(1, 20))
    engine = VoxCPMInferenceEngine(
        tmp_path / "models",
        tmp_path / "design",
        tmp_path / "clone",
        loader=loader,
        name_factory=lambda: next(names),
    )
    return engine, selected, loads


def _reference(path: Path) -> bytes:
    write_pcm_wav(path, b"\x00\x01" * 160, sample_rate=16_000, simulated=False, disclaimer="")
    return path.read_bytes()


def test_loading_keeps_one_model_for_later_requests(tmp_path: Path) -> None:
    engine, model, loads = _engine(tmp_path)
    manager = ModelManager(engine)

    manager.initialize()
    manager.initialize()
    first = manager.generate(GenerationRequest(GenerationType.VOICE_DESIGN, "Hello"))
    second = manager.generate(GenerationRequest(GenerationType.VOICE_DESIGN, "Again"))

    assert loads == [tmp_path / "models"]
    assert model.calls[0]["text"] == "Hello"
    assert "reference_wav_path" not in model.calls[0]
    assert first.simulated is False
    assert first.provider is ModelProvider.VOXCPM
    assert first.audio.sample_rate == 24_000
    assert second.output_path != first.output_path
    assert manager.capabilities.real_inference is True
    assert manager.capabilities.voice_design is CapabilitySupport.AVAILABLE
    assert manager.capabilities.voice_cloning is CapabilitySupport.AVAILABLE
    assert manager.capabilities.progress_reporting is False
    assert manager.capabilities.cancellation is False
    assert "seed" not in manager.capabilities.accepted_parameters
    assert "denoise" not in manager.capabilities.applied_parameters
    assert "voxcpm" not in sys.modules


def test_voice_design_wraps_the_description(tmp_path: Path) -> None:
    engine, model, _loads = _engine(tmp_path)
    engine.initialize()

    result = engine.generate(
        GenerationRequest(
            GenerationType.VOICE_DESIGN,
            "Hello",
            voice_description="warm",
            parameters=InferenceParameters(cfg_value=1.5, inference_timesteps=4, normalize=True),
        )
    )

    assert model.calls[0]["text"] == "(warm)Hello"
    assert model.calls[0]["cfg_value"] == 1.5
    assert model.calls[0]["inference_timesteps"] == 4
    assert model.calls[0]["normalize"] is True
    assert result.message == "Generated with VoxCPM2."
    assert b"ICMT" not in result.output_path.read_bytes()
    assert inspect_wav(result.output_path).sample_rate == 24_000


def test_cloning_uses_the_reference_and_leaves_it_unchanged(tmp_path: Path) -> None:
    reference = tmp_path / "speaker.wav"
    original = _reference(reference)
    engine, model, _loads = _engine(tmp_path)
    engine.initialize()

    plain = engine.generate(
        GenerationRequest(GenerationType.VOICE_CLONING, "Read this.", reference_audio_path=reference)
    )
    detailed = engine.generate(
        GenerationRequest(
            GenerationType.VOICE_CLONING,
            "Read this.",
            reference_audio_path=reference,
            reference_transcript="the spoken reference",
        )
    )

    assert model.calls[0]["reference_wav_path"] == str(reference)
    assert "prompt_wav_path" not in model.calls[0]
    assert model.calls[1]["prompt_wav_path"] == str(reference)
    assert model.calls[1]["prompt_text"] == "the spoken reference"
    assert reference.read_bytes() == original
    assert "used for voice cloning" in plain.message
    assert "transcript" in detailed.message
    assert plain.simulated is False


def test_denoise_is_forwarded_and_described(tmp_path: Path) -> None:
    engine, model, _loads = _engine(tmp_path)
    engine.initialize()

    result = engine.generate(
        GenerationRequest(
            GenerationType.VOICE_DESIGN,
            "Hello",
            parameters=InferenceParameters(denoise=True),
        )
    )

    assert model.calls[0]["denoise"] is True
    assert "denoiser is not loaded" in result.message


def test_a_model_failure_does_not_keep_a_partial_file(tmp_path: Path) -> None:
    engine, model, _loads = _engine(tmp_path)
    model.waveform = []
    engine.initialize()

    with pytest.raises(InferenceError, match="empty audio"):
        engine.generate(GenerationRequest(GenerationType.VOICE_DESIGN, "Hello"))

    assert list((tmp_path / "design").glob("*.wav")) == []
    assert engine.get_status().ready is True


def test_an_unexpected_model_error_hides_its_text(tmp_path: Path) -> None:
    engine, model, _loads = _engine(tmp_path)

    def explode(**_kwargs: object) -> object:
        raise RuntimeError("SECRET_STACK")

    model.generate = explode  # type: ignore[method-assign]
    engine.initialize()

    with pytest.raises(InferenceError, match="could not generate audio") as caught:
        engine.generate(GenerationRequest(GenerationType.VOICE_DESIGN, "Hello"))

    assert "SECRET_STACK" not in str(caught.value)


def test_shutdown_drops_the_model_and_the_next_initialize_loads_again(tmp_path: Path) -> None:
    engine, _model, loads = _engine(tmp_path)
    engine.initialize()
    engine.shutdown()

    with pytest.raises(InferenceError, match="not initialized"):
        engine.generate(GenerationRequest(GenerationType.VOICE_DESIGN, "Hello"))

    engine.initialize()
    assert len(loads) == 2
    assert engine.get_capabilities().sample_rate == 24_000


def test_a_missing_package_stays_unimported(tmp_path: Path) -> None:
    engine = VoxCPMInferenceEngine(tmp_path / "models", tmp_path / "design", tmp_path / "clone")

    with pytest.raises(EngineUnavailableError, match="not installed"):
        engine.initialize()

    assert engine.get_capabilities().real_inference is False
    assert "voxcpm" not in sys.modules
    assert "torch" not in sys.modules
