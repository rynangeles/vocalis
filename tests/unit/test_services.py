"""Voice design and voice cloning services submit typed requests."""

from __future__ import annotations

from pathlib import Path

import pytest

from vocalis.core.exceptions import InferenceError, InvalidAudioError, MissingInputError
from vocalis.engine.mock_engine import MockInferenceEngine
from vocalis.engine.model_manager import ModelManager
from vocalis.engine.types import GenerationType, InferenceParameters
from vocalis.services.audio import AudioService
from vocalis.services.cloning import VoiceCloningService
from vocalis.services.design import VoiceDesignService
from vocalis.services.jobs import JobState
from vocalis.services.queue import GenerationQueue
from vocalis.utils.audio import write_pcm_wav
from tests.unit.test_queue import GateEngine


def _wav(path: Path, frames: int = 160) -> bytes:
    write_pcm_wav(
        path,
        b"\x00\x01" * frames,
        sample_rate=16_000,
        simulated=False,
        disclaimer="",
    )
    return path.read_bytes()


@pytest.fixture
def design_queue(tmp_path: Path):
    engine = MockInferenceEngine(tmp_path / "design", tmp_path / "clone")
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=2)
    queue.start()
    try:
        yield queue
    finally:
        queue.shutdown()
        manager.shutdown()


def test_voice_design_builds_a_request_and_returns_before_waiting(tmp_path: Path) -> None:
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=2)
    queue.start()
    try:
        service = VoiceDesignService(queue)
        job_id = service.submit(
            "  Hello there  ",
            voice_description="  warm and close  ",
            parameters=InferenceParameters(cfg_value=1.5, inference_timesteps=4, normalize=True),
        )
        engine.wait_started(1)
        running = queue.get_job(job_id)
        assert running.state is JobState.RUNNING
        assert running.result is None
        assert running.request.generation_type is GenerationType.VOICE_DESIGN
        assert running.request.text == "Hello there"
        assert running.request.voice_description == "warm and close"
        assert running.request.reference_audio_path is None
        assert running.request.parameters is not None
        assert running.request.parameters.cfg_value == 1.5
        assert running.request.parameters.inference_timesteps == 4
        assert running.request.parameters.normalize is True

        engine.allow(1)
        finished = queue.wait(job_id, timeout=2)
        assert finished.state is JobState.SUCCEEDED
        assert finished.result is not None
        assert finished.result.text == "Hello there"
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_voice_design_rejects_missing_text_without_queueing(design_queue: GenerationQueue) -> None:
    service = VoiceDesignService(design_queue)
    with pytest.raises(MissingInputError, match="must not be empty"):
        service.submit("   ")
    assert design_queue.list_jobs() == ()


def test_voice_design_blank_description_is_omitted(design_queue: GenerationQueue) -> None:
    service = VoiceDesignService(design_queue)
    job_id = service.submit("Hello", voice_description="  ")
    finished = design_queue.wait(job_id, timeout=2)
    assert finished.state is JobState.SUCCEEDED
    assert finished.request.voice_description is None
    assert finished.result is not None
    assert finished.result.simulated is True
    assert finished.result.output_path.is_file()


def test_voice_cloning_submits_the_reference_without_changing_it(tmp_path: Path, design_queue: GenerationQueue) -> None:
    reference = tmp_path / "reference.wav"
    original = _wav(reference)
    references = tmp_path / "references"
    service = VoiceCloningService(
        design_queue,
        AudioService(max_upload_bytes=1024 * 1024, reference_dir=references),
    )
    job_id = service.submit(
        "  Clone this  ",
        reference,
        reference_transcript="  spoken line  ",
        parameters=InferenceParameters(denoise=True),
    )
    finished = design_queue.wait(job_id, timeout=2)

    assert finished.state is JobState.SUCCEEDED
    assert finished.request.generation_type is GenerationType.VOICE_CLONING
    assert finished.request.text == "Clone this"
    stored = finished.request.reference_audio_path
    assert stored is not None
    assert stored != reference.resolve()
    assert stored.parent == references.resolve()
    assert stored.name.startswith("reference-")
    assert stored.suffix == ".wav"
    assert stored.read_bytes() == original
    assert finished.request.reference_transcript == "spoken line"
    assert finished.request.parameters is not None
    assert finished.request.parameters.denoise is True
    assert reference.read_bytes() == original
    assert finished.result is not None
    assert finished.result.reference_audio_path == stored


def test_voice_cloning_returns_before_generation_finishes(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    _wav(reference)
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=1)
    queue.start()
    try:
        service = VoiceCloningService(
            queue,
            AudioService(max_upload_bytes=1024 * 1024, reference_dir=tmp_path / "references"),
        )
        job_id = service.submit("Hello", reference)
        engine.wait_started(1)
        assert queue.get_job(job_id).state is JobState.RUNNING
        assert queue.get_job(job_id).result is None
        engine.allow(1)
        assert queue.wait(job_id, timeout=2).state is JobState.SUCCEEDED
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_voice_cloning_rejects_missing_input_and_invalid_audio(tmp_path: Path, design_queue: GenerationQueue) -> None:
    service = VoiceCloningService(design_queue, AudioService(max_upload_bytes=1024))
    with pytest.raises(MissingInputError, match="reference audio"):
        service.submit("Hello", None)
    with pytest.raises(MissingInputError, match="must not be empty"):
        service.submit(" ", tmp_path / "reference.wav")

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(InvalidAudioError, match="empty"):
        service.submit("Hello", empty)
    with pytest.raises(InferenceError, match="cfg_value"):
        reference = tmp_path / "reference.wav"
        _wav(reference)
        service.submit("Hello", reference, parameters=InferenceParameters(cfg_value=0))
    assert design_queue.list_jobs() == ()
