"""The generation queue records transitions without holding a connection during inference."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from vocalis.core.exceptions import PersistenceError, UnknownJobError
from vocalis.engine.mock_engine import MockInferenceEngine
from vocalis.engine.model_manager import ModelManager
from vocalis.engine.types import GenerationType, InferenceParameters
from vocalis.persistence.database import Database
from vocalis.persistence.models import NewGeneration
from vocalis.persistence.repositories.generations import (
    INTERRUPTED_MESSAGE,
    GenerationRepository,
)
from vocalis.services.audio import AudioService
from vocalis.services.cloning import VoiceCloningService
from vocalis.services.design import VoiceDesignService
from vocalis.services.jobs import JobState
from vocalis.services.queue import GenerationQueue
from vocalis.utils.audio import write_pcm_wav
from tests.unit.test_queue import GateEngine, _request


class _CreateFails:
    def create(self, generation: NewGeneration) -> None:
        raise PersistenceError("The generation could not be saved.")


class _RunningFails:
    def __init__(self, inner: GenerationRepository) -> None:
        self.inner = inner

    def create(self, generation: NewGeneration) -> None:
        self.inner.create(generation)

    def mark_running(self, job_id: str, started_at: object) -> None:
        raise PersistenceError("The generation could not be saved.")

    def mark_failed(self, job_id: str, completed_at: object, error_message: str) -> None:
        self.inner.mark_failed(job_id, completed_at, error_message)  # type: ignore[arg-type]

    def mark_succeeded(self, job_id: str, completed_at: object, result: object) -> None:
        raise AssertionError("inference should not complete")

    def mark_cancelled(self, job_id: str, completed_at: object, error_message: str | None) -> None:
        self.inner.mark_cancelled(job_id, completed_at, error_message)  # type: ignore[arg-type]


class _SuccessFailsOnce:
    def __init__(self, inner: GenerationRepository) -> None:
        self.inner = inner
        self.failed = False

    def create(self, generation: NewGeneration) -> None:
        self.inner.create(generation)

    def mark_running(self, job_id: str, started_at: object) -> None:
        self.inner.mark_running(job_id, started_at)  # type: ignore[arg-type]

    def mark_succeeded(self, job_id: str, completed_at: object, result: object) -> None:
        if not self.failed:
            self.failed = True
            raise PersistenceError("The generation could not be saved.")
        self.inner.mark_succeeded(job_id, completed_at, result)  # type: ignore[arg-type]

    def mark_failed(self, job_id: str, completed_at: object, error_message: str) -> None:
        self.inner.mark_failed(job_id, completed_at, error_message)  # type: ignore[arg-type]

    def mark_cancelled(self, job_id: str, completed_at: object, error_message: str | None) -> None:
        self.inner.mark_cancelled(job_id, completed_at, error_message)  # type: ignore[arg-type]


class _CancelFails:
    def __init__(self, inner: GenerationRepository) -> None:
        self.inner = inner

    def create(self, generation: NewGeneration) -> None:
        self.inner.create(generation)

    def mark_running(self, job_id: str, started_at: object) -> None:
        self.inner.mark_running(job_id, started_at)  # type: ignore[arg-type]

    def mark_succeeded(self, job_id: str, completed_at: object, result: object) -> None:
        self.inner.mark_succeeded(job_id, completed_at, result)  # type: ignore[arg-type]

    def mark_failed(self, job_id: str, completed_at: object, error_message: str) -> None:
        self.inner.mark_failed(job_id, completed_at, error_message)  # type: ignore[arg-type]

    def mark_cancelled(self, job_id: str, completed_at: object, error_message: str | None) -> None:
        raise PersistenceError("The generation could not be saved.")


def _repository(tmp_path: Path) -> GenerationRepository:
    database = Database(tmp_path / "history.db")
    database.migrate()
    return GenerationRepository(database)


def _queue(manager: ModelManager, repository: object, *, max_pending: int = 3) -> GenerationQueue:
    queue = GenerationQueue(manager, max_pending=max_pending, store=repository)  # type: ignore[arg-type]
    queue.start()
    return queue


def test_successful_job_is_stored_before_it_is_reported(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = _queue(manager, repository)
    try:
        job_id = queue.submit(_request("Hello there"))
        stored = repository.get(job_id)
        assert stored is not None
        assert stored.status is JobState.QUEUED
        assert stored.input_text == "Hello there"
        engine.wait_started(1)
        running = repository.get(job_id)
        assert running is not None
        assert running.status is JobState.RUNNING
        assert running.started_at is not None

        engine.allow(1)
        finished = queue.wait(job_id, timeout=2)
        assert finished.state is JobState.SUCCEEDED
        saved = repository.get(job_id)
        assert saved is not None
        assert saved.status is JobState.SUCCEEDED
        assert saved.output_audio_path == "tone.wav"
        assert saved.simulated is True
        assert saved.completed_at is not None
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_failed_and_cancelled_jobs_are_stored(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    engine = GateEngine()
    engine.errors.append(RuntimeError("SECRET_STACK"))
    manager = ModelManager(engine)
    manager.initialize()
    queue = _queue(manager, repository)
    try:
        failed_id = queue.submit(_request("bad"))
        engine.allow(1)
        failed = queue.wait(failed_id, timeout=2)
        assert failed.state is JobState.FAILED
        assert failed.error_message == "Generation failed."
        saved = repository.get(failed_id)
        assert saved is not None
        assert saved.status is JobState.FAILED
        assert saved.error_message == "Generation failed."
        assert "SECRET_STACK" not in saved.error_message

        holding = queue.submit(_request("hold"))
        engine.wait_started(2)
        pending = queue.submit(_request("pending"))
        cancelled = queue.cancel(pending)
        assert cancelled.state is JobState.CANCELLED
        stored_cancel = repository.get(pending)
        assert stored_cancel is not None
        assert stored_cancel.status is JobState.CANCELLED
        assert stored_cancel.started_at is None
        engine.allow(10)
        assert queue.wait(holding, timeout=2).state is JobState.SUCCEEDED
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_insert_failure_does_not_queue_the_job(tmp_path: Path) -> None:
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = _queue(manager, _CreateFails())
    try:
        with pytest.raises(PersistenceError, match="could not be saved"):
            queue.submit(_request("missing"))
        assert queue.list_jobs() == ()
        assert engine.started_count == 0
    finally:
        queue.shutdown()
        manager.shutdown()


def test_running_update_failure_does_not_start_inference(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = _queue(manager, _RunningFails(repository))
    try:
        job_id = queue.submit(_request("blocked"))
        finished = queue.wait(job_id, timeout=2)
        assert finished.state is JobState.FAILED
        assert finished.error_message == (
            "The generation could not be started because its history could not be saved."
        )
        assert engine.started_count == 0
        stored = repository.get(job_id)
        assert stored is not None
        assert stored.status is JobState.FAILED
        assert stored.error_message == finished.error_message
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_result_save_failure_is_not_reported_as_success_and_worker_continues(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    store = _SuccessFailsOnce(repository)
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = _queue(manager, store)
    try:
        engine.allow(2)
        first = queue.submit(_request("first"))
        failed = queue.wait(first, timeout=2)
        assert failed.state is JobState.FAILED
        assert failed.error_message == "The generation finished, but its history could not be saved."
        stored = repository.get(first)
        assert stored is not None
        assert stored.status is JobState.FAILED
        assert stored.output_audio_path is None

        second = queue.submit(_request("second"))
        assert queue.wait(second, timeout=2).state is JobState.SUCCEEDED
        saved = repository.get(second)
        assert saved is not None
        assert saved.status is JobState.SUCCEEDED
        assert engine.started_count == 2
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_cancel_failure_leaves_the_job_queued(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = _queue(manager, _CancelFails(repository))
    try:
        first = queue.submit(_request("first"))
        engine.wait_started(1)
        second = queue.submit(_request("second"))
        with pytest.raises(PersistenceError, match="could not be saved"):
            queue.cancel(second)
        assert queue.get_job(second).state is JobState.QUEUED
        assert repository.get(second) is not None
        assert repository.get(second).status is JobState.QUEUED  # type: ignore[union-attr]
        engine.allow(2)
        assert queue.wait(first, timeout=2).state is JobState.SUCCEEDED
        assert queue.wait(second, timeout=2).state is JobState.SUCCEEDED
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_shutdown_persists_cancellation_of_pending_work(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = _queue(manager, repository)
    try:
        running_id = queue.submit(_request("finish"))
        engine.wait_started(1)
        pending_id = queue.submit(_request("drop"))
        stopper = threading.Thread(target=queue.shutdown)
        stopper.start()
        cancelled = queue.wait(pending_id, timeout=2)
        assert cancelled.state is JobState.CANCELLED
        stored = repository.get(pending_id)
        assert stored is not None
        assert stored.status is JobState.CANCELLED
        assert stored.error_message == "The generation queue shut down before this job started."
        engine.allow(1)
        stopper.join(timeout=2)
        assert not stopper.is_alive()
        assert repository.get(running_id) is not None
        assert repository.get(running_id).status is JobState.SUCCEEDED  # type: ignore[union-attr]
    finally:
        engine.allow(10)
        if queue._worker is not None and queue._worker.is_alive():  # noqa: SLF001
            queue.shutdown()
        manager.shutdown()


def test_restart_does_not_rerun_incomplete_history(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    from datetime import datetime, timezone

    from vocalis.core.types import ModelProvider
    from vocalis.engine.types import InferenceParameters
    from vocalis.persistence.models import NewGeneration

    created = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def seed(job_id: str) -> None:
        repository.create(
            NewGeneration(
                id=job_id,
                generation_type=GenerationType.VOICE_DESIGN,
                provider=ModelProvider.MOCK,
                input_text=job_id,
                voice_description=None,
                reference_audio_path=None,
                reference_transcript=None,
                parameters=InferenceParameters(),
                created_at=created,
            )
        )

    seed("waiting")
    seed("kept")
    seed("orphan")
    repository.mark_cancelled("waiting", created, None)
    repository.mark_running("kept", created)
    repository.mark_succeeded("kept", created, _succeeded_result(tmp_path / "kept.wav"))
    repository.mark_running("orphan", created)

    assert repository.interrupt_incomplete() == 1
    assert repository.get("orphan").status is JobState.FAILED  # type: ignore[union-attr]
    assert repository.get("orphan").error_message == INTERRUPTED_MESSAGE  # type: ignore[union-attr]
    assert repository.get("kept").status is JobState.SUCCEEDED  # type: ignore[union-attr]
    assert repository.get("waiting").status is JobState.CANCELLED  # type: ignore[union-attr]

    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=2, store=repository)
    queue.start()
    try:
        assert engine.started_count == 0
        follow = queue.submit(_request("new"))
        engine.allow(1)
        assert queue.wait(follow, timeout=2).state is JobState.SUCCEEDED
        assert engine.started_count == 1
        assert [request.text for request in engine.seen] == ["new"]
        assert repository.get("kept").status is JobState.SUCCEEDED  # type: ignore[union-attr]
        assert repository.get("orphan").error_message == INTERRUPTED_MESSAGE  # type: ignore[union-attr]
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_services_persist_design_and_cloning_without_logging_prompts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    repository = _repository(tmp_path)
    engine = MockInferenceEngine(tmp_path / "design", tmp_path / "clone")
    manager = ModelManager(engine)
    manager.initialize()
    queue = _queue(manager, repository)
    reference = tmp_path / "reference.wav"
    write_pcm_wav(reference, b"\x00\x01" * 160, sample_rate=16_000, simulated=False, disclaimer="")
    original = reference.read_bytes()
    prompt = "UNIQUE_PROMPT_SHOULD_STAY_OUT_OF_LOGS"
    description = "UNIQUE_DESCRIPTION_SHOULD_STAY_OUT_OF_LOGS"
    transcript = "UNIQUE_TRANSCRIPT_SHOULD_STAY_OUT_OF_LOGS"
    try:
        with caplog.at_level("DEBUG"):
            design_id = VoiceDesignService(queue).submit(
                f"  {prompt}  ",
                voice_description=description,
                parameters=InferenceParameters(cfg_value=1.5, inference_timesteps=4),
            )
            design = queue.wait(design_id, timeout=2)
            references = tmp_path / "references"
            clone_id = VoiceCloningService(
                queue,
                AudioService(max_upload_bytes=1024 * 1024, reference_dir=references),
            ).submit(
                "Clone this",
                reference,
                reference_transcript=transcript,
            )
            clone = queue.wait(clone_id, timeout=2)
        assert prompt not in caplog.text
        assert description not in caplog.text
        assert transcript not in caplog.text
    finally:
        queue.shutdown()
        manager.shutdown()

    assert design.state is JobState.SUCCEEDED
    assert clone.state is JobState.SUCCEEDED
    stored_design = repository.get(design_id)
    stored_clone = repository.get(clone_id)
    assert stored_design is not None and stored_clone is not None
    assert stored_design.generation_type is GenerationType.VOICE_DESIGN
    assert stored_design.input_text == prompt
    assert stored_design.voice_description == description
    assert stored_design.parameters is not None
    assert stored_design.parameters.cfg_value == 1.5
    assert stored_design.simulated is True
    assert stored_design.output_audio_path is not None
    assert Path(stored_design.output_audio_path).is_file()
    assert stored_clone.generation_type is GenerationType.VOICE_CLONING
    stored_reference = Path(stored_clone.reference_audio_path or "")
    assert stored_reference.is_file()
    assert stored_reference != reference.resolve()
    assert stored_reference.parent == references.resolve()
    assert stored_reference.read_bytes() == original
    assert stored_clone.reference_transcript == transcript
    assert reference.read_bytes() == original
    blob = _snapshot(tmp_path / "history.db")
    assert b"tone-bytes" not in blob
    assert original not in blob


def test_a_job_that_leaves_memory_stays_in_history(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    engine = MockInferenceEngine(tmp_path / "design", tmp_path / "clone")
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=3, max_cached=1, store=repository)
    queue.start()
    try:
        first = queue.submit(_request("keep"))
        assert queue.wait(first, timeout=2).state is JobState.SUCCEEDED
        second = queue.submit(_request("newer"))
        assert queue.wait(second, timeout=2).state is JobState.SUCCEEDED
        with pytest.raises(UnknownJobError, match="No generation job"):
            queue.get_job(first)
        stored = repository.get(first)
        assert stored is not None
        assert stored.status is JobState.SUCCEEDED
        assert stored.input_text == "keep"
        assert repository.get(second) is not None
    finally:
        queue.shutdown()
        manager.shutdown()


def _succeeded_result(output: Path):
    from vocalis.core.types import ModelProvider
    from vocalis.engine.types import AudioMetadata, GenerationResult, InferenceParameters

    return GenerationResult(
        provider=ModelProvider.MOCK,
        generation_type=GenerationType.VOICE_DESIGN,
        output_path=output,
        audio=AudioMetadata(
            sample_rate=16_000,
            channels=1,
            sample_width_bytes=2,
            frame_count=16,
            duration_seconds=0.001,
            audio_format="wav",
            simulated=True,
            disclaimer="simulated",
        ),
        simulated=True,
        generation_seconds=0.01,
        parameters=InferenceParameters(),
        text="kept",
        voice_description=None,
        reference_audio_path=None,
        reference_transcript=None,
        message="ok",
    )


def _snapshot(path: Path) -> bytes:
    import sqlite3

    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        connection.close()
    blob = path.read_bytes()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            blob += sidecar.read_bytes()
    return blob
