"""Generation queue: one worker, bounded pending jobs, and explicit states."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from vocalis.core.exceptions import (
    CancellationUnsupportedError,
    InferenceError,
    QueueFullError,
    QueueUnavailableError,
    UnknownJobError,
)
from vocalis.core.types import ModelProvider
from vocalis.engine.model_manager import ModelManager
from vocalis.engine.types import (
    AudioMetadata,
    CapabilitySupport,
    EngineCapabilities,
    EngineStatus,
    GenerationRequest,
    GenerationResult,
    GenerationType,
    InferenceParameters,
)
from vocalis.services.jobs import JobState
from vocalis.services.queue import GenerationQueue

_CAPABILITIES = EngineCapabilities(
    provider=ModelProvider.MOCK,
    voice_design=CapabilitySupport.SIMULATED,
    voice_cloning=CapabilitySupport.SIMULATED,
    real_inference=False,
    progress_reporting=False,
    cancellation=False,
    sample_rate=16_000,
    accepted_parameters=(),
    applied_parameters=(),
    notice="Gate engine for queue tests.",
)


def _request(text: str = "Hello") -> GenerationRequest:
    return GenerationRequest(
        generation_type=GenerationType.VOICE_DESIGN,
        text=text,
        parameters=InferenceParameters(),
    )


def _result(request: GenerationRequest) -> GenerationResult:
    return GenerationResult(
        provider=ModelProvider.MOCK,
        generation_type=request.generation_type,
        output_path=Path("tone.wav"),
        audio=AudioMetadata(
            sample_rate=16_000,
            channels=1,
            sample_width_bytes=2,
            frame_count=16,
            duration_seconds=0.001,
            audio_format="wav",
            simulated=True,
            disclaimer="test",
        ),
        simulated=True,
        generation_seconds=0.0,
        parameters=request.resolved_parameters(),
        text=request.text,
        voice_description=request.voice_description,
        reference_audio_path=request.reference_audio_path,
        reference_transcript=request.reference_transcript,
        message="ok",
    )


class GateEngine:
    """Blocks inside generate until the test releases each call in order."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = threading.Condition(self._lock)
        self.started_count = 0
        self.active = 0
        self.peak = 0
        self.releases = 0
        self.errors: list[BaseException] = []
        self.seen: list[GenerationRequest] = []
        self._ready = False

    def initialize(self) -> None:
        self._ready = True

    def shutdown(self) -> None:
        self._ready = False

    def get_status(self) -> EngineStatus:
        return EngineStatus(provider=ModelProvider.MOCK, ready=self._ready, detail="gate")

    def get_capabilities(self) -> EngineCapabilities:
        return _CAPABILITIES

    def allow(self, count: int) -> None:
        with self._lock:
            self.releases = count
            self._started.notify_all()

    def wait_started(self, count: int, timeout: float = 2.0) -> None:
        with self._lock:
            reached = self._started.wait_for(lambda: self.started_count >= count, timeout)
        assert reached, f"generate was entered {self.started_count} times, expected {count}"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.started_count += 1
            ticket = self.started_count
            self.seen.append(request)
            self._started.notify_all()
            released = self._started.wait_for(lambda: self.releases >= ticket, timeout=5)
            self.active -= 1
            error = self.errors.pop(0) if self.errors else None
        if not released:
            raise TimeoutError("The test gate was not released.")
        if error is not None:
            raise error
        return _result(request)


@pytest.fixture
def gate_queue():
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=3)
    queue.start()
    try:
        yield queue, engine
    finally:
        engine.allow(10_000)
        queue.shutdown()
        manager.shutdown()


def test_submit_processes_one_job_and_records_times(gate_queue) -> None:
    queue, engine = gate_queue
    job_id = queue.submit(_request("one"))
    engine.wait_started(1)

    running = queue.get_job(job_id)
    assert running.state is JobState.RUNNING
    assert running.started_at is not None
    assert running.completed_at is None
    assert running.created_at.tzinfo is not None

    engine.allow(1)
    finished = queue.wait(job_id, timeout=2)
    assert finished.state is JobState.SUCCEEDED
    assert finished.result is not None
    assert finished.result.text == "one"
    assert finished.completed_at is not None
    assert finished.error_message is None
    assert engine.peak == 1
    assert engine.seen == [running.request]


def test_one_worker_runs_jobs_strictly_in_order(gate_queue) -> None:
    queue, engine = gate_queue
    first = queue.submit(_request("first"))
    engine.wait_started(1)
    second = queue.submit(_request("second"))
    third = queue.submit(_request("third"))

    assert queue.get_job(second).state is JobState.QUEUED
    assert queue.get_job(third).state is JobState.QUEUED
    assert engine.started_count == 1
    assert engine.peak == 1
    workers = [thread for thread in threading.enumerate() if thread.name == "vocalis-generation"]
    assert len(workers) == 1

    engine.allow(1)
    engine.wait_started(2)
    assert queue.get_job(first).state is JobState.SUCCEEDED
    assert queue.get_job(second).state is JobState.RUNNING
    assert engine.peak == 1

    engine.allow(3)
    assert queue.wait(third, timeout=2).state is JobState.SUCCEEDED
    assert [request.text for request in engine.seen] == ["first", "second", "third"]
    assert len(engine.seen) == 3


def test_pending_limit_counts_waiting_jobs_only() -> None:
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=2)
    queue.start()
    try:
        running = queue.submit(_request("running"))
        engine.wait_started(1)
        queue.submit(_request("wait-1"))
        queue.submit(_request("wait-2"))

        with pytest.raises(QueueFullError, match="2 jobs waiting"):
            queue.submit(_request("rejected"))

        assert engine.started_count == 1
        assert engine.peak == 1
        assert queue.get_job(running).state is JobState.RUNNING
        engine.allow(3)
        assert queue.wait(running, timeout=2).state is JobState.SUCCEEDED
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_failed_vocalis_error_is_stored_and_worker_continues(gate_queue) -> None:
    queue, engine = gate_queue
    engine.errors.append(InferenceError("The reference could not be used."))
    failed_id = queue.submit(_request("bad"))
    engine.allow(1)
    failed = queue.wait(failed_id, timeout=2)

    assert failed.state is JobState.FAILED
    assert failed.error_message == "The reference could not be used."
    assert failed.result is None
    assert "Traceback" not in (failed.error_message or "")

    engine.allow(2)
    succeeded_id = queue.submit(_request("after"))
    succeeded = queue.wait(succeeded_id, timeout=2)
    assert succeeded.state is JobState.SUCCEEDED
    assert engine.started_count == 2


def test_unexpected_worker_exception_hides_internal_details(gate_queue) -> None:
    queue, engine = gate_queue
    engine.errors.append(RuntimeError("SECRET_STACK frame in engine.py"))
    job_id = queue.submit(_request("boom"))
    engine.allow(1)
    failed = queue.wait(job_id, timeout=2)

    assert failed.state is JobState.FAILED
    assert failed.error_message == "Generation failed."
    assert "SECRET_STACK" not in failed.error_message

    engine.allow(2)
    follow_id = queue.submit(_request("still-alive"))
    assert queue.wait(follow_id, timeout=2).state is JobState.SUCCEEDED


def test_pending_job_can_be_cancelled(gate_queue) -> None:
    queue, engine = gate_queue
    running_id = queue.submit(_request("hold"))
    engine.wait_started(1)
    pending_id = queue.submit(_request("cancel-me"))

    cancelled = queue.cancel(pending_id)
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.started_at is None
    assert cancelled.completed_at is not None
    assert queue.get_job(running_id).state is JobState.RUNNING

    engine.allow(1)
    assert queue.wait(running_id, timeout=2).state is JobState.SUCCEEDED
    assert engine.started_count == 1
    assert queue.get_job(pending_id).state is JobState.CANCELLED


def test_running_job_cannot_be_cancelled(gate_queue) -> None:
    queue, engine = gate_queue
    job_id = queue.submit(_request("active"))
    engine.wait_started(1)

    with pytest.raises(CancellationUnsupportedError, match="cannot be cancelled"):
        queue.cancel(job_id)
    assert queue.get_job(job_id).state is JobState.RUNNING

    engine.allow(1)
    finished = queue.wait(job_id, timeout=2)
    assert finished.state is JobState.SUCCEEDED
    with pytest.raises(CancellationUnsupportedError, match="Only a queued job"):
        queue.cancel(job_id)


def test_unknown_job_id(gate_queue) -> None:
    queue, _engine = gate_queue
    with pytest.raises(UnknownJobError, match="No generation job"):
        queue.get_job("missing")
    with pytest.raises(UnknownJobError, match="No generation job"):
        queue.cancel("missing")


def test_recent_jobs_are_newest_first(gate_queue) -> None:
    queue, engine = gate_queue
    engine.allow(3)
    first = queue.submit(_request("first"))
    second = queue.submit(_request("second"))
    third = queue.submit(_request("third"))
    queue.wait(third, timeout=2)

    listed = queue.list_jobs(limit=2)
    assert [job.id for job in listed] == [third, second]
    assert queue.list_jobs()[2].id == first
    with pytest.raises(QueueUnavailableError, match="limit"):
        queue.list_jobs(limit=0)


def test_shutdown_cancels_pending_work_and_lets_the_running_job_finish(gate_queue) -> None:
    queue, engine = gate_queue
    running_id = queue.submit(_request("finish"))
    engine.wait_started(1)
    pending_id = queue.submit(_request("drop"))

    stopped = threading.Event()

    def _stop() -> None:
        queue.shutdown()
        stopped.set()

    stopper = threading.Thread(target=_stop, name="queue-shutdown-test")
    stopper.start()
    cancelled = queue.wait(pending_id, timeout=2)
    assert cancelled.state is JobState.CANCELLED
    assert cancelled.error_message == "The generation queue shut down before this job started."
    assert stopper.is_alive()

    engine.allow(1)
    stopper.join(timeout=2)
    assert stopped.is_set()
    assert queue.wait(running_id, timeout=2).state is JobState.SUCCEEDED
    with pytest.raises(QueueUnavailableError, match="not accepting"):
        queue.submit(_request("late"))
    workers = [thread for thread in threading.enumerate() if thread.name == "vocalis-generation" and thread.is_alive()]
    assert workers == []


def test_shutdown_before_start_is_a_no_op() -> None:
    engine = GateEngine()
    manager = ModelManager(engine)
    queue = GenerationQueue(manager, max_pending=1)
    queue.shutdown()
    with pytest.raises(QueueUnavailableError, match="not accepting"):
        queue.submit(_request())


def test_older_finished_jobs_leave_the_memory_cache() -> None:
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=3, max_cached=2)
    queue.start()
    try:
        engine.allow(3)
        first = queue.submit(_request("one"))
        second = queue.submit(_request("two"))
        third = queue.submit(_request("three"))
        assert queue.wait(third, timeout=2).state is JobState.SUCCEEDED
        with pytest.raises(UnknownJobError, match="No generation job"):
            queue.get_job(first)
        assert [job.id for job in queue.list_jobs()] == [third, second]
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_queued_and_running_jobs_stay_in_memory() -> None:
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=2, max_cached=1)
    queue.start()
    try:
        finished_id = queue.submit(_request("done"))
        engine.allow(1)
        assert queue.wait(finished_id, timeout=2).state is JobState.SUCCEEDED
        running_id = queue.submit(_request("running"))
        engine.wait_started(2)
        with pytest.raises(UnknownJobError, match="No generation job"):
            queue.get_job(finished_id)
        assert queue.get_job(running_id).state is JobState.RUNNING
        waiting_id = queue.submit(_request("waiting"))
        assert queue.get_job(running_id).state is JobState.RUNNING
        assert queue.get_job(waiting_id).state is JobState.QUEUED
    finally:
        engine.allow(10)
        queue.shutdown()
        manager.shutdown()


def test_job_cache_must_keep_at_least_one_job() -> None:
    manager = ModelManager(GateEngine())
    with pytest.raises(ValueError, match="at least 1"):
        GenerationQueue(manager, max_pending=1, max_cached=0)


def test_submit_rejected_when_queue_was_not_started() -> None:
    engine = GateEngine()
    manager = ModelManager(engine)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=1)
    with pytest.raises(QueueUnavailableError, match="not accepting"):
        queue.submit(_request())
    manager.shutdown()
