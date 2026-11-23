"""One worker thread in front of ModelManager.generate()."""

from __future__ import annotations

import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from vocalis.core.exceptions import (
    CancellationUnsupportedError,
    QueueFullError,
    QueueUnavailableError,
    UnknownJobError,
    VocalisError,
)
from vocalis.engine.model_manager import ModelManager
from vocalis.engine.types import GenerationRequest, GenerationResult
from vocalis.persistence.models import NewGeneration
from vocalis.persistence.store import GenerationStore
from vocalis.services.jobs import GenerationJob, JobState, QueueSnapshot

logger = logging.getLogger("vocalis.services")

# Finished jobs older than this leave memory. SQLite keeps the durable history.
# Queued and running jobs are kept even when this number is already reached.
DEFAULT_CACHED_JOBS = 100
_TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class _Record:
    id: str
    request: GenerationRequest
    created_at: datetime
    sequence: int
    state: JobState = JobState.QUEUED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: GenerationResult | None = None
    error_message: str | None = None
    finished: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> GenerationJob:
        return GenerationJob(
            id=self.id,
            state=self.state,
            request=self.request,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            result=self.result,
            error_message=self.error_message,
        )


class GenerationQueue:
    """Accept generation requests and run them one at a time.

    Pending jobs can be cancelled. A job that has started stays running,
    because the engine does not expose a safe way to stop it.
    """

    def __init__(
        self,
        manager: ModelManager,
        *,
        max_pending: int,
        store: GenerationStore | None = None,
        max_cached: int = DEFAULT_CACHED_JOBS,
    ) -> None:
        """Run jobs one at a time.

        The running application passes a history store. The store records each
        transition and is not asked to run inference. A missing store keeps
        the worker's existing in-memory behavior.

        ``max_cached`` bounds how many jobs stay in memory. Finished jobs are
        dropped oldest first. A queued or running job is never dropped to make
        room, and SQLite remains the record of jobs that have left memory.
        """
        if max_pending < 1:
            raise ValueError("The pending queue limit must be at least 1.")
        if max_cached < 1:
            raise ValueError("The in-memory job cache must keep at least 1 job.")
        self._manager = manager
        self._store = store
        self._max_pending = max_pending
        self._max_cached = max_cached
        self._condition = threading.Condition()
        self._pending: deque[str] = deque()
        self._jobs: dict[str, _Record] = {}
        self._sequence = 0
        self._running_id: str | None = None
        self._worker: threading.Thread | None = None
        self._started = False
        self._stop = False

    def start(self) -> None:
        """Start the single worker. A second call does nothing while it is alive."""
        with self._condition:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop = False
            self._started = True
            self._worker = threading.Thread(
                target=self._worker_main,
                name="vocalis-generation",
                daemon=False,
            )
            self._worker.start()
        logger.info(
            "Generation queue started pending_limit=%s cached_jobs=%s",
            self._max_pending,
            self._max_cached,
        )

    def submit(self, request: GenerationRequest) -> str:
        """Queue a request and return its id without waiting for the result."""
        with self._condition:
            if not self._started or self._stop:
                raise QueueUnavailableError("The generation queue is not accepting jobs.")
            if len(self._pending) >= self._max_pending:
                raise QueueFullError(
                    f"The generation queue is full ({self._max_pending} jobs waiting)."
                )
            job_id = uuid.uuid4().hex
            created_at = _utcnow()
            sequence = self._sequence + 1
            record = _Record(
                id=job_id,
                request=request,
                created_at=created_at,
                sequence=sequence,
            )
            self._persist_create(record)
            self._sequence = sequence
            self._jobs[job_id] = record
            self._pending.append(job_id)
            self._trim_cached()
            pending = len(self._pending)
            self._condition.notify()
        logger.info(
            "Job queued id=%s type=%s pending=%s",
            job_id,
            request.generation_type.value,
            pending,
        )
        return job_id

    def get_job(self, job_id: str) -> GenerationJob:
        with self._condition:
            return self._require(job_id).snapshot()

    def snapshot(self) -> QueueSnapshot:
        """Report whether the worker is accepting work and what it is doing."""
        with self._condition:
            return QueueSnapshot(
                accepting=self._started and not self._stop,
                pending_count=len(self._pending),
                running=self._running_id is not None,
                max_pending=self._max_pending,
            )

    def list_jobs(self, *, limit: int = 20) -> tuple[GenerationJob, ...]:
        """Return the newest jobs still in memory."""
        if limit < 1:
            raise QueueUnavailableError("The job list limit must be at least 1.")
        with self._condition:
            records = sorted(self._jobs.values(), key=lambda record: record.sequence, reverse=True)
            return tuple(record.snapshot() for record in records[:limit])

    def cancel(self, job_id: str) -> GenerationJob:
        """Cancel a job that is still queued. A running job cannot be stopped."""
        with self._condition:
            record = self._require(job_id)
            if record.state is JobState.RUNNING:
                raise CancellationUnsupportedError("A running generation cannot be cancelled.")
            if record.state is not JobState.QUEUED:
                raise CancellationUnsupportedError("Only a queued job can be cancelled.")
            index = self._pending_index(job_id)
            self._pending.remove(job_id)
            completed_at = _utcnow()
            try:
                if self._store is not None:
                    self._store.mark_cancelled(job_id, completed_at, None)
            except Exception:
                self._pending = self._insert_pending(job_id, index)
                raise
            record.state = JobState.CANCELLED
            record.completed_at = completed_at
            record.finished.set()
            snapshot = record.snapshot()
            self._trim_cached()
            self._condition.notify_all()
        logger.info("Job cancelled id=%s", job_id)
        return snapshot

    def wait(self, job_id: str, timeout: float) -> GenerationJob:
        """Block until the job reaches a terminal state or the timeout expires."""
        with self._condition:
            record = self._require(job_id)
            finished = record.finished
        if not finished.wait(timeout):
            raise TimeoutError(f"Timed out waiting for generation job {job_id}.")
        with self._condition:
            current = self._jobs.get(job_id, record)
            return current.snapshot()

    def shutdown(self, *, timeout: float = 5.0) -> None:
        """Stop accepting jobs, cancel work that has not started, and join the worker.

        A job that is already running is allowed to finish. It is not cancelled.
        """
        with self._condition:
            if not self._started and (self._worker is None or not self._worker.is_alive()):
                return
            self._stop = True
            while self._pending:
                job_id = self._pending.popleft()
                record = self._jobs[job_id]
                if record.state is JobState.QUEUED:
                    completed_at = _utcnow()
                    message = "The generation queue shut down before this job started."
                    try:
                        if self._store is not None:
                            self._store.mark_cancelled(job_id, completed_at, message)
                    except Exception:
                        logger.exception("Job history was not marked cancelled id=%s", job_id)
                    record.state = JobState.CANCELLED
                    record.completed_at = completed_at
                    record.error_message = message
                    record.finished.set()
                    logger.info("Job cancelled id=%s reason=shutdown", job_id)
            worker = self._worker
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                raise QueueUnavailableError("The generation worker did not stop.")
        with self._condition:
            self._started = False
            self._worker = None
            self._running_id = None
        logger.info("Generation queue stopped")

    def _require(self, job_id: str) -> _Record:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise UnknownJobError("No generation job exists with that id.") from exc

    def _worker_main(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stop:
                    self._condition.wait()
                if not self._pending:
                    return
                job_id = self._pending.popleft()
                record = self._jobs[job_id]
                started_at = _utcnow()
                record.state = JobState.RUNNING
                record.started_at = started_at
                self._running_id = job_id
                request = record.request
            logger.info("Job running id=%s", job_id)
            if not self._persist_running(job_id, started_at):
                continue
            try:
                result = self._manager.generate(request)
            except VocalisError as exc:
                logger.error("Job failed id=%s error=%s", job_id, exc)
                self._finish_failed(job_id, str(exc))
            except Exception:
                logger.exception("Job failed id=%s", job_id)
                self._finish_failed(job_id, "Generation failed.")
            else:
                self._finish_succeeded(job_id, result)

    def _persist_create(self, record: _Record) -> None:
        if self._store is None:
            return
        request = record.request
        reference = request.reference_audio_path
        self._store.create(
            NewGeneration(
                id=record.id,
                generation_type=request.generation_type,
                provider=self._manager.capabilities.provider,
                input_text=request.text,
                voice_description=request.voice_description,
                reference_audio_path=None if reference is None else str(reference),
                reference_transcript=request.reference_transcript,
                parameters=request.resolved_parameters(),
                created_at=record.created_at,
            )
        )

    def _persist_running(self, job_id: str, started_at: datetime) -> bool:
        """Return false when the running transition could not be stored.

        Inference does not start in that case. The database connection is
        closed before ``generate`` is called.
        """
        if self._store is None:
            return True
        try:
            self._store.mark_running(job_id, started_at)
        except Exception:
            logger.exception("Job history was not marked running id=%s", job_id)
            self._finish_failed(
                job_id,
                "The generation could not be started because its history could not be saved.",
            )
            return False
        return True

    def _finish_succeeded(self, job_id: str, result: GenerationResult) -> None:
        completed_at = _utcnow()
        if self._store is not None:
            try:
                self._store.mark_succeeded(job_id, completed_at, result)
            except Exception:
                logger.exception("Job history was not marked succeeded id=%s", job_id)
                self._finish_failed(
                    job_id,
                    "The generation finished, but its history could not be saved.",
                )
                return
        with self._condition:
            record = self._jobs[job_id]
            record.state = JobState.SUCCEEDED
            record.result = result
            record.completed_at = completed_at
            record.finished.set()
            if self._running_id == job_id:
                self._running_id = None
            self._trim_cached()
            self._condition.notify_all()
        logger.info("Job succeeded id=%s", job_id)

    def _finish_failed(self, job_id: str, message: str) -> None:
        completed_at = _utcnow()
        if self._store is not None:
            try:
                self._store.mark_failed(job_id, completed_at, message)
            except Exception:
                logger.exception("Job history was not marked failed id=%s", job_id)
        with self._condition:
            record = self._jobs[job_id]
            record.state = JobState.FAILED
            record.error_message = message
            record.completed_at = completed_at
            record.finished.set()
            if self._running_id == job_id:
                self._running_id = None
            self._trim_cached()
            self._condition.notify_all()

    def _trim_cached(self) -> None:
        """Drop the oldest finished jobs until the cache fits. Caller holds the lock."""
        if len(self._jobs) <= self._max_cached:
            return
        finished = sorted(
            (
                record
                for record in self._jobs.values()
                if record.state in _TERMINAL_STATES and record.finished.is_set()
            ),
            key=lambda record: record.sequence,
        )
        for record in finished:
            if len(self._jobs) <= self._max_cached:
                return
            del self._jobs[record.id]
            logger.debug("Job left the memory cache id=%s", record.id)

    def _pending_index(self, job_id: str) -> int:
        for index, pending_id in enumerate(self._pending):
            if pending_id == job_id:
                return index
        return 0

    def _insert_pending(self, job_id: str, index: int) -> deque[str]:
        items = list(self._pending)
        items.insert(index, job_id)
        return deque(items)
