"""Generation repository reads, writes, filters, and deletion."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vocalis.core.exceptions import HistoryQueryError, PersistenceError
from vocalis.core.types import ModelProvider
from vocalis.engine.types import (
    AudioMetadata,
    GenerationResult,
    GenerationType,
    InferenceParameters,
)
from vocalis.persistence.database import Database
from vocalis.persistence.models import NewGeneration
from vocalis.persistence.parameters import dump_parameters
from vocalis.persistence.repositories.generations import (
    INTERRUPTED_MESSAGE,
    GenerationRepository,
)
from vocalis.services.history import HistoryService
from vocalis.services.jobs import JobState


def _checkpoint_bytes(path: Path) -> bytes:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        connection.close()
    blob = path.read_bytes()
    wal = Path(str(path) + "-wal")
    if wal.exists():
        blob += wal.read_bytes()
    return blob


def _repository(tmp_path: Path) -> GenerationRepository:
    database = Database(tmp_path / "history.db")
    database.migrate()
    return GenerationRepository(database)


def _moment(offset: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset)


def _new(
    job_id: str,
    *,
    offset: int,
    generation_type: GenerationType = GenerationType.VOICE_DESIGN,
    provider: ModelProvider = ModelProvider.MOCK,
    text: str = "Hello",
) -> NewGeneration:
    return NewGeneration(
        id=job_id,
        generation_type=generation_type,
        provider=provider,
        input_text=text,
        voice_description="warm" if generation_type is GenerationType.VOICE_DESIGN else None,
        reference_audio_path="/tmp/reference.wav" if generation_type is GenerationType.VOICE_CLONING else None,
        reference_transcript="spoken" if generation_type is GenerationType.VOICE_CLONING else None,
        parameters=InferenceParameters(cfg_value=1.25, inference_timesteps=7, normalize=True),
        created_at=_moment(offset),
    )


def _result(job_id: str, output: Path) -> GenerationResult:
    parameters = InferenceParameters(cfg_value=1.25, inference_timesteps=7, normalize=True)
    return GenerationResult(
        provider=ModelProvider.MOCK,
        generation_type=GenerationType.VOICE_DESIGN,
        output_path=output,
        audio=AudioMetadata(
            sample_rate=16_000,
            channels=1,
            sample_width_bytes=2,
            frame_count=1600,
            duration_seconds=0.1,
            audio_format="wav",
            simulated=True,
            disclaimer="simulated",
        ),
        simulated=True,
        generation_seconds=0.05,
        parameters=parameters,
        text="Hello",
        voice_description="warm",
        reference_audio_path=None,
        reference_transcript=None,
        message="ok",
    )


def test_create_round_trip_keeps_parameters_and_omits_audio_bytes(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    created = _new("job-1", offset=1, text="UNIQUE_PROMPT_TEXT")
    repository.create(created)

    stored = repository.get("job-1")
    assert stored is not None
    assert stored.status is JobState.QUEUED
    assert stored.input_text == "UNIQUE_PROMPT_TEXT"
    assert stored.parameters == InferenceParameters(cfg_value=1.25, inference_timesteps=7, normalize=True, denoise=False)
    assert stored.started_at is None
    assert stored.output_audio_path is None
    assert dump_parameters(stored.parameters) == (
        '{"cfg_value":1.25,"denoise":false,"inference_timesteps":7,"normalize":true}'
    )
    database_bytes = _checkpoint_bytes(tmp_path / "history.db")
    assert b"UNIQUE_PROMPT_TEXT" in database_bytes
    assert b"RIFF" not in database_bytes


def test_status_transitions_record_success_failure_and_cancellation(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.create(_new("done", offset=1))
    repository.create(_new("bad", offset=2))
    repository.create(_new("dropped", offset=3))
    output = tmp_path / "tone.wav"
    output.write_bytes(b"RIFFtone-bytes-not-sql")

    repository.mark_running("done", _moment(10))
    repository.mark_succeeded("done", _moment(11), _result("done", output))
    repository.mark_running("bad", _moment(12))
    repository.mark_failed("bad", _moment(13), "The reference could not be used.")
    repository.mark_cancelled("dropped", _moment(14), None)

    done = repository.get("done")
    bad = repository.get("bad")
    dropped = repository.get("dropped")
    assert done is not None and bad is not None and dropped is not None
    assert done.status is JobState.SUCCEEDED
    assert done.output_audio_path == str(output)
    assert done.duration_seconds == pytest.approx(0.1)
    assert done.sample_rate == 16_000
    assert done.simulated is True
    assert done.generation_seconds == pytest.approx(0.05)
    assert done.error_message is None
    assert bad.status is JobState.FAILED
    assert bad.error_message == "The reference could not be used."
    assert "Traceback" not in (bad.error_message or "")
    assert dropped.status is JobState.CANCELLED
    assert dropped.started_at is None
    assert dropped.error_message is None
    assert output.read_bytes() == b"RIFFtone-bytes-not-sql"
    assert b"tone-bytes-not-sql" not in _checkpoint_bytes(tmp_path / "history.db")


def test_history_pages_filters_and_counts(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    history = HistoryService(repository)
    rows = (
        ("a", 1, GenerationType.VOICE_DESIGN, ModelProvider.MOCK, JobState.SUCCEEDED),
        ("b", 2, GenerationType.VOICE_CLONING, ModelProvider.MOCK, JobState.FAILED),
        ("c", 3, GenerationType.VOICE_DESIGN, ModelProvider.VOXCPM, JobState.SUCCEEDED),
        ("d", 4, GenerationType.VOICE_CLONING, ModelProvider.VOXCPM, JobState.CANCELLED),
        ("e", 5, GenerationType.VOICE_DESIGN, ModelProvider.MOCK, JobState.QUEUED),
    )
    for job_id, offset, generation_type, provider, _status in rows:
        repository.create(
            _new(job_id, offset=offset, generation_type=generation_type, provider=provider)
        )
    repository.mark_running("a", _moment(20))
    repository.mark_succeeded("a", _moment(21), _result("a", tmp_path / "a.wav"))
    repository.mark_running("b", _moment(22))
    repository.mark_failed("b", _moment(23), "failed")
    repository.mark_running("c", _moment(24))
    repository.mark_succeeded("c", _moment(25), _result("c", tmp_path / "c.wav"))
    repository.mark_cancelled("d", _moment(26), "cancelled")

    page = history.list_recent(limit=2, offset=0)
    assert [record.id for record in page] == ["e", "d"]
    assert [record.id for record in history.list_recent(limit=2, offset=2)] == ["c", "b"]
    assert history.list_recent(limit=2, offset=8) == ()
    assert history.count() == 5
    assert history.count(generation_type=GenerationType.VOICE_CLONING) == 2
    assert [record.id for record in history.list_recent(status=JobState.SUCCEEDED)] == ["c", "a"]
    assert history.count(provider=ModelProvider.VOXCPM, status=JobState.SUCCEEDED) == 1
    assert history.get("missing") is None

    with pytest.raises(HistoryQueryError, match="limit"):
        history.list_recent(limit=0)
    with pytest.raises(HistoryQueryError, match="limit"):
        history.list_recent(limit=101)
    with pytest.raises(HistoryQueryError, match="offset"):
        history.list_recent(offset=-1)
    with pytest.raises(HistoryQueryError, match="Status"):
        history.list_recent(status="queued")  # type: ignore[arg-type]


def test_delete_removes_the_row_and_leaves_audio_files(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    history = HistoryService(repository)
    output = tmp_path / "generated.wav"
    reference = tmp_path / "reference.wav"
    output.write_bytes(b"generated-audio")
    reference.write_bytes(b"reference-audio")
    repository.create(
        NewGeneration(
            id="gone",
            generation_type=GenerationType.VOICE_CLONING,
            provider=ModelProvider.MOCK,
            input_text="Clone",
            voice_description=None,
            reference_audio_path=str(reference),
            reference_transcript="line",
            parameters=InferenceParameters(),
            created_at=_moment(1),
        )
    )
    repository.mark_running("gone", _moment(2))
    repository.mark_succeeded("gone", _moment(3), _result("gone", output))

    assert history.delete("gone") is True
    assert history.get("gone") is None
    assert history.delete("gone") is False
    assert output.read_bytes() == b"generated-audio"
    assert reference.read_bytes() == b"reference-audio"


def test_restart_recovery_marks_incomplete_rows_and_keeps_finished_rows(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.create(_new("waiting", offset=1))
    repository.create(_new("active", offset=2))
    repository.create(_new("kept", offset=3))
    repository.create(_new("stopped", offset=4))
    repository.mark_running("active", _moment(5))
    repository.mark_running("kept", _moment(6))
    repository.mark_succeeded("kept", _moment(7), _result("kept", tmp_path / "kept.wav"))
    repository.mark_cancelled("stopped", _moment(8), None)

    recovered = repository.interrupt_incomplete(completed_at=_moment(30))

    assert recovered == 2
    waiting = repository.get("waiting")
    active = repository.get("active")
    kept = repository.get("kept")
    stopped = repository.get("stopped")
    assert waiting is not None and active is not None and kept is not None and stopped is not None
    assert waiting.status is JobState.FAILED
    assert active.status is JobState.FAILED
    assert waiting.error_message == INTERRUPTED_MESSAGE
    assert active.error_message == INTERRUPTED_MESSAGE
    assert waiting.completed_at == _moment(30)
    assert active.started_at == _moment(5)
    assert kept.status is JobState.SUCCEEDED
    assert kept.error_message is None
    assert stopped.status is JobState.CANCELLED
    assert repository.interrupt_incomplete() == 0


def test_duplicate_create_does_not_replace_the_row(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    repository.create(_new("same", offset=1, text="first"))
    with pytest.raises(PersistenceError, match="could not be saved"):
        repository.create(_new("same", offset=2, text="second"))
    stored = repository.get("same")
    assert stored is not None
    assert stored.input_text == "first"
    assert repository.count() == 1


def test_concurrent_creates_are_all_visible(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait(timeout=2)
            repository.create(_new(f"job-{index}", offset=index))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert errors == []
    assert repository.count() == 8
    assert {record.id for record in repository.list_recent(limit=20)} == {f"job-{index}" for index in range(8)}
