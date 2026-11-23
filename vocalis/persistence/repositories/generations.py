"""Parameterized SQL for generation history."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

from vocalis.core.exceptions import HistoryQueryError, PersistenceError
from vocalis.core.types import ModelProvider
from vocalis.engine.types import GenerationResult, GenerationType, InferenceParameters
from vocalis.persistence.database import Database
from vocalis.persistence.models import GenerationRecord, NewGeneration
from vocalis.persistence.parameters import dump_parameters, load_parameters
from vocalis.services.jobs import JobState

logger = logging.getLogger("vocalis.persistence")

INTERRUPTED_MESSAGE = "Generation interrupted because Vocalis restarted before completion."
MAX_HISTORY_PAGE_SIZE = 100
DEFAULT_HISTORY_PAGE_SIZE = 20

_COLUMNS = """
    id, generation_type, provider, input_text, voice_description,
    reference_audio_path, reference_transcript, parameters_json, status,
    created_at, started_at, completed_at, output_audio_path, sample_rate,
    channels, sample_width_bytes, frame_count, duration_seconds, audio_format,
    simulated, audio_disclaimer, generation_seconds, error_message
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GenerationRepository:
    """Create and update generation rows. Each method opens its own connection."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def create(self, generation: NewGeneration) -> None:
        with self._database.transaction() as connection:
            try:
                connection.execute(
                    f"""
                    INSERT INTO generations ({_COLUMNS})
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL,
                            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
                    """,
                    (
                        generation.id,
                        generation.generation_type.value,
                        generation.provider.value,
                        generation.input_text,
                        generation.voice_description,
                        generation.reference_audio_path,
                        generation.reference_transcript,
                        dump_parameters(generation.parameters),
                        JobState.QUEUED.value,
                        _iso(generation.created_at),
                    ),
                )
            except sqlite3.Error as exc:
                logger.exception("Generation history insert failed id=%s", generation.id)
                raise PersistenceError("The generation could not be saved.") from exc
        logger.info(
            "Generation record created id=%s type=%s provider=%s",
            generation.id,
            generation.generation_type.value,
            generation.provider.value,
        )

    def mark_running(self, job_id: str, started_at: datetime) -> None:
        self._update(
            job_id,
            """
            UPDATE generations
            SET status = ?, started_at = ?
            WHERE id = ? AND status = ?
            """,
            (JobState.RUNNING.value, _iso(started_at), job_id, JobState.QUEUED.value),
            JobState.RUNNING,
        )

    def mark_succeeded(self, job_id: str, completed_at: datetime, result: GenerationResult) -> None:
        audio = result.audio
        self._update(
            job_id,
            """
            UPDATE generations
            SET status = ?, completed_at = ?, output_audio_path = ?, sample_rate = ?,
                channels = ?, sample_width_bytes = ?, frame_count = ?,
                duration_seconds = ?, audio_format = ?, simulated = ?,
                audio_disclaimer = ?, generation_seconds = ?, error_message = NULL
            WHERE id = ? AND status = ?
            """,
            (
                JobState.SUCCEEDED.value,
                _iso(completed_at),
                str(result.output_path),
                audio.sample_rate,
                audio.channels,
                audio.sample_width_bytes,
                audio.frame_count,
                audio.duration_seconds,
                audio.audio_format,
                1 if result.simulated else 0,
                audio.disclaimer,
                result.generation_seconds,
                job_id,
                JobState.RUNNING.value,
            ),
            JobState.SUCCEEDED,
        )

    def mark_failed(self, job_id: str, completed_at: datetime, error_message: str) -> None:
        self._update(
            job_id,
            """
            UPDATE generations
            SET status = ?, completed_at = ?, error_message = ?
            WHERE id = ? AND status IN (?, ?)
            """,
            (
                JobState.FAILED.value,
                _iso(completed_at),
                error_message,
                job_id,
                JobState.QUEUED.value,
                JobState.RUNNING.value,
            ),
            JobState.FAILED,
        )

    def mark_cancelled(
        self,
        job_id: str,
        completed_at: datetime,
        error_message: str | None,
    ) -> None:
        self._update(
            job_id,
            """
            UPDATE generations
            SET status = ?, completed_at = ?, error_message = ?
            WHERE id = ? AND status = ?
            """,
            (
                JobState.CANCELLED.value,
                _iso(completed_at),
                error_message,
                job_id,
                JobState.QUEUED.value,
            ),
            JobState.CANCELLED,
        )

    def interrupt_incomplete(self, *, completed_at: datetime | None = None) -> int:
        """Mark leftover queued and running rows as failed. Do not rerun them."""
        moment = completed_at or _utcnow()
        with self._database.transaction() as connection:
            try:
                cursor = connection.execute(
                    """
                    UPDATE generations
                    SET status = ?, completed_at = ?, error_message = ?
                    WHERE status IN (?, ?)
                    """,
                    (
                        JobState.FAILED.value,
                        _iso(moment),
                        INTERRUPTED_MESSAGE,
                        JobState.QUEUED.value,
                        JobState.RUNNING.value,
                    ),
                )
                updated = int(cursor.rowcount)
            except sqlite3.Error as exc:
                logger.exception("Could not recover interrupted generations")
                raise PersistenceError("Interrupted generations could not be recovered.") from exc
        if updated:
            logger.info("Interrupted generations recovered count=%s", updated)
        return updated

    def get(self, job_id: str) -> GenerationRecord | None:
        with self._database.connect() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM generations WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return _record(row)

    def list_recent(
        self,
        *,
        limit: int = DEFAULT_HISTORY_PAGE_SIZE,
        offset: int = 0,
        generation_type: GenerationType | None = None,
        status: JobState | None = None,
        provider: ModelProvider | None = None,
    ) -> tuple[GenerationRecord, ...]:
        limit, offset = _page(limit, offset)
        where, params = _filters(generation_type, status, provider)
        query = (
            f"SELECT {_COLUMNS} FROM generations{where} "
            "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        with self._database.connect() as connection:
            rows = connection.execute(query, (*params, limit, offset)).fetchall()
        return tuple(_record(row) for row in rows)

    def count(
        self,
        *,
        generation_type: GenerationType | None = None,
        status: JobState | None = None,
        provider: ModelProvider | None = None,
    ) -> int:
        where, params = _filters(generation_type, status, provider)
        with self._database.connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS total FROM generations{where}",
                params,
            ).fetchone()
        if row is None:
            return 0
        return int(row["total"])

    def delete(self, job_id: str) -> bool:
        """Delete the history row. Associated audio files are left on disk."""
        with self._database.transaction() as connection:
            try:
                cursor = connection.execute("DELETE FROM generations WHERE id = ?", (job_id,))
            except sqlite3.Error as exc:
                logger.exception("Generation history delete failed id=%s", job_id)
                raise PersistenceError("The generation record could not be deleted.") from exc
        deleted = cursor.rowcount == 1
        if deleted:
            logger.info("Generation record deleted id=%s", job_id)
        return deleted

    def _update(self, job_id: str, sql: str, params: tuple[object, ...], status: JobState) -> None:
        with self._database.transaction() as connection:
            try:
                cursor = connection.execute(sql, params)
            except sqlite3.Error as exc:
                logger.exception("Generation history update failed id=%s status=%s", job_id, status.value)
                raise PersistenceError("The generation could not be saved.") from exc
            if cursor.rowcount != 1:
                raise PersistenceError("The generation record was not updated.")
        logger.info("Generation record updated id=%s status=%s", job_id, status.value)


def _filters(
    generation_type: GenerationType | None,
    status: JobState | None,
    provider: ModelProvider | None,
) -> tuple[str, tuple[str, ...]]:
    clauses: list[str] = []
    params: list[str] = []
    kind = _enum(generation_type, GenerationType, "Generation type")
    if kind is not None:
        clauses.append("generation_type = ?")
        params.append(kind.value)
    state = _enum(status, JobState, "Status")
    if state is not None:
        clauses.append("status = ?")
        params.append(state.value)
    selected = _enum(provider, ModelProvider, "Provider")
    if selected is not None:
        clauses.append("provider = ?")
        params.append(selected.value)
    if not clauses:
        return "", tuple(params)
    return " WHERE " + " AND ".join(clauses), tuple(params)


def _enum(value: object, enum_cls: type, label: str):
    if value is None:
        return None
    if not isinstance(value, enum_cls):
        raise HistoryQueryError(f"{label} is not a valid history filter.")
    return value


def _page(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_HISTORY_PAGE_SIZE:
        raise HistoryQueryError(
            f"History limit must be an integer from 1 to {MAX_HISTORY_PAGE_SIZE}."
        )
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise HistoryQueryError("History offset must be an integer of 0 or greater.")
    return limit, offset


def _record(row: sqlite3.Row) -> GenerationRecord:
    try:
        return GenerationRecord(
            id=row["id"],
            generation_type=GenerationType(row["generation_type"]),
            provider=ModelProvider(row["provider"]),
            input_text=row["input_text"],
            voice_description=row["voice_description"],
            reference_audio_path=row["reference_audio_path"],
            reference_transcript=row["reference_transcript"],
            parameters=load_parameters(row["parameters_json"]),
            status=JobState(row["status"]),
            created_at=_required_time(row["created_at"]),
            started_at=_optional_time(row["started_at"]),
            completed_at=_optional_time(row["completed_at"]),
            output_audio_path=row["output_audio_path"],
            sample_rate=row["sample_rate"],
            channels=row["channels"],
            sample_width_bytes=row["sample_width_bytes"],
            frame_count=row["frame_count"],
            duration_seconds=row["duration_seconds"],
            audio_format=row["audio_format"],
            simulated=_optional_bool(row["simulated"]),
            audio_disclaimer=row["audio_disclaimer"],
            generation_seconds=row["generation_seconds"],
            error_message=row["error_message"],
        )
    except (KeyError, ValueError) as exc:
        raise PersistenceError("A stored generation record could not be read.") from exc


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise PersistenceError("Generation timestamps must include a timezone.")
    return value.isoformat()


def _required_time(value: str) -> datetime:
    parsed = _optional_time(value)
    if parsed is None:
        raise PersistenceError("A stored generation record could not be read.")
    return parsed


def _optional_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PersistenceError("A stored generation record could not be read.") from exc
    if parsed.tzinfo is None:
        raise PersistenceError("A stored generation record could not be read.")
    return parsed


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if value in (0, 1):
        return bool(value)
    raise PersistenceError("A stored generation record could not be read.")
