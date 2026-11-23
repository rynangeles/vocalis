"""Studio actions. They call services and do not run inference themselves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vocalis.core.exceptions import (
    CancellationUnsupportedError,
    UnknownJobError,
    VocalisError,
)
from vocalis.engine.types import InferenceParameters
from vocalis.persistence.models import GenerationRecord
from vocalis.services.jobs import GenerationJob, JobState
from vocalis.ui.present import (
    HISTORY_PAGE_SIZE,
    PROVIDER_FILTERS,
    STATUS_FILTERS,
    TYPE_FILTERS,
    format_inspection,
    format_job,
    format_record,
    history_row,
    parse_provider_filter,
    parse_status_filter,
    parse_type_filter,
    playable_file,
    render_monitor,
    render_settings,
)
from vocalis.ui.runtime import StudioServices

_EMPTY_JOB = "No job yet. Generate to queue one. Refresh does not submit a new job."
_MISSING_JOB = "No generation with that id is in this session or in history."


@dataclass(frozen=True, slots=True)
class JobView:
    job_id: str
    detail: str
    audio_path: str | None
    can_cancel: bool
    refresh: bool
    status_label: str


@dataclass(frozen=True, slots=True)
class HistoryPage:
    rows: list[list[str]]
    ids: tuple[str, ...]
    summary: str
    offset: int


@dataclass(frozen=True, slots=True)
class DeleteOutcome:
    deleted: bool
    armed_id: str
    message: str
    selected_id: str


def submit_design(
    services: StudioServices,
    text: str,
    voice_description: str,
    cfg_value: float,
    timesteps: float,
    normalize: bool,
    denoise: bool,
) -> JobView:
    try:
        job_id = services.design.submit(
            text,
            voice_description,
            _parameters(cfg_value, timesteps, normalize, denoise),
        )
    except VocalisError as exc:
        return _error(str(exc))
    return inspect_job(services, job_id)


def submit_clone(
    services: StudioServices,
    text: str,
    reference: str | Path | None,
    transcript: str,
    cfg_value: float,
    timesteps: float,
    normalize: bool,
    denoise: bool,
) -> JobView:
    try:
        job_id = services.cloning.submit(
            text,
            _as_path(reference),
            transcript,
            _parameters(cfg_value, timesteps, normalize, denoise),
        )
    except VocalisError as exc:
        return _error(str(exc))
    return inspect_job(services, job_id)


def inspect_reference(services: StudioServices, reference: str | Path | None) -> tuple[str, str | None]:
    path = _as_path(reference)
    if path is None:
        return "Upload a WAV, FLAC, or MP3 reference.", None
    try:
        inspection = services.audio.inspect(path)
    except VocalisError as exc:
        return str(exc), None
    return format_inspection(inspection), str(inspection.path)


def inspect_job(services: StudioServices, job_id: str) -> JobView:
    token = (job_id or "").strip()
    if not token:
        return JobView("", _EMPTY_JOB, None, False, False, "Idle")
    try:
        job = services.queue.get_job(token)
    except UnknownJobError:
        record = _history_record(services, token)
        if record is None:
            return JobView(token, _MISSING_JOB, None, False, False, "Unknown")
        return _from_record(record)
    except VocalisError as exc:
        return JobView(token, str(exc), None, False, False, "Unavailable")
    return _from_job(job)


def cancel_job(services: StudioServices, job_id: str) -> JobView:
    token = (job_id or "").strip()
    if not token:
        return JobView("", "There is no pending job to cancel.", None, False, False, "Idle")
    try:
        job = services.queue.cancel(token)
    except CancellationUnsupportedError as exc:
        current = inspect_job(services, token)
        detail = f"{exc}\n\n{current.detail}" if current.detail else str(exc)
        return JobView(token, detail, current.audio_path, False, current.refresh, current.status_label)
    except UnknownJobError:
        record = _history_record(services, token)
        if record is None:
            return JobView(token, _MISSING_JOB, None, False, False, "Unknown")
        detail = (
            "This job is no longer in the current session, so it cannot be cancelled.\n\n"
            + format_record(record)
        )
        return _from_record(record, detail=detail, can_cancel=False, refresh=False)
    except VocalisError as exc:
        return JobView(token, str(exc), None, False, False, "Unavailable")
    return _from_job(job)


def history_page(
    services: StudioServices,
    *,
    offset: int,
    type_label: str | None,
    status_label: str | None,
    provider_label: str | None,
) -> HistoryPage:
    if not (
        _known_filter(type_label, TYPE_FILTERS)
        and _known_filter(status_label, STATUS_FILTERS)
        and _known_filter(provider_label, PROVIDER_FILTERS)
    ):
        return HistoryPage([], (), "That filter is not available.", 0)
    generation_type = parse_type_filter(type_label)
    status = parse_status_filter(status_label)
    provider = parse_provider_filter(provider_label)
    try:
        total = services.history.count(
            generation_type=generation_type,
            status=status,
            provider=provider,
        )
        bounded = _bounded_offset(offset, total)
        records = services.history.list_recent(
            limit=HISTORY_PAGE_SIZE,
            offset=bounded,
            generation_type=generation_type,
            status=status,
            provider=provider,
        )
    except VocalisError as exc:
        return HistoryPage([], (), str(exc), 0)
    if total == 0:
        summary = "No generations match these filters." if _filters_active(type_label, status_label, provider_label) else "No generations yet."
        return HistoryPage([], (), summary, 0)
    start = bounded + 1
    end = bounded + len(records)
    summary = f"Showing {start}–{end} of {total}. Newest first."
    return HistoryPage(
        [history_row(record) for record in records],
        tuple(record.id for record in records),
        summary,
        bounded,
    )


def history_detail(services: StudioServices, job_id: str) -> tuple[str, str | None]:
    record = _history_record(services, job_id)
    if record is None:
        return "That history record is no longer available.", None
    return format_record(record), playable_file(record.output_audio_path)


def delete_history(services: StudioServices, selected_id: str, armed_id: str) -> DeleteOutcome:
    selected = (selected_id or "").strip()
    if not selected:
        return DeleteOutcome(False, "", "Select a history record first.", "")
    if armed_id != selected:
        return DeleteOutcome(
            False,
            selected,
            "Confirm delete to remove this history record. The audio files will stay on disk.",
            selected,
        )
    try:
        removed = services.history.delete(selected)
    except VocalisError as exc:
        return DeleteOutcome(False, "", str(exc), selected)
    if not removed:
        return DeleteOutcome(False, "", "That history record is already gone.", "")
    return DeleteOutcome(True, "", "History record deleted. Audio files were left on disk.", "")


def settings_text(services: StudioServices) -> str:
    return render_settings(
        services.settings,
        services.manager,
        services.queue,
        _database_available(services),
    )


def monitor_text(services: StudioServices) -> str:
    cpu, memory_percent, used, total = _resources()
    return render_monitor(
        services.settings,
        services.manager,
        services.queue,
        uptime_seconds=services.uptime_seconds(),
        cpu_percent=cpu,
        memory_percent=memory_percent,
        memory_used_bytes=used,
        memory_total_bytes=total,
        database_available=_database_available(services),
    )


def _from_job(job: GenerationJob) -> JobView:
    audio = None
    if job.result is not None:
        audio = playable_file(str(job.result.output_path))
    live = job.state in {JobState.QUEUED, JobState.RUNNING}
    return JobView(
        job.id,
        format_job(job),
        audio,
        job.state is JobState.QUEUED,
        live,
        _status_word(job.state, job.result.simulated if job.result is not None else None),
    )


def _from_record(
    record: GenerationRecord,
    *,
    detail: str | None = None,
    can_cancel: bool = False,
    refresh: bool | None = None,
) -> JobView:
    live = record.status in {JobState.QUEUED, JobState.RUNNING}
    return JobView(
        record.id,
        detail if detail is not None else format_record(record),
        playable_file(record.output_audio_path),
        can_cancel,
        live if refresh is None else refresh,
        _status_word(record.status, record.simulated),
    )


def _status_word(state: JobState, simulated: bool | None) -> str:
    label = {
        JobState.QUEUED: "Queued",
        JobState.RUNNING: "Running",
        JobState.SUCCEEDED: "Succeeded",
        JobState.FAILED: "Failed",
        JobState.CANCELLED: "Cancelled",
    }[state]
    if state is JobState.SUCCEEDED and simulated is True:
        return "Succeeded · simulated"
    if state is JobState.SUCCEEDED and simulated is False:
        return "Succeeded · real"
    return label


def _error(message: str) -> JobView:
    return JobView("", message, None, False, False, "Needs attention")


def _parameters(cfg_value: float, timesteps: float, normalize: bool, denoise: bool) -> InferenceParameters:
    return InferenceParameters(
        cfg_value=float(cfg_value),
        inference_timesteps=int(timesteps),
        normalize=bool(normalize),
        denoise=bool(denoise),
    )


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.strip():
        return Path(name)
    return None


def _history_record(services: StudioServices, job_id: str) -> GenerationRecord | None:
    try:
        return services.history.get(job_id)
    except VocalisError:
        return None


def _database_available(services: StudioServices) -> bool:
    try:
        services.history.count()
    except VocalisError:
        return False
    return True


def _resources() -> tuple[float | None, float | None, int | None, int | None]:
    try:
        import psutil
    except ImportError:
        return None, None, None, None
    memory = psutil.virtual_memory()
    return (
        float(psutil.cpu_percent(interval=None)),
        float(memory.percent),
        int(memory.used),
        int(memory.total),
    )


def _bounded_offset(offset: int, total: int) -> int:
    if offset < 0 or total <= 0:
        return 0
    last_page = ((total - 1) // HISTORY_PAGE_SIZE) * HISTORY_PAGE_SIZE
    if offset > last_page:
        return last_page
    return offset


def _known_filter(label: str | None, choices: tuple[str, ...]) -> bool:
    return label is None or label == "" or label in choices


def _filters_active(type_label: str | None, status_label: str | None, provider_label: str | None) -> bool:
    return any(
        label not in {None, "", "Any type", "Any status", "Any provider"}
        for label in (type_label, status_label, provider_label)
    )
