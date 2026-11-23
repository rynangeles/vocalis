"""Studio pages call services and do not run inference themselves."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import pytest

from vocalis.core.config import load_settings
from vocalis.engine.mock_engine import MockInferenceEngine
from vocalis.engine.model_manager import ModelManager
from vocalis.services.history import HistoryService
from vocalis.services.queue import GenerationQueue
from vocalis.persistence.database import Database
from vocalis.persistence.repositories.generations import GenerationRepository
from vocalis.ui.actions import (
    cancel_job,
    delete_history,
    history_detail,
    history_page,
    inspect_job,
    inspect_reference,
    monitor_text,
    settings_text,
    submit_clone,
    submit_design,
)
from vocalis.ui.runtime import StudioServices
from vocalis.ui.studio import button_variants, create_app, page_visibility
from vocalis.utils.audio import write_pcm_wav
from tests.unit.test_queue import GateEngine


def _settings(tmp_path: Path):
    return load_settings(
        environ={
            "MODEL_PROVIDER": "mock",
            "VOCALIS_DATA_DIR": str(tmp_path / "data"),
            "VOCALIS_OUTPUT_DIR": str(tmp_path / "output"),
            "VOCALIS_UPLOAD_DIR": str(tmp_path / "uploads"),
            "VOCALIS_LOG_DIR": str(tmp_path / "logs"),
            "VOCALIS_DATABASE_PATH": str(tmp_path / "data" / "history.db"),
            "MAX_QUEUE_SIZE": "3",
            "MAX_CONCURRENT_GENERATIONS": "1",
        },
        project_root=tmp_path,
    )


def _services(tmp_path: Path, engine=None) -> tuple[StudioServices, GenerationRepository]:
    settings = _settings(tmp_path)
    database = Database(settings.database_path)
    database.migrate()
    repository = GenerationRepository(database)
    history = HistoryService(repository)
    selected = engine or MockInferenceEngine(settings.design_output_dir, settings.clone_output_dir)
    manager = ModelManager(selected)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=settings.max_queue_size, store=repository)
    queue.start()
    return (
        StudioServices(
            settings=settings,
            queue=queue,
            history=history,
            manager=manager,
            started_at=time.monotonic(),
        ),
        repository,
    )


def _close(services: StudioServices, gate: GateEngine | None = None) -> None:
    if gate is not None:
        gate.allow(10_000)
    services.queue.shutdown()
    services.manager.shutdown()


def _wav(path: Path) -> bytes:
    write_pcm_wav(path, b"\x00\x01" * 160, sample_rate=16_000, simulated=False, disclaimer="")
    return path.read_bytes()


def _wait(services: StudioServices, job_id: str, label: str) -> object:
    deadline = time.monotonic() + 3
    view = inspect_job(services, job_id)
    while time.monotonic() < deadline and view.status_label != label:
        time.sleep(0.02)
        view = inspect_job(services, job_id)
    return view


def _button(demo: gr.Blocks, label: str):
    matches = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Button) and block.value == label
    ]
    assert len(matches) == 1
    return matches[0]


def _click(demo: gr.Blocks, label: str):
    button = _button(demo, label)
    return next(
        item.fn
        for item in demo.fns.values()
        if item.fn is not None and (button._id, "click") in item.targets
    )


def test_theme_builds() -> None:
    from vocalis.ui.theme import studio_theme

    theme = studio_theme()
    assert theme is not None


def test_shell_has_pages_and_two_timers(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    try:
        demo = create_app(services.settings, services)
        labels = {block.value for block in demo.blocks.values() if isinstance(block, gr.Button)}
        timers = [block for block in demo.blocks.values() if isinstance(block, gr.Timer)]
        assert demo.title == "Vocalis — AI Voice Studio"
        assert {
            "Voice Design",
            "Voice Cloning",
            "Generation History",
            "System Monitor",
            "Settings",
            "Generate",
            "Cancel pending job",
            "Confirm delete",
        } <= labels
        confirm = _button(demo, "Confirm delete")
        remove = _button(demo, "Delete history record")
        wired = {
            block_id
            for item in demo.fns.values()
            if item.fn is not None
            for block_id, event in item.targets
            if event == "click" and block_id in {confirm._id, remove._id}
        }
        assert wired == {confirm._id, remove._id}
        assert len(timers) == 2
        assert {block.value for block in timers} == {2, 5}
        assert all(block.active is False for block in timers)
        shell = next(block.value for block in demo.blocks.values() if isinstance(block, gr.HTML))
        assert "Give your words a voice." in shell
        assert "Powered by VoxCPM2" in shell
        assert "simulated tone, not VoxCPM2 speech" in shell
    finally:
        _close(services)


def test_navigation_switches_pages_without_submitting(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    try:
        demo = create_app(services.settings, services)
        assert page_visibility("history") == [False, False, True, False, False]
        assert button_variants("monitor")[3] == "primary"
        before = services.history.count()
        opened = _click(demo, "System Monitor")(0, "Any type", "Any status", "Any provider")
        visible = [item["visible"] for item in opened[:5]]
        assert visible == [False, False, False, True, False]
        assert opened[11]["active"] is True
        assert "CPU:" in opened[-2]
        assert "GPU monitoring is not configured." in opened[-2]
        assert str(tmp_path) not in opened[-2]
        assert services.history.count() == before
    finally:
        _close(services)


def test_voice_design_returns_before_inference_finishes(tmp_path: Path) -> None:
    gate = GateEngine()
    services, _repository = _services(tmp_path, gate)
    try:
        started = time.monotonic()
        view = submit_design(services, "Hold this line.", "warm", 2, 10, False, False)
        elapsed = time.monotonic() - started
        assert elapsed < 1
        assert view.job_id
        assert view.status_label in {"Queued", "Running"}
        assert view.refresh is True
        assert "Progress is not reported." in view.detail
        before = services.history.count()
        again = inspect_job(services, view.job_id)
        assert again.job_id == view.job_id
        assert services.history.count() == before
    finally:
        _close(services, gate)


def test_voice_design_shows_simulated_audio(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    try:
        submitted = submit_design(services, "Hello from Vocalis.", "", 2, 10, True, False)
        finished = _wait(services, submitted.job_id, "Succeeded · simulated")
        assert finished.audio_path is not None
        assert Path(finished.audio_path).is_file()
        assert "Simulated tone. This is not VoxCPM2 speech." in finished.detail
        assert "Vocalis simulated audio." in finished.detail
        assert finished.refresh is False
        assert finished.can_cancel is False
    finally:
        _close(services)


def test_voice_design_validation_does_not_queue(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    try:
        view = submit_design(services, "   ", "", 2, 10, False, False)
        assert view.job_id == ""
        assert view.status_label == "Needs attention"
        assert view.detail == "Text to generate must not be empty."
        assert services.history.count() == 0
    finally:
        _close(services)


def test_failed_job_hides_the_exception_text(tmp_path: Path) -> None:
    gate = GateEngine()
    gate.errors.append(RuntimeError("SECRET_STACK"))
    services, _repository = _services(tmp_path, gate)
    try:
        view = submit_design(services, "This will fail.", "", 2, 10, False, False)
        gate.allow(1)
        finished = _wait(services, view.job_id, "Failed")
        assert "Generation failed." in finished.detail
        assert "SECRET_STACK" not in finished.detail
        assert finished.audio_path is None
    finally:
        _close(services, gate)


def test_pending_job_can_be_cancelled(tmp_path: Path) -> None:
    gate = GateEngine()
    services, _repository = _services(tmp_path, gate)
    try:
        running = submit_design(services, "Keep the worker busy.", "", 2, 10, False, False)
        gate.wait_started(1)
        waiting = submit_design(services, "Cancel me.", "", 2, 10, False, False)
        assert waiting.can_cancel is True
        assert waiting.status_label == "Queued"
        cancelled = cancel_job(services, waiting.job_id)
        assert cancelled.status_label == "Cancelled"
        assert cancelled.can_cancel is False
        assert inspect_job(services, running.job_id).status_label == "Running"
    finally:
        _close(services, gate)


def test_voice_cloning_submits_without_changing_the_reference(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    reference = tmp_path / "uploads" / "references" / "speaker.wav"
    original = _wav(reference)
    try:
        message, preview = inspect_reference(services, reference)
        assert "WAV" in message
        assert "does not use it to clone a voice" in message
        assert preview == str(reference)
        submitted = submit_clone(services, "Read this line.", reference, "optional words", 2, 10, False, True)
        finished = _wait(services, submitted.job_id, "Succeeded · simulated")
        assert reference.read_bytes() == original
        assert "not used to clone a voice" in finished.detail
        assert finished.audio_path is not None
        assert Path(finished.audio_path) != reference
        record = services.history.get(submitted.job_id)
        assert record is not None and record.reference_audio_path is not None
        stored = Path(record.reference_audio_path)
        assert stored != reference
        assert stored.parent == reference.parent
        assert stored.name.startswith("reference-")
        assert stored.read_bytes() == original
    finally:
        _close(services)


def test_voice_cloning_reports_a_missing_reference(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    try:
        view = submit_clone(services, "Read this line.", None, "", 2, 10, False, False)
        assert view.status_label == "Needs attention"
        assert view.detail == "Voice cloning requires a reference audio file."
        message, preview = inspect_reference(services, None)
        assert message == "Upload a WAV, FLAC, or MP3 reference."
        assert preview is None
    finally:
        _close(services)


def test_unknown_job_uses_history_after_the_queue_forgets_it(tmp_path: Path) -> None:
    services, repository = _services(tmp_path)
    try:
        submitted = submit_design(services, "Remember this line.", "", 2, 10, False, False)
        finished = _wait(services, submitted.job_id, "Succeeded · simulated")
        services.queue.shutdown()
        replacement = GenerationQueue(
            services.manager,
            max_pending=services.settings.max_queue_size,
            store=repository,
        )
        restarted = StudioServices(
            settings=services.settings,
            queue=replacement,
            history=services.history,
            manager=services.manager,
            started_at=services.started_at,
        )
        restored = inspect_job(restarted, finished.job_id)
        assert restored.status_label == "Succeeded · simulated"
        assert "Remember this line." in restored.detail
        missing = inspect_job(restarted, "job-missing")
        assert missing.status_label == "Unknown"
        assert missing.detail == "No generation with that id is in this session or in history."
    finally:
        services.queue.shutdown()
        services.manager.shutdown()


def test_history_lists_filters_and_paginates(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    reference = tmp_path / "uploads" / "references" / "speaker.wav"
    _wav(reference)
    try:
        for index in range(8):
            submitted = submit_design(services, f"design line {index}", "", 2, 10, False, False)
            _wait(services, submitted.job_id, "Succeeded · simulated")
        cloned = submit_clone(services, "clone line", reference, "", 2, 10, False, False)
        _wait(services, cloned.job_id, "Succeeded · simulated")
        first = history_page(services, offset=0, type_label="Any type", status_label="Any status", provider_label="Any provider")
        assert first.summary == "Showing 1–8 of 9. Newest first."
        assert first.rows[0][1] == "Voice cloning"
        assert first.rows[0][2] == "clone line"
        second = history_page(services, offset=8, type_label=None, status_label=None, provider_label=None)
        assert second.summary == "Showing 9–9 of 9. Newest first."
        assert len(second.rows) == 1
        designs = history_page(services, offset=0, type_label="Voice design", status_label="Succeeded", provider_label="mock")
        assert designs.summary == "Showing 1–8 of 8. Newest first."
        assert {row[1] for row in designs.rows} == {"Voice design"}
        empty = history_page(services, offset=0, type_label="Voice cloning", status_label="Failed", provider_label="Any provider")
        assert empty.rows == []
        assert empty.summary == "No generations match these filters."
    finally:
        _close(services)


def test_history_detail_and_confirmed_delete_keep_the_audio(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    long_text = "A" * 90
    try:
        submitted = submit_design(services, long_text, "a calm voice", 2, 10, False, False)
        finished = _wait(services, submitted.job_id, "Succeeded · simulated")
        page = history_page(services, offset=0, type_label="Any type", status_label="Any status", provider_label="Any provider")
        assert page.rows[0][2].endswith("…")
        assert len(page.rows[0][2]) == 72
        detail, audio = history_detail(services, finished.job_id)
        assert long_text in detail
        assert audio == finished.audio_path
        assert "This result is simulated." in detail
        output = Path(audio)
        output.unlink()
        missing_detail, missing_audio = history_detail(services, finished.job_id)
        assert missing_audio is None
        assert "no longer available" in missing_detail
        output.write_bytes(b"kept")
        first = delete_history(services, finished.job_id, "")
        assert first.deleted is False
        assert "Confirm delete" in first.message
        assert services.history.count() == 1
        assert output.read_bytes() == b"kept"
        second = delete_history(services, finished.job_id, first.armed_id)
        assert second.deleted is True
        assert "left on disk" in second.message
        assert services.history.count() == 0
        assert output.read_bytes() == b"kept"
        gone, gone_audio = history_detail(services, finished.job_id)
        assert gone_audio is None
        assert "no longer available" in gone
    finally:
        _close(services)


def test_settings_and_monitor_hide_paths(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    try:
        settings = settings_text(services)
        monitor = monitor_text(services)
        assert "Version: Not set" in settings
        assert "Maximum concurrent generations: 1" in settings
        assert "Pending queue limit: 3" in settings
        assert "Maximum upload size: 20 MiB" in settings
        assert "Output format: WAV" in settings
        assert "Sample rate: 16000" in settings
        assert "History database: Available" in settings
        assert "Uptime:" in monitor
        assert "CPU:" in monitor
        assert "Memory:" in monitor
        assert "Pending jobs: 0" in monitor
        assert "Running: no" in monitor
        assert "GPU monitoring is not configured." in monitor
        assert str(tmp_path) not in settings
        assert str(tmp_path) not in monitor
        services.queue.shutdown()
        assert "Queue accepting jobs: no" in monitor_text(services)
    finally:
        services.manager.shutdown()


def test_empty_history_and_idle_job(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    try:
        page = history_page(services, offset=0, type_label="Any type", status_label="Any status", provider_label="Any provider")
        assert page.summary == "No generations yet."
        idle = inspect_job(services, "")
        assert idle.status_label == "Idle"
        assert "does not submit a new job" in idle.detail
        assert cancel_job(services, "").detail == "There is no pending job to cancel."
    finally:
        _close(services)
