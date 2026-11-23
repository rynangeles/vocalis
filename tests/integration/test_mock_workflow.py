"""A mock generation crosses the studio, queue, history, and audio file."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import gradio as gr
import pytest

from vocalis.core.config import load_settings
from vocalis.engine.mock_engine import MockInferenceEngine
from vocalis.engine.model_manager import ModelManager
from vocalis.services.history import HistoryService
from vocalis.services.jobs import JobState
from vocalis.services.queue import GenerationQueue
from vocalis.persistence.database import Database
from vocalis.persistence.repositories.generations import INTERRUPTED_MESSAGE, GenerationRepository
from vocalis.ui.actions import history_detail, history_page, inspect_job, submit_clone, submit_design
from vocalis.ui.runtime import StudioServices
from vocalis.ui.studio import create_app
from vocalis.utils.audio import write_pcm_wav
from tests.unit.test_queue import GateEngine

_NEEDLE = "UNIQUE_PROMPT_NEEDLE"


def _services(tmp_path: Path, engine=None) -> tuple[StudioServices, GenerationRepository]:
    settings = load_settings(
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
    database = Database(settings.database_path)
    database.migrate()
    repository = GenerationRepository(database)
    repository.interrupt_incomplete()
    selected = engine or MockInferenceEngine(settings.design_output_dir, settings.clone_output_dir)
    manager = ModelManager(selected)
    manager.initialize()
    queue = GenerationQueue(manager, max_pending=settings.max_queue_size, store=repository)
    queue.start()
    services = StudioServices(
        settings=settings,
        queue=queue,
        history=HistoryService(repository),
        manager=manager,
        started_at=time.monotonic(),
    )
    return services, repository


def _close(services: StudioServices, gate: GateEngine | None = None) -> None:
    if gate is not None:
        gate.allow(10_000)
    services.queue.shutdown()
    services.manager.shutdown()


def _wait(services: StudioServices, job_id: str, label: str):
    deadline = time.monotonic() + 3
    view = inspect_job(services, job_id)
    while time.monotonic() < deadline and view.status_label != label:
        time.sleep(0.02)
        view = inspect_job(services, job_id)
    return view


def _wav(path: Path) -> bytes:
    write_pcm_wav(path, b"\x00\x01" * 160, sample_rate=16_000, simulated=False, disclaimer="")
    return path.read_bytes()


def _generate_click(demo: gr.Blocks, label: str):
    button = next(
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Button) and block.value == label
    )
    return next(
        item.fn
        for item in demo.fns.values()
        if item.fn is not None and (button._id, "click") in item.targets
    )


def test_design_and_clone_agree_across_queue_history_and_disk(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    services, repository = _services(tmp_path)
    reference = tmp_path / "speaker.wav"
    original = _wav(reference)
    try:
        with caplog.at_level(logging.INFO, logger="vocalis"):
            design = submit_design(services, f"Hello {_NEEDLE}.", "warm", 2, 10, False, False)
            finished = _wait(services, design.job_id, "Succeeded · simulated")
            cloned = submit_clone(services, "Read the line.", reference, "spoken words", 2, 10, False, False)
            cloned_done = _wait(services, cloned.job_id, "Succeeded · simulated")
        assert finished.audio_path is not None
        assert Path(finished.audio_path).is_file()
        assert finished.audio_path.startswith(str(tmp_path / "output" / "design"))
        assert reference.read_bytes() == original
        assert cloned_done.audio_path is not None
        assert Path(cloned_done.audio_path).is_file()
        clone_record = services.history.get(cloned.job_id)
        assert clone_record is not None and clone_record.reference_audio_path is not None
        stored_reference = Path(clone_record.reference_audio_path)
        assert stored_reference.is_file()
        assert stored_reference != reference.resolve()
        assert stored_reference.parent == (tmp_path / "uploads" / "references").resolve()
        assert stored_reference.read_bytes() == original
        record = services.history.get(finished.job_id)
        assert record is not None
        assert record.status is JobState.SUCCEEDED
        assert record.simulated is True
        assert record.output_audio_path == finished.audio_path
        assert _NEEDLE in record.input_text
        assert _NEEDLE not in caplog.text
        page = history_page(services, offset=0, type_label="Any type", status_label="Succeeded", provider_label="mock")
        assert page.summary == "Showing 1–2 of 2. Newest first."
        assert services.queue.snapshot().pending_count == 0
        assert services.queue.snapshot().running is False
        services.queue.shutdown()
        restarted = GenerationRepository(Database(services.settings.database_path))
        assert restarted.interrupt_incomplete() == 0
        detail, audio = history_detail(
            StudioServices(
                settings=services.settings,
                queue=GenerationQueue(services.manager, max_pending=3, store=restarted),
                history=HistoryService(restarted),
                manager=services.manager,
                started_at=services.started_at,
            ),
            finished.job_id,
        )
        assert audio == finished.audio_path
        assert _NEEDLE in detail
        assert repository.get(finished.job_id) is not None
    finally:
        services.queue.shutdown()
        services.manager.shutdown()


def test_a_failed_generation_does_not_stop_the_next_one(tmp_path: Path) -> None:
    gate = GateEngine()
    gate.errors.append(RuntimeError("SECRET_STACK"))
    services, _repository = _services(tmp_path, gate)
    try:
        failed = submit_design(services, "This fails.", "", 2, 10, False, False)
        gate.allow(1)
        failed_view = _wait(services, failed.job_id, "Failed")
        assert "Generation failed." in failed_view.detail
        assert "SECRET_STACK" not in failed_view.detail
        succeeded = submit_design(services, "This works.", "", 2, 10, False, False)
        gate.allow(2)
        succeeded_view = _wait(services, succeeded.job_id, "Succeeded · simulated")
        assert succeeded_view.status_label == "Succeeded · simulated"
        assert services.queue.snapshot().accepting is True
        assert services.history.count() == 2
    finally:
        _close(services, gate)


def test_a_full_queue_rejects_without_losing_the_waiting_jobs(tmp_path: Path) -> None:
    gate = GateEngine()
    services, _repository = _services(tmp_path, gate)
    try:
        first = submit_design(services, "Hold the worker.", "", 2, 10, False, False)
        gate.wait_started(1)
        waiting = [
            submit_design(services, f"Waiting {index}.", "", 2, 10, False, False)
            for index in range(3)
        ]
        rejected = submit_design(services, "One too many.", "", 2, 10, False, False)
        assert rejected.job_id == ""
        assert "full" in rejected.detail
        assert services.queue.snapshot().pending_count == 3
        assert services.history.count() == 4
        assert inspect_job(services, first.job_id).status_label == "Running"
        assert all(inspect_job(services, job.job_id).status_label == "Queued" for job in waiting)
    finally:
        _close(services, gate)


def test_rejected_submission_keeps_the_current_job_on_screen(tmp_path: Path) -> None:
    services, _repository = _services(tmp_path)
    try:
        demo = create_app(services.settings, services)
        submitted = submit_design(services, "Keep this job.", "", 2, 10, False, False)
        finished = _wait(services, submitted.job_id, "Succeeded · simulated")
        generate = _generate_click(demo, "Generate")
        job_id, status, detail, audio, _cancel, _timer = generate(
            "   ",
            "",
            2,
            10,
            False,
            False,
            finished.job_id,
        )
        assert job_id == finished.job_id
        assert status == "Succeeded · simulated"
        assert detail.startswith("Text to generate must not be empty.")
        assert finished.job_id in detail
        assert audio == finished.audio_path
        assert services.history.count() == 1
    finally:
        _close(services)


def test_a_crashed_job_is_marked_interrupted_and_its_audio_is_kept(tmp_path: Path) -> None:
    services, repository = _services(tmp_path)
    try:
        submitted = submit_design(services, "Completed before the crash.", "", 2, 10, False, False)
        finished = _wait(services, submitted.job_id, "Succeeded · simulated")
        output = Path(finished.audio_path or "")
        services.queue.shutdown()
        from vocalis.persistence.models import NewGeneration
        from vocalis.engine.types import GenerationType, InferenceParameters
        from vocalis.core.types import ModelProvider
        from datetime import datetime, timezone

        repository.create(
            NewGeneration(
                id="left-running",
                generation_type=GenerationType.VOICE_DESIGN,
                provider=ModelProvider.MOCK,
                input_text="still running",
                voice_description=None,
                reference_audio_path=None,
                reference_transcript=None,
                parameters=InferenceParameters(),
                created_at=datetime.now(timezone.utc),
            )
        )
        repository.mark_running("left-running", datetime.now(timezone.utc))
        reopened = GenerationRepository(Database(services.settings.database_path))
        assert reopened.interrupt_incomplete() == 1
        interrupted = reopened.get("left-running")
        assert interrupted is not None
        assert interrupted.status is JobState.FAILED
        assert interrupted.error_message == INTERRUPTED_MESSAGE
        kept = reopened.get(finished.job_id)
        assert kept is not None
        assert kept.status is JobState.SUCCEEDED
        assert output.is_file()
    finally:
        services.queue.shutdown()
        services.manager.shutdown()
