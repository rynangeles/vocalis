"""Application entry point and foundation screen."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import gradio as gr
import pytest

import app as vocalis_app
from vocalis.core.config import load_settings
from vocalis.core.exceptions import InferenceError, PersistenceError
from vocalis.core.logging import configure_logging
from vocalis.persistence.database import Database
from vocalis.engine.types import ModelState
from vocalis.ui.foundation import display_address, render_foundation
from vocalis.ui.runtime import StudioServices
from vocalis.ui.studio import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_foundation_page_identifies_mock_mode(tmp_path: Path) -> None:
    settings = load_settings(environ={"APP_HOST": "::1"}, project_root=tmp_path)
    page = render_foundation(settings)

    assert "<h1 class=\"vocalis-title\">Vocalis</h1>" in page
    assert "Give your words a voice." in page
    assert "Powered by VoxCPM2" in page
    assert "<dd>mock</dd>" in page
    assert "simulated tone, not VoxCPM2 speech" in page
    assert display_address(settings) == "http://[::1]:7860"


def test_foundation_page_escapes_the_application_name(tmp_path: Path) -> None:
    settings = load_settings(environ={"APP_NAME": "<Vocalis>"}, project_root=tmp_path)
    page = render_foundation(settings)

    assert "<Vocalis>" not in page
    assert "&lt;Vocalis&gt;" in page


def test_create_app_returns_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vocalis_app, "_manager", None)
    monkeypatch.setattr(vocalis_app, "_queue", None)
    monkeypatch.setattr(vocalis_app, "_history", None)
    settings = load_settings(environ={}, project_root=tmp_path)
    logger = configure_logging(settings)
    assert vocalis_app._initialize_engine(logger, settings, retain=True)
    try:
        services = StudioServices(
            settings=settings,
            queue=vocalis_app.get_generation_queue(),
            history=vocalis_app.get_history_service(),
            manager=vocalis_app.get_model_manager(),
            started_at=0,
        )
        demo = create_app(settings, services)
        assert isinstance(demo, gr.Blocks)
        assert demo.title == "Vocalis — AI Voice Studio"
    finally:
        vocalis_app.shutdown_runtime()


def test_check_command_prepares_directories(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "app.py"), "--check"],
        cwd=REPO_ROOT,
        env=_process_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "output" / "design").is_dir()
    log_text = (tmp_path / "logs" / "vocalis.log").read_text(encoding="utf-8")
    assert "Configuration check completed." in log_text
    assert "VoxCPM2 will not be imported." in log_text
    assert (tmp_path / "data" / "history.db").is_file()
    assert "voice_design=simulated" in log_text
    assert "real_inference=False" in log_text


def test_check_command_reports_configuration_errors(tmp_path: Path) -> None:
    env = _process_env(tmp_path)
    env["APP_HOST"] = "0.0.0.0"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "app.py"), "--check"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "Configuration error:" in result.stderr
    assert "APP_HOST" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "data").exists()


def test_running_process_keeps_a_ready_mock_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vocalis_app, "_manager", None)
    monkeypatch.setattr(vocalis_app, "_queue", None)
    monkeypatch.setattr(vocalis_app, "_history", None)
    settings = load_settings(environ={}, project_root=tmp_path)
    logger = configure_logging(settings)

    assert vocalis_app._initialize_engine(logger, settings, retain=True)
    try:
        manager = vocalis_app.get_model_manager()
        assert manager.state is ModelState.READY
        assert manager.capabilities.real_inference is False
        assert vocalis_app.get_generation_queue() is not None
        assert vocalis_app.get_history_service().count() == 0
        assert settings.database_path.is_file()
    finally:
        vocalis_app.shutdown_runtime()


def test_get_model_manager_before_startup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vocalis_app, "_manager", None)
    with pytest.raises(InferenceError, match="not initialized"):
        vocalis_app.get_model_manager()


def test_history_service_before_startup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vocalis_app, "_history", None)
    with pytest.raises(PersistenceError, match="not initialized"):
        vocalis_app.get_history_service()


def test_initialize_reports_a_migration_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(vocalis_app, "_manager", None)
    monkeypatch.setattr(vocalis_app, "_queue", None)
    monkeypatch.setattr(vocalis_app, "_history", None)

    def fail(self: Database) -> tuple[int, ...]:
        raise PersistenceError('Migration 0009_bad.sql failed: near "NOPE": syntax error')

    monkeypatch.setattr(Database, "migrate", fail)
    settings = load_settings(environ={}, project_root=tmp_path)
    logger = configure_logging(settings)

    assert vocalis_app._initialize_engine(logger, settings, retain=False) is False
    captured = capsys.readouterr()
    assert "Database error:" in captured.err
    assert "Migration 0009_bad.sql failed" in captured.err
    assert "Traceback" not in captured.err
    assert vocalis_app._manager is None


def test_check_command_rejects_an_unavailable_voxcpm_engine(tmp_path: Path) -> None:
    model = tmp_path / "models"
    model.mkdir()
    env = _process_env(tmp_path)
    env["MODEL_PROVIDER"] = "voxcpm"
    env["VOXCPM_MODEL_PATH"] = str(model)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "app.py"), "--check"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "Engine error:" in result.stderr
    assert "VoxCPM2 engine is unavailable" in result.stderr
    assert "Traceback" not in result.stderr


def _process_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("APP_", "VOCALIS_", "MODEL_", "MAX_", "GRADIO_")):
            del env[key]
    env.update(
        {
            "APP_ENV": "development",
            "APP_HOST": "127.0.0.1",
            "APP_PORT": "7860",
            "MODEL_PROVIDER": "mock",
            "VOXCPM_MODEL_PATH": "",
            "VOCALIS_DATA_DIR": str(root / "data"),
            "VOCALIS_OUTPUT_DIR": str(root / "output"),
            "VOCALIS_UPLOAD_DIR": str(root / "uploads"),
            "VOCALIS_LOG_DIR": str(root / "logs"),
            "VOCALIS_DATABASE_PATH": str(root / "data" / "history.db"),
            "MAX_QUEUE_SIZE": "3",
            "MAX_CONCURRENT_GENERATIONS": "1",
            "GRADIO_SHARE": "false",
            "GRADIO_ANALYTICS_ENABLED": "False",
        }
    )
    return env
