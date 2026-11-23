"""Configuration loading and validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vocalis.core.config import load_settings
from vocalis.core.exceptions import ConfigurationError
from vocalis.core.types import AppEnvironment, ModelProvider


def _load(root: Path, **overrides: str):
    return load_settings(environ=overrides, project_root=root)


def test_development_defaults_create_directories(tmp_path: Path) -> None:
    settings = _load(tmp_path)

    assert settings.app_name == "Vocalis"
    assert settings.app_env is AppEnvironment.DEVELOPMENT
    assert settings.model_provider is ModelProvider.MOCK
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 7860
    assert settings.gradio_share is False
    assert settings.voxcpm_model_path is None
    assert settings.max_queue_size == 3
    assert settings.max_concurrent_generations == 1
    assert settings.max_upload_bytes == 20 * 1024 * 1024
    assert settings.data_dir == (tmp_path / "data").resolve()
    assert settings.database_path == (tmp_path / "data" / "history.db").resolve()
    assert settings.design_output_dir.is_dir()
    assert settings.clone_output_dir.is_dir()
    assert settings.reference_upload_dir.is_dir()
    assert settings.log_dir.is_dir()
    assert not settings.database_path.exists()


def test_relative_paths_resolve_inside_the_project(tmp_path: Path) -> None:
    settings = _load(tmp_path, VOCALIS_OUTPUT_DIR="nested/audio")

    assert settings.output_dir == (tmp_path / "nested" / "audio").resolve()
    assert settings.design_output_dir.is_dir()


def test_absolute_directory_is_accepted(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "data"
    settings = _load(tmp_path, VOCALIS_DATA_DIR=str(target))

    assert settings.data_dir == target.resolve()
    assert settings.data_dir.is_dir()


def test_relative_path_cannot_escape_the_project(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="VOCALIS_DATA_DIR"):
        _load(tmp_path, VOCALIS_DATA_DIR="../outside")

    assert not (tmp_path / "data").exists()


def test_env_file_applies_and_explicit_environ_wins(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "APP_NAME=FromFile",
                "APP_PORT=7870",
                "MODEL_PROVIDER=mock",
            ]
        ),
        encoding="utf-8",
    )

    from_file = load_settings(environ={}, env_file=env_file, project_root=tmp_path)
    overridden = load_settings(
        environ={"APP_PORT": "7871"},
        env_file=env_file,
        project_root=tmp_path,
    )

    assert from_file.app_name == "FromFile"
    assert from_file.app_port == 7870
    assert overridden.app_name == "FromFile"
    assert overridden.app_port == 7871


def test_blank_values_fall_back_to_defaults(tmp_path: Path) -> None:
    settings = _load(tmp_path, APP_PORT="", MODEL_PROVIDER="  ")

    assert settings.app_port == 7860
    assert settings.model_provider is ModelProvider.MOCK


def test_invalid_environment_provider_port_and_bool(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="APP_ENV"):
        _load(tmp_path, APP_ENV="staging")
    with pytest.raises(ConfigurationError, match="MODEL_PROVIDER"):
        _load(tmp_path, MODEL_PROVIDER="cuda")
    with pytest.raises(ConfigurationError, match="APP_PORT"):
        _load(tmp_path, APP_PORT="0")
    with pytest.raises(ConfigurationError, match="APP_PORT"):
        _load(tmp_path, APP_PORT="eighty")
    with pytest.raises(ConfigurationError, match="GRADIO_SHARE"):
        _load(tmp_path, GRADIO_SHARE="maybe")


def test_development_rejects_public_bind_and_sharing(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="APP_HOST"):
        _load(tmp_path, APP_HOST="0.0.0.0")
    with pytest.raises(ConfigurationError, match="GRADIO_SHARE"):
        _load(tmp_path, GRADIO_SHARE="true")
    assert not (tmp_path / "data").exists()


def test_loopback_hosts_are_accepted(tmp_path: Path) -> None:
    assert _load(tmp_path, APP_HOST="localhost").app_host == "localhost"
    assert _load(tmp_path / "v6", APP_HOST="::1").app_host == "::1"


def test_upload_limit(tmp_path: Path) -> None:
    assert _load(tmp_path, MAX_UPLOAD_BYTES="1024").max_upload_bytes == 1024
    with pytest.raises(ConfigurationError, match="MAX_UPLOAD_BYTES"):
        _load(tmp_path, MAX_UPLOAD_BYTES="0")
    with pytest.raises(ConfigurationError, match="MAX_UPLOAD_BYTES"):
        _load(tmp_path, MAX_UPLOAD_BYTES=str(100 * 1024 * 1024 + 1))


def test_queue_limits(tmp_path: Path) -> None:
    assert _load(tmp_path, MAX_QUEUE_SIZE="20").max_queue_size == 20
    with pytest.raises(ConfigurationError, match="MAX_QUEUE_SIZE"):
        _load(tmp_path, MAX_QUEUE_SIZE="0")
    with pytest.raises(ConfigurationError, match="MAX_QUEUE_SIZE"):
        _load(tmp_path, MAX_QUEUE_SIZE="21")
    with pytest.raises(ConfigurationError, match="MAX_CONCURRENT_GENERATIONS must be 1"):
        _load(tmp_path, MAX_CONCURRENT_GENERATIONS="2")


def test_database_path_cannot_be_a_directory(tmp_path: Path) -> None:
    database = tmp_path / "db-dir"
    database.mkdir()
    with pytest.raises(ConfigurationError, match="VOCALIS_DATABASE_PATH"):
        _load(tmp_path, VOCALIS_DATABASE_PATH=str(database))


def test_repeated_loads_are_safe(tmp_path: Path) -> None:
    first = _load(tmp_path)
    second = _load(tmp_path)

    assert first.data_dir == second.data_dir
    assert first.design_output_dir.is_dir()


def test_production_requires_an_existing_model_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="MODEL_PROVIDER=voxcpm"):
        _load(tmp_path, APP_ENV="production")
    assert not (tmp_path / "data").exists()

    missing = tmp_path / "missing-model"
    with pytest.raises(ConfigurationError, match="does not exist"):
        _load(
            tmp_path,
            APP_ENV="production",
            MODEL_PROVIDER="voxcpm",
            VOXCPM_MODEL_PATH=str(missing),
        )

    model_file = tmp_path / "model-file"
    model_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be a directory"):
        _load(
            tmp_path,
            APP_ENV="production",
            MODEL_PROVIDER="voxcpm",
            VOXCPM_MODEL_PATH=str(model_file),
        )


def test_production_accepts_a_model_directory_without_importing_voxcpm(tmp_path: Path) -> None:
    model = tmp_path / "models"
    model.mkdir()
    settings = _load(
        tmp_path,
        APP_ENV="production",
        APP_HOST="0.0.0.0",
        MODEL_PROVIDER="voxcpm",
        VOXCPM_MODEL_PATH=str(model),
    )

    assert settings.app_env is AppEnvironment.PRODUCTION
    assert settings.model_provider is ModelProvider.VOXCPM
    assert settings.voxcpm_model_path == model.resolve()
    assert settings.app_host == "0.0.0.0"
    assert "voxcpm" not in sys.modules
    assert "torch" not in sys.modules


def test_development_voxcpm_provider_still_requires_a_model_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="VOXCPM_MODEL_PATH"):
        _load(tmp_path, MODEL_PROVIDER="voxcpm")


def test_importing_configuration_does_not_load_the_model_stack() -> None:
    import vocalis.core.config  # noqa: F401

    assert "voxcpm" not in sys.modules
    assert "torch" not in sys.modules
    assert "torchaudio" not in sys.modules
