"""Load and validate Vocalis configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from dotenv import dotenv_values

from vocalis.core.exceptions import ConfigurationError
from vocalis.core.types import AppEnvironment, ModelProvider

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_MAX_QUEUE_SIZE = 20
_MAX_APP_NAME_LENGTH = 80
_DEFAULT_UPLOAD_BYTES = 20 * 1024 * 1024
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

_DEFAULTS: dict[str, str] = {
    "APP_NAME": "Vocalis",
    "APP_ENV": AppEnvironment.DEVELOPMENT.value,
    "APP_HOST": "127.0.0.1",
    "APP_PORT": "7860",
    "MODEL_PROVIDER": ModelProvider.MOCK.value,
    "VOCALIS_DATA_DIR": "./data",
    "VOCALIS_OUTPUT_DIR": "./output",
    "VOCALIS_UPLOAD_DIR": "./uploads",
    "VOCALIS_LOG_DIR": "./logs",
    "VOCALIS_DATABASE_PATH": "./data/history.db",
    "MAX_QUEUE_SIZE": "3",
    "MAX_CONCURRENT_GENERATIONS": "1",
    "MAX_UPLOAD_BYTES": str(_DEFAULT_UPLOAD_BYTES),
    "GRADIO_SHARE": "false",
}


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated process settings. Paths are absolute."""

    app_name: str
    app_env: AppEnvironment
    app_host: str
    app_port: int
    model_provider: ModelProvider
    voxcpm_model_path: Path | None
    data_dir: Path
    output_dir: Path
    upload_dir: Path
    log_dir: Path
    database_path: Path
    max_queue_size: int
    max_concurrent_generations: int
    max_upload_bytes: int
    gradio_share: bool
    project_root: Path

    @property
    def design_output_dir(self) -> Path:
        return self.output_dir / "design"

    @property
    def clone_output_dir(self) -> Path:
        return self.output_dir / "clone"

    @property
    def reference_upload_dir(self) -> Path:
        return self.upload_dir / "references"

    def runtime_directories(self) -> tuple[Path, ...]:
        return (
            self.data_dir,
            self.output_dir,
            self.design_output_dir,
            self.clone_output_dir,
            self.upload_dir,
            self.reference_upload_dir,
            self.log_dir,
            self.database_path.parent,
        )


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_settings(
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = None,
    project_root: Path | None = None,
) -> Settings:
    """Load settings and create runtime directories.

    When ``environ`` is omitted, values from the environment file are overlaid
    with the process environment. When ``environ`` is provided, it is the only
    overlay, which keeps tests independent of the developer machine.
    """
    root = (project_root or default_project_root()).resolve()
    source = _settings_source(root, environ, env_file)
    settings = _build_settings(source, root)
    _ensure_runtime_directories(settings)
    return settings


def _settings_source(
    project_root: Path,
    environ: Mapping[str, str] | None,
    env_file: Path | None,
) -> dict[str, str]:
    file_path = project_root / ".env" if env_file is None else env_file
    file_values: dict[str, str] = {}
    if file_path.is_file():
        loaded = dotenv_values(file_path)
        file_values = {key: value for key, value in loaded.items() if value is not None}
    overlay = os.environ if environ is None else environ
    return {**file_values, **dict(overlay)}


def _build_settings(source: Mapping[str, str], project_root: Path) -> Settings:
    app_name = _text(source, "APP_NAME")
    app_env = _parse_enum(AppEnvironment, _text(source, "APP_ENV"), "APP_ENV")
    app_host = _text(source, "APP_HOST")
    app_port = _bounded_int(_text(source, "APP_PORT"), "APP_PORT", minimum=1, maximum=65535)
    model_provider = _parse_enum(
        ModelProvider, _text(source, "MODEL_PROVIDER"), "MODEL_PROVIDER"
    )
    data_dir = _resolve_path(source, "VOCALIS_DATA_DIR", project_root)
    output_dir = _resolve_path(source, "VOCALIS_OUTPUT_DIR", project_root)
    upload_dir = _resolve_path(source, "VOCALIS_UPLOAD_DIR", project_root)
    log_dir = _resolve_path(source, "VOCALIS_LOG_DIR", project_root)
    database_path = _resolve_path(source, "VOCALIS_DATABASE_PATH", project_root)
    max_queue_size = _bounded_int(
        _text(source, "MAX_QUEUE_SIZE"),
        "MAX_QUEUE_SIZE",
        minimum=1,
        maximum=_MAX_QUEUE_SIZE,
    )
    max_concurrent_generations = _bounded_int(
        _text(source, "MAX_CONCURRENT_GENERATIONS"),
        "MAX_CONCURRENT_GENERATIONS",
        minimum=1,
        maximum=1,
    )
    max_upload_bytes = _bounded_int(
        _text(source, "MAX_UPLOAD_BYTES"),
        "MAX_UPLOAD_BYTES",
        minimum=1,
        maximum=_MAX_UPLOAD_BYTES,
    )
    gradio_share = _parse_bool(_text(source, "GRADIO_SHARE"), "GRADIO_SHARE")
    voxcpm_model_path = _optional_model_path(source, project_root)

    if not app_name:
        raise ConfigurationError("APP_NAME must not be empty.")
    if len(app_name) > _MAX_APP_NAME_LENGTH:
        raise ConfigurationError(
            f"APP_NAME must be {_MAX_APP_NAME_LENGTH} characters or fewer."
        )
    if any(character.isspace() for character in app_host):
        raise ConfigurationError("APP_HOST must not contain whitespace.")
    if gradio_share:
        raise ConfigurationError(
            "GRADIO_SHARE must be false. Vocalis does not enable Gradio public sharing."
        )
    if app_env is AppEnvironment.DEVELOPMENT and app_host not in _LOOPBACK_HOSTS:
        raise ConfigurationError(
            "APP_HOST must be 127.0.0.1, localhost, or ::1 when APP_ENV=development."
        )
    if database_path.exists() and database_path.is_dir():
        raise ConfigurationError("VOCALIS_DATABASE_PATH must be a file path, not a directory.")

    _validate_provider(app_env, model_provider, voxcpm_model_path)

    return Settings(
        app_name=app_name,
        app_env=app_env,
        app_host=app_host,
        app_port=app_port,
        model_provider=model_provider,
        voxcpm_model_path=voxcpm_model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        upload_dir=upload_dir,
        log_dir=log_dir,
        database_path=database_path,
        max_queue_size=max_queue_size,
        max_concurrent_generations=max_concurrent_generations,
        max_upload_bytes=max_upload_bytes,
        gradio_share=False,
        project_root=project_root,
    )


def _validate_provider(
    app_env: AppEnvironment,
    model_provider: ModelProvider,
    voxcpm_model_path: Path | None,
) -> None:
    if app_env is AppEnvironment.PRODUCTION and model_provider is not ModelProvider.VOXCPM:
        raise ConfigurationError("Production requires MODEL_PROVIDER=voxcpm.")
    if model_provider is ModelProvider.VOXCPM:
        if voxcpm_model_path is None:
            raise ConfigurationError("MODEL_PROVIDER=voxcpm requires VOXCPM_MODEL_PATH.")
        if not voxcpm_model_path.exists():
            raise ConfigurationError(
                f"VOXCPM_MODEL_PATH does not exist: {voxcpm_model_path}"
            )
        if not voxcpm_model_path.is_dir():
            raise ConfigurationError(
                f"VOXCPM_MODEL_PATH must be a directory: {voxcpm_model_path}"
            )


def _optional_model_path(source: Mapping[str, str], project_root: Path) -> Path | None:
    raw = source.get("VOXCPM_MODEL_PATH")
    if raw is None or not str(raw).strip():
        return None
    return _resolve_path_value(str(raw).strip(), "VOXCPM_MODEL_PATH", project_root)


def _text(source: Mapping[str, str], key: str) -> str:
    raw = source.get(key)
    if raw is None or not str(raw).strip():
        return _DEFAULTS[key]
    return str(raw).strip()


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _parse_enum(enum_cls: type[_EnumT], raw: str, label: str) -> _EnumT:
    try:
        return enum_cls(raw.lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_cls)
        raise ConfigurationError(f"{label} must be one of: {allowed}.") from exc


def _parse_bool(value: str, key: str) -> bool:
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{key} must be true or false.")


def _bounded_int(value: str, key: str, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{key} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        if minimum == maximum:
            raise ConfigurationError(f"{key} must be {minimum}.")
        raise ConfigurationError(f"{key} must be between {minimum} and {maximum}.")
    return parsed


def _resolve_path(source: Mapping[str, str], key: str, project_root: Path) -> Path:
    return _resolve_path_value(_text(source, key), key, project_root)


def _resolve_path_value(value: str, key: str, project_root: Path) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    resolved = (project_root / raw).resolve()
    if not _within_root(resolved, project_root):
        raise ConfigurationError(
            f"{key} must stay inside the project directory when it is a relative path."
        )
    return resolved


def _within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ensure_runtime_directories(settings: Settings) -> None:
    for directory in settings.runtime_directories():
        directory.mkdir(parents=True, exist_ok=True)
