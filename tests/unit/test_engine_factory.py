"""Engine selection from MODEL_PROVIDER."""

import sys
from pathlib import Path

import pytest

from vocalis.core.config import load_settings
from vocalis.core.exceptions import EngineUnavailableError
from vocalis.engine.factory import create_engine
from vocalis.engine.mock_engine import MockInferenceEngine
from vocalis.engine.model_manager import ModelManager
from vocalis.engine.voxcpm_engine import VoxCPMInferenceEngine


def test_mock_provider_returns_the_mock_engine(tmp_path: Path) -> None:
    settings = load_settings(environ={}, project_root=tmp_path)

    engine = create_engine(settings)

    assert isinstance(engine, MockInferenceEngine)
    assert "voxcpm" not in sys.modules
    assert "torch" not in sys.modules
    assert "torchaudio" not in sys.modules

    manager = ModelManager(engine)
    status = manager.initialize()
    assert status.ready is True
    assert status.provider.value == "mock"


def test_voxcpm_provider_returns_the_boundary_and_fails_on_initialize(tmp_path: Path) -> None:
    model = tmp_path / "models"
    model.mkdir()
    settings = load_settings(
        environ={
            "APP_ENV": "development",
            "MODEL_PROVIDER": "voxcpm",
            "VOXCPM_MODEL_PATH": str(model),
        },
        project_root=tmp_path,
    )

    engine = create_engine(settings)

    assert isinstance(engine, VoxCPMInferenceEngine)
    assert engine.model_path == model.resolve()
    with pytest.raises(EngineUnavailableError, match="not installed"):
        ModelManager(engine).initialize()
    assert "voxcpm" not in sys.modules
    assert "torch" not in sys.modules
