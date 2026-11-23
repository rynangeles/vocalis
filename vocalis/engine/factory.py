"""Select an inference engine from application settings."""

from __future__ import annotations

from vocalis.core.config import Settings
from vocalis.core.exceptions import ConfigurationError
from vocalis.core.types import ModelProvider
from vocalis.engine.base import InferenceEngine
from vocalis.engine.mock_engine import MockInferenceEngine


def create_engine(settings: Settings) -> InferenceEngine:
    """Return the engine for ``settings.model_provider``.

    Mock mode constructs ``MockInferenceEngine`` and does not import VoxCPM2.
    The VoxCPM2 provider constructs the adapter and does not import ``voxcpm``
    until that engine is initialized.
    """
    if settings.model_provider is ModelProvider.MOCK:
        return MockInferenceEngine(
            design_output_dir=settings.design_output_dir,
            clone_output_dir=settings.clone_output_dir,
        )
    if settings.model_provider is ModelProvider.VOXCPM:
        from vocalis.engine.voxcpm_engine import VoxCPMInferenceEngine

        return VoxCPMInferenceEngine(
            settings.voxcpm_model_path,
            design_output_dir=settings.design_output_dir,
            clone_output_dir=settings.clone_output_dir,
        )
    raise ConfigurationError(f"Unsupported model provider: {settings.model_provider}.")
