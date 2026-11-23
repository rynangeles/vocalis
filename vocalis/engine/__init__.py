"""Inference engines and the model lifecycle."""

from vocalis.engine.base import InferenceEngine
from vocalis.engine.factory import create_engine
from vocalis.engine.mock_engine import MOCK_CAPABILITIES, MockInferenceEngine
from vocalis.engine.model_manager import ModelManager
from vocalis.engine.types import (
    AudioMetadata,
    CapabilitySupport,
    EngineCapabilities,
    EngineStatus,
    GenerationRequest,
    GenerationResult,
    GenerationType,
    InferenceParameters,
    ModelState,
    ModelStatus,
)
from vocalis.engine.voxcpm_engine import VoxCPMInferenceEngine

__all__ = [
    "AudioMetadata",
    "CapabilitySupport",
    "EngineCapabilities",
    "EngineStatus",
    "GenerationRequest",
    "GenerationResult",
    "GenerationType",
    "InferenceEngine",
    "InferenceParameters",
    "MOCK_CAPABILITIES",
    "MockInferenceEngine",
    "ModelManager",
    "ModelState",
    "ModelStatus",
    "VoxCPMInferenceEngine",
    "create_engine",
]
