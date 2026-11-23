"""Inference protocol and capability reporting."""

import sys

from vocalis.core.types import ModelProvider
from vocalis.engine.base import InferenceEngine
from vocalis.engine.mock_engine import MOCK_CAPABILITIES, MockInferenceEngine
from vocalis.engine.types import CapabilitySupport
from vocalis.engine.voxcpm_engine import VoxCPMInferenceEngine


def test_mock_engine_satisfies_the_protocol(tmp_path) -> None:
    engine = MockInferenceEngine(tmp_path / "design", tmp_path / "clone")

    assert isinstance(engine, InferenceEngine)
    assert engine.get_capabilities() == MOCK_CAPABILITIES


def test_protocol_rejects_an_incomplete_object() -> None:
    class Incomplete:
        def initialize(self) -> None:
            return None

    assert not isinstance(Incomplete(), InferenceEngine)


def test_mock_capabilities_are_simulated_and_do_not_apply_parameters() -> None:
    capabilities = MOCK_CAPABILITIES

    assert capabilities.provider is ModelProvider.MOCK
    assert capabilities.voice_design is CapabilitySupport.SIMULATED
    assert capabilities.voice_cloning is CapabilitySupport.SIMULATED
    assert capabilities.real_inference is False
    assert capabilities.progress_reporting is False
    assert capabilities.cancellation is False
    assert capabilities.sample_rate == 16_000
    assert capabilities.applied_parameters == ()
    assert "cfg_value" in capabilities.accepted_parameters
    assert CapabilitySupport.AVAILABLE not in {
        capabilities.voice_design,
        capabilities.voice_cloning,
    }
    assert "does not synthesize speech" in capabilities.notice


def test_voxcpm_boundary_reports_every_feature_unavailable(tmp_path) -> None:
    engine = VoxCPMInferenceEngine(tmp_path)
    capabilities = engine.get_capabilities()

    assert isinstance(engine, InferenceEngine)
    assert capabilities.provider is ModelProvider.VOXCPM
    assert capabilities.voice_design is CapabilitySupport.UNAVAILABLE
    assert capabilities.voice_cloning is CapabilitySupport.UNAVAILABLE
    assert capabilities.real_inference is False
    assert capabilities.progress_reporting is False
    assert capabilities.cancellation is False
    assert capabilities.sample_rate is None
    assert capabilities.accepted_parameters == ()
    assert capabilities.applied_parameters == ()
    assert engine.get_status().ready is False
    assert "voxcpm" not in sys.modules
    assert "torch" not in sys.modules
    assert "torchaudio" not in sys.modules
