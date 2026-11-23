"""Model manager lifecycle without a queue or background thread."""

from pathlib import Path

import pytest

from vocalis.core.exceptions import InferenceError
from vocalis.core.types import ModelProvider
from vocalis.engine.mock_engine import MockInferenceEngine
from vocalis.engine.model_manager import ModelManager
from vocalis.engine.types import (
    CapabilitySupport,
    EngineCapabilities,
    EngineStatus,
    GenerationRequest,
    GenerationResult,
    GenerationType,
    ModelState,
)
from vocalis.engine.voxcpm_engine import VoxCPMInferenceEngine


class ProbeEngine:
    """Records the manager state observed inside lifecycle calls."""

    def __init__(self) -> None:
        self.manager: ModelManager | None = None
        self.seen: list[ModelState] = []
        self.init_calls = 0
        self.fail_with: Exception | None = None

    def initialize(self) -> None:
        self.init_calls += 1
        assert self.manager is not None
        self.seen.append(self.manager.state)
        if self.fail_with is not None:
            raise self.fail_with

    def shutdown(self) -> None:
        assert self.manager is not None
        self.seen.append(self.manager.state)

    def get_status(self) -> EngineStatus:
        return EngineStatus(provider=ModelProvider.MOCK, ready=False, detail="probe")

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            provider=ModelProvider.MOCK,
            voice_design=CapabilitySupport.SIMULATED,
            voice_cloning=CapabilitySupport.SIMULATED,
            real_inference=False,
            progress_reporting=False,
            cancellation=False,
            sample_rate=16_000,
            accepted_parameters=(),
            applied_parameters=(),
            notice="probe",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        del request
        raise InferenceError("probe does not generate")


def _probe() -> tuple[ModelManager, ProbeEngine]:
    engine = ProbeEngine()
    manager = ModelManager(engine)
    engine.manager = manager
    return manager, engine


def test_successful_lifecycle_transitions() -> None:
    manager, engine = _probe()

    assert manager.get_status().state is ModelState.NOT_INITIALIZED
    assert manager.get_status().ready is False

    status = manager.initialize()
    assert status.state is ModelState.READY
    assert status.ready is True
    assert status.provider is ModelProvider.MOCK
    assert status.error is None
    assert engine.seen == [ModelState.INITIALIZING]

    manager.initialize()
    assert engine.init_calls == 1

    shutdown = manager.shutdown()
    assert shutdown.state is ModelState.NOT_INITIALIZED
    assert shutdown.ready is False
    assert engine.seen == [ModelState.INITIALIZING, ModelState.SHUTTING_DOWN]

    again = manager.shutdown()
    assert again.state is ModelState.NOT_INITIALIZED
    assert engine.seen == [ModelState.INITIALIZING, ModelState.SHUTTING_DOWN]


def test_initialization_failure_is_reported_and_can_be_retried() -> None:
    manager, engine = _probe()
    engine.fail_with = InferenceError("boom")

    with pytest.raises(InferenceError, match="boom"):
        manager.initialize()
    status = manager.get_status()
    assert status.state is ModelState.ERROR
    assert status.ready is False
    assert status.error == "boom"
    with pytest.raises(InferenceError, match="not ready"):
        manager.generate(GenerationRequest(GenerationType.VOICE_DESIGN, "Hello"))

    engine.fail_with = None
    recovered = manager.initialize()
    assert recovered.state is ModelState.READY
    assert recovered.error is None
    assert engine.init_calls == 2


def test_unexpected_initialization_error_is_wrapped() -> None:
    manager, engine = _probe()
    engine.fail_with = RuntimeError("disk")

    with pytest.raises(InferenceError, match="Model initialization failed") as caught:
        manager.initialize()

    assert caught.value.__cause__ is engine.fail_with
    assert manager.state is ModelState.ERROR


def test_generate_is_rejected_until_ready_and_after_shutdown(tmp_path: Path) -> None:
    engine = MockInferenceEngine(tmp_path / "design", tmp_path / "clone")
    manager = ModelManager(engine)
    request = GenerationRequest(GenerationType.VOICE_DESIGN, "Hello")

    with pytest.raises(InferenceError, match="not_initialized"):
        manager.generate(request)

    manager.initialize()
    result = manager.generate(request)
    assert result.simulated is True
    assert manager.state is ModelState.READY

    manager.shutdown()
    with pytest.raises(InferenceError, match="not_initialized"):
        manager.generate(request)


def test_a_rejected_request_leaves_a_ready_manager_ready(tmp_path: Path) -> None:
    manager = ModelManager(MockInferenceEngine(tmp_path / "design", tmp_path / "clone"))
    manager.initialize()

    with pytest.raises(InferenceError, match="must not be empty"):
        manager.generate(GenerationRequest(GenerationType.VOICE_DESIGN, " "))

    assert manager.state is ModelState.READY


def test_voxcpm_initialization_records_error_without_importing_the_package(tmp_path: Path) -> None:
    import sys

    manager = ModelManager(VoxCPMInferenceEngine(tmp_path))

    with pytest.raises(InferenceError, match="VoxCPM2 engine is unavailable"):
        manager.initialize()

    status = manager.get_status()
    assert status.state is ModelState.ERROR
    assert status.ready is False
    assert status.provider is ModelProvider.VOXCPM
    assert manager.capabilities.real_inference is False
    assert manager.capabilities.voice_design is CapabilitySupport.UNAVAILABLE
    assert "voxcpm" not in sys.modules
    assert "torch" not in sys.modules
