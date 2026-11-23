"""Inference engine contract. Implementations must not import Gradio."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vocalis.engine.types import (
    EngineCapabilities,
    EngineStatus,
    GenerationRequest,
    GenerationResult,
)


@runtime_checkable
class InferenceEngine(Protocol):
    """Shared operations for the mock engine and the future VoxCPM2 adapter."""

    def initialize(self) -> None:
        """Prepare the engine. Mock mode must not load a model."""

    def shutdown(self) -> None:
        """Release engine resources. Must be safe to call once."""

    def get_status(self) -> EngineStatus:
        """Report whether this engine instance is ready."""

    def get_capabilities(self) -> EngineCapabilities:
        """Report supported workflows without advertising missing features."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one voice-design or voice-cloning request."""
