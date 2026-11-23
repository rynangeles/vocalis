"""Shared enumerations used across Vocalis."""

from enum import StrEnum


class AppEnvironment(StrEnum):
    """Runtime mode. Development never requires the VoxCPM2 package."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class ModelProvider(StrEnum):
    """Inference backend selected by configuration."""

    MOCK = "mock"
    VOXCPM = "voxcpm"
