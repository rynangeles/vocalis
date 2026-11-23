"""Configuration, logging, and shared application types."""

from vocalis.core.config import Settings, load_settings
from vocalis.core.exceptions import (
    CancellationUnsupportedError,
    ConfigurationError,
    EngineUnavailableError,
    HistoryQueryError,
    InferenceError,
    InvalidAudioError,
    MissingInputError,
    PersistenceError,
    QueueFullError,
    QueueUnavailableError,
    UnknownJobError,
    UnsupportedAudioFormatError,
    VocalisError,
)
from vocalis.core.logging import configure_logging
from vocalis.core.types import AppEnvironment, ModelProvider

__all__ = [
    "AppEnvironment",
    "CancellationUnsupportedError",
    "ConfigurationError",
    "EngineUnavailableError",
    "HistoryQueryError",
    "InferenceError",
    "InvalidAudioError",
    "MissingInputError",
    "PersistenceError",
    "ModelProvider",
    "QueueFullError",
    "QueueUnavailableError",
    "Settings",
    "UnknownJobError",
    "UnsupportedAudioFormatError",
    "VocalisError",
    "configure_logging",
    "load_settings",
]
