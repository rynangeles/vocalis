"""Application exceptions safe to surface to an operator."""


class VocalisError(Exception):
    """Expected Vocalis failure with a message suitable for the operator."""


class ConfigurationError(VocalisError):
    """Configuration is missing or invalid."""


class InferenceError(VocalisError):
    """A generation request was rejected or the engine failed."""


class EngineUnavailableError(InferenceError):
    """The selected engine cannot run in this environment."""


class QueueFullError(VocalisError):
    """The pending generation queue is already at its limit."""


class QueueUnavailableError(VocalisError):
    """The generation queue is stopped or has not been started."""


class UnknownJobError(VocalisError):
    """No generation job exists with the given id."""


class CancellationUnsupportedError(VocalisError):
    """This job cannot be cancelled."""


class MissingInputError(VocalisError):
    """A required generation field is missing."""


class InvalidAudioError(VocalisError):
    """An audio file is empty, corrupt, or does not match its type."""


class UnsupportedAudioFormatError(VocalisError):
    """The audio file extension is not one Vocalis accepts."""


class PersistenceError(VocalisError):
    """The generation history database could not be read or updated."""


class HistoryQueryError(VocalisError):
    """A generation history query was rejected."""
