"""Application services. They talk to the queue, not to Gradio."""

from vocalis.services.audio import AudioInspection, AudioService
from vocalis.services.cloning import VoiceCloningService
from vocalis.services.design import VoiceDesignService
from vocalis.services.history import HistoryService
from vocalis.services.jobs import GenerationJob, JobState
from vocalis.services.queue import GenerationQueue

__all__ = [
    "AudioInspection",
    "AudioService",
    "GenerationJob",
    "GenerationQueue",
    "HistoryService",
    "JobState",
    "VoiceCloningService",
    "VoiceDesignService",
]
