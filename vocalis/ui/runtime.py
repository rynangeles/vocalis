"""Services the studio pages are allowed to call."""

from __future__ import annotations

import time
from dataclasses import dataclass

from vocalis.core.config import Settings
from vocalis.engine.model_manager import ModelManager
from vocalis.services.audio import AudioService
from vocalis.services.cloning import VoiceCloningService
from vocalis.services.design import VoiceDesignService
from vocalis.services.history import HistoryService
from vocalis.services.queue import GenerationQueue


@dataclass
class StudioServices:
    """The UI's handle on the running application. It does not own the worker."""

    settings: Settings
    queue: GenerationQueue
    history: HistoryService
    manager: ModelManager
    started_at: float

    def __post_init__(self) -> None:
        self.audio = AudioService(
            max_upload_bytes=self.settings.max_upload_bytes,
            reference_dir=self.settings.reference_upload_dir,
        )
        self.design = VoiceDesignService(self.queue)
        self.cloning = VoiceCloningService(self.queue, self.audio)

    def uptime_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)
