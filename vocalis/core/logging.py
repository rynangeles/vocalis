"""Rotating file and console logging for the Vocalis process."""

from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler

from vocalis.core.config import Settings

_LOG_BYTES = 1_048_576
_LOG_BACKUPS = 5


def configure_logging(settings: Settings) -> logging.Logger:
    """Attach UTC rotating-file and console handlers to the Vocalis logger.

    Calling this again replaces handlers so tests and restarts in one process
    do not duplicate lines or keep a stale log file open.
    """
    logger = logging.getLogger("vocalis")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime

    file_handler = RotatingFileHandler(
        settings.log_dir / "vocalis.log",
        maxBytes=_LOG_BYTES,
        backupCount=_LOG_BACKUPS,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
