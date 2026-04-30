"""
Centralized logging setup for the KeySG project.

Goals:
- Use Loguru consistently across modules.
- Default to console-only logging (no files).
- Allow optional file logging via env var KeySG_LOG_FILE if needed.
"""

from __future__ import annotations

import os
from loguru import logger


def setup_logging() -> None:
    """Configure Loguru sinks in a minimal, consistent way.

    - Remove pre-existing handlers to avoid duplicates when re-running.
    - Add a single console sink at INFO level with a concise format.
    - If env var KeySG_LOG_FILE is set to a filepath, also log to that file.
    """
    try:
        logger.remove()
    except Exception:
        # If loguru hasn't been configured yet
        pass

    # Console sink
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n",
    )

    # Optional file sink controlled by env var
    log_file = os.environ.get("KeySG_LOG_FILE")
    if log_file:
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except Exception:
            pass
        logger.add(log_file, level="INFO", rotation="10 MB", retention=3)


__all__ = ["setup_logging", "logger"]
