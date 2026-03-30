"""Logging utilities using loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger

# Remove default handler to reconfigure
_logger.remove()

# Default format
_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def get_logger(
    name: str | None = None,
    log_file: str | Path | None = None,
    level: str = "INFO",
    format_str: str | None = None,
) -> _logger:
    """
    Get a configured loguru logger.

    Args:
        name: Module name (unused, kept for compatibility).
        log_file: Optional file path to write logs.
        level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR").
        format_str: Custom log format string.

    Returns:
        Configured logger instance.
    """
    fmt = format_str or _DEFAULT_FORMAT

    # Console output
    _logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=True,
    )

    # File output
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_path,
            format=fmt,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            colorize=False,
        )

    return _logger
