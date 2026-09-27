"""Logging: full detail to a log file (with credentials redacted); concise console output.

In TUI mode nothing is written to the console by the logging system (it would corrupt the
live display); the UI shows concise messages and points to the log file.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from harness.orchestrator.events import redact

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def default_log_path() -> Path:
    return Path(tempfile.gettempdir()) / "ai-coding-harness" / "harness.log"


def configure_logging(level: int, *, log_file: str | None = None, console: bool = True) -> Path | None:
    """Configure the root logger. Returns the log file path (None if it cannot be opened)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)
    redactor = RedactingFilter()

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(level)
        stream.setFormatter(logging.Formatter(LOG_FORMAT))
        stream.addFilter(redactor)
        root.addHandler(stream)

    path = Path(log_file).expanduser() if log_file else default_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        return None
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)
    return path
