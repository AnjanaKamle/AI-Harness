"""Basic logging: human-readable text or JSON lines, to stderr and optionally a file."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from harness.config import LoggingConfig

ROOT_LOGGER = "harness"

# Attributes present on every LogRecord; anything else was passed via `extra=`.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


TEXT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure the ``harness`` logger tree. Idempotent: replaces prior handlers."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(config.level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter: logging.Formatter = (
        JsonFormatter() if config.format == "json" else logging.Formatter(TEXT_FORMAT)
    )

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if config.file is not None:
        config.file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.file, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the harness logger, e.g. get_logger('llm') -> 'harness.llm'."""
    if name == ROOT_LOGGER or name.startswith(ROOT_LOGGER + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")
