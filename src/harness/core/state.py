"""Shared state / context that every agent reads from and writes to."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.task import Task


@dataclass(frozen=True)
class StateEvent:
    """Append-only record of something that happened during a run."""

    source: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class SharedState:
    """Thread-safe key/value context plus an event log, scoped to one task."""

    def __init__(self, task: Task) -> None:
        self.task = task
        self._data: dict[str, Any] = {}
        self._events: list[StateEvent] = []
        self._lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any, *, source: str = "system") -> None:
        with self._lock:
            self._data[key] = value
            self._events.append(StateEvent(source=source, kind="set", data={"key": key}))

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data

    def record(self, source: str, kind: str, **data: Any) -> StateEvent:
        event = StateEvent(source=source, kind=kind, data=data)
        with self._lock:
            self._events.append(event)
        return event

    @property
    def events(self) -> list[StateEvent]:
        with self._lock:
            return list(self._events)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "task": {
                    "id": self.task.id,
                    "description": self.task.description,
                    "status": self.task.status.value,
                    "attempts": self.task.attempts,
                    "metadata": self.task.metadata,
                },
                "data": dict(self._data),
                "events": [asdict(e) for e in self._events],
            }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2, default=str), encoding="utf-8")
