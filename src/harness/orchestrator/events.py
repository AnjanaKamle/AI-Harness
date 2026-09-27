"""Orchestration event log: every scheduling/agent/tool/verification event, timestamped.

Events are logged to ``harness.events`` and kept in memory for the final result. Details are
passed through :func:`redact` so credentials never reach logs.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

log = logging.getLogger("harness.events")

MAX_DETAIL_CHARS = 300
_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,}|AKIA[0-9A-Z]{16}|"
    r"xox[baprs]-[A-Za-z0-9-]{8,}|(?i:(?:api[_-]?key|token|secret|password)\s*[=:]\s*)\S+)"
)


class EventType(StrEnum):
    TASK_CREATED = "TASK_CREATED"
    PLAN_CREATED = "PLAN_CREATED"
    NODE_READY = "NODE_READY"
    AGENT_STARTED = "AGENT_STARTED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_CALLED = "TOOL_CALLED"
    TOOL_FAILED = "TOOL_FAILED"
    TOOL_RETRIED = "TOOL_RETRIED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"
    AGENT_RETRIED = "AGENT_RETRIED"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    STATE_REJECTED = "STATE_REJECTED"
    RESEARCH_FAILED = "RESEARCH_FAILED"
    TEST_PASSED = "TEST_PASSED"
    TEST_FAILED = "TEST_FAILED"
    REPAIR_STARTED = "REPAIR_STARTED"
    NODE_BLOCKED = "NODE_BLOCKED"
    NODE_SKIPPED = "NODE_SKIPPED"
    VERIFICATION_STARTED = "VERIFICATION_STARTED"
    VERIFICATION_PASSED = "VERIFICATION_PASSED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_BLOCKED = "TASK_BLOCKED"


def redact(text: str) -> str:
    text = _SECRET.sub("[REDACTED]", text)
    api_key = os.environ.get("AI_API_KEY", "")
    if len(api_key) >= 6:
        text = text.replace(api_key, "[REDACTED]")
    return text


@dataclass(frozen=True)
class OrchestrationEvent:
    timestamp: str
    task_id: str
    agent: str
    event: EventType
    details: str
    node: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "event": self.event.value}


Listener = Callable[["OrchestrationEvent"], None]


class EventLog:
    def __init__(self, task_id: str = "") -> None:
        self.task_id = task_id
        self._events: list[OrchestrationEvent] = []
        self._lock = threading.Lock()
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> None:
        """Call ``listener(event)`` for every new event (e.g. a UI). Listener errors are
        logged and never affect orchestration."""
        with self._lock:
            self._listeners.append(listener)

    def emit(
        self,
        event: EventType,
        details: str = "",
        *,
        agent: str = "orchestrator",
        node: str | None = None,
        **data: Any,
    ) -> OrchestrationEvent:
        clean = redact(details)[:MAX_DETAIL_CHARS]
        record = OrchestrationEvent(
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
            task_id=self.task_id,
            agent=agent,
            event=event,
            details=clean,
            node=node,
            data={k: redact(v) if isinstance(v, str) else v for k, v in data.items()},
        )
        with self._lock:
            self._events.append(record)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(record)
            except Exception:  # noqa: BLE001 - presentation must never break the core
                log.debug("event listener failed", exc_info=True)
        log.info(
            "%s task=%s agent=%s%s | %s",
            event.value, self.task_id, agent, f" node={node}" if node else "", clean,
        )
        return record

    @property
    def events(self) -> list[OrchestrationEvent]:
        with self._lock:
            return list(self._events)

    def of_type(self, event: EventType) -> list[OrchestrationEvent]:
        return [e for e in self.events if e.event is event]

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]
