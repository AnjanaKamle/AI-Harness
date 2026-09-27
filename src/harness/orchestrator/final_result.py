"""FinalResult: the single, structured answer to 'what happened and is it done?'"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from harness.orchestrator.state import AgentState
from harness.orchestrator.verification_manager import FinalStatus


@dataclass
class FinalResult:
    task_id: str
    status: FinalStatus
    summary: str
    files_changed: list[str] = field(default_factory=list)
    tests_run: list[dict[str, Any]] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    research_performed: bool = False
    tool_calls: int = 0
    retries: int = 0
    retry_history: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    unresolved_issues: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    plan: list[dict[str, Any]] = field(default_factory=list)
    music: dict[str, Any] | None = None
    events: int = 0
    error_code: str | None = None  # e.g. AUTHENTICATION_ERROR, PROVIDER_TIMEOUT
    state: AgentState | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "summary": self.summary,
            "files_changed": self.files_changed,
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "research_performed": self.research_performed,
            "tool_calls": self.tool_calls,
            "retries": self.retries,
            "retry_history": self.retry_history,
            "duration_seconds": self.duration_seconds,
            "unresolved_issues": self.unresolved_issues,
            "verification": self.verification,
            "plan": self.plan,
            "music": self.music,
            "events": self.events,
            "error_code": self.error_code,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
