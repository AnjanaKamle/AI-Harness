"""AgentState: the shared, JSON-serializable record of one task run."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from harness.verification.models import VerificationStatus


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"  # changes exist AND verification passed
    BLOCKED = "BLOCKED"  # stopped without verified success (limits, environment, agent)


class StepStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class PlanStep:
    id: str
    description: str
    agent: str
    status: StepStatus = StepStatus.PENDING


def _new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class AgentState:
    task: str
    task_id: str = field(default_factory=_new_task_id)
    status: TaskStatus = TaskStatus.PENDING
    plan: list[PlanStep] = field(default_factory=list)
    active_agent: str | None = None
    completed_tasks: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)
    research_findings: list[dict[str, Any]] = field(default_factory=list)
    code_changes: list[dict[str, Any]] = field(default_factory=list)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    final_result: str | None = None
    created_at: str = field(default_factory=_now)
    # --- verification / repair loop (Phase 3) ---
    verification_status: VerificationStatus = VerificationStatus.NOT_VERIFIED
    verification_attempts: int = 0
    repair_attempts: int = 0
    latest_test_result: dict[str, Any] | None = None
    failure_history: list[dict[str, Any]] = field(default_factory=list)
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    outcome: dict[str, Any] | None = None
    # --- research / observability (Phase 4) ---
    research: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    # --- autonomous controller (Phase 5) ---
    task_graph: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None  # test results before any change (Tester)
    retry_history: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        data = dict(data)
        data["status"] = TaskStatus(data.get("status", TaskStatus.PENDING))
        data["verification_status"] = VerificationStatus(
            data.get("verification_status", VerificationStatus.NOT_VERIFIED)
        )
        data["plan"] = [
            PlanStep(**{**step, "status": StepStatus(step.get("status", StepStatus.PENDING))})
            for step in data.get("plan", [])
        ]
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> AgentState:
        return cls.from_dict(json.loads(text))
