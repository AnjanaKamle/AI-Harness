"""RecoveryManager: decides what happens after a failure, with a reason, within limits.

    TOOL FAILURE      retry the tool if the failure is transient   (MAX_TOOL_RETRIES, in registry)
    AGENT FAILURE     retry the agent if safe                       (MAX_AGENT_RETRIES)
    MALFORMED OUTPUT  retry the agent                               (MAX_AGENT_RETRIES)
    TEST FAILURE      repair: Coder receives the failure evidence   (MAX_REPAIR_ATTEMPTS)
    RESEARCH FAILURE  retry if transient, else proceed if research is optional, else block
    TIMEOUT           operation terminated -> retry if safe, otherwise BLOCKED

Every decision - including every retry - is recorded in the retry history with its reason.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from harness.config.settings import Settings
from harness.orchestrator.graph import TaskNode


class FailureKind(StrEnum):
    TOOL_FAILURE = "TOOL_FAILURE"
    AGENT_FAILURE = "AGENT_FAILURE"
    MALFORMED_OUTPUT = "MALFORMED_OUTPUT"
    LLM_ERROR = "LLM_ERROR"
    TEST_FAILURE = "TEST_FAILURE"
    RESEARCH_FAILURE = "RESEARCH_FAILURE"
    TIMEOUT = "TIMEOUT"
    BLOCKED_BY_AGENT = "BLOCKED_BY_AGENT"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
    INTEGRITY_VIOLATION = "INTEGRITY_VIOLATION"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    OPTIONAL_TASK_FAILURE = "OPTIONAL_TASK_FAILURE"


class RecoveryAction(StrEnum):
    RETRY = "RETRY"  # run the same node again
    REPAIR = "REPAIR"  # add Coder repair + re-test nodes with the failure evidence
    PROCEED = "PROCEED"  # record the failure; dependents continue (optional work)
    BLOCK = "BLOCK"  # stop this branch; the task cannot be verified


@dataclass(frozen=True)
class RetryPolicy:
    max_tool_retries: int = 2
    max_agent_retries: int = 2
    max_repair_attempts: int = 5

    @classmethod
    def from_settings(cls, settings: Settings) -> RetryPolicy:
        return cls(settings.max_tool_retries, settings.max_agent_retries,
                   settings.max_repair_attempts)


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    kind: FailureKind
    reason: str


class RecoveryManager:
    def __init__(self, policy: RetryPolicy) -> None:
        self.policy = policy
        self._history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def record(self, **entry: Any) -> None:
        entry.setdefault("timestamp", datetime.now(UTC).isoformat(timespec="milliseconds"))
        with self._lock:
            self._history.append(entry)

    def record_tool_retry(self, node_id: str, agent: str, tool: str, attempt: int, reason: str) -> None:
        self.record(node=node_id, agent=agent, level="tool", kind=FailureKind.TOOL_FAILURE.value,
                    action=RecoveryAction.RETRY.value, attempt=attempt, tool=tool,
                    reason=f"transient tool failure: {reason}")

    def decide(
        self,
        node: TaskNode,
        kind: FailureKind,
        *,
        detail: str,
        retryable: bool = True,
        repairable: bool = False,
        repairs_done: int = 0,
        can_proceed: bool = False,
    ) -> RecoveryDecision:
        agent_retries_left = node.retry_count < self.policy.max_agent_retries

        def decision(action: RecoveryAction, reason: str) -> RecoveryDecision:
            made = RecoveryDecision(action, kind, reason)
            self.record(node=node.id, agent=node.agent, level="node", kind=kind.value,
                        action=action.value, attempt=node.retry_count + 1, reason=reason,
                        detail=detail[:300])
            return made

        if not node.required:
            if retryable and agent_retries_left and kind in (FailureKind.LLM_ERROR, FailureKind.TIMEOUT):
                return decision(RecoveryAction.RETRY, f"optional task hit a transient {kind.value}; retrying")
            return decision(RecoveryAction.PROCEED,
                            f"optional task failed ({kind.value}); it does not affect the main task")

        if kind in (FailureKind.BLOCKED_BY_AGENT, FailureKind.VERIFICATION_UNAVAILABLE,
                    FailureKind.INTEGRITY_VIOLATION, FailureKind.PERMISSION_DENIED):
            return decision(RecoveryAction.BLOCK, f"{kind.value} cannot be recovered automatically")

        if kind is FailureKind.TEST_FAILURE:
            if not repairable:
                return decision(RecoveryAction.BLOCK, "failure is not fixable by code changes")
            if repairs_done >= self.policy.max_repair_attempts:
                return decision(RecoveryAction.BLOCK,
                                f"maximum repair attempts ({self.policy.max_repair_attempts}) reached")
            return decision(RecoveryAction.REPAIR,
                            f"repairable test failure; repair {repairs_done + 1} of "
                            f"{self.policy.max_repair_attempts} with the failure evidence")

        if kind is FailureKind.RESEARCH_FAILURE:
            if retryable and agent_retries_left:
                return decision(RecoveryAction.RETRY, "research failed transiently; retrying")
            if can_proceed:
                return decision(RecoveryAction.PROCEED,
                                "research failed but the Coder can proceed on local evidence")
            return decision(RecoveryAction.BLOCK,
                            "research failed and is required - refusing to guess")

        if kind is FailureKind.TIMEOUT:
            if retryable and agent_retries_left:
                return decision(RecoveryAction.RETRY,
                                "operation timed out and was terminated; safe to retry")
            return decision(RecoveryAction.BLOCK, "operation timed out and retrying is not possible")

        if kind is FailureKind.LLM_ERROR and not retryable:
            return decision(RecoveryAction.BLOCK, "non-retryable model/provider error")

        # AGENT_FAILURE, MALFORMED_OUTPUT, retryable LLM_ERROR, TOOL_FAILURE
        if retryable and agent_retries_left:
            return decision(RecoveryAction.RETRY,
                            f"{kind.value}; retry {node.retry_count + 1} of {self.policy.max_agent_retries}")
        return decision(RecoveryAction.BLOCK,
                        f"{kind.value}; agent retries exhausted ({self.policy.max_agent_retries})")

    def to_dict(self) -> dict[str, Any]:
        return {"policy": asdict(self.policy), "history": self.history}
