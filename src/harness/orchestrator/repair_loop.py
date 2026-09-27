"""Coder -> Tester repair loop with evidence-based completion.

    Coder -> Tester -> PASS? --yes--> VERIFIED_SUCCESS
                         \\--no--> classify -> (repairable & attempts left) -> Coder (with evidence)
                                          \\-> otherwise BLOCKED

VERIFIED_SUCCESS requires ALL of:
  1. code changes exist (made through the Coder's tools, and visible in git when available)
  2. relevant verification actually executed
  3. that verification passed
  4. no unresolved critical error (Coder must report SUCCESS - which itself requires an
     inspected diff - and verification must not have modified the repository)

Nothing the Coder *says* ("everything works") counts as evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from harness.agents.base import AgentResult, AgentStatus
from harness.context.manager import ContextCategory
from harness.orchestrator.state import AgentState, StepStatus, TaskStatus
from harness.tools.git import GitDiffTool
from harness.tools.repository import RepositoryContext
from harness.verification.models import (
    FailureCategory,
    FailureClassification,
    TestResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    from harness.agents.coder import CoderAgent
    from harness.agents.tester import TesterAgent, TesterResult

log = logging.getLogger("harness.orchestrator.repair")

MAX_EVIDENCE_CHARS = 6_000
STDERR_TAIL_CHARS = 1_500


@dataclass
class _Failure:
    """One failed attempt, as recorded in failure_history."""

    attempt: int
    category: FailureCategory
    repairable: bool
    reason: str
    evidence: list[str] = field(default_factory=list)
    command: str | None = None
    exit_code: int | None = None
    counts: str = ""
    failed_tests: list[str] = field(default_factory=list)
    failure_summary: str = ""
    stderr_tail: str = ""
    coder_status: str = ""
    files_changed: list[str] = field(default_factory=list)

    def signature(self) -> tuple[Any, ...]:
        return (self.category, self.command, tuple(sorted(self.failed_tests)), self.counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "category": self.category.value,
            "repairable": self.repairable,
            "reason": self.reason,
            "evidence": self.evidence,
            "command": self.command,
            "exit_code": self.exit_code,
            "counts": self.counts,
            "failed_tests": self.failed_tests,
            "failure_summary": self.failure_summary,
            "stderr_tail": self.stderr_tail,
            "coder_status": self.coder_status,
            "files_changed": self.files_changed,
        }


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n...[{len(text) - limit} more chars]"


class RepairLoop:
    def __init__(
        self,
        coder: CoderAgent,
        tester: TesterAgent,
        repo: RepositoryContext,
        *,
        max_repair_attempts: int,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be >= 0")
        self.coder = coder
        self.tester = tester
        self.repo = repo
        self.max_repair_attempts = max_repair_attempts

    # --- public ----------------------------------------------------------------------------

    def run(self, state: AgentState) -> AgentState:
        original_task = state.task
        coder_task = original_task
        files_changed: list[str] = []
        state.status = TaskStatus.IN_PROGRESS

        attempt = 0
        while True:
            attempt += 1
            if attempt > 1:
                self._compress_context(state)
            action = "implement" if attempt == 1 else "repair"
            log.info("Attempt %d | agent=coder | action=%s", attempt, action)
            self._set_step(state, "coder", StepStatus.IN_PROGRESS)
            coder_result = self.coder.run(coder_task, state)
            new_files = list(getattr(coder_result, "files_changed", []) or [])
            files_changed.extend(f for f in new_files if f not in files_changed)
            log.info(
                "Attempt %d | agent=coder | action=%s | result=%s | files_changed=%s",
                attempt, action, coder_result.status, new_files or "none",
            )
            record: dict[str, Any] = {
                "attempt": attempt,
                "action": action,
                "coder": {
                    "status": coder_result.status.value,
                    "summary": coder_result.summary,
                    "files_changed": new_files,
                    "errors": coder_result.errors,
                },
                "tester": None,
                "classification": None,
            }
            state.attempt_history.append(record)

            if coder_result.status is AgentStatus.BLOCKED:
                return self._stop(state, f"Coder is blocked: {coder_result.summary}", files_changed)
            if coder_result.metadata.get("llm_error"):
                return self._stop(
                    state, f"Language model failure: {'; '.join(coder_result.errors)}", files_changed
                )

            # --- verification -------------------------------------------------------------
            state.status = TaskStatus.VERIFYING
            self._set_step(state, "coder", StepStatus.DONE)
            self._set_step(state, "tester", StepStatus.IN_PROGRESS)
            log.info("Attempt %d | agent=tester | action=verify", attempt)
            tester_result = self._as_tester_result(self.tester.run(original_task, state))
            self._log_tests(attempt, tester_result)
            state.verification_status = tester_result.verification_status
            record["tester"] = {
                "verification_status": tester_result.verification_status.value,
                "summary": tester_result.summary,
                "tests": [
                    {"command": r.command, "status": r.status.value, "counts": r.counts_text(),
                     "gating": r.gating}
                    for r in tester_result.test_results
                ],
                "integrity_violations": tester_result.integrity_violations,
            }

            # --- evidence-based decision -------------------------------------------------
            changes_exist, change_evidence = self._changes_exist(files_changed)
            if (
                tester_result.verification_status is VerificationStatus.PASSED
                and changes_exist
                and coder_result.status is AgentStatus.SUCCESS
                and not tester_result.integrity_violations
            ):
                record["outcome"] = "VERIFIED"
                log.info("Attempt %d | result=VERIFIED_SUCCESS | %s", attempt, tester_result.summary)
                return self._finish(state, files_changed, change_evidence, tester_result)

            failure = self._failure(
                attempt, coder_result, tester_result, changes_exist, change_evidence, new_files
            )
            state.failure_history.append(failure.to_dict())
            record["classification"] = {
                "category": failure.category.value,
                "repairable": failure.repairable,
                "reason": failure.reason,
            }
            log.info(
                "Attempt %d | result=FAILED | classification=%s | repairable=%s | %s",
                attempt, failure.category, failure.repairable, failure.reason,
            )

            # --- stop conditions ------------------------------------------------------------
            if tester_result.integrity_violations:
                record["outcome"] = "STOP"
                return self._stop(state, "Verification modified repository files", files_changed)
            if tester_result.verification_status is VerificationStatus.NOT_AVAILABLE:
                record["outcome"] = "STOP"
                return self._stop(
                    state, f"Cannot verify the change: {tester_result.summary}", files_changed
                )
            if not failure.repairable:
                record["outcome"] = "STOP"
                return self._stop(
                    state,
                    f"{failure.category.value} cannot be fixed by code changes: {failure.reason}",
                    files_changed,
                )
            if state.repair_attempts >= self.max_repair_attempts:
                record["outcome"] = "STOP"
                return self._stop(
                    state,
                    f"Maximum repair attempts ({self.max_repair_attempts}) reached without "
                    "passing verification",
                    files_changed,
                )
            if self._no_progress(state, failure, new_files):
                record["outcome"] = "STOP"
                return self._stop(
                    state,
                    "No progress: the Coder made no further changes and the failure is "
                    "unchanged - the task appears impossible with the available information",
                    files_changed,
                )

            record["outcome"] = "RETRY"
            state.repair_attempts += 1
            state.retry_count = state.repair_attempts
            coder_task = self.repair_prompt(original_task, state, failure, files_changed)

    # --- evidence ----------------------------------------------------------------------------

    def _changes_exist(self, files_changed: list[str]) -> tuple[bool, str]:
        if not files_changed:
            return False, "the Coder has not modified any file"
        if self.repo.is_git_repo:
            diff = GitDiffTool(self.repo).run({})
            if not diff.ok:
                return False, f"could not inspect git diff: {diff.error}"
            if not diff.data.get("has_changes"):
                return False, "git diff shows no changes (edits were reverted?)"
            return True, str(diff.data.get("summary"))
        return True, f"files modified through tools: {', '.join(files_changed)}"

    def _failure(
        self,
        attempt: int,
        coder_result: AgentResult,
        tester_result: TesterResult,
        changes_exist: bool,
        change_evidence: str,
        new_files: list[str],
    ) -> _Failure:
        decisive: TestResult | None = tester_result.decisive_result
        classification: FailureClassification | None = tester_result.failure_classification
        common: dict[str, Any] = {
            "attempt": attempt,
            "coder_status": coder_result.status.value,
            "files_changed": new_files,
        }
        if decisive is not None:
            common.update(
                command=decisive.command,
                exit_code=decisive.exit_code,
                counts=decisive.counts_text(),
                failed_tests=list(decisive.failed_tests),
                failure_summary=_clip(decisive.failure_summary, MAX_EVIDENCE_CHARS),
                stderr_tail=decisive.stderr[-STDERR_TAIL_CHARS:],
            )

        if tester_result.integrity_violations:
            return _Failure(
                category=FailureCategory.ENVIRONMENT_FAILURE, repairable=False,
                reason="; ".join(tester_result.integrity_violations), **common,
            )
        if tester_result.verification_status is not VerificationStatus.PASSED:
            if classification is None:
                return _Failure(
                    category=FailureCategory.ENVIRONMENT_FAILURE, repairable=False,
                    reason=tester_result.summary, **common,
                )
            return _Failure(
                category=classification.category,
                repairable=classification.repairable,
                reason=tester_result.summary,
                evidence=list(classification.evidence),
                **common,
            )
        # Verification passed, but completion evidence is still missing.
        problems = []
        if not changes_exist:
            problems.append(change_evidence)
        if coder_result.status is not AgentStatus.SUCCESS:
            problems.append(
                f"Coder reported {coder_result.status.value}: {'; '.join(coder_result.errors) or coder_result.summary}"
            )
        return _Failure(
            category=FailureCategory.UNKNOWN_FAILURE,
            repairable=True,
            reason="Tests pass but completion is not proven: " + "; ".join(problems),
            **common,
        )

    @staticmethod
    def _no_progress(state: AgentState, failure: _Failure, new_files: list[str]) -> bool:
        if new_files or len(state.failure_history) < 2:
            return False
        previous = state.failure_history[-2]
        return (
            previous["category"],
            previous["command"],
            tuple(sorted(previous["failed_tests"])),
            previous["counts"],
        ) == failure.signature()

    # --- prompts -----------------------------------------------------------------------------

    def repair_prompt(
        self, original_task: str, state: AgentState, failure: _Failure, files_changed: list[str]
    ) -> str:
        history = "\n".join(
            f"- attempt {f['attempt']}: {f['category']} - {f['counts'] or f['reason'][:120]}"
            for f in state.failure_history
        )
        parts = [
            f"ORIGINAL TASK:\n{original_task}",
            f"REPAIR ATTEMPT {state.repair_attempts} of {self.max_repair_attempts}. The previous "
            "attempt did NOT pass objective verification. Use the evidence below to find the root "
            "cause and fix it. Do not repeat a change that already failed.",
            f"FAILURE CLASSIFICATION: {failure.category.value}\n"
            + "\n".join(f"  - {e}" for e in failure.evidence),
            f"REASON: {failure.reason}",
        ]
        if failure.command:
            parts.append(
                f"VERIFICATION COMMAND: {failure.command}\n"
                f"EXIT CODE: {failure.exit_code}\nRESULT: {failure.counts}"
            )
        if failure.failed_tests:
            parts.append("FAILING TESTS:\n" + "\n".join(f"  - {t}" for t in failure.failed_tests))
        if failure.failure_summary:
            parts.append(f"FAILURE DETAILS:\n{failure.failure_summary}")
        if failure.stderr_tail.strip() and failure.stderr_tail.strip() not in failure.failure_summary:
            parts.append(f"STDERR (tail):\n{failure.stderr_tail}")
        parts.append(f"ATTEMPT HISTORY:\n{history}")
        parts.append(
            "FILES CHANGED SO FAR: " + (", ".join(files_changed) if files_changed else "none")
        )
        parts.append(
            "Fix the production code (not the tests, unless the task requires it), then inspect "
            "git_diff and return your structured result."
        )
        return "\n\n".join(parts)

    # --- bookkeeping -------------------------------------------------------------------------

    @staticmethod
    def _as_tester_result(result: AgentResult) -> TesterResult:
        from harness.agents.tester import TesterResult

        if isinstance(result, TesterResult):
            return result
        # The tester crashed (BaseAgent converted the exception): no evidence was produced.
        return TesterResult(
            agent_name=result.agent_name,
            status=AgentStatus.BLOCKED,
            summary=f"Tester failed to run: {'; '.join(result.errors) or result.summary}",
            errors=result.errors,
            verification_status=VerificationStatus.NOT_AVAILABLE,
        )

    @staticmethod
    def _log_tests(attempt: int, tester_result: TesterResult) -> None:
        category = (
            tester_result.failure_classification.category.value
            if tester_result.failure_classification else "-"
        )
        for r in tester_result.test_results:
            log.info(
                "Attempt %d | agent=tester | action=run %s | command=%s | result=%s (%s) | "
                "classification=%s",
                attempt, r.kind.value.lower(), r.command, r.status.value, r.counts_text(),
                category if r is tester_result.decisive_result and not r.ok else "-",
            )
        if not tester_result.test_results:
            log.info("Attempt %d | agent=tester | result=%s | %s", attempt,
                     tester_result.verification_status, tester_result.summary)

    @staticmethod
    def _set_step(state: AgentState, agent: str, status: StepStatus) -> None:
        for step in state.plan:
            if step.agent == agent and step.status is not StepStatus.SKIPPED:
                step.status = status

    def _compress_context(self, state: AgentState) -> None:
        """Replace older per-attempt entries with structured summaries (bounded store)."""
        store = self.coder.context
        for category in (ContextCategory.TEST_RESULT, ContextCategory.AGENT_OUTPUT,
                         ContextCategory.RELEVANT_FILE):
            summary = store.compress(category, keep_last=3, task_id=state.task_id)
            if summary is not None:
                log.debug("compressed %s context into summary (%s entries)", category,
                          summary.metadata.get("compressed_entries"))

    def _tests_run(self, state: AgentState) -> list[dict[str, Any]]:
        return [
            {"attempt": t.get("attempt"), "command": t["command"], "status": t["status"],
             "passed": t.get("passed"), "failed": t.get("failed"), "exit_code": t.get("exit_code")}
            for t in state.test_results
        ]

    def _finish(
        self,
        state: AgentState,
        files_changed: list[str],
        change_evidence: str,
        tester_result: TesterResult,
    ) -> AgentState:
        state.status = TaskStatus.VERIFIED_SUCCESS
        state.verification_status = VerificationStatus.PASSED
        self._set_step(state, "tester", StepStatus.DONE)
        state.completed_tasks = [s.id for s in state.plan]
        state.pending_tasks = []
        state.outcome = {
            "status": TaskStatus.VERIFIED_SUCCESS.value,
            "reason": "Code changes exist and verification passed",
            "evidence": {
                "changes": change_evidence,
                "verification": tester_result.summary,
                "commands": tester_result.commands_run,
            },
            "attempts": len(state.attempt_history),
            "repair_attempts": state.repair_attempts,
            "files_changed": files_changed,
            "tests_run": self._tests_run(state),
            "failure_history": state.failure_history,
        }
        state.final_result = (
            f"VERIFIED_SUCCESS after {len(state.attempt_history)} attempt(s): "
            f"{tester_result.summary}. Files changed: {', '.join(files_changed)}"
        )
        return state

    def _stop(self, state: AgentState, reason: str, files_changed: list[str]) -> AgentState:
        state.status = TaskStatus.BLOCKED
        self._set_step(state, "tester", StepStatus.FAILED)
        latest = state.failure_history[-1] if state.failure_history else None
        state.outcome = {
            "status": TaskStatus.BLOCKED.value,
            "reason": reason,
            "attempted": [
                {
                    "attempt": a["attempt"],
                    "action": a["action"],
                    "coder": a["coder"]["summary"],
                    "verification": (a["tester"] or {}).get("verification_status"),
                    "outcome": a.get("outcome"),
                }
                for a in state.attempt_history
            ],
            "latest_failure": latest,
            "previous_failures": state.failure_history[:-1],
            "files_changed": files_changed,
            "tests_run": self._tests_run(state),
            "verification_status": state.verification_status.value,
            "repair_attempts": state.repair_attempts,
        }
        state.final_result = f"BLOCKED: {reason}"
        log.warning("Task %s BLOCKED: %s", state.task_id, reason)
        return state
