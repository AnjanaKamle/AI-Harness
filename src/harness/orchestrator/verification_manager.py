"""VerificationManager: the only component that can declare VERIFIED_SUCCESS.

Checks are computed from the task graph, shared state and the repository itself - never
from an agent's or model's claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from harness.orchestrator.graph import NodeKind, NodeStatus, TaskGraph
from harness.orchestrator.state import AgentState
from harness.tools.git import GitDiffTool, GitStatusTool
from harness.tools.repository import RepositoryContext
from harness.verification.models import VerificationStatus


class FinalStatus(StrEnum):
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class VerificationDecision:
    status: FinalStatus
    checks: list[VerificationCheck] = field(default_factory=list)
    unresolved_issues: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def verified(self) -> bool:
        return self.status is FinalStatus.VERIFIED_SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "summary": self.summary,
                "checks": [asdict(c) for c in self.checks],
                "unresolved_issues": self.unresolved_issues}


def changed_files(state: AgentState) -> list[str]:
    files: list[str] = []
    for change in state.code_changes:
        files.extend(change.get("files_changed", []))
    return list(dict.fromkeys(files))


class VerificationManager:
    def __init__(self, repo: RepositoryContext) -> None:
        self.repo = repo

    def _task_behavior_check(self, state: AgentState, latest: dict[str, Any]) -> VerificationCheck:
        """Passing tests only prove the task if some test exercises the change: a test that
        failed at baseline now passes, a test related to the changed code ran, or the change
        includes tests."""
        from harness.verification.detection import inspect_project, related_python_tests

        baseline = state.baseline or {}
        if not baseline.get("available"):
            return VerificationCheck("task_behavior_verified", True,
                                     "baseline unavailable; relying on the test results")
        fixed = (latest.get("baseline_comparison") or {}).get("fixed") or []
        files = changed_files(state)
        changed_tests = [f for f in files if f.rsplit("/", 1)[-1].startswith("test_")
                         or f.endswith("_test.py")]
        related = related_python_tests(self.repo, files, inspect_project(self.repo)) if files else []
        if fixed:
            return VerificationCheck("task_behavior_verified", True,
                                     f"failing before the change, passing now: {', '.join(fixed[:5])}")
        if changed_tests:
            return VerificationCheck("task_behavior_verified", True,
                                     f"the change includes tests: {', '.join(changed_tests[:5])}")
        if related:
            return VerificationCheck("task_behavior_verified", True,
                                     f"tests exercising the changed code ran: {', '.join(related[:5])}")
        return VerificationCheck(
            "task_behavior_verified", False,
            "no test exercises the changed code (nothing failed before and passes now, no related "
            "or new tests) - the requested behaviour is not verified",
        )

    def evaluate(
        self, state: AgentState, graph: TaskGraph, *, fatal_error: str | None = None
    ) -> VerificationDecision:
        checks: list[VerificationCheck] = []
        work = [n for n in graph if n.kind is not NodeKind.VERIFY]

        # 1. required agent tasks succeeded (a failed node superseded by a repair is resolved)
        unresolved = [
            n for n in work
            if n.required and n.status is not NodeStatus.SUCCESS and not n.superseded_by
        ]
        checks.append(VerificationCheck(
            "required_tasks_succeeded", not unresolved,
            "all required tasks succeeded" if not unresolved else
            "; ".join(f"{n.id} {n.status.value}: {n.error or ''}".strip() for n in unresolved),
        ))

        coding = bool(graph.by_kind(NodeKind.IMPLEMENT))
        if coding:
            # 2. required code changes were applied (and are visible in the repository)
            files = changed_files(state)
            ok, detail = bool(files), f"files changed: {', '.join(files)}" if files else "no file was changed"
            if files and self.repo.is_git_repo:
                diff = GitDiffTool(self.repo).run({})
                ok = diff.ok and bool(diff.data.get("has_changes"))
                detail = (str(diff.data.get("summary")) if ok else
                          f"git diff shows no changes ({diff.error or 'reverted?'})")
            checks.append(VerificationCheck("code_changes_applied", ok, detail))

        tests = [n for n in graph.by_kind(NodeKind.TEST) if not n.superseded_by]
        if tests:
            # 3. relevant tests actually ran and passed (evidence from the Tester only)
            latest = state.latest_test_result or {}
            accepted = bool(latest.get("baseline_accepted"))
            passed = (
                all(n.status is NodeStatus.SUCCESS for n in tests)
                and state.verification_status is VerificationStatus.PASSED
                and (latest.get("status") == "PASS" or accepted)
            )
            detail = (
                f"{latest.get('command', '?')} -> {latest.get('status', 'not run')} "
                f"(passed={latest.get('passed')}, failed={latest.get('failed')})"
            )
            if accepted:
                pre = (latest.get("baseline_comparison") or {}).get("preexisting_failures", [])
                detail += f"; only pre-existing unrelated failures remain: {', '.join(pre[:5])}"
            checks.append(VerificationCheck("tests_passed", passed, detail))
            if coding and state.baseline is not None:
                checks.append(self._task_behavior_check(state, latest))

        # 4. no blocking failures remain
        blocking = [n.id for n in work if n.required and n.status is NodeStatus.BLOCKED]
        violations = [v for n in tests for v in (n.result or {}).get("integrity_violations", [])]
        coder_nodes = [n for n in graph if n.kind in (NodeKind.IMPLEMENT, NodeKind.REPAIR)
                       and n.status is not NodeStatus.PENDING]
        last_coder = coder_nodes[-1] if coder_nodes else None
        problems = [f"blocked: {', '.join(blocking)}"] if blocking else []
        problems += violations
        if fatal_error:
            problems.append(f"fatal error: {fatal_error}")
        if last_coder is not None and last_coder.status is not NodeStatus.SUCCESS:
            problems.append(f"last coding step {last_coder.id} is {last_coder.status.value}")
        checks.append(VerificationCheck(
            "no_blocking_failures", not problems, "; ".join(problems) or "none"
        ))

        # 5. the final repository state is inspectable
        if self.repo.is_git_repo:
            status = GitStatusTool(self.repo).run({})
            inspectable, detail = status.ok, (
                f"git status ok ({len(status.data.get('files', []))} changed path(s))" if status.ok
                else f"git status failed: {status.error}"
            )
        else:
            inspectable = self.repo.root.is_dir()
            detail = "repository directory readable (not a git repository)"
        checks.append(VerificationCheck("repository_inspectable", inspectable, detail))

        issues = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
        optional_failures = [
            f"optional task {n.id} {n.status.value}: {n.error or ''}".strip()
            for n in work if not n.required and n.status is not NodeStatus.SUCCESS
        ]
        if fatal_error:
            status = FinalStatus.FAILED
        elif not issues:
            status = FinalStatus.VERIFIED_SUCCESS
        elif blocking or any(n.status is NodeStatus.BLOCKED for n in work if n.required):
            status = FinalStatus.BLOCKED
        elif {c.name for c in checks if not c.passed} <= {"task_behavior_verified",
                                                          "required_tasks_succeeded"}:
            status = FinalStatus.BLOCKED  # the change could not be verified, not a crash
        else:
            status = FinalStatus.FAILED
        summary = (
            "All verification checks passed." if status is FinalStatus.VERIFIED_SUCCESS
            else f"{len(issues)} verification check(s) failed."
        )
        return VerificationDecision(status, checks, issues + optional_failures, summary)
