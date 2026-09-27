"""TesterAgent: produces verification evidence. It never modifies the repository.

The Tester is deterministic by design: which commands to run is decided from repository
evidence (TestRunner detection), and PASS/FAIL comes only from commands that actually ran.
An LLM opinion is never evidence, so the Tester does not ask one.

Order of checks (relevant first):
  1. tests related to the changed files (pytest only) - fast feedback
  2. the project test suite
  3. build (gating) and lint / type checks (advisory) when the project declares them
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from harness.agents.base import AgentResult, AgentStatus, BaseAgent
from harness.config.settings import DEFAULT_TEST_TIMEOUT_SECONDS, Settings
from harness.context.builder import ContextBuilder
from harness.context.manager import ContextCategory, ContextEntry, ContextManager
from harness.llm.client import LLMClient
from harness.orchestrator.state import AgentState
from harness.tools.base import BaseTool
from harness.tools.registry import TESTER_TOOL_NAMES, ToolRegistry
from harness.tools.repository import RepositoryContext, is_ignored_dir
from harness.tools.testing import RunTestsTool, build_testing_tools
from harness.verification.classifier import classify_failure
from harness.verification.detection import related_python_tests
from harness.verification.models import (
    CheckKind,
    FailureClassification,
    TestCommand,
    TestResult,
    TestStatus,
    VerificationStatus,
)
from harness.verification.runner import TestRunner

FORBIDDEN_TESTER_TOOLS = frozenset({"write_file", "edit_file", "terminal"})
MAX_SNAPSHOT_FILES = 20_000
MAX_HASH_BYTES = 5_000_000
_FAILED = (TestStatus.FAIL, TestStatus.ERROR, TestStatus.TIMEOUT)

Snapshot = dict[str, tuple[int, str]]


def snapshot_repository(repo: RepositoryContext) -> Snapshot:
    """Content fingerprint of every non-generated file (used to prove nothing was edited)."""
    snap: Snapshot = {}
    for dirpath, dirnames, filenames in os.walk(repo.root):
        dirnames[:] = [d for d in dirnames if not is_ignored_dir(d)]
        for name in filenames:
            if len(snap) >= MAX_SNAPSHOT_FILES:
                return snap
            path = Path(dirpath, name)
            try:
                size = path.stat().st_size
                digest = (
                    hashlib.sha1(path.read_bytes()).hexdigest()
                    if size <= MAX_HASH_BYTES
                    else f"mtime:{path.stat().st_mtime_ns}"
                )
            except OSError:
                continue
            snap[repo.relative(path)] = (size, digest)
    return snap


@dataclass
class TesterResult(AgentResult):
    __test__ = False  # not a pytest test class

    verification_status: VerificationStatus = VerificationStatus.NOT_VERIFIED
    test_results: list[TestResult] = field(default_factory=list)
    detected_commands: list[dict[str, Any]] = field(default_factory=list)
    failure_classification: FailureClassification | None = None
    decisive_result: TestResult | None = None
    integrity_violations: list[str] = field(default_factory=list)
    generated_files: list[str] = field(default_factory=list)

    @property
    def commands_run(self) -> list[str]:
        return [r.command for r in self.test_results]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_classification"] = (
            self.failure_classification.to_dict() if self.failure_classification else None
        )
        return data


class TesterAgent(BaseAgent):
    __test__ = False  # not a pytest test class

    name = "tester"
    description = (
        "Verifies changes with evidence: detects the project's test/build/lint commands from "
        "repository metadata, runs the relevant ones, parses results and classifies failures. "
        "Never modifies source files."
    )

    def __init__(
        self,
        llm: LLMClient,
        context: ContextManager,
        repo: RepositoryContext,
        tools: Sequence[BaseTool] | None = None,
        *,
        settings: Settings | None = None,
        run_static_checks: bool = True,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        if tools is None:
            from harness.tools.registry import build_default_tools

            timeout = settings.test_timeout_seconds if settings else DEFAULT_TEST_TIMEOUT_SECONDS
            base = [
                t for t in build_default_tools(repo, settings)
                if t.name in TESTER_TOOL_NAMES and t.name not in ("detect_tests", "run_tests")
            ]
            tools = [*base, *build_testing_tools(repo, timeout)]
        forbidden = FORBIDDEN_TESTER_TOOLS.intersection(t.name for t in tools)
        if forbidden:
            raise ValueError(f"TesterAgent must not have modifying tools: {sorted(forbidden)}")
        super().__init__(llm, context, tools)
        self.repo = repo
        self.run_static_checks = run_static_checks
        self.registry = ToolRegistry(self.tools.values())
        run_tool = self.tools.get("run_tests")
        if not isinstance(run_tool, RunTestsTool):
            raise ValueError("TesterAgent requires the run_tests tool")
        self.runner: TestRunner = run_tool.runner
        self.builder = context_builder or ContextBuilder(
            repo, context, max_chars=settings.max_context_chars if settings else 16_000
        )

    # --- helpers -------------------------------------------------------------------------

    @staticmethod
    def changed_files(state: AgentState) -> list[str]:
        files: list[str] = []
        for change in state.code_changes:
            files.extend(change.get("files_changed", []))
        return list(dict.fromkeys(files))

    def _run(self, spec: TestCommand) -> TestResult:
        """Execute through the run_tests tool so every run is logged like any tool call."""
        outcome = self.registry.execute("run_tests", {"command": spec.command, "kind": spec.kind.value})
        if not outcome.ok:
            return TestResult(
                spec.command, TestStatus.ERROR, spec.kind, gating=spec.gating,
                targeted=spec.targeted, failure_summary=outcome.error or "run_tests failed",
            )
        result = TestResult.from_dict(outcome.data)
        result.gating, result.targeted = spec.gating, spec.targeted
        return result

    # --- main ----------------------------------------------------------------------------

    def execute(self, task: str, state: AgentState) -> TesterResult:
        changed = self.changed_files(state)
        brief = self.builder.for_tester(state)  # the Tester's targeted context
        before = snapshot_repository(self.repo)

        profile = self.runner.inspect()
        detected = self.runner.detect(profile)
        results: list[TestResult] = []

        primary_tests = [c for c in detected if c.kind is CheckKind.TEST and c.gating]
        fallback_tests = [c for c in detected if c.kind is CheckKind.TEST and not c.gating]
        builds = [c for c in detected if c.kind is not CheckKind.TEST and c.gating]
        advisory = [c for c in detected if c.kind is not CheckKind.TEST and not c.gating]

        gating_failed = False
        for primary in primary_tests[:1] or fallback_tests[:1]:
            targeted = self.runner.targeted_command(primary, changed, profile)
            if targeted is not None:
                first = self._run(targeted)
                results.append(first)
                if first.status in _FAILED:
                    gating_failed = True
                    break
            full = self._run(primary)
            results.append(full)
            if full.status is TestStatus.NOT_AVAILABLE and fallback_tests and primary.gating:
                full = self._run(replace(fallback_tests[0], gating=True))
                results.append(full)
            gating_failed = full.status in _FAILED
        # Other declared test commands (e.g. a second ecosystem) are also gating evidence.
        for extra in primary_tests[1:]:
            if gating_failed:
                break
            res = self._run(extra)
            results.append(res)
            gating_failed = res.status in _FAILED
        if not gating_failed:
            for spec in builds:
                res = self._run(spec)
                results.append(res)
                if res.status in _FAILED:
                    gating_failed = True
                    break
        if not gating_failed and self.run_static_checks:
            results.extend(self._run(spec) for spec in advisory)

        after = snapshot_repository(self.repo)
        violations = sorted(
            f"{path} was {'deleted' if path not in after else 'modified'} during verification"
            for path, fingerprint in before.items()
            if after.get(path) != fingerprint
        )
        generated = sorted(set(after) - set(before))

        result = self._conclude(detected, results, changed, violations, generated, profile.metadata_files)
        if state.baseline and result.decisive_result is not None:
            self._compare_with_baseline(result, state.baseline, changed, task or state.task, profile)
        result.metadata["context"] = brief.stats
        self._record(state, result)
        return result

    # --- baseline (before any change) ------------------------------------------------------

    def baseline(self, task: str, state: AgentState) -> TesterResult:
        """Run the project's test command before the Coder changes anything and record the
        outcome in ``state.baseline``. Failing tests here are expected (that may be the bug);
        only an inability to run tests makes this (optional) step fail."""
        profile = self.runner.inspect()
        detected = self.runner.detect(profile)
        tests = [c for c in detected if c.kind is CheckKind.TEST]
        primary = next((c for c in tests if c.gating), tests[0] if tests else None)
        if primary is None:
            state.baseline = {"available": False, "reason": "no test command detected"}
            return TesterResult(self.name, AgentStatus.FAILURE,
                                "Baseline skipped: no test command detected from repository evidence",
                                errors=["no test command"], verification_status=VerificationStatus.NOT_AVAILABLE)
        result = self._run(replace(primary, gating=True))
        available = result.status in (TestStatus.PASS, TestStatus.FAIL)
        state.baseline = {
            "available": available,
            "command": result.command,
            "status": result.status.value,
            "passed": result.passed,
            "failed": result.failed,
            "errors": result.errors,
            "failed_tests": list(result.failed_tests),
        }
        summary = (
            f"Baseline before any change: {primary.command} -> {result.status.value} "
            f"({result.counts_text()})"
            + (f"; already failing: {', '.join(result.failed_tests[:5])}" if result.failed_tests else "")
        )
        return TesterResult(
            self.name, AgentStatus.SUCCESS if available else AgentStatus.FAILURE, summary,
            errors=[] if available else [result.failure_summary[:300] or result.status.value],
            verification_status=VerificationStatus.NOT_VERIFIED, test_results=[result],
            decisive_result=result,
        )

    def _compare_with_baseline(
        self,
        result: TesterResult,
        baseline: dict[str, Any],
        changed: list[str],
        task: str,
        profile: Any,
    ) -> None:
        """Before/after comparison. Pre-existing failures that are unrelated to the change are
        not blamed on the agent; new failures are; baseline failures now passing are evidence
        the change works."""
        decisive = result.decisive_result
        if decisive is None or not baseline.get("available"):
            return
        before = set(baseline.get("failed_tests") or [])
        after = set(decisive.failed_tests) if decisive.status is not TestStatus.PASS else set()
        new, still, fixed = sorted(after - before), sorted(after & before), sorted(before - after)
        related = set(related_python_tests(self.repo, changed, profile))
        related |= {c for c in changed if Path(c).name.startswith("test_") or c.endswith("_test.py")}

        def is_related(test_id: str) -> bool:
            file, _, name = test_id.partition("::")
            return file in related or file in task or bool(name and name in task)

        still_related = [t for t in still if is_related(t)]
        accepted = (
            result.verification_status is VerificationStatus.FAILED
            and not result.integrity_violations
            and decisive.status is TestStatus.FAIL
            and not decisive.targeted
            and bool(after) and not new and not still_related
            and not decisive.errors
        )
        comparison = {"new_failures": new, "preexisting_failures": still, "fixed": fixed,
                      "preexisting_related": still_related, "accepted": accepted}
        result.metadata["baseline_comparison"] = comparison
        if accepted:
            result.verification_status = VerificationStatus.PASSED
            result.status = AgentStatus.SUCCESS
            result.failure_classification = None
            result.summary = (
                f"Verification PASSED for the change: {decisive.command} has only pre-existing "
                f"failures unrelated to it ({', '.join(still[:5])}), which also failed before any change."
            )
        elif result.verification_status is VerificationStatus.FAILED and decisive.status is TestStatus.FAIL:
            if new:
                result.summary += f" New failures introduced since the baseline: {', '.join(new[:8])}."
            if still:
                result.summary += (f" Already failing before any change: {', '.join(still[:8])}"
                                   + (" (related to this task)." if still_related else "."))

    def _conclude(
        self,
        detected: list[TestCommand],
        results: list[TestResult],
        changed: list[str],
        violations: list[str],
        generated: list[str],
        metadata_files: list[str],
    ) -> TesterResult:
        gating = [r for r in results if r.gating]
        executed = [r for r in gating if r.status is not TestStatus.NOT_AVAILABLE]
        failing = next((r for r in executed if r.status is not TestStatus.PASS), None)
        errors: list[str] = list(violations)

        if not detected:
            status = VerificationStatus.NOT_AVAILABLE
            where = ", ".join(metadata_files) or "no project metadata files"
            summary = (
                "No verification command could be determined from repository evidence "
                f"(inspected: {where}). Refusing to invent one."
            )
            decisive = None
        elif violations:
            status = VerificationStatus.FAILED
            summary = "Verification modified existing repository files - results are not trusted."
            decisive = failing or (gating[-1] if gating else None)
        elif failing is not None:
            status = VerificationStatus.FAILED
            decisive = failing
            summary = f"Verification FAILED: {failing.command} -> {failing.status} ({failing.counts_text()})"
        elif executed:
            status = VerificationStatus.PASSED
            decisive = executed[-1]
            summary = "Verification PASSED: " + "; ".join(
                f"{r.command} -> {r.status} ({r.counts_text()})" for r in executed
            )
        else:
            status = VerificationStatus.NOT_AVAILABLE
            decisive = gating[-1] if gating else None
            summary = "Verification commands were detected but none could run: " + "; ".join(
                f"{r.command} -> {r.status}: {r.failure_summary[:200]}" for r in gating
            )

        classification = None
        if status is not VerificationStatus.PASSED and decisive is not None:
            classification = classify_failure(decisive, self.repo, changed)
            summary += f" [classification: {classification.category}]"
        advisory_failures = [
            f"advisory check failed: {r.command} ({r.status})"
            for r in results if not r.gating and r.status in _FAILED
        ]

        agent_status = {
            VerificationStatus.PASSED: AgentStatus.SUCCESS,
            VerificationStatus.FAILED: AgentStatus.FAILURE,
        }.get(status, AgentStatus.BLOCKED)
        return TesterResult(
            agent_name=self.name,
            status=agent_status,
            summary=summary,
            errors=errors,
            metadata={"advisory": advisory_failures, "changed_files": changed},
            verification_status=status,
            test_results=results,
            detected_commands=[c.to_dict() for c in detected],
            failure_classification=classification,
            decisive_result=decisive,
            integrity_violations=violations,
            generated_files=generated,
        )

    def _record(self, state: AgentState, result: TesterResult) -> None:
        state.verification_attempts += 1
        for r in result.test_results:
            state.test_results.append({"attempt": state.verification_attempts, **r.to_dict()})
        if result.decisive_result is not None:
            latest = result.decisive_result.to_dict()
            comparison = result.metadata.get("baseline_comparison")
            if comparison:
                latest["baseline_comparison"] = comparison
                latest["baseline_accepted"] = bool(comparison.get("accepted"))
            state.latest_test_result = latest
        self.context.put(
            ContextEntry(
                ContextCategory.TEST_RESULT,
                f"{state.task_id}:verification:{state.verification_attempts}",
                result.summary,
                source=self.name,
                metadata={
                    "verification_status": result.verification_status.value,
                    "commands": result.commands_run,
                },
                task_id=state.task_id,
            )
        )
