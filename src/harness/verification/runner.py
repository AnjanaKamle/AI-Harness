"""TestRunner: execute verification commands and turn their output into TestResults."""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Sequence

from harness.tools.base import truncate_text
from harness.tools.policy import CommandPolicy
from harness.tools.repository import RepositoryContext
from harness.tools.terminal import TerminalTool
from harness.verification.detection import (
    ProjectProfile,
    detect_commands,
    inspect_project,
    related_python_tests,
)
from harness.verification.models import CheckKind, TestCommand, TestResult, TestStatus
from harness.verification.parsing import (
    ParsedOutput,
    parse_generic,
    parse_javascript,
    parse_pytest,
    parse_unittest,
)

log = logging.getLogger("harness.verification")

DEFAULT_TEST_TIMEOUT_SECONDS = 300.0
RAW_OUTPUT_LIMIT = 2_000_000  # captured in full, then summarized
STORED_STREAM_CHARS = 8_000  # per stream, tail-weighted (summaries live at the end)
PYTEST_FLAGS = ("-rfE", "--tb=short", "--color=no", "-p", "no:cacheprovider")

_NOT_AVAILABLE_PATTERNS = re.compile(
    r"(No module named '?pytest'?|No module named unittest|npm (?:ERR!|error) Missing script|"
    r"Missing script: \"[\w:-]+\"|make: \*\*\* No rule to make target)",
    re.IGNORECASE,
)

# Commands a verification run may execute: runners, builds, linters, type checkers.
_VERIFICATION_ARGV: tuple[tuple[str, ...], ...] = (
    ("python3", "-m", "pytest"),
    ("python", "-m", "pytest"),
    ("python3", "-m", "unittest"),
    ("python", "-m", "unittest"),
    ("pytest",),
    ("npm", "test"),
    ("npm", "t"),
    ("npm", "run"),
    ("make", "test"),
    ("make", "check"),
    ("make", "lint"),
    ("ruff", "check"),
    ("mypy",),
    ("tsc",),
    ("eslint",),
)


def is_verification_command(command: str) -> bool:
    """True if ``command`` is a test/build/lint/typecheck invocation (not arbitrary code)."""
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    return any(tuple(argv[: len(prefix)]) == prefix for prefix in _VERIFICATION_ARGV)


def _tail_weighted(text: str, limit: int = STORED_STREAM_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = limit // 4
    marker = f"\n...[{len(text) - limit} chars omitted]...\n"
    return text[:head] + marker + text[-(limit - head - len(marker)) :], True


def _parser_for(command: str, ecosystem: str):  # type: ignore[no-untyped-def]
    if "pytest" in command:
        return parse_pytest
    if "unittest" in command:
        return parse_unittest
    if ecosystem == "node" and ("test" in command.split()):
        return parse_javascript
    return parse_generic


class TestRunner:
    __test__ = False

    def __init__(
        self,
        repo: RepositoryContext,
        *,
        timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS,
        policy: CommandPolicy | None = None,
    ) -> None:
        self.repo = repo
        self.timeout_seconds = timeout_seconds
        self.terminal = TerminalTool(
            repo,
            policy,
            default_timeout=timeout_seconds,
            max_timeout=timeout_seconds,
            max_output_chars=RAW_OUTPUT_LIMIT,
        )

    # --- detection ---------------------------------------------------------------------

    def inspect(self) -> ProjectProfile:
        return inspect_project(self.repo)

    def detect(self, profile: ProjectProfile | None = None) -> list[TestCommand]:
        return detect_commands(self.repo, profile)

    def targeted_command(
        self, base: TestCommand, changed_files: Sequence[str], profile: ProjectProfile
    ) -> TestCommand | None:
        """A pytest run restricted to tests related to ``changed_files``, if any exist."""
        if "pytest" not in base.command:
            return None
        related = related_python_tests(self.repo, changed_files, profile)
        if not related or set(related) >= set(profile.python_test_files):
            return None  # nothing related, or "related" is the whole suite (no double run)
        return TestCommand(
            f"{base.command} " + " ".join(shlex.quote(p) for p in related),
            CheckKind.TEST,
            base.ecosystem,
            (f"tests related to changed files: {', '.join(related)}",),
            gating=True,
            targeted=True,
        )

    # --- execution ---------------------------------------------------------------------

    def run(self, spec: TestCommand) -> TestResult:
        command = spec.command
        if not is_verification_command(command):
            return TestResult(
                command, TestStatus.ERROR, spec.kind, gating=spec.gating, targeted=spec.targeted,
                failure_summary="Not a recognized test/build/lint/typecheck command; refused.",
            )
        if "pytest" in command and "--tb" not in command:
            command = command.replace("-m pytest", "-m pytest " + " ".join(PYTEST_FLAGS), 1)
            if command.startswith("pytest"):
                command = "pytest " + " ".join(PYTEST_FLAGS) + command[len("pytest"):]

        log.info("running %s check: %s", spec.kind.value.lower(), command)
        outcome = self.terminal.run({"command": command})
        data = outcome.data
        base = {"kind": spec.kind, "gating": spec.gating, "targeted": spec.targeted}

        if data.get("rejected"):
            return TestResult(command, TestStatus.ERROR, **base, failure_summary=outcome.error or "")
        if "exit_code" not in data and "timed_out" not in data:
            # the command never started (e.g. executable missing)
            status = (
                TestStatus.NOT_AVAILABLE
                if outcome.error and "not found" in outcome.error
                else TestStatus.ERROR
            )
            return TestResult(
                command, status, **base, stderr=outcome.error or "",
                failure_summary=outcome.error or "command could not be started",
            )

        raw_out, raw_err = data.get("stdout", ""), data.get("stderr", "")
        exit_code = data.get("exit_code")
        parser = _parser_for(command, spec.ecosystem)
        parsed: ParsedOutput = parser(raw_out, raw_err, exit_code)
        status = self._status(spec, exit_code, bool(data.get("timed_out")), parsed, raw_out + raw_err)

        stdout, cut_out = _tail_weighted(raw_out)
        stderr, cut_err = _tail_weighted(raw_err)
        summary = parsed.failure_summary
        if status is TestStatus.TIMEOUT:
            summary = f"Timed out after {self.timeout_seconds:g}s.\n" + summary
        if status is TestStatus.NOT_AVAILABLE and not summary:
            summary = truncate_text((raw_err or raw_out).strip(), 2_000)[0] or "no tests found"
        result = TestResult(
            command=command,
            status=status,
            exit_code=exit_code,
            passed=parsed.passed,
            failed=parsed.failed,
            skipped=parsed.skipped,
            errors=parsed.errors,
            stdout=stdout,
            stderr=stderr,
            duration=float(data.get("duration_seconds", 0.0)),
            failure_summary=summary,
            failed_tests=parsed.failed_tests,
            output_truncated=cut_out or cut_err or bool(data.get("truncated")),
            **base,
        )
        log.info("%s -> %s (%s)", command, status.value, result.counts_text())
        return result

    @staticmethod
    def _status(
        spec: TestCommand,
        exit_code: int | None,
        timed_out: bool,
        parsed: ParsedOutput,
        output: str,
    ) -> TestStatus:
        if timed_out:
            return TestStatus.TIMEOUT
        if _NOT_AVAILABLE_PATTERNS.search(output):
            return TestStatus.NOT_AVAILABLE
        if exit_code == 0:
            if spec.kind is CheckKind.TEST and parsed.no_tests:
                return TestStatus.NOT_AVAILABLE
            return TestStatus.PASS
        if "pytest" in spec.command:
            # pytest: 1 = tests failed, 2 = interrupted/collection errors, 3 = internal error,
            # 4 = usage error, 5 = no tests collected
            return {1: TestStatus.FAIL, 5: TestStatus.NOT_AVAILABLE}.get(
                exit_code or -1, TestStatus.ERROR
            )
        if parsed.no_tests and spec.kind is CheckKind.TEST:
            return TestStatus.NOT_AVAILABLE
        if (parsed.failed or 0) > 0 or spec.kind is not CheckKind.TEST:
            return TestStatus.FAIL
        return TestStatus.ERROR if not parsed.recognized else TestStatus.FAIL
