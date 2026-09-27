"""Verification tools: detect_tests and run_tests.

run_tests accepts only test/build/lint/typecheck invocations - it cannot be used to run
arbitrary code (e.g. ``python3 -c``), which keeps the Tester read-only with respect to
the source tree.
"""

from __future__ import annotations

from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult
from harness.tools.repository import RepositoryContext
from harness.verification.models import CheckKind, TestCommand
from harness.verification.runner import TestRunner, is_verification_command


class DetectTestsTool(BaseTool):
    name = "detect_tests"
    description = (
        "Inspect repository metadata (pyproject.toml, package.json, requirements files, "
        "Makefile, test directories) and list the verification commands it supports, with "
        "the evidence for each. Returns nothing if the project declares no tests."
    )
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, runner: TestRunner) -> None:
        self.runner = runner

    def execute(self, **_: Any) -> ToolExecutionResult:
        profile = self.runner.inspect()
        commands = self.runner.detect(profile)
        return ToolExecutionResult.success(
            self.name,
            {"profile": profile.to_dict(), "commands": [c.to_dict() for c in commands]},
        )


class RunTestsTool(BaseTool):
    name = "run_tests"
    description = (
        "Run ONE verification command (tests, build, lint or type check), e.g. "
        "'python3 -m pytest tests/test_x.py' or 'npm test'. Returns a structured TestResult: "
        "status (PASS/FAIL/ERROR/TIMEOUT/NOT_AVAILABLE), counts, failure summary, stdout, "
        "stderr, exit code and duration."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "kind": {"type": "string", "enum": [k.value for k in CheckKind]},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, runner: TestRunner) -> None:
        self.runner = runner

    def execute(self, command: str, kind: str = "TEST", **_: Any) -> ToolExecutionResult:
        if not is_verification_command(command):
            raise ToolError(
                "run_tests only runs test/build/lint/typecheck commands "
                "(pytest, unittest, npm test/run, make test, ruff, mypy, tsc, eslint)"
            )
        result = self.runner.run(TestCommand(command, CheckKind(kind), _ecosystem(command), ()))
        return ToolExecutionResult.success(self.name, result.to_dict())


def _ecosystem(command: str) -> str:
    first = command.split()[0] if command.split() else ""
    return {"npm": "node", "make": "make"}.get(first, "python")


def build_testing_tools(repo: RepositoryContext, timeout_seconds: float) -> list[BaseTool]:
    runner = TestRunner(repo, timeout_seconds=timeout_seconds)
    return [DetectTestsTool(runner), RunTestsTool(runner)]
