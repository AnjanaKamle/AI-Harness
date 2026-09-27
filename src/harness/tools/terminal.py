"""terminal: run one policy-approved development command inside the repository."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult, truncate_text
from harness.tools.policy import CommandPolicy, sanitized_environment
from harness.tools.repository import RepositoryContext, bump_generation

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_OUTPUT_CHARS = 12_000


class TerminalTool(BaseTool):
    name = "terminal"
    description = (
        "Run ONE development command (e.g. 'pytest -q', 'python app.py', 'git status') inside "
        "the repository. No shell: pipes, redirects, '&&' and substitutions are not supported. "
        "Only allow-listed commands run; destructive commands (rm, sudo, git push/reset...) "
        "are rejected. Returns stdout, stderr, exit_code and duration."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string", "description": "Working dir relative to repo root."},
            "timeout_seconds": {"type": "number", "minimum": 1},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        repo: RepositoryContext,
        policy: CommandPolicy | None = None,
        *,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_timeout: float | None = None,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self.repo = repo
        self.policy = policy or CommandPolicy()
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout if max_timeout is not None else default_timeout
        self.max_output_chars = max_output_chars

    def execute(
        self, command: str, cwd: str = ".", timeout_seconds: float | None = None, **_: Any
    ) -> ToolExecutionResult:
        workdir = self.repo.resolve(cwd)
        if not workdir.is_dir():
            raise ToolError(f"Working directory not found: {cwd}")
        argv = self.policy.parse(command)
        decision = self.policy.evaluate(argv, self.repo, workdir)
        if not decision.allowed:
            return ToolExecutionResult.failure(
                self.name,
                f"Command rejected by policy ({decision.risk}): {decision.reason}",
                {"command": command, "risk": decision.risk.value, "rejected": True},
            )

        timeout = min(timeout_seconds or self.default_timeout, self.max_timeout)
        bump_generation(self.repo.root)  # an arbitrary command may create/modify files
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=sanitized_environment(),
                start_new_session=True,  # own process group, so a timeout kills children too
            )
        except FileNotFoundError:
            raise ToolError(f"Executable not found: {argv[0]}") from None
        except PermissionError:
            raise ToolError(f"Executable not permitted: {argv[0]}") from None

        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
        duration = round(time.perf_counter() - started, 3)

        half = self.max_output_chars // 2
        stdout, out_trunc = truncate_text(out.decode("utf-8", "replace"), half)
        stderr, err_trunc = truncate_text(err.decode("utf-8", "replace"), half)
        data = {
            "command": command,
            "cwd": self.repo.relative(workdir),
            "exit_code": None if timed_out else proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration_seconds": duration,
            "timed_out": timed_out,
            "truncated": out_trunc or err_trunc,
        }
        if timed_out:
            return ToolExecutionResult.failure(
                self.name, f"Command timed out after {timeout:g}s and was killed", data
            )
        if proc.returncode != 0:
            return ToolExecutionResult.failure(
                self.name, f"Command exited with code {proc.returncode}", data
            )
        return ToolExecutionResult.success(self.name, data)
