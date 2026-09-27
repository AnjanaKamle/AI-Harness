"""Demo mode (opt-in: AI_PROVIDER=demo).

This is NOT a language model. It is a small scripted client that knows how to fix the
bundled sample repository's two deliberate bugs - one per Coder run - by reading the real
tool results it is given. Everything else is real: tools edit files, the Tester runs pytest,
failures trigger the repair loop, and the verification gate decides the outcome.

It exists so the full pipeline (and the TUI) can be demonstrated without the official
evaluation model, which has not been specified yet. It runs on a temporary copy of the
sample repository, never on the user's code, and is only used when explicitly selected.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harness.llm.client import LLMClient, register_provider
from harness.llm.models import LLMResponse, Message, Role, StopReason, ToolCall, ToolDefinition

DEMO_PROVIDER = "demo"
SAMPLE_DIR = Path(__file__).parent / "sample"
SAMPLE_TASK = "Fix the bug in this repository and play Beethoven Symphony No. 5 while you work."

# The sample's known bugs: (file, broken snippet, fixed snippet). One is fixed per Coder run.
KNOWN_FIXES: tuple[tuple[str, str, str], ...] = (
    ("calc.py", "def multiply(a, b):\n    return a + b\n", "def multiply(a, b):\n    return a * b\n"),
    ("calc.py", "def divide(a, b):\n    return a // b\n", "def divide(a, b):\n    return a / b\n"),
)


def _call(name: str, **arguments: Any) -> LLMResponse:
    return LLMResponse("", StopReason.TOOL_USE, tool_calls=(ToolCall(f"demo-{name}", name, arguments),),
                       model="demo-scripted")


def _final(body: dict[str, Any]) -> LLMResponse:
    return LLMResponse(json.dumps(body), StopReason.END_TURN, model="demo-scripted")


DEMO_STEP_ENV = "AI_DEMO_STEP_SECONDS"


class DemoScriptedClient(LLMClient):
    """Deterministic scripted client for the bundled sample repository (not an AI model).

    ``step_seconds`` paces each reply so the live dashboard is watchable (0 in tests).
    """

    model = "demo-scripted"

    def __init__(self, step_seconds: float = 0.0) -> None:
        self.step_seconds = step_seconds

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if self.step_seconds > 0:
            time.sleep(self.step_seconds)
        system = messages[0].content if messages else ""
        if "Researcher agent" in system:
            return _final({"status": "BLOCKED", "summary": "The demo client performs no research.",
                           "findings": [], "open_questions": []})
        turn = sum(1 for m in messages if m.role is Role.ASSISTANT)
        last = self._last_tool_data(messages)
        if tools is None:  # asked to finalize without tools
            return _final({"status": "FAILURE", "summary": "Demo client stopped.", "files_changed": [],
                           "next_action": "", "errors": ["demo client finalized without tools"]})
        if turn == 0:
            return _call("list_files", recursive=True)
        if turn == 1:
            return _call("read_file", path="tests/test_calc.py")
        if turn == 2:
            return _call("read_file", path="calc.py")
        if turn == 3:
            content = last.get("content", "")
            for path, broken, fixed in KNOWN_FIXES:
                if broken in content:
                    return _call("edit_file", path=path, old_text=broken, new_text=fixed)
            return _final({"status": "FAILURE", "summary": "No known sample bug remains to fix.",
                           "files_changed": [], "next_action": "Inspect the failing tests manually.",
                           "errors": ["demo client has no further scripted fix"]})
        if turn == 4:
            return _call("git_diff")
        changed = sorted({path for path, _, _ in KNOWN_FIXES})
        return _final({"status": "SUCCESS",
                       "summary": "Fixed one sample bug and reviewed the diff (scripted demo).",
                       "files_changed": changed, "next_action": "Run the tests.", "errors": []})

    @staticmethod
    def _last_tool_data(messages: Sequence[Message]) -> dict[str, Any]:
        for message in reversed(messages):
            if message.tool_results:
                try:
                    return json.loads(message.tool_results[0].content).get("data", {})
                except (json.JSONDecodeError, AttributeError):
                    return {}
        return {}


def register_demo_provider() -> None:
    try:
        step = float(os.environ.get(DEMO_STEP_ENV, "0.4"))
    except ValueError:
        step = 0.4
    register_provider(DEMO_PROVIDER, lambda settings: DemoScriptedClient(max(step, 0.0)))


def prepare_sample_repo(base: Path | None = None) -> Path:
    """Copy the sample into a fresh temporary git repository and return its path."""
    root = Path(tempfile.mkdtemp(prefix="harness-demo-", dir=base)) / "sample-calculator"
    shutil.copytree(SAMPLE_DIR, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    from harness.tools.policy import sanitized_environment

    env = {**sanitized_environment(), "GIT_AUTHOR_NAME": "harness-demo", "GIT_AUTHOR_EMAIL": "demo@example.invalid",
           "GIT_COMMITTER_NAME": "harness-demo", "GIT_COMMITTER_EMAIL": "demo@example.invalid",
           "GIT_CEILING_DIRECTORIES": str(root.parent)}
    if shutil.which("git"):
        for args in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "sample repository"]):
            subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True,
                           timeout=30)
    return root
