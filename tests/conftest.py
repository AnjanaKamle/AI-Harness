from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from typing import Any
from pathlib import Path

import pytest

from harness.config.settings import Settings
from harness.llm.client import LLMClient
from harness.llm.models import LLMResponse, Message, StopReason, ToolDefinition
from harness.tools.repository import RepositoryContext

FAKE_KEY = "test-key-not-real"

# Snapshot of the environment pytest started with (only the optional live tests read it;
# every other test runs with AI_* variables removed).
ORIGINAL_ENV = dict(os.environ)

Scripted = LLMResponse | Callable[[list[Message]], LLMResponse]


class MockLLMClient(LLMClient):
    """Test double: returns scripted responses in order and records every call.

    A scripted item may be a callable receiving the conversation so far, which lets tests
    assert on tool results the "model" has seen.
    """

    def __init__(self, responses: Sequence[Scripted] = ()) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[Message], list[ToolDefinition] | None]] = []

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), list(tools) if tools is not None else None))
        if self.responses:
            item = self.responses.pop(0)
            return item(list(messages)) if callable(item) else item
        return LLMResponse(content="ok", stop_reason=StopReason.END_TURN)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never see the developer's real credentials or config."""
    for key in list(os.environ):
        if key.startswith("AI_") or key in ("LOG_LEVEL", "TASK"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _reset_logging() -> Any:
    """main() configures the root logger; never let a test's handlers leak into the next."""
    import logging

    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


@pytest.fixture
def settings() -> Settings:
    return Settings(api_key=FAKE_KEY)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


# --- temporary repositories ---------------------------------------------------------------

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

BUGGY_APP = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"


def git(root: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_CEILING_DIRECTORIES": str(root.parent),
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    return subprocess.run(
        ["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def plain_repo(tmp_path: Path) -> RepositoryContext:
    """A non-git temporary project directory."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text(BUGGY_APP)
    (root / "pkg").mkdir()
    (root / "pkg" / "util.py").write_text("def helper():\n    return 'help'\n")
    (root / "README.md").write_text("# demo\n")
    return RepositoryContext(root)


@pytest.fixture
def git_repo(plain_repo: RepositoryContext) -> RepositoryContext:
    """The same project, as a git repository with one commit."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    git(plain_repo.root, "init", "-q", "-b", "main")
    git(plain_repo.root, "add", ".")
    git(plain_repo.root, "commit", "-q", "-m", "initial")
    return plain_repo
