"""Phase 7 failure modes: every one ends in a controlled result, never a hang or traceback."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from harness.config.settings import Settings
from harness.llm import LLMClient, LLMError, LLMNotConfiguredError, LLMResponse, Message, create_llm_client
from harness.main import main
from harness.orchestrator.verification_manager import FinalStatus
from harness.tools import EditFileTool, RepositoryContext, TerminalTool
from harness.verification import CheckKind, FailureCategory, TestCommand, TestRunner, TestStatus, classify_failure

from .conftest import FAKE_KEY, MockLLMClient
from .test_orchestration_flows import BUGGY, TESTS, build, edit, make_repo, node_status


@pytest.fixture
def repo(tmp_path: Path) -> RepositoryContext:
    return make_repo(tmp_path, {"calc.py": BUGGY, "tests/test_calc.py": TESTS})


# --- credentials / provider configuration -------------------------------------------------------


def test_missing_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "--task", "x"]) == 2
    assert "AI_API_KEY is not set" in capsys.readouterr().err


def test_api_key_without_provider_reports_live_execution_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], repo: RepositoryContext
) -> None:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    assert main(["--json", "--task", "Fix multiply", "--repo", str(repo.root)]) == 0
    out = capsys.readouterr()
    assert "Live model execution is unavailable" in out.err
    assert json.loads(out.out)["status"] == "PLANNED"
    assert FAKE_KEY not in out.out + out.err


def test_unknown_provider_is_a_clear_configuration_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], repo: RepositoryContext
) -> None:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("AI_PROVIDER", "some-unspecified-provider")
    assert main(["--json", "--task", "Fix multiply", "--repo", str(repo.root)]) == 2
    err = capsys.readouterr().err
    assert "Unknown provider 'some-unspecified-provider'" in err and "Traceback" not in err


def test_plugin_provider_requires_model_and_loads_without_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = tmp_path / "plugins" / "organizer_adapter.py"
    plugin.parent.mkdir()
    plugin.write_text(
        "from harness.llm import LLMClient, LLMResponse, StopReason\n"
        "class Adapter(LLMClient):\n"
        "    def __init__(self, settings):\n"
        "        self.model = settings.model\n"
        "    def generate(self, messages, tools=None, **kw):\n"
        "        return LLMResponse('ok', StopReason.END_TURN, model=self.model)\n"
        "def make(settings):\n"
        "    return Adapter(settings)\n"
    )
    monkeypatch.syspath_prepend(str(plugin.parent))
    with pytest.raises(LLMNotConfiguredError, match="requires AI_MODEL"):
        create_llm_client(Settings(api_key=FAKE_KEY, provider="organizer_adapter:make"))
    client = create_llm_client(Settings(api_key=FAKE_KEY, provider="organizer_adapter:make",
                                        model="organizer-model"))
    assert client.generate([Message.user("hi")]).model == "organizer-model"
    with pytest.raises(LLMNotConfiguredError, match="cannot import"):
        create_llm_client(Settings(api_key=FAKE_KEY, provider="missing_module:make", model="m"))


def test_invalid_api_key_blocks_cleanly(repo: RepositoryContext) -> None:
    def rejected(messages: list[Message]) -> LLMResponse:
        raise LLMError("401 Unauthorized: invalid API key", retryable=False)

    orch, agents, _ = build(repo, [rejected])
    final = orch.execute("Fix multiply.", repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED
    assert any("non-retryable model/provider error" in r["reason"] for r in final.retry_history)
    assert (repo.root / "calc.py").read_text() == BUGGY


# --- model behaviour ----------------------------------------------------------------------------------


def test_model_request_timeouts_are_retried_then_blocked(repo: RepositoryContext) -> None:
    def timeout(messages: list[Message]) -> LLMResponse:
        raise LLMError("request timed out after 60s", retryable=True)

    orch, agents, _ = build(repo, [timeout] * 3, max_agent_retries=2)
    final = orch.execute("Fix multiply.", repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED
    retries = [r for r in final.retry_history if r["node"] == "implement-1"]
    assert [r["action"] for r in retries] == ["RETRY", "RETRY", "BLOCK"]


class HangingModel(LLMClient):
    def generate(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
        time.sleep(3)  # a provider call that never returns in time
        raise LLMError("too late")


def test_hanging_model_call_does_not_hang_the_harness(repo: RepositoryContext) -> None:
    orch, agents, _ = build(repo, [], agent_timeout_seconds=0.5, max_agent_retries=0)
    agents["coder"].llm = HangingModel()
    started = time.monotonic()
    final = orch.execute("Fix multiply.", repo, agents=agents, grace_seconds=0.5)
    assert time.monotonic() - started < 10
    assert final.status is FinalStatus.BLOCKED
    assert node_status(final)["implement-1"] == "BLOCKED"


def test_malformed_model_response_is_bounded(repo: RepositoryContext) -> None:
    garbage = LLMResponse("<<not json>>", __import__("harness.llm", fromlist=["StopReason"]).StopReason.END_TURN)
    orch, agents, _ = build(repo, [garbage] * 20, max_agent_retries=1)
    final = orch.execute("Fix multiply.", repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED
    assert [r["kind"] for r in final.retry_history if r["node"] == "implement-1"] == ["MALFORMED_OUTPUT"] * 2


# --- tools and environment ------------------------------------------------------------------------------


def test_missing_command_is_reported(repo: RepositoryContext, monkeypatch: pytest.MonkeyPatch,
                                     tmp_path: Path) -> None:
    import harness.tools.terminal as terminal_module

    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setattr(terminal_module, "sanitized_environment", lambda: {"PATH": str(empty)})
    result = TerminalTool(repo).run({"command": "eslint ."})
    assert not result.ok and "Executable not found: eslint" in (result.error or "")


def test_missing_test_framework(repo: RepositoryContext, monkeypatch: pytest.MonkeyPatch,
                                tmp_path: Path) -> None:
    import harness.tools.terminal as terminal_module

    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python3"
    fake_python.write_text("#!/bin/sh\necho '/usr/bin/python3: No module named pytest' >&2\nexit 1\n")
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(terminal_module, "sanitized_environment", lambda: {"PATH": str(fake_bin)})
    result = TestRunner(repo).run(TestCommand("python3 -m pytest", CheckKind.TEST, "python", ()))
    assert result.status is TestStatus.NOT_AVAILABLE
    assert classify_failure(result, repo).category is FailureCategory.DEPENDENCY_FAILURE


def test_repository_path_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                               tmp_path: Path) -> None:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("AI_PROVIDER", "demo")
    monkeypatch.setenv("AI_MUSIC_BACKEND", "none")
    code = main(["--json", "--task", "Fix it", "--repo", str(tmp_path / "does-not-exist")])
    err = capsys.readouterr().err
    assert code == 2 and "Repository root does not exist" in err and "Traceback" not in err


@pytest.mark.skipif(sys.platform.startswith("win") or os.geteuid() == 0, reason="needs POSIX non-root")
def test_permission_error_is_a_structured_tool_failure(repo: RepositoryContext) -> None:
    target = repo.root / "calc.py"
    target.chmod(0o444)
    try:
        result = EditFileTool(repo).run({"path": "calc.py", "old_text": "a + b", "new_text": "a * b"})
    finally:
        target.chmod(0o644)
    assert not result.ok and "PermissionError" in (result.error or "")
    assert target.read_text() == BUGGY


def test_tool_timeout_and_failed_test_and_research_are_controlled(repo: RepositoryContext) -> None:
    # failed test -> repair; failed research is covered in test_orchestration_flows (E);
    # tool timeout in test_orchestration_flows (F). Here: everything at once stays bounded.
    script = (edit("return a + b", "return a - b") + edit("return a - b", "return a + b")
              + edit("return a + b", "return a - b"))  # every attempt changes code, none fixes it
    orch, agents, _ = build(repo, script, max_repair_attempts=2)
    started = time.monotonic()
    final = orch.execute("Fix multiply.", repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED and time.monotonic() - started < 60
    assert final.state.repair_attempts == 2  # type: ignore[union-attr]
    assert any("maximum repair attempts (2)" in r["reason"] for r in final.retry_history)


def test_scripted_mock_remains_available_for_tests() -> None:
    assert isinstance(MockLLMClient(), LLMClient)
