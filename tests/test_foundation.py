from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from harness.agents import AgentResult, AgentStatus, BaseAgent
from harness.config.settings import Settings
from harness.context import ContextCategory, ContextEntry, InMemoryContextManager
from harness.main import EXIT_CONFIG_ERROR, EXIT_NO_TASK, EXIT_OK, main
from harness.orchestrator import AgentState, Orchestrator, StepStatus, TaskStatus
from harness.tools import BaseTool, ToolExecutionResult, ToolStatus

from .conftest import FAKE_KEY, MockLLMClient

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Tools ------------------------------------------------------------------------------


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the given text"
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def execute(self, **arguments: Any) -> ToolExecutionResult:
        return ToolExecutionResult(self.name, ToolStatus.SUCCESS, output=arguments["text"])


class BrokenTool(BaseTool):
    name = "broken"
    description = "Always raises"

    def execute(self, **arguments: Any) -> ToolExecutionResult:
        raise OSError("disk on fire")


def test_base_tool_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseTool()  # type: ignore[abstract]


def test_tool_subclass_run_and_definition() -> None:
    tool = EchoTool()
    result = tool.run({"text": "hi"})
    assert result.ok and result.output == "hi"
    assert "duration_ms" in result.metadata
    definition = tool.definition()
    assert (definition.name, definition.input_schema["required"]) == ("echo", ["text"])


@pytest.mark.parametrize("arguments", [{}, {"text": "hi", "extra": 1}])
def test_tool_input_validation(arguments: dict[str, Any]) -> None:
    result = EchoTool().run(arguments)
    assert result.status is ToolStatus.FAILURE and "ToolInputError" in (result.error or "")


def test_tool_exception_becomes_failure_result() -> None:
    result = BrokenTool().run({})
    assert result.status is ToolStatus.FAILURE and result.error == "OSError: disk on fire"


# --- Agents -----------------------------------------------------------------------------


class DemoAgent(BaseAgent):
    name = "demo"
    description = "Test agent"

    def execute(self, task: str, state: AgentState) -> AgentResult:
        assert state.active_agent == self.name
        return AgentResult(self.name, AgentStatus.SUCCESS, summary=f"did {task}")


class CrashingAgent(BaseAgent):
    name = "crasher"
    description = "Always raises"

    def execute(self, task: str, state: AgentState) -> AgentResult:
        raise RuntimeError("boom")


def test_base_agent_is_abstract(mock_llm: MockLLMClient) -> None:
    with pytest.raises(TypeError):
        BaseAgent(mock_llm, InMemoryContextManager())  # type: ignore[abstract]


def test_agent_subclass(mock_llm: MockLLMClient) -> None:
    agent = DemoAgent(mock_llm, InMemoryContextManager(), tools=[EchoTool()])
    state = AgentState(task="t")
    result = agent.run("t", state)

    assert (agent.name, agent.description, agent.available_tools) == ("demo", "Test agent", ["echo"])
    assert result.status is AgentStatus.SUCCESS and result.summary == "did t"
    assert (result.artifacts, result.errors, result.metadata) == ({}, [], {})
    assert state.active_agent is None


def test_agent_exception_becomes_failure(mock_llm: MockLLMClient) -> None:
    result = CrashingAgent(mock_llm, InMemoryContextManager()).run("t", AgentState(task="t"))
    assert result.status is AgentStatus.FAILURE and result.errors == ["RuntimeError: boom"]


def test_agent_statuses() -> None:
    assert {s.value for s in AgentStatus} == {"SUCCESS", "FAILURE", "BLOCKED"}


# --- State ------------------------------------------------------------------------------


def test_agent_state_creation_defaults() -> None:
    state = AgentState(task="Fix bug")
    assert state.status is TaskStatus.PENDING and len(state.task_id) == 12
    assert state.plan == [] and state.retry_count == 0 and state.final_result is None
    for name in ("completed_tasks", "pending_tasks", "research_findings", "code_changes",
                 "test_results", "failures"):
        assert getattr(state, name) == []


def test_agent_state_json_round_trip(settings: Settings) -> None:
    state = Orchestrator(settings).submit("Fix bug")
    state.research_findings.append({"source": "docs", "note": "x"})
    state.failures.append({"step": "step-2", "error": "boom"})

    payload = json.loads(state.to_json())
    assert payload["status"] == "PLANNED" and payload["plan"][0]["status"] == "PENDING"

    restored = AgentState.from_json(state.to_json())
    assert restored == state
    assert restored.plan[0].status is StepStatus.PENDING


# --- Context ----------------------------------------------------------------------------


def test_context_manager_store_and_retrieve() -> None:
    ctx = InMemoryContextManager()
    ctx.put(ContextEntry(ContextCategory.RESEARCH, "k1", "v1"))
    ctx.put(ContextEntry(ContextCategory.TEST_RESULT, "k2", "v2"))
    ctx.put(ContextEntry(ContextCategory.RESEARCH, "k1", "v1-updated"))

    assert ctx.get(ContextCategory.RESEARCH, "k1").content == "v1-updated"  # type: ignore[union-attr]
    assert [e.key for e in ctx.entries(ContextCategory.RESEARCH)] == ["k1"]
    assert len(ctx.entries()) == 2
    assert ctx.remove(ContextCategory.TEST_RESULT, "k2") and not ctx.remove(
        ContextCategory.TEST_RESULT, "k2"
    )
    ctx.clear()
    assert len(ctx) == 0


# --- Orchestrator -----------------------------------------------------------------------


def test_orchestrator_accepts_task(settings: Settings, mock_llm: MockLLMClient) -> None:
    ctx = InMemoryContextManager()
    state = Orchestrator(settings, llm=mock_llm, context=ctx).submit("  Fix the auth bug ")

    assert state.task == "Fix the auth bug" and state.status is TaskStatus.PLANNED
    assert [s.agent for s in state.plan] == ["researcher", "coder", "tester"]
    assert state.pending_tasks == ["step-1", "step-2", "step-3"]
    assert ctx.get(ContextCategory.TASK, state.task_id) is not None
    assert mock_llm.calls == []  # no model calls in phase 1


def test_orchestrator_rejects_empty_task(settings: Settings) -> None:
    with pytest.raises(ValueError):
        Orchestrator(settings).submit("   ")


# --- Entry point ------------------------------------------------------------------------


def test_main_with_task_arg(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    assert main(["--task", "Foundation smoke test"]) == EXIT_OK
    out = capsys.readouterr()
    assert json.loads(out.out)["task"] == "Foundation smoke test"
    assert FAKE_KEY not in out.out + out.err


def test_main_reads_task_env(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("TASK", "from env")
    assert main([]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["task"] == "from env"


def test_main_reads_piped_stdin(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import io

    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(sys, "stdin", io.StringIO("from stdin\n"))
    assert main([]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["task"] == "from stdin"


def test_main_without_task(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    monkeypatch.setenv("AI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert main([]) == EXIT_NO_TASK


def test_main_without_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--task", "x"]) == EXIT_CONFIG_ERROR
    assert "AI_API_KEY is not set" in capsys.readouterr().err


# --- Makefile ---------------------------------------------------------------------------

needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")


@needs_make
@pytest.mark.parametrize("target", ["setup", "run", "test", "clean"])
def test_makefile_target_exists(target: str) -> None:
    proc = subprocess.run(
        ["make", "-n", target], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


@needs_make
def test_make_run_starts_without_api_request() -> None:
    """Runs the real `make run` target; phase 1 makes no network/model calls."""
    if not (REPO_ROOT / ".venv" / ".installed").exists():
        pytest.skip("run `make setup` first")
    env = {**os.environ, "AI_API_KEY": FAKE_KEY}
    env.pop("MAKEFLAGS", None)
    proc = subprocess.run(
        ["make", "run", "TASK=Foundation smoke test"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert payload["task"] == "Foundation smoke test" and payload["status"] == "PLANNED"
