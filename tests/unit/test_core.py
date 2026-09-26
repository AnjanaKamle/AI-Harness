from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.config import HarnessConfig
from harness.core import (
    AgentRole,
    Orchestrator,
    SharedState,
    Task,
    TaskStatus,
    VerificationResult,
)
from harness.llm import MockLLMClient
from tests.conftest import CrashingAgent, EchoAgent


def test_task_defaults() -> None:
    task = Task("x")
    assert task.status is TaskStatus.PENDING
    assert len(task.id) == 12
    assert TaskStatus.PASSED.is_terminal and not TaskStatus.RUNNING.is_terminal


def test_shared_state_set_get_and_events(state: SharedState) -> None:
    state.set("plan", ["a", "b"], source="orchestrator")
    state.record("coder", "note", detail="hi")
    assert state.get("plan") == ["a", "b"]
    assert "plan" in state and "missing" not in state
    assert state.get("missing", 42) == 42
    kinds = [(e.source, e.kind) for e in state.events]
    assert kinds == [("orchestrator", "set"), ("coder", "note")]


def test_shared_state_save_roundtrip(state: SharedState, tmp_path: Path) -> None:
    state.set("k", "v")
    out = tmp_path / "runs" / "state.json"
    state.save(out)
    data = json.loads(out.read_text())
    assert data["task"]["description"] == "add two numbers"
    assert data["data"] == {"k": "v"}


def test_agent_run_success(state: SharedState) -> None:
    agent = EchoAgent(MockLLMClient(["done"]))
    result = agent.run(state)
    assert result.success and result.output == "done"
    assert agent.name == "coder"
    assert state.get("echo") == "done"
    assert [e.kind for e in state.events] == ["agent_started", "set", "agent_finished"]


def test_agent_run_captures_exceptions(state: SharedState) -> None:
    result = CrashingAgent(MockLLMClient()).run(state)
    assert not result.success
    assert result.error == "RuntimeError: boom"
    assert state.events[-1].data == {"success": False, "error": "RuntimeError: boom"}


def test_verification_result_helpers() -> None:
    assert VerificationResult.ok("fine").passed
    failed = VerificationResult.fail("test_a", "test_b")
    assert not failed.passed and failed.failures == ["test_a", "test_b"]


def test_orchestrator_registry(config: HarnessConfig, mock_llm: MockLLMClient) -> None:
    orch = Orchestrator(config, mock_llm)
    coder = EchoAgent(mock_llm)
    orch.register(coder)
    assert orch.get(AgentRole.CODER) is coder
    with pytest.raises(ValueError):
        orch.register(EchoAgent(mock_llm))
    with pytest.raises(KeyError):
        orch.get(AgentRole.MUSIC)


def test_orchestrator_run_not_implemented(config: HarnessConfig, mock_llm: MockLLMClient) -> None:
    with pytest.raises(NotImplementedError):
        Orchestrator(config, mock_llm).run(Task("x"))
