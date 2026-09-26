from __future__ import annotations

import pytest

from harness.config import HarnessConfig, LLMConfig
from harness.core.agent import AgentResult, AgentRole, BaseAgent
from harness.core.state import SharedState
from harness.core.task import Task
from harness.llm.mock_client import MockLLMClient


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory) -> None:
    """Keep tests independent of the developer's shell env and .env file."""
    import os

    for key in list(os.environ):
        if key.startswith("HARNESS_") or key.startswith("ANTHROPIC_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def config() -> HarnessConfig:
    return HarnessConfig(llm=LLMConfig(provider="mock"))


@pytest.fixture
def task() -> Task:
    return Task(description="add two numbers")


@pytest.fixture
def state(task: Task) -> SharedState:
    return SharedState(task)


class EchoAgent(BaseAgent):
    """Minimal concrete agent used to exercise BaseAgent plumbing."""

    role = AgentRole.CODER

    def execute(self, state: SharedState) -> AgentResult:
        reply = self.llm.ask(state.task.description)
        state.set("echo", reply, source=self.name)
        return AgentResult(agent=self.name, success=True, output=reply)


class CrashingAgent(BaseAgent):
    role = AgentRole.TESTER

    def execute(self, state: SharedState) -> AgentResult:
        raise RuntimeError("boom")
