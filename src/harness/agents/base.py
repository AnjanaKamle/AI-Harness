"""BaseAgent: the contract for Coder, Researcher, Tester and Music agents (later phases)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from harness.context.manager import ContextManager
from harness.context.usage import accumulate_usage
from harness.llm.client import LLMClient
from harness.orchestrator.state import AgentState
from harness.tools.base import BaseTool


class AgentStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    BLOCKED = "BLOCKED"


@dataclass
class AgentResult:
    agent_name: str
    status: AgentStatus
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    def __init__(
        self,
        llm: LLMClient,
        context: ContextManager,
        tools: Sequence[BaseTool] = (),
    ) -> None:
        self.llm = llm
        self.context = context
        self.tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.log = logging.getLogger(f"harness.agents.{self.name}")

    @property
    def available_tools(self) -> list[str]:
        return sorted(self.tools)

    @abstractmethod
    def execute(self, task: str, state: AgentState) -> AgentResult:
        """Carry out ``task`` using ``state`` and return a structured result."""

    def run(self, task: str, state: AgentState) -> AgentResult:
        """Call :meth:`execute`, tracking the active agent and converting crashes to FAILURE."""
        state.active_agent = self.name
        self.log.info("started: %s", task)
        try:
            result = self.execute(task, state)
        except Exception as exc:  # boundary: one agent must not crash the orchestrator
            self.log.exception("agent raised")
            result = AgentResult(
                self.name,
                AgentStatus.FAILURE,
                summary="Agent raised an unexpected exception",
                errors=[f"{type(exc).__name__}: {exc}"],
            )
        finally:
            state.active_agent = None
        usage = result.metadata.get("usage")
        if isinstance(usage, dict):
            accumulate_usage(state.usage, self.name, usage)
        self.log.info("finished: %s", result.status)
        return result
