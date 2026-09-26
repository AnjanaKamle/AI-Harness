"""Base class for the specialist agents (Coder, Researcher, Tester, Music)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from harness.core.state import SharedState
from harness.llm.base import LLMClient
from harness.logging_setup import get_logger


class AgentRole(str, Enum):
    CODER = "coder"
    RESEARCHER = "researcher"
    TESTER = "tester"
    MUSIC = "music"


@dataclass
class AgentResult:
    agent: str
    success: bool
    output: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseAgent(ABC):
    """Common agent plumbing: name, role, LLM handle, logger, and a safe ``run`` wrapper.

    Subclasses implement :meth:`execute`; callers use :meth:`run`, which records start/finish
    events on the shared state and converts unexpected exceptions into a failed result.
    """

    role: AgentRole

    def __init__(self, llm: LLMClient, *, name: str | None = None) -> None:
        self.llm = llm
        self.name = name or self.role.value
        self.log = get_logger(f"agent.{self.name}")

    @abstractmethod
    def execute(self, state: SharedState) -> AgentResult:
        """Do the agent's work against ``state``."""

    def run(self, state: SharedState) -> AgentResult:
        state.record(self.name, "agent_started")
        self.log.info("agent started", extra={"task_id": state.task.id})
        try:
            result = self.execute(state)
        except Exception as exc:  # noqa: BLE001 - boundary: never let one agent crash the run
            self.log.exception("agent failed")
            result = AgentResult(
                agent=self.name, success=False, error=f"{type(exc).__name__}: {exc}"
            )
        state.record(self.name, "agent_finished", success=result.success, error=result.error)
        return result
