"""Orchestrator skeleton: owns the agent registry and (in later phases) the run loop."""

from __future__ import annotations

from harness.config import HarnessConfig
from harness.core.agent import AgentRole, BaseAgent
from harness.core.state import SharedState
from harness.core.task import Task
from harness.core.verification import Verifier
from harness.llm.base import LLMClient
from harness.logging_setup import get_logger


class Orchestrator:
    def __init__(
        self,
        config: HarnessConfig,
        llm: LLMClient,
        *,
        verifier: Verifier | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.verifier = verifier
        self.log = get_logger("orchestrator")
        self._agents: dict[AgentRole, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if agent.role in self._agents:
            raise ValueError(f"An agent for role {agent.role.value!r} is already registered")
        self._agents[agent.role] = agent
        self.log.debug("registered agent", extra={"role": agent.role.value, "agent": agent.name})

    def get(self, role: AgentRole) -> BaseAgent:
        try:
            return self._agents[role]
        except KeyError:
            raise KeyError(f"No agent registered for role {role.value!r}") from None

    @property
    def agents(self) -> dict[AgentRole, BaseAgent]:
        return dict(self._agents)

    def new_state(self, task: Task) -> SharedState:
        return SharedState(task)

    def run(self, task: Task) -> SharedState:
        """Plan -> dispatch agents -> verify -> recover/retry. Implemented in a later phase."""
        raise NotImplementedError("Orchestrator.run is implemented in a later phase")
