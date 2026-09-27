"""Orchestrator.

``submit`` accepts a task and returns its initial state with a placeholder plan.
``run`` plans the task, runs the Researcher when the task needs external knowledge
(storing structured findings in shared state), then drives the Coder -> Tester repair loop
until the change is VERIFIED_SUCCESS or the loop stops as BLOCKED (see ``repair_loop``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from harness.config.settings import Settings
from harness.context.manager import (
    ContextCategory,
    ContextEntry,
    ContextManager,
    InMemoryContextManager,
)
from harness.llm.client import LLMClient, create_llm_client
from harness.orchestrator.state import AgentState, PlanStep, StepStatus, TaskStatus
from harness.tools.repository import RepositoryContext

if TYPE_CHECKING:
    from harness.orchestrator.final_result import FinalResult
    from harness.agents.coder import CoderAgent
    from harness.agents.researcher import ResearcherAgent
    from harness.agents.tester import TesterAgent
    from harness.orchestrator.planner import ResearchPlan

log = logging.getLogger("harness.orchestrator")

# Placeholder plan used until the planner exists: (step description, responsible agent).
PLACEHOLDER_PLAN: tuple[tuple[str, str], ...] = (
    ("Gather context about the task and repository", "researcher"),
    ("Implement the required code changes", "coder"),
    ("Run tests and verify the result", "tester"),
)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient | None = None,
        context: ContextManager | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.context = context if context is not None else InMemoryContextManager()

    def submit(self, task: str) -> AgentState:
        """Accept ``task`` and return its initial state with a placeholder plan."""
        task = task.strip()
        if not task:
            raise ValueError("Task must be a non-empty string")

        state = AgentState(task=task)
        log.info("accepted task %s: %s", state.task_id, task)

        state.plan = [
            PlanStep(id=f"step-{i}", description=description, agent=agent)
            for i, (description, agent) in enumerate(PLACEHOLDER_PLAN, start=1)
        ]
        state.pending_tasks = [step.id for step in state.plan]
        state.status = TaskStatus.PLANNED
        self.context.put(ContextEntry(ContextCategory.TASK, state.task_id, task, "orchestrator"))
        log.info("created placeholder plan with %d steps", len(state.plan))
        return state

    def run(
        self,
        task: str,
        repo: RepositoryContext | str | Path,
        *,
        coder: CoderAgent | None = None,
        tester: TesterAgent | None = None,
        researcher: ResearcherAgent | None = None,
        research: bool | None = None,
    ) -> AgentState:
        """Plan -> (Researcher) -> Coder -> Tester repair loop for ``task`` on ``repo``.

        ``research``: None = decide from the task, True = always research, False = never.
        """
        # Imported here: the agents package itself depends on orchestrator.state.
        from harness.agents.coder import CoderAgent
        from harness.agents.researcher import ResearcherAgent
        from harness.agents.tester import TesterAgent
        from harness.orchestrator.planner import build_plan, plan_research
        from harness.orchestrator.repair_loop import RepairLoop
        from harness.verification.detection import detect_commands, inspect_project

        repo_ctx = repo if isinstance(repo, RepositoryContext) else RepositoryContext(Path(repo))
        state = self.submit(task)

        # 1. detect the repository
        profile = inspect_project(repo_ctx)
        commands = [c.command for c in detect_commands(repo_ctx, profile)]
        self.context.put(
            ContextEntry(
                ContextCategory.REPOSITORY,
                f"{state.task_id}:profile",
                json.dumps({**profile.to_dict(), "verification_commands": commands}),
                "orchestrator",
                task_id=state.task_id,
            )
        )
        log.info(
            "repository %s: ecosystems=%s, metadata=%s, verification=%s",
            repo_ctx.name, profile.ecosystems or ["unknown"], profile.metadata_files or ["none"],
            commands or ["none detected"],
        )

        # 2. plan (does the task need research?)
        research_plan = plan_research(task, repo_ctx, force=research)
        state.plan = build_plan(research_plan)
        state.pending_tasks = [s.id for s in state.plan if s.status is not StepStatus.SKIPPED]
        self.context.put(
            ContextEntry(ContextCategory.PLAN, f"{state.task_id}:plan",
                         json.dumps([asdict(s) for s in state.plan], default=str),
                         "orchestrator", task_id=state.task_id)
        )
        log.info("plan: research %s (%s); %d steps",
                 "needed" if research_plan.needed else "skipped", research_plan.reason,
                 len(state.plan))

        needs_llm = coder is None or tester is None or (research_plan.needed and researcher is None)
        if needs_llm and self.llm is None:
            self.llm = create_llm_client(self.settings)
        llm = self.llm

        # 3. research (optional) -> structured findings in shared state
        if research_plan.needed:
            assert llm is not None or researcher is not None
            researcher = researcher or ResearcherAgent(
                llm, self.context, repo_ctx, settings=self.settings  # type: ignore[arg-type]
            )
            researcher.topics = research_plan.libraries
            blocked = self._research(state, researcher, research_plan)
            if blocked is not None:
                return blocked
        else:
            state.research = {"status": "SKIPPED", **research_plan.to_dict()}

        # 4. Coder -> Tester repair loop (unchanged architecture)
        coder = coder or CoderAgent(llm, self.context, repo_ctx, settings=self.settings)  # type: ignore[arg-type]
        tester = tester or TesterAgent(llm, self.context, repo_ctx, settings=self.settings)  # type: ignore[arg-type]
        log.info(
            "starting repair loop for %s on repository %s (max repair attempts: %d)",
            state.task_id, repo_ctx.name, self.settings.max_repair_attempts,
        )
        loop = RepairLoop(
            coder, tester, repo_ctx, max_repair_attempts=self.settings.max_repair_attempts
        )
        return loop.run(state)

    # --- autonomous controller (Phase 5) ---------------------------------------------------

    def build_agents(
        self, repo: RepositoryContext, *, include_music: bool = True
    ) -> dict[str, Any]:
        """The four worker agents, created from settings (requires a configured LLM)."""
        from harness.agents.coder import CoderAgent
        from harness.agents.music import MusicAgent
        from harness.agents.researcher import ResearcherAgent
        from harness.agents.tester import TesterAgent

        if self.llm is None:
            self.llm = create_llm_client(self.settings)
        agents: dict[str, Any] = {
            "coder": CoderAgent(self.llm, self.context, repo, settings=self.settings),
            "researcher": ResearcherAgent(self.llm, self.context, repo, settings=self.settings),
            "tester": TesterAgent(self.llm, self.context, repo, settings=self.settings),
        }
        if include_music:
            agents["music"] = MusicAgent(self.llm, self.context, settings=self.settings)
        return agents

    def plan_graph(
        self,
        task: str,
        repo: RepositoryContext | str | Path,
        *,
        research: bool | None = None,
        listeners: list[Any] | None = None,
    ) -> AgentState:
        """Plan only: returns the state with its dependency-aware task graph (no agent runs)."""
        from harness.orchestrator.controller import ExecutionController

        repo_ctx = repo if isinstance(repo, RepositoryContext) else RepositoryContext(Path(repo))
        return ExecutionController(self, repo_ctx, {}, listeners=listeners).plan(task, research=research)

    def execute(
        self,
        task: str,
        repo: RepositoryContext | str | Path,
        *,
        agents: dict[str, Any] | None = None,
        research: bool | None = None,
        grace_seconds: float = 30.0,
        listeners: list[Any] | None = None,
    ) -> FinalResult:
        """The autonomous controller: plan a task graph, schedule agents (concurrently where
        independent), recover from failures, repair, and pass the verification gate.

        Never raises for agent/tool failures: the outcome is always a FinalResult
        (VERIFIED_SUCCESS / FAILED / BLOCKED).
        """
        from harness.orchestrator.controller import ExecutionController

        repo_ctx = repo if isinstance(repo, RepositoryContext) else RepositoryContext(Path(repo))
        workers = agents if agents is not None else self.build_agents(repo_ctx)
        controller = ExecutionController(
            self, repo_ctx, workers, grace_seconds=grace_seconds, listeners=listeners
        )
        self.last_controller = controller
        return controller.execute(task, research=research)

    # --- research stage ------------------------------------------------------------------

    def _research(
        self, state: AgentState, researcher: ResearcherAgent, plan: ResearchPlan
    ) -> AgentState | None:
        """Run the Researcher (API questions, then project-specific questions). Returns a
        BLOCKED state if the Coder cannot proceed, else None. Never raises."""
        from harness.agents.base import AgentStatus

        runs: list[dict[str, Any]] = []
        research_steps = [s for s in state.plan if s.agent == "researcher"]
        for step, questions in zip(
            research_steps, (plan.api_questions, plan.project_questions), strict=False
        ):
            if not questions:
                continue
            step.status = StepStatus.IN_PROGRESS
            log.info("research | agent=researcher | questions=%d | %s", len(questions), step.description[:120])
            result = researcher.run("\n".join(questions), state)  # BaseAgent.run never raises
            findings = getattr(result, "findings", [])
            runs.append(
                {
                    "step": step.id,
                    "status": result.status.value,
                    "summary": result.summary,
                    "findings": len(findings),
                    "usable_findings": len([f for f in findings if f.usable]),
                    "rejected_findings": len(getattr(result, "rejected_findings", [])),
                    "sources": getattr(result, "sources_consulted", []),
                    "errors": result.errors,
                }
            )
            step.status = StepStatus.DONE if result.status is AgentStatus.SUCCESS else StepStatus.FAILED
            log.info("research | result=%s | findings=%d | %s", result.status, len(findings),
                     result.summary[:160])

        usable = [
            f for f in state.research_findings
            if f.get("kind") == "FACT" or (f.get("kind") == "INFERENCE" and f.get("confidence") != "LOW")
        ]
        if usable:
            decision, reason = "proceed", f"{len(usable)} usable finding(s) handed to the Coder"
        elif plan.can_proceed_without_research:
            decision = "proceed"
            reason = (
                "research produced no usable findings, but the Coder can inspect local evidence: "
                + "; ".join(f"{lib}: {', '.join(ev) or 'n/a'}" for lib, ev in plan.local_evidence.items())
                if plan.libraries else "research was optional"
            )
        else:
            decision = "blocked"
            reason = (
                "research failed and the required librar"
                f"{'ies' if len(plan.libraries) > 1 else 'y'} {', '.join(plan.libraries)} "
                "has no local evidence (not declared, imported or installed) - refusing to guess"
            )
        state.research = {
            "status": "SUCCESS" if usable else "FAILED",
            **plan.to_dict(),
            "runs": runs,
            "usable_findings": len(usable),
            "decision": decision,
            "decision_reason": reason,
        }
        log.info("research decision: %s - %s", decision, reason)
        if decision == "proceed":
            return None

        state.status = TaskStatus.BLOCKED
        for step in state.plan:
            if step.status is StepStatus.PENDING:
                step.status = StepStatus.SKIPPED
        state.outcome = {
            "status": TaskStatus.BLOCKED.value,
            "reason": f"Research failed: {reason}",
            "research": state.research,
            "attempted": [],
            "latest_failure": state.failures[-1] if state.failures else None,
            "previous_failures": state.failures[:-1],
            "files_changed": [],
            "tests_run": [],
            "verification_status": state.verification_status.value,
        }
        state.final_result = f"BLOCKED: Research failed: {reason}"
        log.warning("Task %s BLOCKED: %s", state.task_id, state.final_result)
        return state
