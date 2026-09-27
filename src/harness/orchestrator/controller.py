"""ExecutionController: the Orchestrator's autonomous control loop.

    understand + decompose  -> Planner builds a TaskGraph (DAG)
    select agents           -> each node names an agent; capabilities are enforced
    schedule                -> Scheduler runs independent READY nodes concurrently
    manage shared state     -> agents work on private state copies; the controller
                               validates their changes and commits only what each node kind
                               is allowed to contribute
    handle failures/retries -> RecoveryManager (bounded, every retry has a reason)
    repair                  -> failed tests add Coder-repair + re-test nodes with evidence
    verify / finish         -> VerificationManager is the only path to VERIFIED_SUCCESS

Agents are workers; the controller owns AgentState.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from harness.agents.base import AgentResult, AgentStatus, BaseAgent
from harness.context.manager import ContextCategory, ContextEntry
from harness.context.usage import accumulate_usage
from harness.orchestrator.events import EventLog, EventType
from harness.orchestrator.final_result import FinalResult
from harness.orchestrator.graph import NodeKind, NodeStatus, TaskGraph, TaskNode
from harness.orchestrator.permissions import (
    PermissionViolation,
    check_assignment,
    check_tools,
    repository_access,
)
from harness.orchestrator.planner import Planner, PlannedTask
from harness.orchestrator.recovery import (
    FailureKind,
    RecoveryAction,
    RecoveryManager,
    RetryPolicy,
)
from harness.orchestrator.scheduler import NodeOutcome, Scheduler
from harness.orchestrator.state import AgentState, PlanStep, StepStatus, TaskStatus
from harness.orchestrator.verification_manager import (
    FinalStatus,
    VerificationDecision,
    VerificationManager,
    changed_files,
)
from harness.research.models import FindingError, ResearchFinding
from harness.tools.registry import CancellationToken
from harness.tools.repository import RepositoryContext
from harness.verification.models import TestStatus, VerificationStatus

if TYPE_CHECKING:
    from harness.orchestrator.orchestrator import Orchestrator

log = logging.getLogger("harness.orchestrator.controller")

# What each node kind may contribute to shared state. Everything else is rejected.
COMMITTABLE: dict[NodeKind, frozenset[str]] = {
    NodeKind.IMPLEMENT: frozenset({"code_changes", "failures"}),
    NodeKind.REPAIR: frozenset({"code_changes", "failures"}),
    NodeKind.TEST: frozenset({"test_results", "latest_test_result", "verification_attempts", "failures"}),
    NodeKind.RESEARCH: frozenset({"research_findings", "failures"}),
    NodeKind.INSPECT: frozenset(),
    NodeKind.BASELINE: frozenset({"baseline"}),
    NodeKind.MUSIC: frozenset(),
    NodeKind.VERIFY: frozenset(),
}
IGNORED_FIELDS = frozenset({"active_agent", "usage"})  # bookkeeping done by BaseAgent.run
_STEP_STATUS = {
    NodeStatus.PENDING: StepStatus.PENDING, NodeStatus.READY: StepStatus.PENDING,
    NodeStatus.RUNNING: StepStatus.IN_PROGRESS, NodeStatus.SUCCESS: StepStatus.DONE,
    NodeStatus.FAILED: StepStatus.FAILED, NodeStatus.BLOCKED: StepStatus.FAILED,
    NodeStatus.SKIPPED: StepStatus.SKIPPED,
}
_TASK_STATUS = {
    FinalStatus.VERIFIED_SUCCESS: TaskStatus.VERIFIED_SUCCESS,
    FinalStatus.BLOCKED: TaskStatus.BLOCKED,
    FinalStatus.FAILED: TaskStatus.FAILED,
}


@dataclass
class _Prepared:
    baseline: AgentState  # state at dispatch (for validating the agent's changes)
    working: AgentState  # private copy the agent may modify
    graph: TaskGraph | None = None  # for the VERIFY node


class ExecutionController:
    def __init__(
        self,
        orchestrator: Orchestrator,
        repo: RepositoryContext,
        agents: dict[str, BaseAgent],
        *,
        planner: Planner | None = None,
        recovery: RecoveryManager | None = None,
        verifier: VerificationManager | None = None,
        grace_seconds: float = 30.0,
        listeners: list[Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.settings = orchestrator.settings
        self.context = orchestrator.context
        self.repo = repo
        self.agents = agents
        self.planner = planner or Planner()
        self.recovery = recovery or RecoveryManager(RetryPolicy.from_settings(self.settings))
        self.verifier = verifier or VerificationManager(repo)
        self.grace_seconds = grace_seconds
        self.events = EventLog()
        for listener in listeners or []:
            self.events.subscribe(listener)
        self.state: AgentState | None = None
        self.graph = TaskGraph()
        self.planned: PlannedTask | None = None
        self.scheduler: Scheduler | None = None
        self._research_runs: list[dict[str, Any]] = []
        self._repair_helper: Any = None

    # --- public ----------------------------------------------------------------------------

    def plan(self, task: str, *, research: bool | None = None) -> AgentState:
        """Plan only (no agent runs): state with the task graph."""
        state = self.orchestrator.submit(task)
        self.events.task_id = state.task_id
        self.state = state
        self._plan(task, research)
        return state

    def execute(self, task: str, *, research: bool | None = None) -> FinalResult:
        started = time.monotonic()
        fatal: str | None = None
        decision: VerificationDecision | None = None
        try:
            state = self.plan(task, research=research)
            self._enforce_permissions()
            self._configure_agents()
            self.scheduler = Scheduler(
                self.graph,
                prepare=self._prepare,
                execute=self._run_node,
                on_complete=self._on_complete,
                access=lambda n: repository_access(n.agent, n.kind),
                agent_key=lambda n: n.agent,
                max_concurrent=self.settings.max_concurrent_agents,
                node_timeout=self.settings.agent_timeout_seconds,
                grace_seconds=self.grace_seconds,
                events=self.events,
            )
            self.scheduler.run_sync()
            decision = self._final_decision()
        except PermissionViolation as exc:
            fatal = f"permission violation: {exc}"
            log.error("%s", fatal)
        except Exception as exc:  # noqa: BLE001 - clean final error instead of a traceback
            fatal = f"{type(exc).__name__}: {exc}"
            log.error("orchestration failed: %s", fatal)
            log.debug("orchestration traceback", exc_info=True)
        if self.state is None:  # failed before a task existed
            self.state = AgentState(task=task or "<empty>")
            self.events.task_id = self.state.task_id
        if fatal is not None:
            try:
                decision = self.verifier.evaluate(self.state, self.graph, fatal_error=fatal)
            except Exception as exc:  # noqa: BLE001
                decision = VerificationDecision(FinalStatus.FAILED, [], [f"fatal error: {fatal}", str(exc)],
                                                "Orchestration failed.")
        assert decision is not None
        return self._finish(decision, time.monotonic() - started)

    # --- planning --------------------------------------------------------------------------

    def _plan(self, task: str, research: bool | None) -> None:
        from harness.verification.detection import detect_commands, inspect_project

        state = self.state
        assert state is not None
        self.events.emit(EventType.TASK_CREATED, task[:200])
        profile = inspect_project(self.repo)
        commands = [c.command for c in detect_commands(self.repo, profile)]
        self.context.put(ContextEntry(
            ContextCategory.REPOSITORY, f"{state.task_id}:profile",
            json.dumps({**profile.to_dict(), "verification_commands": commands}),
            "orchestrator", task_id=state.task_id,
        ))
        self.planned = self.planner.plan(task, self.repo, research=research)
        self.graph = self.planned.graph
        state.research = (
            {"status": "PLANNED", **self.planned.research.to_dict()}
            if self.planned.research.needed
            else {"status": "SKIPPED", **self.planned.research.to_dict()}
        )
        self._sync_plan()
        self.context.put(ContextEntry(
            ContextCategory.PLAN, f"{state.task_id}:plan", self.graph.describe(),
            "orchestrator", task_id=state.task_id,
        ))
        self.events.emit(
            EventType.PLAN_CREATED,
            f"{len(self.graph)} tasks; research={'yes' if self.planned.research.needed else 'no'}; "
            f"music={'yes' if self.planned.music_command else 'no'}",
            nodes=[{"id": n.id, "agent": n.agent, "kind": n.kind.value, "description": n.description,
                    "dependencies": list(n.dependencies), "required": n.required}
                   for n in (self.graph[i] for i in self.graph.topological_order())],
            agents={name: getattr(agent, "status_text", "available")
                    for name, agent in self.agents.items()},
            coding_task=self.planned.coding_task,
            music_command=self.planned.music_command,
        )
        log.info("plan:\n%s", self.graph.describe())

    def _sync_plan(self) -> None:
        state = self.state
        assert state is not None
        state.plan = [
            PlanStep(n.id, n.description, n.agent, _STEP_STATUS[n.status])
            for n in (self.graph[i] for i in self.graph.topological_order())
        ]
        state.pending_tasks = [n.id for n in self.graph if not n.status.is_terminal]
        state.completed_tasks = [n.id for n in self.graph if n.status is NodeStatus.SUCCESS]
        state.task_graph = self.graph.to_dict()

    def _enforce_permissions(self) -> None:
        for name, agent in self.agents.items():
            if agent.name != name:
                raise PermissionViolation(f"agent registered as {name!r} is {agent.name!r}")
            check_tools(name, agent.tools)
        for node in self.graph:
            if node.agent == "orchestrator":
                if node.kind is not NodeKind.VERIFY:
                    raise PermissionViolation(f"orchestrator may only verify, not {node.kind}")
                continue
            if node.agent not in self.agents:
                raise PermissionViolation(f"no agent available for {node.id} ({node.agent})")
            check_assignment(node.agent, node.kind)

    def _configure_agents(self) -> None:
        for agent in self.agents.values():
            registry = getattr(agent, "registry", None)
            if registry is not None:
                registry.max_tool_retries = self.settings.max_tool_retries

    # --- dispatch (scheduler thread boundary) ----------------------------------------------

    def _prepare(self, node: TaskNode) -> _Prepared:
        """Runs on the event-loop thread: snapshot state for the agent's private use."""
        assert self.state is not None
        baseline = copy.deepcopy(self.state)
        prepared = _Prepared(baseline=baseline, working=copy.deepcopy(baseline))
        if node.kind is NodeKind.VERIFY:
            prepared.graph = copy.deepcopy(self.graph)
        agent = self.agents.get(node.agent)
        registry = getattr(agent, "registry", None)
        if registry is not None:
            registry.observer = self._tool_observer(node.id, node.agent)
            registry.before = self._tool_start_observer(node.id, node.agent)
        return prepared

    def _tool_start_observer(self, node_id: str, agent: str):  # type: ignore[no-untyped-def]
        def started(tool: str, arguments: Any) -> None:
            target = _tool_target(arguments)
            self.events.emit(EventType.TOOL_STARTED, f"{tool} {target}".strip(), agent=agent,
                             node=node_id, tool=tool, target=target)
        return started

    def _tool_observer(self, node_id: str, agent: str):  # type: ignore[no-untyped-def]
        def observe(tool: str, arguments: Any, result: Any, attempt: int) -> None:
            target = _tool_target(arguments)
            if attempt > 1:
                self.events.emit(EventType.TOOL_RETRIED, f"{tool} attempt {attempt}", agent=agent,
                                 node=node_id, tool=tool, target=target)
            if result.ok:
                self.events.emit(EventType.TOOL_CALLED, f"{tool} {target}".strip(), agent=agent,
                                 node=node_id, tool=tool, target=target,
                                 summary=_tool_summary(tool, result.data))
            else:
                self.events.emit(EventType.TOOL_FAILED, f"{tool}: {result.error}", agent=agent,
                                 node=node_id, tool=tool, target=target)
                if result.metadata.get("retryable"):
                    self.recovery.record_tool_retry(node_id, agent, tool, attempt, result.error or "")
        return observe

    def _run_node(self, node: TaskNode, token: CancellationToken, prepared: _Prepared) -> NodeOutcome:
        """Runs on a worker thread. Touches only the private state copy."""
        state = prepared.working
        if node.kind is NodeKind.VERIFY:
            assert prepared.graph is not None
            return NodeOutcome(result=self.verifier.evaluate(state, prepared.graph),
                               snapshot=state, baseline=prepared.baseline)
        agent = self.agents[node.agent]
        registry = getattr(agent, "registry", None)
        if registry is not None:
            registry.cancel_token = token
        assert self.planned is not None
        coding_task = self.planned.coding_task or self.planned.original_task
        if node.kind is NodeKind.INSPECT:
            try:
                result: AgentResult = agent.inspect(coding_task, state)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - agent boundary
                result = AgentResult(agent.name, AgentStatus.FAILURE, "Inspection raised an exception",
                                     errors=[f"{type(exc).__name__}: {exc}"])
        elif node.kind is NodeKind.RESEARCH:
            if hasattr(agent, "topics") and self.planned.research.libraries:
                agent.topics = self.planned.research.libraries  # type: ignore[attr-defined]
            result = agent.run("\n".join(node.payload.get("questions", [])), state)
        elif node.kind is NodeKind.REPAIR:
            result = agent.run(node.payload["prompt"], state)
        elif node.kind is NodeKind.BASELINE:
            try:
                result = agent.baseline(coding_task, state)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - agent boundary; the baseline is optional
                result = AgentResult(agent.name, AgentStatus.FAILURE, "Baseline raised an exception",
                                     errors=[f"{type(exc).__name__}: {exc}"])
        elif node.kind is NodeKind.MUSIC:
            result = agent.run(node.payload["command"], state)
        else:  # IMPLEMENT, TEST
            result = agent.run(coding_task, state)
        return NodeOutcome(result=result, snapshot=state, baseline=prepared.baseline)

    # --- completion (event-loop thread: the only place shared state changes) -----------------

    def _on_complete(self, node: TaskNode, outcome: NodeOutcome) -> None:
        state = self.state
        assert state is not None
        if outcome.snapshot is not None and outcome.result is not None:
            self._commit(node, outcome)
        result = outcome.result
        node.result = self._compact(node, result, outcome)

        succeeded, kind, detail, retryable, repairable = self._assess(node, result, outcome)
        if succeeded:
            node.mark(NodeStatus.SUCCESS)
            self.events.emit(EventType.AGENT_COMPLETED, detail, agent=node.agent, node=node.id,
                             duration=outcome.duration, kind=node.kind.value,
                             usage=_usage_of(result), result=_event_result(node.result))
            if node.kind is NodeKind.TEST:
                self.events.emit(EventType.TEST_PASSED, detail, agent=node.agent, node=node.id)
                self._record_attempt(node, "VERIFIED")
            if node.kind is NodeKind.RESEARCH:
                self._record_research(node, result, "SUCCESS")
            if node.kind is NodeKind.VERIFY:
                self.events.emit(EventType.VERIFICATION_PASSED, detail, node=node.id)
            self._sync_plan()
            return

        assert kind is not None
        self.events.emit(
            EventType.TEST_FAILED if kind is FailureKind.TEST_FAILURE
            else EventType.RESEARCH_FAILED if kind is FailureKind.RESEARCH_FAILURE
            else EventType.VERIFICATION_FAILED if node.kind is NodeKind.VERIFY
            else EventType.AGENT_FAILED,
            f"{kind.value}: {detail}", agent=node.agent, node=node.id, kind=node.kind.value,
            failure=kind.value, usage=_usage_of(result), result=_event_result(node.result),
        )
        can_proceed = False
        if node.kind is NodeKind.RESEARCH:
            can_proceed = not node.required or self._usable_findings() > 0
        decision = self.recovery.decide(
            node, kind, detail=detail, retryable=retryable, repairable=repairable,
            repairs_done=state.repair_attempts, can_proceed=can_proceed,
        )
        if node.kind is NodeKind.RESEARCH:
            self._record_research(node, result, decision.action.value)
        log.info("recovery for %s: %s (%s)", node.id, decision.action, decision.reason)

        if decision.action is RecoveryAction.RETRY:
            node.history.append({"attempt": node.retry_count + 1, "kind": kind.value,
                                 "reason": decision.reason, "detail": detail[:300]})
            node.retry_count += 1
            node.status = NodeStatus.READY
            self.events.emit(EventType.AGENT_RETRIED, decision.reason, agent=node.agent, node=node.id)
        elif decision.action is RecoveryAction.REPAIR:
            node.mark(NodeStatus.FAILED, error=detail)
            self._schedule_repair(node, result)
        elif decision.action is RecoveryAction.PROCEED:
            node.mark(NodeStatus.FAILED, error=f"{detail} (proceeding: {decision.reason})")
            node.required = False  # dependents may continue without it
        else:
            node.mark(NodeStatus.BLOCKED, error=f"{detail} ({decision.reason})")
            self.events.emit(EventType.NODE_BLOCKED, decision.reason, agent=node.agent, node=node.id)
            if node.kind is NodeKind.TEST:
                if kind is FailureKind.TEST_FAILURE:
                    self._test_failure(node, result)  # keep the evidence of why it stopped
                self._record_attempt(node, "STOP")
        self._sync_plan()

    def _assess(
        self, node: TaskNode, result: Any, outcome: NodeOutcome
    ) -> tuple[bool, FailureKind | None, str, bool, bool]:
        """(succeeded, failure kind, detail, retryable, repairable) - from evidence only."""
        from harness.agents.coder import CoderResult
        from harness.agents.tester import TesterResult

        if outcome.error is not None:
            kind = FailureKind.TIMEOUT if outcome.timed_out else FailureKind.AGENT_FAILURE
            return False, kind, outcome.error, not outcome.abandoned, False
        if node.kind is NodeKind.VERIFY:
            ok = isinstance(result, VerificationDecision) and result.verified
            detail = result.summary if isinstance(result, VerificationDecision) else "no decision"
            return ok, None if ok else FailureKind.BLOCKED_BY_AGENT, detail, False, False
        if not isinstance(result, AgentResult):
            return False, FailureKind.MALFORMED_OUTPUT, f"agent returned {type(result).__name__}", True, False

        errors = "; ".join(map(str, result.errors)) or result.summary
        if outcome.timed_out or "OperationCancelled" in errors:
            return False, FailureKind.TIMEOUT, errors, True, False

        if node.kind is NodeKind.TEST:
            if not isinstance(result, TesterResult):
                return False, FailureKind.AGENT_FAILURE, errors, True, False
            if result.verification_status is VerificationStatus.PASSED:
                return True, None, result.summary, False, False
            if result.integrity_violations:
                return False, FailureKind.INTEGRITY_VIOLATION, "; ".join(result.integrity_violations), False, False
            if result.verification_status is VerificationStatus.NOT_AVAILABLE:
                return False, FailureKind.VERIFICATION_UNAVAILABLE, result.summary, False, False
            classification = result.failure_classification
            return (False, FailureKind.TEST_FAILURE, result.summary, False,
                    bool(classification and classification.repairable))

        if result.status is AgentStatus.SUCCESS:
            if node.kind in (NodeKind.IMPLEMENT, NodeKind.REPAIR) and not isinstance(result, CoderResult):
                return False, FailureKind.MALFORMED_OUTPUT, "coder returned an unexpected result", True, False
            return True, None, result.summary, False, False
        if result.status is AgentStatus.BLOCKED and node.kind is not NodeKind.RESEARCH:
            return False, FailureKind.BLOCKED_BY_AGENT, errors, False, False
        if result.metadata.get("llm_error"):
            code = result.metadata.get("error_code", "PROVIDER_ERROR")
            return (False, FailureKind.LLM_ERROR, f"{code}: {errors}",
                    bool(result.metadata.get("retryable")), False)
        if node.kind is NodeKind.RESEARCH:
            transient = bool(result.metadata.get("llm_error") and result.metadata.get("retryable"))
            return False, FailureKind.RESEARCH_FAILURE, errors, transient, False
        if node.kind is NodeKind.MUSIC:
            return False, FailureKind.OPTIONAL_TASK_FAILURE, errors, False, False
        if "structured final result" in errors or "Tool call limit reached" in errors:
            return False, FailureKind.MALFORMED_OUTPUT, errors, True, False
        return False, FailureKind.AGENT_FAILURE, errors, True, False

    # --- state commit ------------------------------------------------------------------------

    def _commit(self, node: TaskNode, outcome: NodeOutcome) -> None:
        """Validate the agent's state changes and apply only the allowed ones."""
        from harness.agents.coder import CoderResult
        from harness.agents.tester import TesterResult

        state, before, after, result = self.state, outcome.baseline, outcome.snapshot, outcome.result
        assert state is not None
        allowed = COMMITTABLE[node.kind]
        rejected: list[str] = []
        for name in AgentState.__dataclass_fields__:
            old, new = getattr(before, name), getattr(after, name)
            if old == new or name in IGNORED_FIELDS:
                continue
            if name not in allowed:
                rejected.append(f"{name} (not writable by {node.kind.value})")
                continue
            if isinstance(new, list):
                if new[: len(old)] != old:
                    rejected.append(f"{name} (existing entries were modified)")
                    continue
                for item in new[len(old):]:
                    problem = self._validate_item(name, item, node, result)
                    if problem:
                        rejected.append(f"{name} entry ({problem})")
                    else:
                        getattr(state, name).append(item)
            elif name == "verification_attempts":
                state.verification_attempts += max(new - old, 0)
            elif name == "latest_test_result":
                if isinstance(result, TesterResult) and result.decisive_result is not None:
                    state.latest_test_result = new if isinstance(new, dict) else result.decisive_result.to_dict()
                else:
                    rejected.append("latest_test_result (no decisive test result)")
            elif name == "baseline":
                if isinstance(new, dict) and isinstance(result, TesterResult):
                    state.baseline = new
                else:
                    rejected.append("baseline (not produced by the Tester)")
        usage = result.metadata.get("usage") if isinstance(result, AgentResult) else None
        if isinstance(usage, dict):
            accumulate_usage(state.usage, node.agent, usage)
        if isinstance(result, TesterResult) and node.kind is NodeKind.TEST:
            state.verification_status = result.verification_status
        if isinstance(result, CoderResult) and result.files_changed:
            pass  # code_changes already validated against result.files_changed
        if rejected:
            self.events.emit(EventType.STATE_REJECTED, "; ".join(rejected)[:280],
                             agent=node.agent, node=node.id)
            log.warning("rejected state changes from %s: %s", node.id, rejected)

    @staticmethod
    def _validate_item(field_name: str, item: Any, node: TaskNode, result: Any) -> str | None:
        from harness.agents.coder import CoderResult
        from harness.agents.tester import TesterResult

        if not isinstance(item, dict):
            return "not an object"
        if field_name == "failures":
            return None if item.get("agent") == node.agent else "attributed to another agent"
        if field_name == "code_changes":
            if not isinstance(result, CoderResult):
                return "no coder result"
            extra = set(item.get("files_changed", [])) - set(result.files_changed)
            return f"files not changed by tools: {sorted(extra)}" if extra else None
        if field_name == "test_results":
            if not isinstance(result, TesterResult):
                return "no tester result"
            try:
                TestStatus(item.get("status"))
            except ValueError:
                return "invalid status"
            return None if item.get("command") in result.commands_run else "command was not run"
        if field_name == "research_findings":
            try:
                finding = ResearchFinding.from_dict(item)
            except FindingError as exc:
                return str(exc)
            if finding.kind.value == "FACT" and not finding.source_verified:
                return "FACT without a verified source"
            return None
        return None

    # --- repair ------------------------------------------------------------------------------

    def _test_failure(self, test_node: TaskNode, tester_result: Any) -> Any:
        """Build the failure evidence for a failed test node and record it in failure_history."""
        from harness.orchestrator.repair_loop import RepairLoop

        state = self.state
        assert state is not None
        coder_nodes = [n for n in self.graph if n.kind in (NodeKind.IMPLEMENT, NodeKind.REPAIR)
                       and n.status is NodeStatus.SUCCESS]
        previous = coder_nodes[-1] if coder_nodes else None
        coder_result = _AgentResultView((previous.result if previous else None) or {})
        if self._repair_helper is None:
            self._repair_helper = RepairLoop(
                self.agents["coder"], self.agents["tester"], self.repo,  # type: ignore[arg-type]
                max_repair_attempts=self.recovery.policy.max_repair_attempts,
            )
        files = changed_files(state)
        exists, evidence = self._repair_helper._changes_exist(files)  # noqa: SLF001
        failure = self._repair_helper._failure(  # noqa: SLF001
            state.verification_attempts, coder_result, tester_result, exists, evidence,
            previous.result.get("files_changed", []) if previous and previous.result else [],
        )
        state.failure_history.append(failure.to_dict())
        return failure

    def _schedule_repair(self, test_node: TaskNode, tester_result: Any) -> None:
        """Add Coder-repair and re-test nodes carrying the real failure evidence."""
        state = self.state
        assert state is not None
        coder_nodes = [n for n in self.graph if n.kind in (NodeKind.IMPLEMENT, NodeKind.REPAIR)
                       and n.status is NodeStatus.SUCCESS]
        previous = coder_nodes[-1]
        files = changed_files(state)
        failure = self._test_failure(test_node, tester_result)
        state.repair_attempts += 1
        state.retry_count = state.repair_attempts
        prompt = self._repair_helper.repair_prompt(self.planned.coding_task or state.task, state, failure, files)  # type: ignore[union-attr]
        self._record_attempt(test_node, "REPAIR", failure.category.value)

        repair = self.graph.add(TaskNode(
            self.graph.next_id("repair"), f"Repair attempt {state.repair_attempts}: fix {failure.category.value}",
            "coder", NodeKind.REPAIR, dependencies=[previous.id], priority=3,
            payload={"prompt": prompt, "failure_attempt": failure.attempt},
        ))
        retest = self.graph.add(TaskNode(
            self.graph.next_id("test"), f"Re-run tests after {repair.id}", "tester", NodeKind.TEST,
            dependencies=[repair.id], priority=2,
        ))
        test_node.superseded_by = repair.id
        for other in self.graph:
            if test_node.id in other.dependencies and other.id not in (repair.id, retest.id):
                other.dependencies = [retest.id if d == test_node.id else d for d in other.dependencies]
        self.graph.validate()
        self.events.emit(EventType.REPAIR_STARTED,
                         f"{repair.id} + {retest.id} after {failure.category.value}: {failure.counts}",
                         agent="coder", node=repair.id, attempt=state.repair_attempts,
                         new_nodes=[{"id": n.id, "agent": n.agent, "kind": n.kind.value,
                                     "description": n.description} for n in (repair, retest)])

    def _record_attempt(self, test_node: TaskNode, outcome: str, category: str | None = None) -> None:
        state = self.state
        assert state is not None
        coder = next((self.graph[d] for d in reversed(test_node.dependencies)), None)
        state.attempt_history.append({
            "attempt": len(state.attempt_history) + 1,
            "action": "implement" if coder and coder.kind is NodeKind.IMPLEMENT else "repair",
            "coder": {
                "node": coder.id if coder else None,
                "status": (coder.result or {}).get("status") if coder else None,
                "summary": (coder.result or {}).get("summary", "") if coder else "",
                "files_changed": (coder.result or {}).get("files_changed", []) if coder else [],
                "errors": (coder.result or {}).get("errors", []) if coder else [],
            },
            "tester": {"node": test_node.id, **{k: (test_node.result or {}).get(k) for k in
                       ("verification_status", "summary")}},
            "classification": {"category": category} if category else None,
            "outcome": outcome,
        })

    # --- research bookkeeping ------------------------------------------------------------------

    def _usable_findings(self) -> int:
        state = self.state
        assert state is not None
        return sum(
            1 for f in state.research_findings
            if f.get("kind") == "FACT" or (f.get("kind") == "INFERENCE" and f.get("confidence") != "LOW")
        )

    def _record_research(self, node: TaskNode, result: Any, decision: str) -> None:
        state = self.state
        assert state is not None and state.research is not None
        self._research_runs.append({
            "node": node.id, "attempt": node.retry_count + 1,
            "status": getattr(getattr(result, "status", None), "value", "FAILURE"),
            "summary": getattr(result, "summary", ""),
            "findings": len(getattr(result, "findings", []) or []),
            "rejected_findings": len(getattr(result, "rejected_findings", []) or []),
            "sources": getattr(result, "sources_consulted", []),
            "errors": getattr(result, "errors", []),
            "decision": decision,
        })
        usable = self._usable_findings()
        state.research.update(
            runs=list(self._research_runs), usable_findings=usable,
            status="SUCCESS" if usable else "FAILED",
        )

    # --- results -------------------------------------------------------------------------------

    @staticmethod
    def _compact(node: TaskNode, result: Any, outcome: NodeOutcome) -> dict[str, Any]:
        data: dict[str, Any] = {"duration": outcome.duration}
        if outcome.error:
            data["error"] = outcome.error
        if isinstance(result, VerificationDecision):
            data.update(result.to_dict())
            return data
        if isinstance(result, AgentResult):
            data.update(status=result.status.value, summary=result.summary[:500],
                        errors=[str(e)[:300] for e in result.errors][:10])
            for attr in ("files_changed", "files_inspected", "commands_run", "integrity_violations",
                         "sources_consulted", "relevant_files", "action", "query", "playback"):
                value = getattr(result, attr, None)
                if value is not None and value != [] and value != {}:
                    data[attr] = value
            verification = getattr(result, "verification_status", None)
            if verification is not None:
                data["verification_status"] = verification.value
            usage = result.metadata.get("usage")
            if isinstance(usage, dict):
                data["tool_calls"] = usage.get("tool_calls", 0)
            if result.metadata.get("error_code"):
                data["error_code"] = result.metadata["error_code"]
        return data

    def _final_decision(self) -> VerificationDecision:
        state = self.state
        assert state is not None
        self.events.emit(EventType.VERIFICATION_STARTED, "evaluating completion evidence")
        decision = self.verifier.evaluate(state, self.graph)
        for check in decision.checks:
            log.info("verification check %s: %s - %s", check.name,
                     "PASS" if check.passed else "FAIL", check.detail)
        log.info("verification decision: %s (%s)", decision.status.value, decision.summary)
        self.events.emit(
            EventType.VERIFICATION_PASSED if decision.verified else EventType.VERIFICATION_FAILED,
            decision.summary + ("" if decision.verified else " " + "; ".join(decision.unresolved_issues))[:250],
        )
        return decision

    def _finish(self, decision: VerificationDecision, duration: float) -> FinalResult:
        state = self.state
        assert state is not None
        state.status = _TASK_STATUS[decision.status]
        state.retry_history = self.recovery.history
        if len(self.graph):
            self._sync_plan()
        tests = state.test_results
        latest = state.latest_test_result or {}
        music_node = next(iter(self.graph.by_kind(NodeKind.MUSIC)), None)
        # every tool execution reported by the agents' registries (all agents, incl. retries)
        tool_calls = sum(
            1 for e in self.events.events if e.event in (EventType.TOOL_CALLED, EventType.TOOL_FAILED)
        )
        files = changed_files(state)
        retries = [r for r in state.retry_history if r.get("action") in ("RETRY", "REPAIR")]
        summary = self._summary(decision, files, latest)
        final = FinalResult(
            task_id=state.task_id,
            status=decision.status,
            summary=summary,
            files_changed=files,
            tests_run=[{"attempt": t.get("attempt"), "command": t.get("command"), "status": str(t.get("status")),
                        "passed": t.get("passed"), "failed": t.get("failed")} for t in tests],
            tests_passed=int(latest.get("passed") or 0),
            tests_failed=int(latest.get("failed") or 0),
            research_performed=any(n.kind is NodeKind.RESEARCH and n.status is not NodeStatus.PENDING
                                   and n.status is not NodeStatus.SKIPPED for n in self.graph),
            tool_calls=tool_calls,
            retries=len(retries),
            retry_history=state.retry_history,
            duration_seconds=round(duration, 3),
            unresolved_issues=decision.unresolved_issues,
            verification=decision.to_dict(),
            plan=[{"id": n.id, "agent": n.agent, "kind": n.kind.value, "status": n.status.value,
                   "dependencies": n.dependencies, "required": n.required, "retry_count": n.retry_count}
                  for n in (self.graph[i] for i in self.graph.topological_order())] if len(self.graph) else [],
            error_code=next((str((n.result or {}).get("error_code")) for n in self.graph
                             if n.status is NodeStatus.BLOCKED and (n.result or {}).get("error_code")),
                            None) if decision.status is not FinalStatus.VERIFIED_SUCCESS else None,
            music=(music_node.result | {"node_status": music_node.status.value}) if music_node and music_node.result
            else ({"node_status": music_node.status.value} if music_node else None),
            state=state,
        )
        event = {FinalStatus.VERIFIED_SUCCESS: EventType.TASK_COMPLETED,
                 FinalStatus.BLOCKED: EventType.TASK_BLOCKED,
                 FinalStatus.FAILED: EventType.TASK_FAILED}[decision.status]
        self.events.emit(event, summary[:250])
        final.events = len(self.events.events)
        state.events = self.events.to_list()
        state.outcome = final.to_dict()
        state.final_result = f"{decision.status.value}: {summary}"
        return final

    @staticmethod
    def _summary(decision: VerificationDecision, files: list[str], latest: dict[str, Any]) -> str:
        if decision.verified:
            command = str(latest.get("command", "tests")).split(" -rfE")[0]
            return (f"Verified: changed {', '.join(files) or 'nothing'}; "
                    f"{command} passed ({latest.get('passed', 0)} passed).")
        return f"{decision.status.value}: " + ("; ".join(decision.unresolved_issues[:3]) or decision.summary)


_TARGET_KEYS = ("path", "query", "command", "target", "url", "name", "level")


def _tool_target(arguments: Any) -> str:
    """A short, safe description of what a tool acted on (never file contents)."""
    if not isinstance(arguments, dict):
        return ""
    for key in _TARGET_KEYS:
        value = arguments.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            text = str(value).strip().splitlines()[0]
            return text if len(text) <= 80 else text[:77] + "..."
    return ""


def _tool_summary(tool: str, data: dict[str, Any]) -> str:
    if tool == "run_tests":
        return f"{data.get('status')} ({data.get('passed') or 0} passed, {data.get('failed') or 0} failed)"
    if tool in ("edit_file", "write_file"):
        return "changed" if data.get("changed") else "unchanged"
    if tool == "play_music":
        return str(data.get("title") or "")
    return ""


def _usage_of(result: Any) -> dict[str, Any]:
    usage = getattr(result, "metadata", {}).get("usage") if result is not None else None
    if not isinstance(usage, dict):
        return {}
    keys = ("llm_turns", "tool_calls", "estimated_input_tokens", "estimated_output_tokens",
            "reported_input_tokens", "reported_output_tokens")
    return {k: usage[k] for k in keys if k in usage}


def _event_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    keys = ("status", "verification_status", "files_changed", "commands_run", "action", "query",
            "errors")
    return {k: result[k] for k in keys if k in result}


class _AgentResultView:
    """Adapter so RepairLoop's evidence helpers can read a compacted coder node result."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.status = AgentStatus(data.get("status", "FAILURE"))
        self.summary = data.get("summary", "")
        self.errors = data.get("errors", [])
        self.files_changed = data.get("files_changed", [])
        self.metadata: dict[str, Any] = {}
