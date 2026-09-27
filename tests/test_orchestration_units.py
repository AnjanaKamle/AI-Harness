"""Phase 5 building blocks: graph, planner, permissions, scheduler, recovery, events,
verification gate."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from harness.agents import CoderAgent, MusicAgent, ResearcherAgent, TesterAgent
from harness.agents.music import parse_music_command
from harness.config.settings import ConfigurationError, Settings
from harness.context import InMemoryContextManager
from harness.orchestrator import AgentState
from harness.orchestrator.events import EventLog, EventType, redact
from harness.orchestrator.graph import GraphError, NodeKind, NodeStatus, TaskGraph, TaskNode
from harness.orchestrator.permissions import (
    AGENT_CAPABILITIES,
    PermissionViolation,
    check_assignment,
    check_tools,
    repository_access,
)
from harness.orchestrator.planner import Planner, split_music_request, third_party_libraries
from harness.orchestrator.recovery import FailureKind, RecoveryAction, RecoveryManager, RetryPolicy
from harness.orchestrator.scheduler import NodeOutcome, Scheduler
from harness.orchestrator.verification_manager import FinalStatus, VerificationManager
from harness.tools import RepositoryContext, ToolExecutionResult, ToolStatus
from harness.tools.base import BaseTool, ToolError
from harness.tools.registry import CancellationToken, ToolRegistry

from .conftest import FAKE_KEY, MockLLMClient, git


def node(nid: str, *deps: str, agent: str = "coder", kind: NodeKind = NodeKind.IMPLEMENT,
         required: bool = True, priority: int = 0) -> TaskNode:
    return TaskNode(nid, nid, agent, kind, dependencies=list(deps), required=required, priority=priority)


# --- graph -----------------------------------------------------------------------------------


def test_graph_validation() -> None:
    with pytest.raises(GraphError, match="unknown node"):
        TaskGraph([node("a", "missing")]).validate()
    with pytest.raises(GraphError, match="cycle"):
        TaskGraph([node("a", "b"), node("b", "a")]).validate()
    with pytest.raises(GraphError, match="duplicate"):
        TaskGraph([node("a"), node("a")])
    graph = TaskGraph([node("c", "a", "b"), node("a", priority=5), node("b")])
    assert graph.topological_order() == ["a", "b", "c"]
    assert graph.independent("a", "b") and not graph.independent("a", "c")


def test_graph_refresh_semantics() -> None:
    graph = TaskGraph([node("a"), node("b", "a"), node("m", required=False, kind=NodeKind.MUSIC),
                       node("c", "b", "m")])
    graph.refresh()
    assert [n.id for n in graph.ready()] == ["a", "m"]
    graph["a"].mark(NodeStatus.SUCCESS)
    graph["m"].mark(NodeStatus.FAILED)  # optional dependency failed -> does not hold c back
    graph.refresh()
    assert graph["b"].status is NodeStatus.READY
    graph["b"].mark(NodeStatus.BLOCKED)
    graph.refresh()
    assert graph["c"].status is NodeStatus.BLOCKED and "b" in (graph["c"].error or "")
    assert graph.finished


def test_optional_node_with_broken_dependency_is_skipped() -> None:
    graph = TaskGraph([node("a"), node("opt", "a", required=False)])
    graph["a"].mark(NodeStatus.BLOCKED)
    graph.refresh()
    assert graph["opt"].status is NodeStatus.SKIPPED


# --- planner ---------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> RepositoryContext:
    root = tmp_path / "r"
    (root / "tests").mkdir(parents=True)
    (root / "calc.py").write_text("def f():\n    return 1\n")
    (root / "tests" / "test_calc.py").write_text("from calc import f\n\ndef test_f():\n    assert f() == 1\n")
    return RepositoryContext(root)


@pytest.mark.parametrize(
    ("task", "music", "coding"),
    [
        ("Play Beethoven while you inspect and fix this bug.", "play Beethoven", "Inspect and fix this bug."),
        ("Please fix the parser and play some jazz.", "play some jazz", "Please fix the parser."),
        ("Fix the play button handler", None, "Fix the play button handler"),
        ("Stop the music and fix the tests", "stop music", "Fix the tests"),
    ],
)
def test_split_music_request(task: str, music: str | None, coding: str) -> None:
    assert split_music_request(task) == (music, coding)


def test_planner_is_dynamic(repo: RepositoryContext) -> None:
    planner = Planner()
    plain = planner.plan("Fix the bug in calc.py", repo)
    assert [n.id for n in plain.graph] == ["inspect-1", "baseline-1", "implement-1", "test-1", "verify-1"]
    assert plain.graph["implement-1"].dependencies == ["inspect-1", "baseline-1"]
    assert not plain.graph["baseline-1"].required  # optional: a missing test runner never blocks
    assert plain.graph.independent("inspect-1", "baseline-1")

    research = planner.plan("Use library fancywidgetz9 to draw charts", repo)
    assert research.graph["research-1"].dependencies == []  # runs alongside inspection
    assert research.graph.independent("research-1", "inspect-1")
    assert research.graph["research-2"].dependencies == ["research-1", "inspect-1"]
    assert research.graph["implement-1"].dependencies == ["inspect-1", "baseline-1", "research-1", "research-2"]
    assert research.graph["research-1"].required  # no local evidence -> required

    music = planner.plan("Play Beethoven while you inspect and fix this bug.", repo)
    assert not music.graph["music-1"].required and music.graph["music-1"].dependencies == []
    assert music.coding_task == "Inspect and fix this bug."
    assert all("music" not in d for n in music.graph for d in n.dependencies)

    music_only = planner.plan("Play Beethoven", repo)
    assert [n.id for n in music_only.graph] == ["music-1", "verify-1"]
    assert music_only.graph["music-1"].required


def test_conditional_research_uses_real_project_dependencies(repo: RepositoryContext) -> None:
    task = "Fix authentication and use the correct library API if necessary."
    assert not Planner().plan(task, repo).research.needed  # no external library in the project
    (repo.root / "auth.py").write_text("import jwt\n\ndef check(t):\n    return jwt.decode(t)\n")
    assert third_party_libraries(repo) == ["jwt"]
    planned = Planner().plan(task, repo)
    assert planned.research.needed and planned.research.libraries == ["jwt"]
    assert not planned.graph["research-1"].required  # conditional research never blocks


# --- permissions -----------------------------------------------------------------------------


def test_capability_matrix_matches_spec() -> None:
    c = {name: caps.to_dict() for name, caps in AGENT_CAPABILITIES.items()}
    assert c["coder"] == {"repository_read": True, "repository_write": True, "terminal": True,
                          "web": False, "music": False}
    assert c["researcher"] == {"repository_read": True, "repository_write": False, "terminal": False,
                               "web": True, "music": False}
    assert c["tester"] == {"repository_read": True, "repository_write": False, "terminal": True,
                           "web": False, "music": False}
    assert c["music"] == {"repository_read": False, "repository_write": False, "terminal": False,
                          "web": False, "music": True}


def test_real_agents_satisfy_their_capabilities(repo: RepositoryContext) -> None:
    settings = Settings(api_key=FAKE_KEY)
    llm, ctx = MockLLMClient(), InMemoryContextManager()
    for agent in (CoderAgent(llm, ctx, repo, settings=settings),
                  ResearcherAgent(llm, ctx, repo, settings=settings),
                  TesterAgent(llm, ctx, repo, settings=settings),
                  MusicAgent(None, ctx)):
        check_tools(agent.name, agent.tools)  # must not raise


def test_permission_violations() -> None:
    with pytest.raises(PermissionViolation, match="write_file"):
        check_tools("tester", ["read_file", "write_file"])
    with pytest.raises(PermissionViolation, match="read_file"):
        check_tools("music", ["read_file"])
    with pytest.raises(PermissionViolation, match="unknown tool"):
        check_tools("coder", ["format_disk"])
    with pytest.raises(PermissionViolation):
        check_assignment("researcher", NodeKind.IMPLEMENT)
    with pytest.raises(PermissionViolation):
        check_assignment("tester", NodeKind.REPAIR)
    check_assignment("coder", NodeKind.INSPECT)
    assert repository_access("coder", NodeKind.IMPLEMENT) == "write"
    assert repository_access("coder", NodeKind.INSPECT) == "read"
    assert repository_access("tester", NodeKind.TEST) == "read"
    assert repository_access("music", NodeKind.MUSIC) == "none"


# --- scheduler -------------------------------------------------------------------------------


def run_scheduler(graph: TaskGraph, work: dict[str, float], *, max_concurrent: int = 3,
                  timeout: float = 30.0, fail: set[str] = frozenset()) -> tuple[Scheduler, list[tuple[str, str, float]]]:  # type: ignore[assignment]
    log: list[tuple[str, str, float]] = []
    lock = threading.Lock()

    def execute(n: TaskNode, token: CancellationToken, prepared: Any) -> NodeOutcome:
        with lock:
            log.append(("start", n.id, time.monotonic()))
        end = time.monotonic() + work.get(n.id, 0.05)
        while time.monotonic() < end and not token.cancelled:
            time.sleep(0.01)
        with lock:
            log.append(("end", n.id, time.monotonic()))
        return NodeOutcome(result=n.id)

    def complete(n: TaskNode, outcome: NodeOutcome) -> None:
        n.mark(NodeStatus.FAILED if n.id in fail or outcome.timed_out else NodeStatus.SUCCESS)

    scheduler = Scheduler(graph, prepare=lambda n: None, execute=execute, on_complete=complete,
                          access=lambda n: repository_access(n.agent, n.kind),
                          max_concurrent=max_concurrent, node_timeout=timeout, grace_seconds=2)
    scheduler.run_sync()
    return scheduler, log


def overlapping(log: list[tuple[str, str, float]], a: str, b: str) -> bool:
    t = {(kind, nid): ts for kind, nid, ts in log}
    return t[("start", a)] < t[("end", b)] and t[("start", b)] < t[("end", a)]


def test_scheduler_runs_independent_nodes_concurrently() -> None:
    graph = TaskGraph([
        node("inspect", agent="coder", kind=NodeKind.INSPECT),
        node("research", agent="researcher", kind=NodeKind.RESEARCH),
        node("music", agent="music", kind=NodeKind.MUSIC, required=False),
        node("implement", "inspect", "research"),
    ])
    scheduler, log = run_scheduler(graph, {"inspect": 0.3, "research": 0.3, "music": 0.3})
    assert overlapping(log, "inspect", "research") and overlapping(log, "inspect", "music")
    assert scheduler.max_observed_concurrency == 3
    starts = {nid: ts for kind, nid, ts in log if kind == "start"}
    ends = {nid: ts for kind, nid, ts in log if kind == "end"}
    assert starts["implement"] >= max(ends["inspect"], ends["research"])  # dependencies respected


def test_scheduler_respects_max_concurrency() -> None:
    graph = TaskGraph([node(f"r{i}", agent=f"reader{i}", kind=NodeKind.INSPECT) for i in range(5)])
    scheduler, _ = run_scheduler(graph, {f"r{i}": 0.15 for i in range(5)}, max_concurrent=2)
    assert scheduler.max_observed_concurrency == 2


def test_scheduler_never_overlaps_repository_writer() -> None:
    graph = TaskGraph([
        node("write-a", agent="coder"),
        TaskNode("write-b", "b", "coder2", NodeKind.IMPLEMENT),
        node("read", agent="tester", kind=NodeKind.TEST),
        node("music", agent="music", kind=NodeKind.MUSIC, required=False),
    ])
    from harness.orchestrator import permissions

    permissions.AGENT_CAPABILITIES["coder2"] = permissions.CODER
    try:
        _, log = run_scheduler(graph, {"write-a": 0.2, "write-b": 0.2, "read": 0.2, "music": 0.5})
    finally:
        del permissions.AGENT_CAPABILITIES["coder2"]
    assert not overlapping(log, "write-a", "write-b")
    assert not overlapping(log, "write-a", "read") and not overlapping(log, "write-b", "read")
    assert overlapping(log, "music", "write-a") or overlapping(log, "music", "write-b")


def test_scheduler_timeout_cancels_and_reports() -> None:
    graph = TaskGraph([node("slow", agent="tester", kind=NodeKind.TEST)])
    scheduler, log = run_scheduler(graph, {"slow": 10}, timeout=0.2)
    assert graph["slow"].status is NodeStatus.FAILED
    duration = log[1][2] - log[0][2]
    assert duration < 2  # the operation was terminated, not left running
    assert scheduler.events.of_type(EventType.AGENT_TIMEOUT)


def test_scheduler_blocks_dependents_of_failures() -> None:
    graph = TaskGraph([node("a"), node("b", "a"), node("c", "b")])
    run_scheduler(graph, {}, fail={"a"})
    assert [graph[n].status for n in "abc"] == [NodeStatus.FAILED, NodeStatus.BLOCKED, NodeStatus.BLOCKED]


# --- recovery --------------------------------------------------------------------------------


def test_recovery_decisions_and_history() -> None:
    rm = RecoveryManager(RetryPolicy(max_tool_retries=1, max_agent_retries=1, max_repair_attempts=2))
    n = node("x")
    assert rm.decide(n, FailureKind.MALFORMED_OUTPUT, detail="bad json").action is RecoveryAction.RETRY
    n.retry_count = 1
    assert rm.decide(n, FailureKind.MALFORMED_OUTPUT, detail="bad json").action is RecoveryAction.BLOCK
    t = node("t", kind=NodeKind.TEST, agent="tester")
    assert rm.decide(t, FailureKind.TEST_FAILURE, detail="1 failed", repairable=True,
                     repairs_done=1).action is RecoveryAction.REPAIR
    assert rm.decide(t, FailureKind.TEST_FAILURE, detail="1 failed", repairable=True,
                     repairs_done=2).action is RecoveryAction.BLOCK
    assert rm.decide(t, FailureKind.TEST_FAILURE, detail="missing dep",
                     repairable=False).action is RecoveryAction.BLOCK
    r = node("r", kind=NodeKind.RESEARCH, agent="researcher")
    assert rm.decide(r, FailureKind.RESEARCH_FAILURE, detail="x", retryable=False,
                     can_proceed=True).action is RecoveryAction.PROCEED
    assert rm.decide(r, FailureKind.RESEARCH_FAILURE, detail="x", retryable=False,
                     can_proceed=False).action is RecoveryAction.BLOCK
    music = node("m", kind=NodeKind.MUSIC, agent="music", required=False)
    assert rm.decide(music, FailureKind.OPTIONAL_TASK_FAILURE, detail="no speaker",
                     retryable=False).action is RecoveryAction.PROCEED
    assert rm.decide(n, FailureKind.LLM_ERROR, detail="auth", retryable=False).action is RecoveryAction.BLOCK
    history = rm.history
    assert len(history) == 9 and all(h["reason"] and h["timestamp"] for h in history)


def test_retry_limits_are_configurable() -> None:
    s = Settings.from_env({"AI_API_KEY": FAKE_KEY, "AI_MAX_TOOL_RETRIES": "0",
                           "AI_MAX_AGENT_RETRIES": "4", "AI_MAX_REPAIR_ATTEMPTS": "1",
                           "AI_MAX_CONCURRENT_AGENTS": "5"})
    assert RetryPolicy.from_settings(s) == RetryPolicy(0, 4, 1)
    assert s.max_concurrent_agents == 5 and Settings(api_key=FAKE_KEY).max_concurrent_agents == 3
    with pytest.raises(ConfigurationError):
        Settings(api_key=FAKE_KEY, max_concurrent_agents=0)


class FlakyTool(BaseTool):
    name = "read_file"
    description = "flaky"

    def __init__(self, failures: int, retryable: bool = True) -> None:
        self.failures, self.retryable, self.calls = failures, retryable, 0

    def execute(self, **_: Any) -> ToolExecutionResult:
        self.calls += 1
        if self.calls <= self.failures:
            raise ToolError("connection reset", retryable=self.retryable)
        return ToolExecutionResult.success(self.name, {"ok": True})


def test_tool_retries_only_transient_failures() -> None:
    seen: list[int] = []
    registry = ToolRegistry([FlakyTool(2)])
    registry.max_tool_retries = 2
    registry.observer = lambda name, args, result, attempt: seen.append(attempt)
    result = registry.execute("read_file", {})
    assert result.ok and seen == [1, 2, 3]
    assert [r["attempt"] for r in result.metadata["retries"]] == [1, 2]

    deterministic = ToolRegistry([FlakyTool(5, retryable=False)])
    deterministic.max_tool_retries = 3
    assert not deterministic.execute("read_file", {}).ok
    assert deterministic.get("read_file").calls == 1  # type: ignore[attr-defined]

    exhausted = ToolRegistry([FlakyTool(9)])
    exhausted.max_tool_retries = 1
    assert not exhausted.execute("read_file", {}).ok
    assert exhausted.get("read_file").calls == 2  # type: ignore[attr-defined]


def test_cancelled_registry_refuses_tools() -> None:
    registry = ToolRegistry([FlakyTool(0)])
    registry.cancel_token = CancellationToken()
    registry.cancel_token.cancel("time limit exceeded")
    result = registry.execute("read_file", {})
    assert result.status is ToolStatus.FAILURE and "cancelled" in (result.error or "")
    assert registry.get("read_file").calls == 0  # type: ignore[attr-defined]


# --- events ----------------------------------------------------------------------------------


def test_event_log_redacts_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "real-looking-key-12345")
    log = EventLog("task-1")
    event = log.emit(EventType.TOOL_FAILED, "auth failed with real-looking-key-12345 and sk-FAKE0000FAKE0000FAKE",
                     agent="coder", node="implement-1", extra="token=supersecret")
    assert "real-looking-key-12345" not in event.details and "sk-FAKE" not in event.details
    assert event.data["extra"] == "[REDACTED]"
    assert event.task_id == "task-1" and event.timestamp and event.agent == "coder"
    assert redact("password: hunter2") == "[REDACTED]"


# --- music -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "parsed"),
    [("play Beethoven", ("play", "Beethoven", {})),
     ("play some jazz at 30% volume", ("play", "jazz", {"volume": 30})),
     ("stop music", ("stop", None, {})),
     ("dance", ("unknown", None, {}))],
)
def test_parse_music_command(command: str, parsed: tuple[Any, ...]) -> None:
    assert parse_music_command(command) == parsed


def test_music_agent_sees_only_the_command() -> None:
    agent = MusicAgent(None, InMemoryContextManager())
    state = AgentState(task="Fix the SECRET_PROJECT_BUG in payments.py")
    result = agent.run("play Beethoven", state)
    assert result.status.value == "FAILURE"  # no backend configured
    assert result.context_chars < 200  # type: ignore[attr-defined]


# --- verification gate -----------------------------------------------------------------------


def test_verification_gate_ignores_claims(tmp_path: Path) -> None:
    root = tmp_path / "v"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "init")
    repo = RepositoryContext(root)
    graph = TaskGraph([node("implement-1"), node("test-1", "implement-1", agent="tester", kind=NodeKind.TEST)])
    for n in graph:
        n.mark(NodeStatus.SUCCESS)
    state = AgentState(task="t")
    state.code_changes.append({"files_changed": ["a.py"]})  # claimed, but git shows nothing
    decision = VerificationManager(repo).evaluate(state, graph)
    assert decision.status is not FinalStatus.VERIFIED_SUCCESS
    failed = {c.name for c in decision.checks if not c.passed}
    assert failed == {"code_changes_applied", "tests_passed"}
