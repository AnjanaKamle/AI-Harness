"""Phase 5 integration: the autonomous controller (plan -> schedule -> recover -> verify).

Real repositories, tools, git and pytest; only the model's replies are scripted.
Scenarios A-J from the spec plus the two acceptance tasks.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from harness.agents import (
    AgentStatus,
    CoderAgent,
    MusicAgent,
    ResearcherAgent,
    TesterAgent,
)
from harness.agents.base import AgentResult
from harness.config.settings import Settings
from harness.context import InMemoryContextManager
from harness.llm import LLMError, LLMResponse, Message, StopReason, ToolCall
from harness.orchestrator import Orchestrator, TaskStatus
from harness.orchestrator.events import EventType
from harness.orchestrator.final_result import FinalResult
from harness.orchestrator.graph import NodeStatus
from harness.orchestrator.verification_manager import FinalStatus
from harness.tools import ListFilesTool, ReadFileTool, RepositoryContext, SearchCodeTool
from harness.tools.base import ToolError
from harness.tools.music import MusicPlayer, PlaybackState
from harness.tools.research import LookupPythonApiTool

from .conftest import FAKE_KEY, MockLLMClient, git

BUGGY = "def multiply(a, b):\n    return a + b\n"
TESTS = (
    "from calc import multiply\n\n\n"
    "def test_multiply():\n    assert multiply(2, 3) == 6\n\n\n"
    "def test_zero():\n    assert multiply(0, 5) == 0\n"
)


def make_repo(tmp_path: Path, files: dict[str, str], name: str = "proj") -> RepositoryContext:
    root = tmp_path / name
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return RepositoryContext(root)


@pytest.fixture
def calc_repo(tmp_path: Path) -> RepositoryContext:
    return make_repo(tmp_path, {"calc.py": BUGGY, "tests/test_calc.py": TESTS,
                                "legacy/unrelated.py": "def old_billing():\n    pass\n"})


# --- scripted model helpers ----------------------------------------------------------------


def tools(*calls: ToolCall) -> LLMResponse:
    return LLMResponse("", StopReason.TOOL_USE, tool_calls=calls)


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(f"id-{name}", name, args)


def done(status: str = "SUCCESS", files: tuple[str, ...] = ("calc.py",), summary: str = "Everything works.") -> LLMResponse:
    return LLMResponse(json.dumps({"status": status, "summary": summary, "files_changed": list(files),
                                   "next_action": "", "errors": []}), StopReason.END_TURN)


def edit(old: str, new: str, path: str = "calc.py", check: Any = None) -> list[Any]:
    def start(messages: list[Message]) -> LLMResponse:
        if check:
            check(messages[1].content)
        return tools(call("read_file", path=path))

    return [start, tools(call("edit_file", path=path, old_text=old, new_text=new)),
            tools(call("git_diff")), done(files=(path,))]


class FakePlayer(MusicPlayer):
    name = "fake"

    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        self.fail, self.delay = fail, delay
        self.played: list[str] = []
        self.active = threading.Event()

    def play(self, query: str, parameters: dict[str, Any]) -> PlaybackState:
        self.active.set()
        time.sleep(self.delay)
        self.active.clear()
        if self.fail:
            raise ToolError("speaker unplugged")
        self.played.append(query)
        return PlaybackState("play", query, True, self.name)

    def stop(self) -> PlaybackState:
        return PlaybackState("stop", None, False, self.name)


def build(
    repo: RepositoryContext,
    script: list[Any],
    *,
    player: MusicPlayer | None = None,
    researcher_tools: bool = False,
    **settings_overrides: Any,
) -> tuple[Orchestrator, dict[str, Any], MockLLMClient]:
    options: dict[str, Any] = {"test_timeout_seconds": 60, "research_backend": "none",
                               **settings_overrides}
    settings = Settings(api_key=FAKE_KEY, **options)
    llm = MockLLMClient(script)
    ctx = InMemoryContextManager()
    agents: dict[str, Any] = {
        "coder": CoderAgent(llm, ctx, repo, settings=settings),
        "tester": TesterAgent(llm, ctx, repo, settings=settings),
        "researcher": ResearcherAgent(
            llm, ctx, repo, settings=settings,
            tools=[ListFilesTool(repo), ReadFileTool(repo), SearchCodeTool(repo), LookupPythonApiTool(repo)]
            if researcher_tools else None,
        ),
        "music": MusicAgent(None, ctx, player=player or FakePlayer()),
    }
    return Orchestrator(settings, llm=llm, context=ctx), agents, llm


def events_of(final: FinalResult, kind: EventType) -> list[dict[str, Any]]:
    assert final.state is not None
    return [e for e in final.state.events if e["event"] == kind.value]


def node_status(final: FinalResult) -> dict[str, str]:
    return {n["id"]: n["status"] for n in final.plan}


# --- A. sequential: Research -> Code -> Test ------------------------------------------------


def test_a_sequential_research_code_test(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {
        "stats.py": "def average(values):\n    raise NotImplementedError\n",
        "tests/test_stats.py": "from stats import average\n\n\ndef test_avg():\n    assert average([1, 2, 6]) == 3\n",
    })

    def api_answer(messages: list[Message]) -> LLMResponse:
        data = json.loads(messages[-1].tool_results[0].content)["data"]
        return LLMResponse(json.dumps({"status": "SUCCESS", "summary": "use statistics.mean", "findings": [
            {"question": "api", "kind": "FACT", "finding": "statistics.mean(data) returns the mean",
             "source": data["source"], "relevance": "needed", "confidence": "HIGH",
             "technical_details": "statistics.mean(data)", "recommended_action": "return statistics.mean(values)"}],
            "open_questions": []}), StopReason.END_TURN)

    def project_answer(messages: list[Message]) -> LLMResponse:
        return LLMResponse(json.dumps({"status": "SUCCESS", "summary": "stub in stats.py", "findings": [
            {"question": "project", "kind": "FACT", "finding": "stats.py has an average() stub",
             "source": "stats.py", "relevance": "target", "confidence": "HIGH"}], "open_questions": []}),
            StopReason.END_TURN)

    def coder_start(messages: list[Message]) -> LLMResponse:
        assert "[FACT/HIGH] statistics.mean(data) returns the mean" in messages[1].content
        assert "REPOSITORY INSPECTION" in messages[1].content
        return tools(call("read_file", path="stats.py"))

    script = [
        tools(call("lookup_python_api", target="statistics.mean")), api_answer,
        tools(call("read_file", path="stats.py")), project_answer,
        coder_start,
        tools(call("write_file", path="stats.py",
                   content="import statistics\n\n\ndef average(values):\n    return statistics.mean(values)\n")),
        tools(call("git_diff")), done(files=("stats.py",)),
    ]
    orch, agents, llm = build(repo, script, researcher_tools=True)
    final = orch.execute("Use the `statistics` library to implement average() in stats.py.", repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    assert final.research_performed and llm.responses == []
    statuses = node_status(final)
    assert statuses == {"inspect-1": "SUCCESS", "baseline-1": "SUCCESS", "research-1": "SUCCESS",
                        "research-2": "SUCCESS", "implement-1": "SUCCESS", "test-1": "SUCCESS",
                        "verify-1": "SUCCESS"}
    # dependencies respected: research finished before implementation started, tests after that
    order = [(e["event"], e["node"]) for e in final.state.events if e["node"]]  # type: ignore[union-attr]
    idx = {pair: i for i, pair in enumerate(order)}
    assert idx[("AGENT_COMPLETED", "research-2")] < idx[("AGENT_STARTED", "implement-1")]
    assert idx[("AGENT_COMPLETED", "implement-1")] < idx[("AGENT_STARTED", "test-1")]
    assert idx[("AGENT_COMPLETED", "test-1")] < idx[("AGENT_STARTED", "verify-1")]


# --- B. parallel: coder inspection + independent music --------------------------------------


def test_b_parallel_inspection_and_music(calc_repo: RepositoryContext) -> None:
    player = FakePlayer(delay=0.6)

    def coder_start(messages: list[Message]) -> LLMResponse:
        prompt = messages[1].content
        assert "Beethoven" not in prompt and "play" not in prompt.lower().split("## task")[1][:80]
        return tools(call("read_file", path="calc.py"))

    script = [coder_start, *edit("return a + b", "return a * b")[1:]]
    orch, agents, llm = build(calc_repo, script, player=player)
    final = orch.execute("Play Beethoven while you inspect and fix this bug.", calc_repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    assert player.played == ["Beethoven"]
    controller = orch.last_controller
    assert ("inspect-1", "music-1") in controller.scheduler.concurrent_pairs  # ran simultaneously
    assert controller.scheduler.max_observed_concurrency >= 2
    assert final.music is not None and final.music["node_status"] == "SUCCESS"
    # the coder never ran alongside a repository reader/writer
    for pair in controller.scheduler.concurrent_pairs:
        if "implement-1" in pair:
            assert set(pair) - {"implement-1"} <= {"music-1"}


def test_music_failure_does_not_invalidate_coding(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a * b"), player=FakePlayer(fail=True))
    final = orch.execute("Play Beethoven while you inspect and fix this bug.", calc_repo, agents=agents)
    assert final.status is FinalStatus.VERIFIED_SUCCESS
    assert node_status(final)["music-1"] == "FAILED"
    assert any("music-1" in issue and "speaker unplugged" in issue for issue in final.unresolved_issues)


def test_default_music_backend_unavailable_is_harmless(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a * b"))
    agents["music"] = MusicAgent(None, InMemoryContextManager())  # AI_MUSIC_BACKEND=none
    final = orch.execute("Play Beethoven while you inspect and fix this bug.", calc_repo, agents=agents)
    assert final.status is FinalStatus.VERIFIED_SUCCESS
    assert "No music backend is configured" in final.music["errors"][0]  # type: ignore[index]


# --- C. failed test -> repair -> pass ------------------------------------------------------


def test_c_failed_test_repair_pass(calc_repo: RepositoryContext) -> None:
    def has_evidence(prompt: str) -> None:
        assert "FAILURE CLASSIFICATION: TEST_FAILURE" in prompt and "assert 8 == 6" in prompt

    script = edit("return a + b", "return a ** b") + edit("return a ** b", "return a * b", check=has_evidence)
    orch, agents, llm = build(calc_repo, script)
    final = orch.execute("Fix the deliberately broken multiply function.", calc_repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    assert llm.responses == []
    assert node_status(final) == {"inspect-1": "SUCCESS", "baseline-1": "SUCCESS", "implement-1": "SUCCESS",
                                  "test-1": "FAILED", "repair-1": "SUCCESS", "test-2": "SUCCESS",
                                  "verify-1": "SUCCESS"}
    plan = {n["id"]: n for n in final.plan}
    assert plan["verify-1"]["dependencies"] == ["test-2"]  # rewired to the re-test
    # the only test file is the related one, so each attempt runs the suite exactly once
    assert [(t["attempt"], t["status"]) for t in final.tests_run] == [(1, "FAIL"), (2, "PASS")]
    assert events_of(final, EventType.TEST_FAILED) and events_of(final, EventType.REPAIR_STARTED)
    assert final.state is not None and len(final.state.failure_history) == 1
    assert final.retries == 1 and final.retry_history[0]["action"] == "REPAIR"
    assert final.tests_passed == 2 and final.tests_failed == 0


# --- D. maximum retry reached ---------------------------------------------------------------


def test_d_maximum_repairs_reached_blocks(calc_repo: RepositoryContext) -> None:
    (calc_repo.root / "tests/test_calc.py").write_text(TESTS + "\n\ndef test_impossible():\n    assert multiply(2, 3) == 7\n")
    git(calc_repo.root, "commit", "-qam", "impossible")
    script = edit("return a + b", "return a * b") + edit("return a * b", "return b * a")
    orch, agents, llm = build(calc_repo, script, max_repair_attempts=1)
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)

    assert final.status is FinalStatus.BLOCKED
    assert llm.responses == []  # exactly 1 implementation + 1 repair
    statuses = node_status(final)
    assert statuses["test-2"] == "BLOCKED" and statuses["verify-1"] == "BLOCKED"
    assert "repair-2" not in statuses
    assert any("maximum repair attempts (1)" in r["reason"] for r in final.retry_history)
    assert any("tests_passed" in issue for issue in final.unresolved_issues)


# --- E. researcher failure ------------------------------------------------------------------


def test_e_required_research_failure_blocks_before_coding(calc_repo: RepositoryContext) -> None:
    blocked = LLMResponse(json.dumps({"status": "BLOCKED", "summary": "no docs", "findings": [],
                                      "open_questions": []}), StopReason.END_TURN)
    orch, agents, llm = build(calc_repo, [blocked])
    final = orch.execute("Use library fancywidgetz9 to render charts in calc.py.", calc_repo, agents=agents)

    assert final.status is FinalStatus.BLOCKED
    statuses = node_status(final)
    assert statuses["research-1"] == "BLOCKED" and statuses["implement-1"] == "BLOCKED"
    assert statuses["research-2"] == "BLOCKED"
    assert not [e for e in final.state.events if e["event"] == "AGENT_STARTED" and e["node"] == "implement-1"]  # type: ignore[union-attr]
    assert events_of(final, EventType.RESEARCH_FAILED)
    assert final.state.research["runs"][0]["decision"] == "BLOCK"  # type: ignore[index, union-attr]
    assert (calc_repo.root / "calc.py").read_text() == BUGGY


def test_e_transient_research_error_is_retried(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": BUGGY, "tests/test_calc.py": TESTS})

    def flaky(messages: list[Message]) -> LLMResponse:
        raise LLMError("rate limited", retryable=True)

    blocked = LLMResponse(json.dumps({"status": "BLOCKED", "summary": "none", "findings": [],
                                      "open_questions": []}), StopReason.END_TURN)
    # statistics is installed locally -> research optional; after the retry it proceeds
    script = [flaky, blocked, blocked, *edit("return a + b", "return a * b")]
    orch, agents, _ = build(repo, script)
    final = orch.execute("Use the `statistics` library to fix multiply in calc.py.", repo, agents=agents)
    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    assert events_of(final, EventType.AGENT_RETRIED)[0]["node"] == "research-1"
    assert node_status(final)["research-1"] == "FAILED"  # recorded, but optional
    assert final.state.failures and final.state.failures[0]["agent"] == "researcher"  # type: ignore[union-attr]


# --- F. tool timeout / agent timeout --------------------------------------------------------


def test_f_test_command_timeout_is_classified_and_repaired(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {
        "calc.py": "import time\n\n\ndef multiply(a, b):\n    time.sleep(30)\n    return a * b\n",
        "tests/test_calc.py": TESTS,
    })
    # first implementation changes the wrong thing -> the test run times out; repair fixes it
    script = [tools(call("read_file", path="calc.py")),
              tools(call("edit_file", path="calc.py", old_text="return a * b", new_text="return b * a")),
              tools(call("git_diff")), done(),
              tools(call("read_file", path="calc.py")),
              tools(call("write_file", path="calc.py", content="def multiply(a, b):\n    return b * a\n")),
              tools(call("git_diff")), done()]
    orch, agents, _ = build(repo, script, test_timeout_seconds=2)
    final = orch.execute("Fix the slow multiply function.", repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    assert final.tests_run[0]["status"] == "TIMEOUT"
    assert final.state.failure_history[0]["category"] == "TIMEOUT"  # type: ignore[union-attr]


class SlowCoder(CoderAgent):
    """Inspection that hangs until cancelled (cooperatively, like a tool loop)."""

    def inspect(self, task: str, state: Any) -> Any:
        while not (self.registry.cancel_token and self.registry.cancel_token.cancelled):
            time.sleep(0.02)
        raise RuntimeError("OperationCancelled: time limit exceeded")


def test_f_agent_timeout_terminates_retries_then_blocks(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, [], agent_timeout_seconds=0.3, max_agent_retries=1)
    agents["coder"] = SlowCoder(orch.llm, orch.context, calc_repo, settings=orch.settings)  # type: ignore[arg-type]
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)

    assert final.status is FinalStatus.BLOCKED
    assert len(events_of(final, EventType.AGENT_TIMEOUT)) == 2  # first attempt + one retry
    assert events_of(final, EventType.AGENT_RETRIED)[0]["node"] == "inspect-1"
    assert node_status(final)["inspect-1"] == "BLOCKED"
    assert any(r["kind"] == "TIMEOUT" for r in final.retry_history)
    assert (calc_repo.root / "calc.py").read_text() == BUGGY


# --- G. malformed agent result --------------------------------------------------------------


def test_g_malformed_output_is_retried(calc_repo: RepositoryContext) -> None:
    garbage = LLMResponse("I fixed it, trust me", StopReason.END_TURN)
    script = [garbage, garbage, *edit("return a + b", "return a * b")]  # 1st run: reply + repair prompt
    orch, agents, _ = build(calc_repo, script)
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)
    assert final.status is FinalStatus.VERIFIED_SUCCESS
    retried = events_of(final, EventType.AGENT_RETRIED)
    assert retried[0]["node"] == "implement-1" and "MALFORMED_OUTPUT" in retried[0]["details"]


class WeirdTester(TesterAgent):
    def run(self, task: str, state: Any) -> Any:  # type: ignore[override]
        return {"status": "PASS", "note": "trust me"}  # not an AgentResult


def test_g_non_result_object_never_counts_as_success(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a * b"), max_agent_retries=0)
    agents["tester"] = WeirdTester(orch.llm, orch.context, calc_repo, settings=orch.settings)  # type: ignore[arg-type]
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED
    assert node_status(final)["test-1"] == "BLOCKED"


# --- H. blocked task ------------------------------------------------------------------------


def test_h_blocked_coder_blocks_task(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, [done("BLOCKED", files=(), summary="Need the product spec")])
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED
    assert node_status(final)["implement-1"] == "BLOCKED" and node_status(final)["test-1"] == "BLOCKED"
    assert final.state is not None and final.state.status is TaskStatus.BLOCKED


def test_llm_claim_cannot_override_verification(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a - b"), max_repair_attempts=0)
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)  # coder says "Everything works."
    assert final.status is FinalStatus.BLOCKED
    assert any(c["name"] == "tests_passed" and not c["passed"] for c in final.verification["checks"])


# --- I. successful verification + J. final result -------------------------------------------


def test_i_j_verification_and_final_result(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a * b"))
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS
    checks = {c["name"]: c["passed"] for c in final.verification["checks"]}
    assert checks == {"required_tasks_succeeded": True, "code_changes_applied": True,
                      "tests_passed": True, "task_behavior_verified": True,
                      "no_blocking_failures": True, "repository_inspectable": True}
    data = json.loads(final.to_json())
    for key in ("status", "summary", "files_changed", "tests_run", "tests_passed", "tests_failed",
                "research_performed", "tool_calls", "retries", "duration_seconds", "unresolved_issues"):
        assert key in data, key
    assert data["status"] == "VERIFIED_SUCCESS" and data["files_changed"] == ["calc.py"]
    assert data["tests_passed"] == 2 and data["tests_failed"] == 0 and data["research_performed"] is False
    assert data["tool_calls"] >= 5 and data["retries"] == 0 and data["duration_seconds"] > 0
    assert data["unresolved_issues"] == []
    # observability: the whole lifecycle is in the event log, without credentials
    kinds = [e["event"] for e in final.state.events]  # type: ignore[union-attr]
    for kind in ("TASK_CREATED", "PLAN_CREATED", "AGENT_STARTED", "TOOL_CALLED", "AGENT_COMPLETED",
                 "TEST_PASSED", "VERIFICATION_STARTED", "VERIFICATION_PASSED", "TASK_COMPLETED"):
        assert kind in kinds, kind
    assert all({"timestamp", "task_id", "agent", "event", "details"} <= set(e) for e in final.state.events)  # type: ignore[union-attr]
    assert FAKE_KEY not in json.dumps(final.state.events)  # type: ignore[union-attr]
    assert final.state.to_json()  # type: ignore[union-attr]


# --- crash resilience -----------------------------------------------------------------------


class ExplodingTester(TesterAgent):
    def execute(self, task: str, state: Any) -> Any:
        raise RuntimeError("tester exploded")


def test_agent_crash_becomes_structured_failure(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a * b"), max_agent_retries=1)
    agents["tester"] = ExplodingTester(orch.llm, orch.context, calc_repo, settings=orch.settings)  # type: ignore[arg-type]
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED
    assert any("tester exploded" in e["details"] for e in events_of(final, EventType.AGENT_FAILED))
    assert len(events_of(final, EventType.AGENT_RETRIED)) == 1


def test_unexpected_fatal_error_gives_clean_failed_result(calc_repo: RepositoryContext, monkeypatch: pytest.MonkeyPatch) -> None:
    orch, agents, _ = build(calc_repo, [])
    from harness.orchestrator import planner as planner_module

    def broken_plan(self: Any, *a: Any, **k: Any) -> Any:
        raise KeyError("planner bug")

    monkeypatch.setattr(planner_module.Planner, "plan", broken_plan)
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)  # must not raise
    assert final.status is FinalStatus.FAILED
    assert any("KeyError" in issue for issue in final.unresolved_issues)


def test_permission_violation_fails_cleanly(calc_repo: RepositoryContext) -> None:
    from harness.tools import WriteFileTool

    orch, agents, _ = build(calc_repo, [])
    agents["music"] = MusicAgent(None, InMemoryContextManager(), tools=[WriteFileTool(calc_repo)])
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)
    assert final.status is FinalStatus.FAILED
    assert any("permission violation" in i and "write_file" in i for i in final.unresolved_issues)
    assert (calc_repo.root / "calc.py").read_text() == BUGGY


class MeddlingResearcher(ResearcherAgent):
    """Tries to corrupt shared state it does not own."""

    def run(self, task: str, state: Any) -> Any:
        state.status = TaskStatus.VERIFIED_SUCCESS
        state.code_changes.append({"agent": "researcher", "files_changed": ["calc.py"]})
        state.research_findings.append({"question": "q", "kind": "FACT", "finding": "made up",
                                        "source": "https://nowhere.example", "relevance": "",
                                        "confidence": "HIGH", "source_verified": False})
        return AgentResult("researcher", AgentStatus.SUCCESS, "done")


def test_orchestrator_rejects_unauthorized_state_changes(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a * b"))
    agents["researcher"] = MeddlingResearcher(orch.llm, orch.context, calc_repo, settings=orch.settings)  # type: ignore[arg-type]
    final = orch.execute("Use the `statistics` library to fix multiply in calc.py.", calc_repo, agents=agents)
    state = final.state
    assert state is not None
    rejected = " ".join(e["details"] for e in events_of(final, EventType.STATE_REJECTED))
    assert "status (not writable by RESEARCH)" in rejected
    assert "code_changes (not writable by RESEARCH)" in rejected
    assert "FACT without a verified source" in rejected
    assert all(f.get("source") != "https://nowhere.example" for f in state.research_findings)
    assert all(c.get("agent") != "researcher" for c in state.code_changes)


# --- acceptance ------------------------------------------------------------------------------


def test_acceptance_fix_broken_repository(calc_repo: RepositoryContext) -> None:
    task = "Fix the deliberately broken test repository. Research the relevant library if needed."
    script = edit("return a + b", "return a - b") + edit("return a - b", "return a * b")
    orch, agents, llm = build(calc_repo, script)
    final = orch.execute(task, calc_repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    assert final.research_performed is False  # no library named -> research not needed
    assert node_status(final)["test-1"] == "FAILED" and node_status(final)["test-2"] == "SUCCESS"
    assert "a * b" in (calc_repo.root / "calc.py").read_text()


def test_acceptance_music_while_fixing(calc_repo: RepositoryContext) -> None:
    player = FakePlayer(delay=0.3)
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a * b"), player=player)
    final = orch.execute("Play Beethoven while you inspect and fix this bug.", calc_repo, agents=agents)
    assert final.status is FinalStatus.VERIFIED_SUCCESS
    assert player.played == ["Beethoven"]
    assert node_status(final)["music-1"] == "SUCCESS"
    tester_ctx = next(e for e in orch.context.entries() if e.source == "tester")
    assert "Beethoven" not in tester_ctx.content


def test_graph_nodes_have_required_fields(calc_repo: RepositoryContext) -> None:
    orch, agents, _ = build(calc_repo, edit("return a + b", "return a * b"))
    final = orch.execute("Fix multiply.", calc_repo, agents=agents)
    node = final.state.task_graph["nodes"][0]  # type: ignore[index, union-attr]
    for key in ("id", "description", "agent", "status", "dependencies", "result", "retry_count", "priority"):
        assert key in node
    assert {n["status"] for n in final.state.task_graph["nodes"]} <= {s.value for s in NodeStatus}  # type: ignore[index, union-attr]
