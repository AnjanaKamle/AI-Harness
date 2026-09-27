"""Evaluation scenarios 1-5 on deterministic repositories (scripted model, real everything else).

Each scenario runs the full autonomous controller (Orchestrator.execute): planner, scheduler,
real tools, real git, real pytest, recovery and the verification gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from harness.llm import LLMResponse, Message, StopReason
from harness.orchestrator.verification_manager import FinalStatus

from .test_orchestration_flows import (
    BUGGY,
    TESTS,
    FakePlayer,
    build,
    call,
    done,
    edit,
    make_repo,
    node_status,
    tools,
)


def order_of(final: Any, event: str) -> list[str]:
    return [e["node"] for e in final.state.events if e["event"] == event and e["node"]]


# --- Scenario 1: simple bug -> Coder -> Tester -> VERIFIED_SUCCESS ------------------------------


def test_scenario_1_simple_bug(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": BUGGY, "tests/test_calc.py": TESTS})
    orch, agents, llm = build(repo, edit("return a + b", "return a * b"))
    final = orch.execute("Fix the deliberately broken multiply function.", repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS
    completed = order_of(final, "AGENT_COMPLETED")
    assert [n for n in completed if n != "baseline-1"][:3] == ["inspect-1", "implement-1", "test-1"]
    # the baseline ran (and finished) before the Coder changed anything
    assert completed.index("baseline-1") < order_of(final, "AGENT_STARTED").index("implement-1")
    assert final.state.baseline["failed_tests"] == ["tests/test_calc.py::test_multiply", "tests/test_calc.py::test_zero"]  # type: ignore[index, union-attr]
    assert final.retries == 0 and final.tests_failed == 0 and final.files_changed == ["calc.py"]
    assert llm.responses == []


# --- Scenario 2: insufficient first fix -> Tester FAIL -> Coder -> Tester PASS --------------------


def test_scenario_2_test_failure_then_repair(tmp_path: Path) -> None:
    source = BUGGY + "\n\ndef square(a):\n    return a + a\n"
    tests = TESTS + "\n\ndef test_square():\n    from calc import square\n    assert square(3) == 9\n"
    repo = make_repo(tmp_path, {"calc.py": source, "tests/test_calc.py": tests})
    # first run fixes only multiply; the repair fixes square using the failure evidence
    script = edit("return a + b", "return a * b") + edit(
        "return a + a", "return a * a",
        check=lambda prompt: "test_square" in prompt and "assert 6 == 9" in prompt,
    )
    orch, agents, _ = build(repo, script)
    final = orch.execute("Fix the calculator.", repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS
    statuses = node_status(final)
    assert (statuses["test-1"], statuses["repair-1"], statuses["test-2"]) == ("FAILED", "SUCCESS", "SUCCESS")
    assert [t["status"] for t in final.tests_run] == ["FAIL", "PASS"]
    assert final.state.failure_history[0]["failed_tests"] == ["tests/test_calc.py::test_square"]  # type: ignore[union-attr]


# --- Scenario 3: research -> Coder -> Tester -> VERIFIED_SUCCESS -----------------------------------


def test_scenario_3_research_then_code(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {
        "textutil.py": "def wrap_text(text, width):\n    raise NotImplementedError\n",
        "tests/test_textutil.py": "from textutil import wrap_text\n\n\ndef test_wrap():\n"
                                  "    assert wrap_text('aaa bbb ccc', 7) == ['aaa bbb', 'ccc']\n",
    })

    def api_answer(messages: list[Message]) -> LLMResponse:
        data = json.loads(messages[-1].tool_results[0].content)["data"]
        return LLMResponse(json.dumps({"status": "SUCCESS", "summary": "textwrap.wrap splits text", "findings": [{
            "question": "api", "kind": "FACT", "finding": "textwrap.wrap(text, width) returns a list of lines",
            "source": data["source"], "relevance": "wrap_text must return lines", "confidence": "HIGH",
            "technical_details": f"textwrap.wrap{data['signature']}",
            "recommended_action": "return textwrap.wrap(text, width)"}], "open_questions": []}), StopReason.END_TURN)

    def project_answer(messages: list[Message]) -> LLMResponse:
        return LLMResponse(json.dumps({"status": "SUCCESS", "summary": "stub", "findings": [{
            "question": "project", "kind": "FACT", "finding": "textutil.py has a wrap_text stub",
            "source": "textutil.py", "relevance": "target", "confidence": "HIGH"}], "open_questions": []}),
            StopReason.END_TURN)

    def coder_start(messages: list[Message]) -> LLMResponse:
        assert "return textwrap.wrap(text, width)" in messages[1].content  # the handoff
        return tools(call("read_file", path="textutil.py"))

    script = [
        tools(call("lookup_python_api", target="textwrap.wrap")), api_answer,
        tools(call("read_file", path="textutil.py")), project_answer,
        coder_start,
        tools(call("write_file", path="textutil.py",
                   content="import textwrap\n\n\ndef wrap_text(text, width):\n    return textwrap.wrap(text, width)\n")),
        tools(call("git_diff")), done(files=("textutil.py",)),
    ]
    orch, agents, llm = build(repo, script, researcher_tools=True)
    final = orch.execute("Use the `textwrap` library to implement wrap_text in textutil.py.", repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    assert final.research_performed and llm.responses == []
    completed = order_of(final, "AGENT_COMPLETED")
    assert completed.index("research-2") < completed.index("implement-1") < completed.index("test-1")


# --- Scenario 4: parallel coding + music ------------------------------------------------------------


def test_scenario_4_parallel_music(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": BUGGY, "tests/test_calc.py": TESTS})
    orch, agents, _ = build(repo, edit("return a + b", "return a * b"), player=FakePlayer(delay=0.3))
    final = orch.execute("Fix the bug and play Beethoven.", repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS
    assert node_status(final)["music-1"] == "SUCCESS"
    assert ("inspect-1", "music-1") in orch.last_controller.scheduler.concurrent_pairs
    assert all("music" not in d for n in final.plan if n["agent"] != "orchestrator" for d in n["dependencies"])


# --- Scenario 5: unsolvable (required dependency unavailable) -> bounded -> BLOCKED ----------------


def test_scenario_5_unsolvable_dependency(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {
        "calc.py": BUGGY,
        "tests/test_calc.py": "import quantum_numpy_unavailable_xyz\n" + TESTS,
    })
    script = edit("return a + b", "return a * b") * 10  # plenty: the loop must not use them all
    orch, agents, llm = build(repo, script)
    started = time.monotonic()
    final = orch.execute("Fix multiply.", repo, agents=agents)

    assert final.status is FinalStatus.BLOCKED
    assert time.monotonic() - started < 60
    assert final.state.failure_history[0]["category"] == "DEPENDENCY_FAILURE"  # type: ignore[union-attr]
    assert "repair-1" not in node_status(final)  # not fixable by code: no pointless repairs
    assert len(llm.responses) == 36  # exactly one coding run consumed its 4 replies
    assert any("DEPENDENCY_FAILURE" in r["kind"] or "not fixable" in r["reason"] for r in final.retry_history)
