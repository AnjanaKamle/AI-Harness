"""Acceptance: USER TASK -> ORCHESTRATOR -> RESEARCHER -> findings -> CODER -> TESTER -> VERIFIED.

Real tools, real pytest, real repositories; the LLM is scripted (no provider configured).
Also shows irrelevant repository files never reach any model call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from harness.agents import CoderAgent, ResearcherAgent, TesterAgent
from harness.config.settings import Settings
from harness.context import ContextCategory, InMemoryContextManager
from harness.llm import LLMResponse, Message, StopReason, ToolCall
from harness.orchestrator import AgentState, Orchestrator, StepStatus, TaskStatus
from harness.orchestrator.planner import build_plan, extract_libraries, plan_research
from harness.tools import ListFilesTool, ReadFileTool, RepositoryContext, SearchCodeTool
from harness.tools.research import LookupPythonApiTool
from harness.verification import VerificationStatus

from .conftest import FAKE_KEY, MockLLMClient, git

TASK = "Use the `statistics` library to implement average() in stats.py (the mean of a list)."
IRRELEVANT = ("legacy/billing_engine.py", "ZZ_BILLING_SECRET_MARKER", "docs/ops_runbook.md",
              "quarterly_invoice_totals")


@pytest.fixture
def stats_repo(tmp_path: Path) -> RepositoryContext:
    root = tmp_path / "stats_project"
    files = {
        "stats.py": "def average(values):\n    raise NotImplementedError\n",
        "tests/test_stats.py": "from stats import average\n\n\ndef test_average():\n"
                               "    assert average([1, 2, 3, 6]) == 3\n",
        "legacy/billing_engine.py": "# ZZ_BILLING_SECRET_MARKER\ndef quarterly_invoice_totals():\n"
                                    "    return 0\n",
        "docs/ops_runbook.md": "# Ops\nquarterly_invoice_totals runs nightly.\n",
    }
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")
    return RepositoryContext(root)


def tools(*calls: ToolCall) -> LLMResponse:
    return LLMResponse("", StopReason.TOOL_USE, tool_calls=calls)


def json_reply(body: dict[str, Any]) -> LLMResponse:
    return LLMResponse(json.dumps(body), StopReason.END_TURN)


def last_tool_data(messages: list[Message]) -> dict[str, Any]:
    tool_msg = next(m for m in reversed(messages) if m.tool_results)
    return json.loads(tool_msg.tool_results[0].content)["data"]


def build(repo: RepositoryContext, script: list[Any]) -> tuple[Orchestrator, dict[str, Any], MockLLMClient]:
    settings = Settings(api_key=FAKE_KEY, test_timeout_seconds=60, research_backend="none")
    llm = MockLLMClient(script)
    ctx = InMemoryContextManager()
    researcher = ResearcherAgent(
        llm, ctx, repo,
        tools=[ListFilesTool(repo), ReadFileTool(repo), SearchCodeTool(repo), LookupPythonApiTool(repo)],
        settings=settings,
    )
    agents = {
        "researcher": researcher,
        "coder": CoderAgent(llm, ctx, repo, settings=settings),
        "tester": TesterAgent(llm, ctx, repo, settings=settings),
    }
    return Orchestrator(settings, llm=llm, context=ctx), agents, llm


def research_script() -> list[Any]:
    """Researcher run 1 (API) + run 2 (project-specific), citing only retrieved sources."""

    def api_answer(messages: list[Message]) -> LLMResponse:
        data = last_tool_data(messages)  # the real lookup result
        assert data["kind"] == "function" and data["signature"].startswith("(data")
        return json_reply({
            "status": "SUCCESS",
            "summary": "statistics.mean(data) returns the arithmetic mean.",
            "findings": [
                {"question": "supported API", "kind": "FACT",
                 "finding": "statistics.mean(data) returns the arithmetic mean of data",
                 "source": data["source"], "relevance": "average() must return the mean",
                 "confidence": "HIGH", "technical_details": f"statistics.mean{data['signature']}",
                 "recommended_action": "import statistics and return statistics.mean(values)"},
                {"question": "supported API", "kind": "INFERENCE",
                 "finding": "statistics.mean raises StatisticsError on empty input",
                 "source": data["source"], "relevance": "edge case", "confidence": "MEDIUM"},
            ],
            "open_questions": [],
        })

    def project_answer(messages: list[Message]) -> LLMResponse:
        assert "FINDINGS SO FAR" in messages[1].content  # builds on run 1
        data = last_tool_data(messages)
        assert "raise NotImplementedError" in data["content"]
        return json_reply({
            "status": "SUCCESS",
            "summary": "stats.py has an average() stub; statistics is not yet imported.",
            "findings": [
                {"question": "project usage", "kind": "FACT",
                 "finding": "stats.py defines average(values) as a NotImplementedError stub",
                 "source": "stats.py", "relevance": "the function to implement",
                 "confidence": "HIGH", "technical_details": "def average(values)",
                 "recommended_action": "replace the stub body"},
                {"question": "project usage", "kind": "UNCERTAINTY",
                 "finding": "expected behaviour for an empty list is not specified",
                 "source": None, "relevance": "edge case", "confidence": "LOW"},
            ],
            "open_questions": ["empty-list behaviour"],
        })

    return [
        tools(ToolCall("r1", "lookup_python_api", {"target": "statistics.mean"})),
        api_answer,
        tools(ToolCall("r2", "read_file", {"path": "stats.py"})),
        project_answer,
    ]


def coder_script() -> list[Any]:
    def start(messages: list[Message]) -> LLMResponse:
        prompt = messages[1].content
        # structured findings were handed to the Coder
        assert "## RESEARCH FINDINGS" in prompt
        assert "[FACT/HIGH] statistics.mean(data) returns the arithmetic mean" in prompt
        assert "import statistics and return statistics.mean(values)" in prompt
        assert "expected behaviour for an empty list is not specified" in prompt  # as uncertainty
        assert "- stats.py" in prompt  # relevant file
        return tools(ToolCall("c1", "search_code", {"query": "def average"}))

    return [
        start,
        tools(ToolCall("c2", "read_file", {"path": "stats.py"})),
        tools(ToolCall("c3", "write_file", {
            "path": "stats.py",
            "content": "import statistics\n\n\ndef average(values):\n    return statistics.mean(values)\n",
        })),
        tools(ToolCall("c4", "git_diff", {})),
        json_reply({"status": "SUCCESS", "summary": "average() now uses statistics.mean",
                    "files_changed": ["stats.py"], "next_action": "run tests", "errors": []}),
    ]


# --- acceptance ------------------------------------------------------------------------------


def test_research_coder_tester_verified_success(stats_repo: RepositoryContext) -> None:
    orchestrator, agents, llm = build(stats_repo, research_script() + coder_script())

    state = orchestrator.run(TASK, stats_repo, **agents)

    # USER TASK -> RESEARCHER -> findings -> CODER -> TESTER -> VERIFIED SUCCESS
    assert state.status is TaskStatus.VERIFIED_SUCCESS, state.final_result
    assert state.verification_status is VerificationStatus.PASSED
    assert llm.responses == []
    assert "statistics.mean(values)" in (stats_repo.root / "stats.py").read_text()

    # plan: Researcher -> Coder -> Researcher -> Coder -> Tester, all executed
    assert [(s.agent, s.status) for s in state.plan] == [
        ("researcher", StepStatus.DONE), ("coder", StepStatus.DONE),
        ("researcher", StepStatus.DONE), ("coder", StepStatus.DONE), ("tester", StepStatus.DONE),
    ]
    # structured findings in shared state and context
    kinds = [f["kind"] for f in state.research_findings]
    assert kinds == ["FACT", "INFERENCE", "FACT", "UNCERTAINTY"]
    assert all(f["source_verified"] for f in state.research_findings if f["kind"] == "FACT")
    assert state.research is not None and state.research["decision"] == "proceed"
    assert state.research["libraries"] == ["statistics"]
    assert len(state.research["runs"]) == 2
    assert len(orchestrator.context.entries(ContextCategory.RESEARCH)) == 4

    # observability: per-agent usage with estimates, no fake exact token counts
    assert state.usage["researcher"]["runs"] == 2 and state.usage["coder"]["runs"] == 1
    assert state.usage["total"]["tool_calls"] == 2 + 4
    assert state.usage["total"]["estimated_input_tokens"] > 0
    assert state.usage["total"]["reported_input_tokens"] == 0  # the mock reports none

    # state survives serialization
    restored = AgentState.from_json(state.to_json())
    assert restored.research == state.research and restored.usage == state.usage
    assert restored.research_findings == state.research_findings


def test_irrelevant_files_never_reach_any_model_call(stats_repo: RepositoryContext) -> None:
    orchestrator, agents, llm = build(stats_repo, research_script() + coder_script())
    state = orchestrator.run(TASK, stats_repo, **agents)
    assert state.status is TaskStatus.VERIFIED_SUCCESS

    assert len(llm.calls) == 9  # 4 researcher turns + 5 coder turns
    for messages, _tools in llm.calls:
        sent = "\n".join(
            [m.content for m in messages]
            + [r.content for m in messages for r in m.tool_results]
        )
        for marker in IRRELEVANT:
            assert marker not in sent, marker
    coder_prompt = llm.calls[4][0][1].content
    assert coder_prompt.count("- ") < 40  # a targeted list, not a repository dump


# --- research failure handling ---------------------------------------------------------------


def test_failed_research_without_local_evidence_blocks(stats_repo: RepositoryContext) -> None:
    task = "Use the `fancywidgetz9` library to render the report in stats.py."
    blocked_answer = json_reply({"status": "BLOCKED", "summary": "No documentation reachable.",
                                 "findings": [], "open_questions": []})
    orchestrator, agents, llm = build(stats_repo, [blocked_answer, blocked_answer])
    state = orchestrator.run(task, stats_repo, **agents)

    assert state.status is TaskStatus.BLOCKED
    assert "no local evidence" in (state.final_result or "")
    assert state.research is not None and state.research["decision"] == "blocked"
    assert state.research_findings == []  # nothing invented
    assert state.attempt_history == [] and state.verification_attempts == 0  # coder never ran
    assert [f["agent"] for f in state.failures] == ["researcher", "researcher"]
    assert state.outcome is not None and state.outcome["research"]["runs"][0]["status"] == "BLOCKED"
    assert (stats_repo.root / "stats.py").read_text().endswith("raise NotImplementedError\n")


def test_failed_research_with_local_evidence_lets_coder_proceed(stats_repo: RepositoryContext) -> None:
    failed = json_reply({"status": "BLOCKED", "summary": "offline", "findings": [],
                         "open_questions": []})
    orchestrator, agents, llm = build(stats_repo, [failed, failed] + coder_script()[1:])
    state = orchestrator.run(TASK, stats_repo, **agents)  # statistics is installed locally
    assert state.research is not None and state.research["decision"] == "proceed"
    assert "installed in the Python environment" in state.research["decision_reason"]
    assert state.status is TaskStatus.VERIFIED_SUCCESS
    assert state.failures[0]["agent"] == "researcher"  # the failure is still recorded


def test_researcher_crash_does_not_crash_harness(stats_repo: RepositoryContext) -> None:
    def explode(messages: list[Message]) -> LLMResponse:
        raise RuntimeError("unexpected bug in research tooling")

    orchestrator, agents, llm = build(stats_repo, [explode, explode] + coder_script()[1:])
    state = orchestrator.run(TASK, stats_repo, **agents)
    assert state.research is not None and state.research["runs"][0]["status"] == "FAILURE"
    assert "RuntimeError" in state.research["runs"][0]["errors"][0]
    assert state.status is TaskStatus.VERIFIED_SUCCESS  # local evidence allowed proceeding


def test_no_research_for_plain_tasks(stats_repo: RepositoryContext) -> None:
    plan = plan_research("Fix the off-by-one error in average()", stats_repo)
    assert not plan.needed
    steps = build_plan(plan)
    assert [(s.agent, s.status) for s in steps] == [
        ("researcher", StepStatus.SKIPPED), ("coder", StepStatus.PENDING),
        ("coder", StepStatus.PENDING), ("tester", StepStatus.PENDING),
    ]
    assert [s.agent for s in steps if s.status is not StepStatus.SKIPPED] == ["coder", "coder", "tester"]


@pytest.mark.parametrize(
    ("task", "libraries"),
    [
        ("Use library httpx to add retries", ["httpx"]),
        ("Implement OAuth using the authlib library", ["authlib"]),
        ("Migrate to `pydantic` v2 models", ["pydantic"]),
        ("Use the requests package and the tenacity library", ["requests", "tenacity"]),
        ("Fix the bug in the parser", []),
        ("Use a loop instead of recursion", []),
    ],
)
def test_extract_libraries(task: str, libraries: list[str]) -> None:
    assert extract_libraries(task) == libraries


def test_plan_research_questions_and_requirements(stats_repo: RepositoryContext) -> None:
    plan = plan_research("Use library fancywidgetz9 to draw charts", stats_repo)
    assert plan.needed and plan.required and not plan.can_proceed_without_research
    assert "fancywidgetz9" in plan.api_questions[0] and "repository" in plan.project_questions[0]
    docs = plan_research("Check the documentation for the deprecated flag", stats_repo)
    assert docs.needed and not docs.required and docs.can_proceed_without_research
    assert plan_research("Use library httpx", stats_repo, force=False).needed is False
    assert plan_research("Tidy things up", stats_repo, force=True).needed is True
