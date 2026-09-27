"""Context management: relevance filtering, truncation, compression, packaging, usage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.config.settings import Settings
from harness.context import ContextCategory, ContextEntry, InMemoryContextManager
from harness.context.builder import ContextBuilder, ContextPackage
from harness.context.relevance import RelevanceRanker, outline, search_terms
from harness.context.summaries import summarize_failures, summarize_research, summarize_tests
from harness.context.usage import UsageStats, accumulate_usage, estimated_tokens
from harness.llm import LLMResponse, Message, StopReason, ToolCall
from harness.orchestrator import AgentState, Orchestrator
from harness.tools import ReadFileTool, RepositoryContext, ToolExecutionResult, ToolStatus
from harness.tools.base import bound_value, truncate_lines, truncate_text
from harness.tools.registry import ToolRegistry

from .conftest import FAKE_KEY

IRRELEVANT_MARKER = "ZZ_UNRELATED_BILLING_MARKER"


@pytest.fixture
def project(tmp_path: Path) -> RepositoryContext:
    root = tmp_path / "shop"
    files = {
        "shop/cart.py": "from shop.pricing import apply_discount\n\n\ndef cart_total(items):\n"
                        "    return apply_discount(sum(items))\n",
        "shop/pricing.py": "def apply_discount(amount):\n    return amount\n",
        "shop/__init__.py": "",
        "tests/test_cart.py": "from shop.cart import cart_total\n\n\ndef test_total():\n"
                              "    assert cart_total([1, 2]) == 3\n",
        "legacy/billing_engine.py": f"# {IRRELEVANT_MARKER}\ndef invoice_run():\n    pass\n",
        "legacy/reports.py": "def quarterly_report():\n    pass\n",
        "docs/ops_runbook.md": "# Runbook\nRestart the billing cron.\n",
        "README.md": "# Shop\nUse cart_total to compute totals.\n",
    }
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    return RepositoryContext(root)


# --- relevance ------------------------------------------------------------------------------


def test_search_terms_pick_code_identifiers() -> None:
    terms = search_terms(["Fix `cart_total` so apply_discount() is used; the function should work"])
    assert terms[:2] == ["cart_total", "apply_discount"]
    assert "function" not in terms and "should" not in terms


def test_relevance_prioritizes_and_excludes_unrelated(project: RepositoryContext) -> None:
    ranked = RelevanceRanker(project).rank(
        ["Fix cart_total in shop/cart.py so discounts apply"], changed_files=["shop/pricing.py"]
    )
    by_path = {f.path: f for f in ranked}
    assert ranked[0].path == "shop/cart.py"  # explicitly mentioned -> highest
    assert "explicitly mentioned" in by_path["shop/cart.py"].reasons
    assert "modified during this task" in by_path["shop/pricing.py"].reasons
    assert "test for relevant code" in by_path["tests/test_cart.py"].reasons
    assert any(r.startswith("documentation") for r in by_path["README.md"].reasons)
    assert by_path["README.md"].score < by_path["tests/test_cart.py"].score
    # unrelated files never make it in
    assert not {"legacy/billing_engine.py", "legacy/reports.py", "docs/ops_runbook.md"} & set(by_path)


def test_relevance_follows_imports(project: RepositoryContext) -> None:
    ranked = RelevanceRanker(project).rank(["Update cart_total"])
    by_path = {f.path: f for f in ranked}
    assert "imported by shop/cart.py" in by_path["shop/pricing.py"].reasons


def test_relevance_with_no_signal_returns_nothing(project: RepositoryContext) -> None:
    assert RelevanceRanker(project).rank(["Make it better"]) == []


def test_relevance_limits_file_count(project: RepositoryContext) -> None:
    assert len(RelevanceRanker(project, max_files=2).rank(["cart_total apply_discount"])) == 2


def test_outline_has_signatures_not_bodies(project: RepositoryContext) -> None:
    lines = outline(project.root / "shop/cart.py")
    assert lines == ["4: def cart_total(items):"]


# --- truncation -----------------------------------------------------------------------------


def test_truncation_markers_are_explicit() -> None:
    text, cut = truncate_text("x" * 5_000, 1_000)
    assert cut and "[OUTPUT TRUNCATED: " in text and len(text) <= 1_000
    lines, cut = truncate_lines("\n".join(f"line {i}" for i in range(1_000)), 30)
    assert cut and "[OUTPUT TRUNCATED: 970 lines omitted]" in lines
    assert lines.startswith("line 0") and lines.rstrip().endswith("line 999")


def test_bound_value_caps_nested_payloads() -> None:
    payload = {"stdout": "\n".join(str(i) for i in range(5_000)), "items": list(range(500)),
               "nested": {"diff": "y" * 50_000}, "n": 3}
    bounded = bound_value(payload, max_chars=2_000, max_lines=50)
    assert "[OUTPUT TRUNCATED" in bounded["stdout"] and bounded["stdout"].count("\n") <= 52
    assert bounded["items"][-1] == "[OUTPUT TRUNCATED: 300 more items]"
    assert len(bounded["nested"]["diff"]) <= 2_000 and bounded["n"] == 3


def test_tool_output_limits_in_registry(project: RepositoryContext) -> None:
    (project.root / "big.txt").write_text("\n".join(f"row {i}" for i in range(10_000)))
    registry = ToolRegistry([ReadFileTool(project)], max_llm_chars=3_000, max_llm_lines=40)
    _, tool_result = registry.execute_call(ToolCall("c1", "read_file", {"path": "big.txt"}))
    content = json.loads(tool_result.content) if tool_result.content.endswith("}") else None
    assert len(tool_result.content) <= 3_000
    assert "[OUTPUT TRUNCATED" in tool_result.content
    if content is not None:
        assert content["data"]["content"].count("\n") <= 42


def test_search_results_capped_by_settings(project: RepositoryContext) -> None:
    from harness.tools import build_registry

    registry = build_registry(project, Settings(api_key=FAKE_KEY, max_search_results=2))
    result = registry.execute("search_code", {"query": "def", "max_results": 100})
    assert result.ok and result.data["count"] == 2 and result.data["truncated"]


def test_context_package_budget_and_markers() -> None:
    package = ContextPackage("coder", max_chars=2_500)
    package.add("TASK", "do the thing", priority=0)
    package.add("BIG", "z" * 10_000, priority=1)
    package.add("LOW PRIORITY", "w" * 3_000, priority=5)
    text = package.render()
    assert len(text) <= 2_600
    assert text.startswith("## TASK\ndo the thing")
    assert "[OUTPUT TRUNCATED" in text
    stats = package.stats
    assert stats["truncated_sections"] == ["BIG"] and stats["dropped_sections"] == ["LOW PRIORITY"]
    assert stats["estimated_context_tokens"] == estimated_tokens(stats["context_chars"])


# --- compression ---------------------------------------------------------------------------


def test_summarize_tests_matches_spec_shape() -> None:
    results = [
        {"attempt": 1, "command": "python3 -m pytest -rfE --tb=short", "status": "FAIL", "failed": 3, "passed": 1},
        {"attempt": 2, "command": "python3 -m pytest", "status": "FAIL", "failed": 1, "passed": 3},
        {"attempt": 3, "command": "python3 -m pytest tests/t.py", "status": "PASS", "passed": 4},
        {"attempt": 3, "command": "python3 -m pytest", "status": "PASS", "passed": 4},
    ]
    assert summarize_tests(results) == (
        "Test summary:\n"
        "Attempt 1: FAIL - python3 -m pytest (1 passed, 3 failed)\n"
        "Attempt 2: FAIL - python3 -m pytest (3 passed, 1 failed)\n"
        "Attempt 3: PASS - python3 -m pytest (4 passed)"
    )


def test_summarize_failures_keeps_latest_detail_only() -> None:
    history = [
        {"attempt": i, "category": "TEST_FAILURE", "counts": f"{3 - i} failed",
         "failed_tests": [f"t::{i}"], "failure_summary": f"DETAIL-{i}\n" + "E assert 1 == 2\n" * 200}
        for i in range(1, 4)
    ]
    text = summarize_failures(history)
    assert "Attempt 1: TEST_FAILURE - 2 failed" in text and "Attempt 3" in text
    assert "DETAIL-3" in text and "DETAIL-1" not in text and "DETAIL-2" not in text
    assert "[OUTPUT TRUNCATED" in text and len(text) < 4_000


def test_context_manager_compress() -> None:
    ctx = InMemoryContextManager()
    for i in range(6):
        ctx.put(ContextEntry(ContextCategory.TEST_RESULT, f"t1:v{i}", f"attempt {i}: FAIL\nraw output...", task_id="t1"))
    ctx.put(ContextEntry(ContextCategory.TEST_RESULT, "other:v0", "other task", task_id="other"))
    summary = ctx.compress(ContextCategory.TEST_RESULT, keep_last=2, task_id="t1")
    assert summary is not None and summary.metadata["compressed_entries"] == 4
    remaining = [e.key for e in ctx.for_task("t1", ContextCategory.TEST_RESULT)]
    assert remaining == ["t1:v4", "t1:v5"]
    assert "t1:v0: attempt 0: FAIL" in summary.content and "raw output" not in summary.content
    assert ctx.get(ContextCategory.TEST_RESULT, "other:v0") is not None  # other task untouched
    assert ctx.compress(ContextCategory.TEST_RESULT, keep_last=2, task_id="t1") is None
    for i in range(6, 9):
        ctx.put(ContextEntry(ContextCategory.TEST_RESULT, f"t1:v{i}", f"attempt {i}", task_id="t1"))
    again = ctx.compress(ContextCategory.TEST_RESULT, keep_last=2, task_id="t1")
    assert again is not None and again.metadata["compressed_entries"] == 7
    assert "t1:v0" in again.content and "t1:v6" in again.content


def test_context_manager_holds_all_categories() -> None:
    names = {c.value for c in ContextCategory}
    assert {"TASK", "PLAN", "REPOSITORY", "RELEVANT_FILE", "RESEARCH", "CODE_CHANGE",
            "TEST_RESULT", "FAILURE", "AGENT_OUTPUT"} <= names
    ctx = InMemoryContextManager()
    ctx.put(ContextEntry(ContextCategory.PLAN, "a", "p1", task_id="t"))
    ctx.put(ContextEntry(ContextCategory.PLAN, "b", "p2", task_id="t"))
    assert ctx.latest(ContextCategory.PLAN, "t").content == "p2"  # type: ignore[union-attr]


# --- per-agent packages ----------------------------------------------------------------------


def _state_with_history() -> AgentState:
    state = Orchestrator(Settings(api_key=FAKE_KEY)).submit("Fix cart_total in shop/cart.py")
    state.research_findings.append({"kind": "FACT", "confidence": "HIGH", "finding": "use Decimal",
                                    "source": "python:decimal.Decimal@stdlib-3.13",
                                    "technical_details": "Decimal('0.1')", "recommended_action": "use it"})
    state.research_findings.append({"kind": "UNCERTAINTY", "confidence": "LOW",
                                    "finding": "rounding mode unclear", "source": None})
    state.code_changes.append({"files_changed": ["shop/cart.py"], "diff_summary": {"summary": "1 file(s) changed, +1 -1"}})
    state.test_results += [
        {"attempt": 1, "command": "python3 -m pytest", "status": "FAIL", "failed": 1, "stdout": "RAW-TEST-OUTPUT " * 500},
    ]
    state.failure_history.append({"attempt": 1, "category": "TEST_FAILURE", "counts": "1 failed",
                                  "failed_tests": ["tests/test_cart.py::test_total"],
                                  "failure_summary": "E   assert 4 == 3"})
    return state


def test_coder_package_contents(project: RepositoryContext) -> None:
    state = _state_with_history()
    package = ContextBuilder(project).for_coder(state.task, state, ["read_file", "edit_file"])
    text = package.render()
    for expected in ("## TASK", "## PLAN", "## REPOSITORY", "## RELEVANT FILES", "## RESEARCH FINDINGS",
                     "## CURRENT FAILURES", "shop/cart.py", "[FACT/HIGH] use Decimal",
                     "Open uncertainties", "rounding mode unclear", "E   assert 4 == 3",
                     "Test summary:", "Attempt 1: FAIL"):
        assert expected in text, expected
    assert "RAW-TEST-OUTPUT" not in text  # raw outputs are summarized, not replayed
    assert IRRELEVANT_MARKER not in text and "billing_engine" not in text
    assert "return apply_discount(sum(items))" not in text  # no file bodies


def test_tester_package_contents(project: RepositoryContext) -> None:
    state = _state_with_history()
    text = ContextBuilder(project).for_tester(state).render()
    for expected in ("## TASK", "## CHANGED FILES", "shop/cart.py", "## TEST CONFIGURATION",
                     "python3 -m pytest", "## PREVIOUS FAILURE", "## VERIFICATION REQUIREMENTS"):
        assert expected in text, expected
    assert "use Decimal" not in text and "RELEVANT FILES" not in text


def test_researcher_package_contents(project: RepositoryContext) -> None:
    (project.root / "requirements.txt").write_text("requests==2.31\n")
    state = _state_with_history()
    text = ContextBuilder(project).for_researcher(
        ["What is the requests timeout API?"], state, topics=["cart_total"]
    ).render()
    assert "What is the requests timeout API?" in text
    assert "requirements.txt: requests==2.31" in text
    assert "shop/cart.py (mentions cart_total)" in text
    assert "E   assert 4 == 3" not in text and IRRELEVANT_MARKER not in text


def test_music_package_contains_only_playback() -> None:
    package = ContextBuilder.for_music("play", {"genre": "lofi", "volume": 40})
    text = package.render()
    assert text == '## PLAYBACK COMMAND\nplay\n\n## MUSIC PARAMETERS\n{"genre": "lofi", "volume": 40}'


# --- usage ----------------------------------------------------------------------------------


def test_usage_stats_estimates_and_reported() -> None:
    from harness.llm import Usage

    stats = UsageStats()
    stats.record_turn([Message.user("x" * 400)], None,
                      LLMResponse("y" * 40, StopReason.END_TURN, usage=Usage(120, 12)))
    data = stats.to_dict()
    assert data["llm_turns"] == 1 and data["input_chars"] == 400
    assert data["estimated_input_tokens"] == 100 and data["estimated_output_tokens"] == 10
    assert (data["reported_input_tokens"], data["reported_output_tokens"]) == (120, 12)


def test_accumulate_usage_per_agent_and_total() -> None:
    totals: dict[str, dict[str, int]] = {}
    accumulate_usage(totals, "coder", {"llm_turns": 3, "tool_calls": 5, "input_chars": 900})
    accumulate_usage(totals, "coder", {"llm_turns": 1, "tool_calls": 0, "input_chars": 100})
    accumulate_usage(totals, "researcher", {"llm_turns": 2, "tool_calls": 2})
    assert totals["coder"] == {"llm_turns": 4, "tool_calls": 5, "input_chars": 1000, "runs": 2}
    assert totals["total"]["llm_turns"] == 6 and totals["total"]["runs"] == 3


def test_research_summary_bounded() -> None:
    findings = [{"kind": "FACT", "confidence": "HIGH", "finding": f"f{i}", "source": "s"} for i in range(20)]
    text = summarize_research(findings, limit=5)
    assert text.count("[FACT/HIGH]") == 5 and "[OUTPUT TRUNCATED: 15 more findings]" in text


def test_tool_status_unchanged() -> None:
    assert ToolExecutionResult("x", ToolStatus.SUCCESS).ok
