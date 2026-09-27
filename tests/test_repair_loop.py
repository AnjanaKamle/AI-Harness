"""Tester agent + Coder -> Tester repair loop, end to end on temporary git repositories.

The Coder's "model" is scripted (no provider is configured), but everything else is real:
tools edit real files, the Tester runs real pytest, and verification decides the outcome.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from harness.agents import AgentStatus, CoderAgent, TesterAgent, TesterResult
from harness.config.settings import DEFAULT_MAX_REPAIR_ATTEMPTS, Settings
from harness.context import InMemoryContextManager
from harness.llm import LLMError, LLMResponse, Message, StopReason, ToolCall
from harness.orchestrator import AgentState, Orchestrator, TaskStatus
from harness.orchestrator.repair_loop import RepairLoop
from harness.tools import ReadFileTool, RepositoryContext, WriteFileTool
from harness.verification import FailureCategory, TestStatus, VerificationStatus

from .conftest import FAKE_KEY, MockLLMClient, git

TASK = "Fix multiply so it returns the product of its arguments."
BUGGY = "def multiply(a, b):\n    return a + b\n"
TESTS = (
    "from calc import multiply\n\n\n"
    "def test_multiply():\n    assert multiply(2, 3) == 6\n\n\n"
    "def test_multiply_zero():\n    assert multiply(0, 5) == 0\n"
)


@pytest.fixture
def calc_repo(tmp_path: Path) -> RepositoryContext:
    root = tmp_path / "calc"
    (root / "tests").mkdir(parents=True)
    (root / "calc.py").write_text(BUGGY)
    (root / "tests" / "test_calc.py").write_text(TESTS)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "initial")
    return RepositoryContext(root)


# --- scripted coder "model" ----------------------------------------------------------------


def tools(*calls: ToolCall) -> LLMResponse:
    return LLMResponse("", StopReason.TOOL_USE, tool_calls=calls)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(f"id-{name}", name, arguments)


def final(status: str = "SUCCESS", summary: str = "Everything works.", files: tuple[str, ...] = ("calc.py",)) -> LLMResponse:
    body = {"status": status, "summary": summary, "files_changed": list(files),
            "next_action": "Run tests.", "errors": []}
    return LLMResponse(json.dumps(body), StopReason.END_TURN)


def edit_attempt(old: str, new: str, check: Callable[[str], None] | None = None) -> list[Any]:
    """One coder run: read -> edit -> git_diff -> final SUCCESS claim."""

    def start(messages: list[Message]) -> LLMResponse:
        if check is not None:
            check(messages[1].content)  # the task/repair prompt the Coder received
        return tools(call("read_file", path="calc.py"))

    return [start, tools(call("edit_file", path="calc.py", old_text=old, new_text=new)),
            tools(call("git_diff")), final()]


def build(
    repo: RepositoryContext, script: list[Any], max_repairs: int = 5
) -> tuple[Orchestrator, CoderAgent, TesterAgent, MockLLMClient]:
    settings = Settings(api_key=FAKE_KEY, max_repair_attempts=max_repairs, test_timeout_seconds=60)
    llm = MockLLMClient(script)
    ctx = InMemoryContextManager()
    coder = CoderAgent(llm, ctx, repo, settings=settings)
    tester = TesterAgent(llm, ctx, repo, settings=settings)
    return Orchestrator(settings, llm=llm, context=ctx), coder, tester, llm


# --- integration: buggy -> coder -> tester -> failure -> coder -> tester -> verified --------


def test_repair_loop_reaches_verified_success(
    calc_repo: RepositoryContext, caplog: pytest.LogCaptureFixture
) -> None:
    def repair_prompt_has_evidence(prompt: str) -> None:
        assert "REPAIR ATTEMPT 1 of 5" in prompt
        assert "FAILURE CLASSIFICATION: TEST_FAILURE" in prompt
        assert "tests/test_calc.py::test_multiply" in prompt
        assert "assert 8 == 6" in prompt  # the real assertion from the failed run
        assert "where 8 = multiply(2, 3)" in prompt
        assert "python3 -m pytest" in prompt and "EXIT CODE: 1" in prompt
        assert "Try again" not in prompt

    script = [
        # attempt 1: a wrong "fix" (2 ** 3 == 8) that the Coder claims works
        *edit_attempt("return a + b", "return a ** b"),
        # attempt 2: sees the failure evidence, fixes it properly
        *edit_attempt("return a ** b", "return a * b", check=repair_prompt_has_evidence),
    ]
    orchestrator, coder, tester, llm = build(calc_repo, script)
    caplog.set_level(logging.INFO, logger="harness")

    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)

    assert state.status is TaskStatus.VERIFIED_SUCCESS, state.final_result
    assert state.verification_status is VerificationStatus.PASSED
    assert (calc_repo.root / "calc.py").read_text() == BUGGY.replace("a + b", "a * b")
    assert llm.responses == []  # exactly two coder runs were needed

    # attempt history: attempt 1 failed (1 test), attempt 2 passed
    assert [a["outcome"] for a in state.attempt_history] == ["RETRY", "VERIFIED"]
    assert state.repair_attempts == 1 and state.verification_attempts == 2
    (failure,) = state.failure_history
    assert failure["attempt"] == 1 and failure["category"] == "TEST_FAILURE"
    assert failure["failed_tests"] == ["tests/test_calc.py::test_multiply"]
    assert failure["counts"] == "1 passed, 1 failed" and "assert 8 == 6" in failure["failure_summary"]

    # the related test file is the whole suite, so each attempt runs it exactly once
    commands = [(t["attempt"], t["status"]) for t in state.test_results]
    assert commands == [(1, "FAIL"), (2, "PASS")]
    assert state.latest_test_result is not None and state.latest_test_result["status"] == "PASS"

    # completion evidence is recorded
    assert state.outcome is not None
    assert state.outcome["evidence"]["changes"] == "1 file(s) changed, +1 -1"
    assert "Verification PASSED" in state.outcome["evidence"]["verification"]
    assert json.loads(state.to_json())["status"] == "VERIFIED_SUCCESS"

    # every cycle is logged: attempt, agent, action, command, result, classification
    text = caplog.text
    assert "Attempt 1 | agent=coder | action=implement" in text
    assert "Attempt 1 | agent=tester | action=run test | command=python3 -m pytest" in text
    assert "result=FAIL (1 passed, 1 failed) | classification=TEST_FAILURE" in text
    assert "Attempt 2 | agent=coder | action=repair" in text
    assert "Attempt 2 | result=VERIFIED_SUCCESS" in text
    assert FAKE_KEY not in text


def test_unsolvable_task_blocks_after_max_repair_attempts(calc_repo: RepositoryContext) -> None:
    # Contradictory requirements: no implementation can satisfy both tests.
    (calc_repo.root / "tests" / "test_calc.py").write_text(
        TESTS + "\n\ndef test_contradiction():\n    assert multiply(2, 3) == 7\n"
    )
    git(calc_repo.root, "commit", "-qam", "impossible spec")

    variants = ["a * b", "a * b + 0", "(a * b)", "b * a"]
    script: list[Any] = []
    previous = "a + b"
    for variant in variants[:3]:  # 1 initial attempt + 2 repairs
        script += edit_attempt(f"return {previous}", f"return {variant}")
        previous = variant
    orchestrator, coder, tester, llm = build(calc_repo, script, max_repairs=2)

    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)

    assert state.status is TaskStatus.BLOCKED
    assert "Maximum repair attempts (2) reached" in (state.final_result or "")
    assert llm.responses == []  # terminated: no fourth coder run
    assert len(state.attempt_history) == 3 and state.repair_attempts == 2
    assert [f["attempt"] for f in state.failure_history] == [1, 2, 3]
    assert all("test_contradiction" in " ".join(f["failed_tests"]) for f in state.failure_history)

    outcome = state.outcome
    assert outcome is not None and outcome["status"] == "BLOCKED"
    assert [a["attempt"] for a in outcome["attempted"]] == [1, 2, 3]
    assert outcome["latest_failure"]["attempt"] == 3
    assert [f["attempt"] for f in outcome["previous_failures"]] == [1, 2]
    assert outcome["files_changed"] == ["calc.py"]
    assert outcome["tests_run"] and all(t["status"] == "FAIL" for t in outcome["tests_run"])


def test_default_max_repair_attempts_is_five() -> None:
    assert DEFAULT_MAX_REPAIR_ATTEMPTS == 5
    assert Settings(api_key=FAKE_KEY).max_repair_attempts == 5
    assert Settings.from_env({"AI_API_KEY": FAKE_KEY, "AI_MAX_REPAIR_ATTEMPTS": "2"}).max_repair_attempts == 2


def test_repair_loop_rejects_negative_limit(calc_repo: RepositoryContext) -> None:
    _, coder, tester, _ = build(calc_repo, [])
    with pytest.raises(ValueError):
        RepairLoop(coder, tester, calc_repo, max_repair_attempts=-1)


# --- success requires evidence --------------------------------------------------------------


def test_coder_claim_without_passing_tests_is_not_success(calc_repo: RepositoryContext) -> None:
    # The Coder edits something irrelevant and says "Everything works".
    script = edit_attempt("def multiply", "def multiply")  # no-op edit -> nothing changed
    orchestrator, coder, tester, _ = build(calc_repo, script, max_repairs=0)
    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)
    assert state.status is TaskStatus.BLOCKED
    assert state.verification_status is VerificationStatus.FAILED
    assert state.failure_history[0]["category"] == "TEST_FAILURE"


def test_passing_tests_without_changes_is_not_success(calc_repo: RepositoryContext) -> None:
    (calc_repo.root / "calc.py").write_text(BUGGY.replace("a + b", "a * b"))
    git(calc_repo.root, "commit", "-qam", "already fixed")
    script = [tools(call("read_file", path="calc.py")), tools(call("git_diff")),
              final(summary="Already correct, everything works.", files=())]
    orchestrator, coder, tester, _ = build(calc_repo, script, max_repairs=0)
    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)
    assert state.verification_status is VerificationStatus.PASSED  # tests really pass...
    assert state.status is TaskStatus.BLOCKED  # ...but no change was made, so not verified
    assert "not modified any file" in state.failure_history[0]["reason"]


def test_no_verification_available_blocks(tmp_path: Path) -> None:
    root = tmp_path / "notests"
    root.mkdir()
    (root / "calc.py").write_text(BUGGY)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "init")
    repo = RepositoryContext(root)
    orchestrator, coder, tester, _ = build(repo, edit_attempt("a + b", "a * b"))
    state = orchestrator.run(TASK, repo, coder=coder, tester=tester)
    assert state.status is TaskStatus.BLOCKED
    assert state.verification_status is VerificationStatus.NOT_AVAILABLE
    assert "Refusing to invent" in (state.final_result or "")
    assert state.test_results == []  # nothing was run, nothing was claimed


def test_dependency_failure_stops_without_retry(calc_repo: RepositoryContext) -> None:
    (calc_repo.root / "tests" / "test_calc.py").write_text("import not_a_real_pkg_xyz\n" + TESTS)
    git(calc_repo.root, "commit", "-qam", "needs dep")
    orchestrator, coder, tester, llm = build(
        calc_repo, edit_attempt("a + b", "a * b") + edit_attempt("x", "y")
    )
    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)
    assert state.status is TaskStatus.BLOCKED
    assert state.failure_history[0]["category"] == "DEPENDENCY_FAILURE"
    assert state.repair_attempts == 0 and len(llm.responses) == 4  # no repair attempted
    assert "cannot be fixed by code changes" in (state.final_result or "")


def test_blocked_coder_stops_loop(calc_repo: RepositoryContext) -> None:
    orchestrator, coder, tester, _ = build(
        calc_repo, [final("BLOCKED", "Need access to the spec", files=())]
    )
    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)
    assert state.status is TaskStatus.BLOCKED and state.verification_attempts == 0
    assert "Coder is blocked" in (state.final_result or "")


def test_llm_error_stops_loop(calc_repo: RepositoryContext) -> None:
    def boom(messages: list[Message]) -> LLMResponse:
        raise LLMError("provider down")

    orchestrator, coder, tester, _ = build(calc_repo, [boom])
    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)
    assert state.status is TaskStatus.BLOCKED and "Language model failure" in (state.final_result or "")


def test_no_progress_stops_early(calc_repo: RepositoryContext) -> None:
    fail_same_way = [tools(call("git_diff")), final("FAILURE", "Unsure", files=())]
    script = edit_attempt("a + b", "a - b") + fail_same_way * 2
    orchestrator, coder, tester, llm = build(calc_repo, script, max_repairs=5)
    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)
    # attempt 2 changed nothing and failed identically -> stop before attempt 3
    assert state.status is TaskStatus.BLOCKED and "No progress" in (state.final_result or "")
    assert len(state.attempt_history) == 2 and len(llm.responses) == 2


# --- tester restrictions -------------------------------------------------------------------


def test_tester_has_no_modifying_tools(calc_repo: RepositoryContext) -> None:
    _, _, tester, _ = build(calc_repo, [])
    assert not {"write_file", "edit_file", "terminal"} & set(tester.available_tools)
    assert {"run_tests", "detect_tests", "git_diff", "read_file"} <= set(tester.available_tools)
    with pytest.raises(ValueError, match="must not have modifying tools"):
        TesterAgent(MockLLMClient(), InMemoryContextManager(), calc_repo,
                    tools=[ReadFileTool(calc_repo), WriteFileTool(calc_repo)])


def test_tester_run_tests_refuses_arbitrary_code(calc_repo: RepositoryContext) -> None:
    _, _, tester, _ = build(calc_repo, [])
    result = tester.registry.execute(
        "run_tests", {"command": "python3 -c \"open('calc.py','w').write('pwned')\""}
    )
    assert not result.ok and "only runs test/build/lint/typecheck" in (result.error or "")
    assert (calc_repo.root / "calc.py").read_text() == BUGGY


def test_tester_detects_verification_that_modifies_source(calc_repo: RepositoryContext) -> None:
    sneaky = TESTS + (
        "\n\ndef test_rewrites_source():\n"
        "    open('calc.py', 'w').write('def multiply(a, b):\\n    return a * b\\n')\n"
    )
    (calc_repo.root / "tests" / "test_calc.py").write_text(sneaky)
    git(calc_repo.root, "commit", "-qam", "sneaky test")
    _, _, tester, _ = build(calc_repo, [])
    state = AgentState(task=TASK)
    result = tester.run(TASK, state)
    assert isinstance(result, TesterResult)
    assert result.integrity_violations == ["calc.py was modified during verification"]
    assert result.verification_status is VerificationStatus.FAILED
    assert result.status is AgentStatus.FAILURE


def test_integrity_violation_blocks_loop(calc_repo: RepositoryContext) -> None:
    (calc_repo.root / "tests" / "test_calc.py").write_text(
        TESTS + "\n\ndef test_touch():\n    open('calc.py', 'a').write('# touched\\n')\n"
    )
    git(calc_repo.root, "commit", "-qam", "touching test")
    orchestrator, coder, tester, _ = build(calc_repo, edit_attempt("a + b", "a * b"))
    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)
    assert state.status is TaskStatus.BLOCKED
    assert "Verification modified repository files" in (state.final_result or "")


def test_tester_reports_evidence_and_does_not_hide_stderr(calc_repo: RepositoryContext) -> None:
    (calc_repo.root / "tests" / "test_calc.py").write_text(
        "import sys\nfrom calc import multiply\n\n\ndef test_m():\n"
        "    print('diagnostic on stderr', file=sys.stderr)\n    assert multiply(2, 3) == 6\n"
    )
    _, _, tester, _ = build(calc_repo, [])
    state = AgentState(task=TASK)
    result = tester.run(TASK, state)
    assert isinstance(result, TesterResult)
    decisive = result.decisive_result
    assert decisive is not None and decisive.status is TestStatus.FAIL and decisive.exit_code == 1
    assert "diagnostic on stderr" in decisive.stdout + decisive.stderr + decisive.failure_summary
    assert result.failure_classification is not None
    assert result.failure_classification.category is FailureCategory.TEST_FAILURE
    assert state.verification_attempts == 1 and state.latest_test_result is not None
    assert state.test_results[0]["attempt"] == 1


def test_tester_state_is_serializable_after_loop(calc_repo: RepositoryContext) -> None:
    orchestrator, coder, tester, _ = build(calc_repo, edit_attempt("a + b", "a * b"))
    state = orchestrator.run(TASK, calc_repo, coder=coder, tester=tester)
    restored = AgentState.from_json(state.to_json())
    assert restored.status is TaskStatus.VERIFIED_SUCCESS
    assert restored.verification_status is VerificationStatus.PASSED
    assert restored.attempt_history == state.attempt_history


def test_orchestrator_run_requires_provider_when_agents_not_given(calc_repo: RepositoryContext) -> None:
    from harness.llm import LLMNotConfiguredError

    with pytest.raises(LLMNotConfiguredError):
        Orchestrator(Settings(api_key=FAKE_KEY)).run(TASK, calc_repo)
