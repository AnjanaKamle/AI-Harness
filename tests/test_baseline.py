"""Baseline awareness: tests are run before any change and compared afterwards."""

from __future__ import annotations

from pathlib import Path

from harness.orchestrator.verification_manager import FinalStatus

from .test_orchestration_flows import BUGGY, TESTS, build, edit, make_repo, node_status

UNRELATED_BROKEN = "from legacy import total\n\n\ndef test_total():\n    assert total() == 42\n"


def test_a_preexisting_unrelated_failure_is_not_blamed_on_the_agent(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": BUGGY, "tests/test_calc.py": TESTS,
                                "legacy.py": "def total():\n    return 0\n",
                                "tests/test_legacy.py": UNRELATED_BROKEN})
    orch, agents, _ = build(repo, edit("return a + b", "return a * b"))
    final = orch.execute("Fix multiply in calc.py.", repo, agents=agents)

    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
    comparison = final.state.latest_test_result["baseline_comparison"]  # type: ignore[index, union-attr]
    assert comparison["preexisting_failures"] == ["tests/test_legacy.py::test_total"]
    assert comparison["new_failures"] == [] and comparison["accepted"] is True
    assert set(comparison["fixed"]) == {"tests/test_calc.py::test_multiply", "tests/test_calc.py::test_zero"}
    tests_check = next(c for c in final.verification["checks"] if c["name"] == "tests_passed")
    assert "pre-existing unrelated failures remain: tests/test_legacy.py::test_total" in tests_check["detail"]


def test_b_failure_introduced_by_the_agent_is_detected(tmp_path: Path) -> None:
    fixed = BUGGY.replace("a + b", "a * b") + "\n\ndef add(a, b):\n    return a + b\n"
    tests = TESTS + "\n\ndef test_add():\n    from calc import add\n    assert add(2, 2) == 4\n"
    repo = make_repo(tmp_path, {"calc.py": fixed, "tests/test_calc.py": tests})
    # the first edit breaks add(); the repair fixes it with a real (net) change
    script = edit("return a + b", "return a - b") + edit("return a - b", "return b + a")
    orch, agents, _ = build(repo, script)
    final = orch.execute("Tidy up add() in calc.py.", repo, agents=agents)

    first = final.state.failure_history[0]  # type: ignore[union-attr]
    assert "New failures introduced since the baseline: tests/test_calc.py::test_add" in first["reason"]
    assert final.status is FinalStatus.VERIFIED_SUCCESS  # repaired back to a passing state


def test_c_related_preexisting_failure_must_be_fixed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": BUGGY, "tests/test_calc.py": TESTS})
    # the "fix" changes calc.py but multiply is still wrong: its failing test is related
    orch, agents, _ = build(repo, edit("return a + b", "return b + a"), max_repair_attempts=0)
    final = orch.execute("Fix multiply in calc.py.", repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED
    comparison = final.state.latest_test_result["baseline_comparison"]  # type: ignore[index, union-attr]
    assert comparison["accepted"] is False and comparison["preexisting_related"]


def test_d_infrastructure_failure_at_baseline_does_not_block(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": BUGGY, "tests/test_calc.py": "import not_installed_xyz\n" + TESTS})
    orch, agents, _ = build(repo, edit("return a + b", "return a * b"))
    final = orch.execute("Fix multiply.", repo, agents=agents)
    assert node_status(final)["baseline-1"] in ("SUCCESS", "FAILED")  # optional either way
    assert final.status is FinalStatus.BLOCKED
    assert final.state.failure_history[0]["category"] == "DEPENDENCY_FAILURE"  # type: ignore[union-attr]


def test_passing_suite_without_test_evidence_is_not_verified(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": BUGGY.replace("a + b", "a * b"), "tests/test_calc.py": TESTS,
                                "notes.py": "GREETING = 'hi'\n"})
    orch, agents, _ = build(repo, edit("GREETING = 'hi'", "GREETING = 'hello'", path="notes.py"))
    final = orch.execute("Change the greeting in notes.py.", repo, agents=agents)
    assert final.status is FinalStatus.BLOCKED  # suite passes, but nothing tests notes.py
    check = next(c for c in final.verification["checks"] if c["name"] == "task_behavior_verified")
    assert check["passed"] is False and "no test exercises the changed code" in check["detail"]


def test_new_test_counts_as_evidence(tmp_path: Path) -> None:
    from .test_orchestration_flows import call, done, tools

    repo = make_repo(tmp_path, {"calc.py": BUGGY.replace("a + b", "a * b"), "tests/test_calc.py": TESTS,
                                "notes.py": "GREETING = 'hi'\n"})
    script = [tools(call("edit_file", path="notes.py", old_text="'hi'", new_text="'hello'")),
              tools(call("write_file", path="tests/test_notes.py",
                         content="from notes import GREETING\n\n\ndef test_g():\n    assert GREETING == 'hello'\n")),
              tools(call("git_diff")), done(files=("notes.py", "tests/test_notes.py"))]
    orch, agents, _ = build(repo, script)
    final = orch.execute("Change the greeting in notes.py.", repo, agents=agents)
    assert final.status is FinalStatus.VERIFIED_SUCCESS, final.unresolved_issues
