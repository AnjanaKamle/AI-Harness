from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from harness.tools import RepositoryContext
from harness.verification import (
    CheckKind,
    FailureCategory,
    TestCommand,
    TestResult,
    TestRunner,
    TestStatus,
    classify_failure,
    detect_commands,
    inspect_project,
    is_verification_command,
    related_python_tests,
)
from harness.verification.parsing import parse_javascript, parse_pytest, parse_unittest


def make_repo(tmp_path: Path, files: dict[str, str]) -> RepositoryContext:
    root = tmp_path / "proj"
    root.mkdir(parents=True)
    for rel, content in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(content)
    return RepositoryContext(root)


CALC = "def multiply(a, b):\n    return a + b\n"
CALC_TEST = "from calc import multiply\n\ndef test_multiply():\n    assert multiply(2, 3) == 6\n"


# --- detection: evidence only, never invented ----------------------------------------------


def test_no_evidence_no_commands(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"main.py": "print(1)\n", "README.md": "# x\n"})
    assert detect_commands(repo) == []


def test_python_test_files_imply_pytest(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": CALC, "tests/test_calc.py": CALC_TEST})
    (cmd,) = detect_commands(repo)
    assert cmd.command == "python3 -m pytest" and cmd.kind is CheckKind.TEST and cmd.gating
    assert any("tests/test_calc.py" in e for e in cmd.evidence)


def test_pyproject_pytest_and_static_checks(tmp_path: Path) -> None:
    repo = make_repo(
        tmp_path,
        {
            "pyproject.toml": "[tool.pytest.ini_options]\n[tool.ruff]\n[tool.mypy]\n",
            "tests/test_a.py": "def test_a():\n    pass\n",
        },
    )
    commands = {c.command: c for c in detect_commands(repo)}
    assert "pyproject.toml [tool.pytest]" in commands["python3 -m pytest"].evidence
    assert commands["ruff check ."].kind is CheckKind.LINT and not commands["ruff check ."].gating
    assert commands["mypy ."].kind is CheckKind.TYPECHECK and not commands["mypy ."].gating


def test_requirements_pytest_evidence(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"requirements-dev.txt": "pytest==8.0\n", "app.py": ""})
    profile = inspect_project(repo)
    assert "pytest listed in requirements-dev.txt" in profile.pytest_evidence
    assert [c.command for c in detect_commands(repo, profile)] == ["python3 -m pytest"]


def test_unittest_only_project(tmp_path: Path) -> None:
    repo = make_repo(
        tmp_path,
        {"test_x.py": "import unittest\nclass T(unittest.TestCase):\n    def test(self): pass\n"},
    )
    # no pytest evidence -> the declared unittest style is used, pytest is not assumed
    (cmd,) = detect_commands(repo)
    assert cmd.command == "python3 -m unittest discover" and cmd.gating


def test_package_json_scripts(tmp_path: Path) -> None:
    pkg = {"scripts": {"test": "jest", "build": "tsc", "lint": "eslint .", "start": "node x"}}
    repo = make_repo(tmp_path, {"package.json": json.dumps(pkg)})
    commands = {c.command: c for c in detect_commands(repo)}
    assert set(commands) == {"npm test", "npm run build", "npm run lint"}
    assert commands["npm run build"].gating and not commands["npm run lint"].gating


def test_npm_placeholder_test_script_ignored(tmp_path: Path) -> None:
    placeholder = 'echo "Error: no test specified" && exit 1'
    repo = make_repo(tmp_path, {"package.json": json.dumps({"scripts": {"test": placeholder}})})
    assert detect_commands(repo) == []


def test_makefile_test_target_only_as_fallback(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"Makefile": "test:\n\techo ok\nbuild:\n\techo b\n"})
    assert [c.command for c in detect_commands(repo)] == ["make test"]
    both = make_repo(tmp_path / "b", {"Makefile": "test:\n\techo\n", "test_a.py": ""})
    assert [c.command for c in detect_commands(both)] == ["python3 -m pytest"]


def test_ignored_dirs_not_counted_as_tests(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {".venv/lib/test_x.py": "", "node_modules/a/test_y.py": ""})
    assert detect_commands(repo) == []


def test_related_python_tests(tmp_path: Path) -> None:
    repo = make_repo(
        tmp_path,
        {
            "calc.py": CALC,
            "pkg/geometry.py": "",
            "tests/test_calc.py": CALC_TEST,
            "tests/test_shapes.py": "from pkg.geometry import area\n",
            "tests/test_other.py": "import json\n",
        },
    )
    profile = inspect_project(repo)
    assert related_python_tests(repo, ["calc.py"], profile) == ["tests/test_calc.py"]
    assert related_python_tests(repo, ["pkg/geometry.py"], profile) == ["tests/test_shapes.py"]
    assert related_python_tests(repo, ["tests/test_other.py"], profile) == ["tests/test_other.py"]
    assert related_python_tests(repo, ["README.md"], profile) == []


# --- parsing ---------------------------------------------------------------------------------

PYTEST_FAIL = """\
============================= test session starts ==============================
collected 3 items

tests/test_calc.py F.s                                                   [100%]

=================================== FAILURES ===================================
________________________________ test_multiply _________________________________
tests/test_calc.py:4: in test_multiply
    assert multiply(2, 3) == 6
E   assert 5 == 6
E    +  where 5 = multiply(2, 3)
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_multiply - assert 5 == 6
==================== 1 failed, 1 passed, 1 skipped in 0.02s ====================
"""


def test_parse_pytest_failure() -> None:
    parsed = parse_pytest(PYTEST_FAIL, "", 1)
    assert (parsed.passed, parsed.failed, parsed.skipped, parsed.errors) == (1, 1, 1, 0)
    assert parsed.failed_tests == ["tests/test_calc.py::test_multiply"]
    assert "E   assert 5 == 6" in parsed.failure_summary
    assert "where 5 = multiply(2, 3)" in parsed.failure_summary
    assert "tests/test_calc.py:4" in parsed.failure_summary


def test_parse_pytest_keeps_important_lines_of_long_failures() -> None:
    noise = "\n".join(f"    line {i} of irrelevant context" for i in range(500))
    output = PYTEST_FAIL.replace("E   assert 5 == 6", noise + "\nE   assert 5 == 6")
    parsed = parse_pytest(output, "", 1)
    assert "E   assert 5 == 6" in parsed.failure_summary
    assert len(parsed.failure_summary) < 3_000


def test_parse_pytest_passed_and_no_tests() -> None:
    ok = parse_pytest("3 passed in 0.01s\n", "", 0)
    assert (ok.passed, ok.failed, ok.failure_summary) == (3, 0, "")
    none = parse_pytest("no tests ran in 0.01s\n", "", 5)
    assert none.no_tests


def test_parse_unittest() -> None:
    out = (
        "F.\n" + "=" * 70 + "\nFAIL: test_mul (test_x.T.test_mul)\n" + "-" * 70
        + "\nTraceback (most recent call last):\n  File \"test_x.py\", line 5, in test_mul\n"
        "AssertionError: 5 != 6\n\n" + "-" * 70 + "\nRan 2 tests in 0.001s\n\nFAILED (failures=1)\n"
    )
    parsed = parse_unittest("", out, 1)
    assert (parsed.passed, parsed.failed, parsed.errors) == (1, 1, 0)
    assert "AssertionError: 5 != 6" in parsed.failure_summary


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Tests:       1 failed, 4 passed, 5 total\n", (4, 1)),
        ("  3 passing (10ms)\n  2 failing\n", (3, 2)),
        (" Tests  1 failed | 2 passed (3)\n", (2, 1)),
        ("# pass 5\n# fail 0\n", (5, 0)),
    ],
)
def test_parse_javascript_counts(output: str, expected: tuple[int, int]) -> None:
    parsed = parse_javascript(output, "", 1 if expected[1] else 0)
    assert (parsed.passed, parsed.failed) == expected


def test_parse_javascript_failure_summary() -> None:
    out = (
        "  ● math › adds\n\n    expect(received).toBe(expected)\n\n    Expected: 6\n"
        "    Received: 5\n\n      at Object.<anonymous> (math.test.js:4:20)\n"
        "Tests:       1 failed, 1 total\n"
    )
    parsed = parse_javascript(out, "", 1)
    assert "Expected: 6" in parsed.failure_summary and "Received: 5" in parsed.failure_summary
    assert parsed.failed_tests == ["math › adds"]


# --- classification --------------------------------------------------------------------------


def _result(status: TestStatus = TestStatus.FAIL, **kw: object) -> TestResult:
    return TestResult(command="python3 -m pytest", status=status, exit_code=1, **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "category", "repairable"),
    [
        ({"failed": 1, "failure_summary": "E   assert 5 == 6"}, FailureCategory.TEST_FAILURE, True),
        ({"failed": 1, "failure_summary": "E   TypeError: unsupported operand"},
         FailureCategory.CODE_FAILURE, True),
        ({"stdout": "ERROR collecting tests/test_a.py\nE   SyntaxError: invalid syntax"},
         FailureCategory.CODE_FAILURE, True),
        ({"stderr": "ModuleNotFoundError: No module named 'requests'"},
         FailureCategory.DEPENDENCY_FAILURE, False),
        ({"stderr": "ModuleNotFoundError: No module named 'calc'"},
         FailureCategory.CODE_FAILURE, True),
        ({"stderr": "sh: jest: command not found"}, FailureCategory.DEPENDENCY_FAILURE, False),
        ({"stderr": "Error: Cannot find module 'lodash'"}, FailureCategory.DEPENDENCY_FAILURE, False),
        ({"stderr": "OSError: [Errno 28] No space left on device"},
         FailureCategory.ENVIRONMENT_FAILURE, False),
        ({"stderr": "PermissionError: [Errno 13] Permission denied: '/x'"},
         FailureCategory.ENVIRONMENT_FAILURE, False),
        ({"stdout": "something odd happened"}, FailureCategory.UNKNOWN_FAILURE, True),
    ],
)
def test_classification(
    tmp_path: Path, kwargs: dict[str, object], category: FailureCategory, repairable: bool
) -> None:
    repo = make_repo(tmp_path, {"calc.py": CALC})
    classification = classify_failure(_result(**kwargs), repo)
    assert classification.category is category, classification.evidence
    assert classification.repairable is repairable
    assert "exit code: 1" in classification.evidence


def test_classification_timeout(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {})
    classification = classify_failure(_result(TestStatus.TIMEOUT), repo)
    assert classification.category is FailureCategory.TIMEOUT and classification.repairable


def test_missing_package_introduced_by_coder_is_code_failure(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": "import numpyy\n"})
    result = _result(stderr="ModuleNotFoundError: No module named 'numpyy'")
    assert classify_failure(result, repo).category is FailureCategory.DEPENDENCY_FAILURE
    classification = classify_failure(result, repo, changed_files=["calc.py"])
    assert classification.category is FailureCategory.CODE_FAILURE
    assert any("calc.py (changed) imports" in e for e in classification.evidence)


# --- runner (real execution) -------------------------------------------------------------


def test_runner_pass_and_fail(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": CALC, "tests/test_calc.py": CALC_TEST})
    runner = TestRunner(repo, timeout_seconds=60)
    failing = runner.run(TestCommand("python3 -m pytest", CheckKind.TEST, "python", ()))
    assert failing.status is TestStatus.FAIL and failing.exit_code == 1
    assert (failing.passed, failing.failed) == (0, 1)
    assert "assert 5 == 6" in failing.failure_summary and failing.duration > 0
    assert failing.stdout  # output is kept, not hidden

    (repo.root / "calc.py").write_text(CALC.replace("a + b", "a * b"))
    passing = runner.run(TestCommand("python3 -m pytest", CheckKind.TEST, "python", ()))
    assert passing.status is TestStatus.PASS and (passing.passed, passing.failed) == (1, 0)


def test_runner_collection_error_is_error_status(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": "def multiply(a, b)\n", "tests/test_calc.py": CALC_TEST})
    result = TestRunner(repo).run(TestCommand("python3 -m pytest", CheckKind.TEST, "python", ()))
    assert result.status is TestStatus.ERROR and result.exit_code == 2
    assert classify_failure(result, repo).category is FailureCategory.CODE_FAILURE


def test_runner_no_tests_is_not_available(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": CALC})
    result = TestRunner(repo).run(TestCommand("python3 -m pytest", CheckKind.TEST, "python", ()))
    assert result.status is TestStatus.NOT_AVAILABLE and result.exit_code == 5


def test_runner_timeout(tmp_path: Path) -> None:
    slow = "import time\n\ndef test_slow():\n    time.sleep(30)\n"
    repo = make_repo(tmp_path, {"tests/test_slow.py": slow})
    result = TestRunner(repo, timeout_seconds=2).run(
        TestCommand("python3 -m pytest", CheckKind.TEST, "python", ())
    )
    assert result.status is TestStatus.TIMEOUT and result.exit_code is None
    assert "Timed out" in result.failure_summary


def test_runner_missing_executable_not_available(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {})
    runner = TestRunner(repo)
    runner.terminal.policy.safe["tsc"]  # allowed by policy...
    result = runner.run(TestCommand("tsc --definitely-missing-binary", CheckKind.TYPECHECK, "node", ()))
    if shutil.which("tsc") is None:  # ...but not installed here
        assert result.status is TestStatus.NOT_AVAILABLE


def test_runner_refuses_non_verification_commands(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": CALC})
    result = TestRunner(repo).run(
        TestCommand("python3 -c \"open('calc.py','w').write('x')\"", CheckKind.TEST, "python", ())
    )
    assert result.status is TestStatus.ERROR
    assert (repo.root / "calc.py").read_text() == CALC


@pytest.mark.parametrize(
    ("command", "allowed"),
    [
        ("python3 -m pytest tests/", True),
        ("python -m pytest", True),
        ("npm test", True),
        ("npm run build", True),
        ("make test", True),
        ("ruff check .", True),
        ("python3 -c 'print(1)'", False),
        ("python3 app.py", False),
        ("npm install", False),
        ("make deploy", False),
        ("rm -rf .", False),
    ],
)
def test_is_verification_command(command: str, allowed: bool) -> None:
    assert is_verification_command(command) is allowed


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm not installed")
def test_runner_real_npm_test(tmp_path: Path) -> None:
    pkg = {"name": "demo", "version": "1.0.0", "scripts": {"test": "node --test"}}
    test_js = (
        "const test = require('node:test');\nconst assert = require('node:assert');\n"
        "const { multiply } = require('./calc');\n"
        "test('multiply', () => { assert.strictEqual(multiply(2, 3), 6); });\n"
    )
    repo = make_repo(
        tmp_path,
        {
            "package.json": json.dumps(pkg),
            "calc.js": "exports.multiply = (a, b) => a + b;\n",
            "calc.test.js": test_js,
        },
    )
    (cmd,) = detect_commands(repo)
    failing = TestRunner(repo, timeout_seconds=120).run(cmd)
    assert failing.status is TestStatus.FAIL and failing.failed == 1
    assert "6" in failing.failure_summary and "5" in failing.failure_summary
    (repo.root / "calc.js").write_text("exports.multiply = (a, b) => a * b;\n")
    passing = TestRunner(repo, timeout_seconds=120).run(cmd)
    assert passing.status is TestStatus.PASS and passing.passed == 1


def test_test_result_roundtrip() -> None:
    result = TestResult("npm test", TestStatus.FAIL, CheckKind.BUILD, exit_code=2, failed=1)
    assert TestResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result


def test_targeted_run_only_when_it_is_a_real_subset(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"calc.py": CALC, "tests/test_calc.py": CALC_TEST,
                                "other.py": "X = 1\n", "tests/test_other.py": "def test_x():\n    pass\n"})
    runner = TestRunner(repo)
    profile = runner.inspect()
    (base,) = [c for c in runner.detect(profile) if c.kind is CheckKind.TEST]
    targeted = runner.targeted_command(base, ["calc.py"], profile)
    assert targeted is not None and targeted.command.endswith("tests/test_calc.py") and targeted.targeted
    # when the related tests are the entire suite, a targeted run would just repeat it
    assert runner.targeted_command(base, ["calc.py", "other.py"], profile) is None


def test_project_detection_cache_invalidated_by_tool_writes(tmp_path: Path) -> None:
    from harness.tools import WriteFileTool

    repo = make_repo(tmp_path, {"app.py": "X = 1\n"})
    assert detect_commands(repo) == []
    WriteFileTool(repo).run({"path": "tests/test_app.py", "content": "def test_a():\n    pass\n"})
    assert [c.command for c in detect_commands(repo)] == ["python3 -m pytest"]
