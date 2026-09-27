"""Evidence-based failure classification.

The category is derived only from the exit code, the command output and the repository
state - never from an LLM's opinion.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from harness.tools.repository import RepositoryContext
from harness.verification.models import (
    REPAIRABLE_CATEGORIES,
    FailureCategory,
    FailureClassification,
    TestResult,
    TestStatus,
)

_ENVIRONMENT = re.compile(
    r"(Permission denied|No space left on device|Read-only file system|EACCES|ENOSPC|EMFILE|"
    r"Too many open files|Cannot allocate memory|Executable not found|command not found|"
    r"Address already in use|INTERNALERROR|Command rejected by policy)",
    re.IGNORECASE,
)
# "ModuleNotFoundError: No module named 'x'" and "python3: No module named pytest" (python -m)
_PY_MISSING_MODULE = re.compile(
    r"(?:(?:ModuleNotFoundError|ImportError):|python[\d.]*:)\s*No module named '?([\w.]+)'?"
)
_PY_CANNOT_IMPORT = re.compile(r"ImportError: cannot import name '(\w+)' from '([\w.]+)'")
_JS_MISSING_MODULE = re.compile(r"Cannot find (?:module|package) '([^']+)'")
_DEPENDENCY = re.compile(
    r"(npm ERR! (?:missing|code E404|code ERESOLVE|code ENOENT)|Could not resolve dependency|"
    r"Could not find a version that satisfies|No matching distribution found|"
    r"sh: [\w.-]+: (?:command )?not found|ERR_MODULE_NOT_FOUND)",
    re.IGNORECASE,
)
_CODE_ERROR = re.compile(
    r"\b(SyntaxError|IndentationError|TabError|NameError|TypeError|AttributeError|"
    r"UnboundLocalError|ZeroDivisionError|RecursionError|KeyError|IndexError|ValueError|"
    r"ReferenceError|RangeError|error TS\d+)\b"
)
_COLLECTION_ERROR = re.compile(r"ERROR collecting|errors? during collection|ImportError while importing")
_ASSERTION = re.compile(
    r"(AssertionError|^E\s+assert\b|AssertionError \[ERR_ASSERTION\]|expect\(.*\)\.|"
    r"Expected:|Received:|assert\.\w+\()",
    re.MULTILINE,
)


def _is_local_module(repo: RepositoryContext, dotted: str) -> bool:
    top = dotted.split(".")[0].lstrip("./")
    if not top:
        return True
    for base in (repo.root, repo.root / "src", repo.root / "lib"):
        if (base / f"{top}.py").is_file() or (base / top).is_dir():
            return True
    return False


def _imported_by_changed_files(
    repo: RepositoryContext, module: str, changed_files: Sequence[str]
) -> str | None:
    pattern = re.compile(
        rf"^\s*(?:import\s+{re.escape(module)}\b|from\s+{re.escape(module)}\b|"
        rf".*require\(['\"]{re.escape(module)}['\"]\)|.*from\s+['\"]{re.escape(module)}['\"])",
        re.MULTILINE,
    )
    for rel in changed_files:
        try:
            text = (repo.resolve(rel)).read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        if pattern.search(text):
            return rel
    return None


def classify_failure(
    result: TestResult,
    repo: RepositoryContext,
    changed_files: Sequence[str] = (),
) -> FailureClassification:
    """Classify a non-passing TestResult."""
    text = "\n".join((result.stdout, result.stderr, result.failure_summary))
    evidence: list[str] = [f"command: {result.command}", f"exit code: {result.exit_code}"]

    def done(category: FailureCategory, *reasons: str) -> FailureClassification:
        return FailureClassification(
            category, tuple(evidence + list(reasons)), category in REPAIRABLE_CATEGORIES
        )

    if result.status is TestStatus.TIMEOUT:
        return done(FailureCategory.TIMEOUT, "command exceeded its time limit and was killed")

    dep = _DEPENDENCY.search(text)
    if dep:
        return done(FailureCategory.DEPENDENCY_FAILURE, f"dependency problem: {dep.group(0)}")

    # Missing modules: a dependency problem - unless the module is part of the repo (a code
    # bug), or the Coder itself just added the import (also fixable in code).
    missing = [m.group(1) for m in _PY_MISSING_MODULE.finditer(text)]
    missing += [
        m.group(1) for m in _JS_MISSING_MODULE.finditer(text) if not m.group(1).startswith(".")
    ]
    for module in dict.fromkeys(missing):
        top = module.split("/")[0] if "/" in module and not module.startswith("@") else module
        if _is_local_module(repo, top):
            return done(FailureCategory.CODE_FAILURE, f"local module {module!r} failed to import")
        introduced_by = _imported_by_changed_files(repo, top.split(".")[0], changed_files)
        if introduced_by:
            return done(
                FailureCategory.CODE_FAILURE,
                f"{introduced_by} (changed) imports missing package {module!r}",
            )
        return done(FailureCategory.DEPENDENCY_FAILURE, f"missing dependency {module!r}")
    for m in _PY_CANNOT_IMPORT.finditer(text):
        if _is_local_module(repo, m.group(2)):
            return done(FailureCategory.CODE_FAILURE, f"cannot import {m.group(1)} from {m.group(2)}")
        return done(FailureCategory.DEPENDENCY_FAILURE, f"incompatible package {m.group(2)!r}")
    env = _ENVIRONMENT.search(text)
    if env:
        return done(FailureCategory.ENVIRONMENT_FAILURE, f"environment problem: {env.group(0)}")

    if result.status is TestStatus.NOT_AVAILABLE:
        return done(FailureCategory.ENVIRONMENT_FAILURE, "verification command not available")

    code_error = _CODE_ERROR.search(result.failure_summary or text)
    if _COLLECTION_ERROR.search(text):
        return done(
            FailureCategory.CODE_FAILURE,
            "tests could not be collected/imported"
            + (f" ({code_error.group(1)})" if code_error else ""),
        )
    if code_error and not _ASSERTION.search(result.failure_summary or ""):
        return done(FailureCategory.CODE_FAILURE, f"runtime error in code: {code_error.group(1)}")
    if (result.failed or 0) > 0 or _ASSERTION.search(text):
        detail = f"{result.failed} test(s) failed" if result.failed else "assertion failures"
        return done(FailureCategory.TEST_FAILURE, detail)
    if code_error:
        return done(FailureCategory.CODE_FAILURE, f"error in code: {code_error.group(1)}")
    return done(FailureCategory.UNKNOWN_FAILURE, "non-zero exit without a recognizable cause")
