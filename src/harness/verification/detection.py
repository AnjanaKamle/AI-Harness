"""Project-aware detection of verification commands.

Commands are proposed only when the repository provides evidence for them (config files,
declared scripts, test files). If there is no evidence, nothing is proposed - the harness
never invents a test command.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.tools.repository import RepositoryContext, is_ignored_dir
from harness.verification.models import CheckKind, TestCommand

PYTHON = "python3"  # the harness runs commands with its own interpreter first on PATH
MAX_SCAN_FILES = 20_000

_PY_TEST_FILE = re.compile(r"^(test_.*|.*_test)\.py$")
_JS_TEST_FILE = re.compile(r"\.(test|spec)\.[cm]?[jt]sx?$")
_NPM_PLACEHOLDER = "no test specified"
_MAKE_TARGET = re.compile(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", re.MULTILINE)


@dataclass
class ProjectProfile:
    """What the repository says about itself."""

    ecosystems: list[str] = field(default_factory=list)
    python_test_files: list[str] = field(default_factory=list)
    js_test_files: list[str] = field(default_factory=list)
    uses_pytest: bool = False
    pytest_evidence: list[str] = field(default_factory=list)
    unittest_evidence: list[str] = field(default_factory=list)
    npm_scripts: dict[str, str] = field(default_factory=dict)
    make_targets: list[str] = field(default_factory=list)
    ruff_config: str | None = None
    mypy_config: str | None = None
    metadata_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ecosystems": self.ecosystems,
            "metadata_files": self.metadata_files,
            "python_test_files": self.python_test_files[:50],
            "js_test_files": self.js_test_files[:50],
            "pytest_evidence": self.pytest_evidence,
            "unittest_evidence": self.unittest_evidence,
            "npm_scripts": sorted(self.npm_scripts),
            "make_targets": self.make_targets,
        }


def _read(path: Path, limit: int = 1_000_000) -> str:
    try:
        if path.is_file() and path.stat().st_size <= limit:
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return ""


def _walk_files(repo: RepositoryContext) -> Iterable[Path]:
    count = 0
    for dirpath, dirnames, filenames in os.walk(repo.root):
        dirnames[:] = sorted(d for d in dirnames if not is_ignored_dir(d))
        for name in sorted(filenames):
            count += 1
            if count > MAX_SCAN_FILES:
                return
            yield Path(dirpath, name)


_CACHE_SECONDS = 30.0
_profile_cache: dict[Path, tuple[int, float, ProjectProfile]] = {}


def inspect_project(repo: RepositoryContext) -> ProjectProfile:
    """Project profile, cached per repository until a tool modifies it (or 30s pass)."""
    import copy
    import time

    from harness.tools.repository import generation

    cached = _profile_cache.get(repo.root)
    now = time.monotonic()
    if cached and cached[0] == generation(repo.root) and now - cached[1] < _CACHE_SECONDS:
        return copy.deepcopy(cached[2])
    profile = _inspect_project(repo)
    _profile_cache[repo.root] = (generation(repo.root), now, copy.deepcopy(profile))
    return profile


def _inspect_project(repo: RepositoryContext) -> ProjectProfile:
    root = repo.root
    profile = ProjectProfile()

    for path in _walk_files(repo):
        rel = repo.relative(path)
        if _PY_TEST_FILE.match(path.name):
            profile.python_test_files.append(rel)
        elif _JS_TEST_FILE.search(path.name) and "node_modules" not in path.parts:
            profile.js_test_files.append(rel)
        if path.name == "conftest.py":
            profile.pytest_evidence.append(f"{rel} present")

    # --- Python metadata ---
    pyproject_text = _read(root / "pyproject.toml")
    if pyproject_text:
        profile.metadata_files.append("pyproject.toml")
        try:
            pyproject = tomllib.loads(pyproject_text)
        except tomllib.TOMLDecodeError:
            pyproject = {}
        tool = pyproject.get("tool", {})
        if "pytest" in tool:
            profile.pytest_evidence.append("pyproject.toml [tool.pytest]")
        if "ruff" in tool:
            profile.ruff_config = "pyproject.toml [tool.ruff]"
        if "mypy" in tool:
            profile.mypy_config = "pyproject.toml [tool.mypy]"
        if re.search(r"[\"']pytest\b", pyproject_text):
            profile.pytest_evidence.append("pytest declared in pyproject.toml dependencies")
    if (root / "pytest.ini").is_file():
        profile.metadata_files.append("pytest.ini")
        profile.pytest_evidence.append("pytest.ini")
    for name, marker in (("setup.cfg", "[tool:pytest]"), ("tox.ini", "[pytest]")):
        text = _read(root / name)
        if text:
            profile.metadata_files.append(name)
            if marker in text:
                profile.pytest_evidence.append(f"{name} {marker}")
            if "[mypy" in text and profile.mypy_config is None:
                profile.mypy_config = f"{name} [mypy]"
    for req in sorted(root.glob("requirements*.txt")):
        profile.metadata_files.append(req.name)
        if re.search(r"^\s*pytest\b", _read(req), re.MULTILINE):
            profile.pytest_evidence.append(f"pytest listed in {req.name}")
    for name in ("ruff.toml", ".ruff.toml"):
        if (root / name).is_file():
            profile.ruff_config = name
    if (root / "mypy.ini").is_file():
        profile.mypy_config = "mypy.ini"

    for rel in profile.python_test_files[:200]:
        text = _read(root / rel, 200_000)
        if re.search(r"^\s*import pytest|^\s*from pytest", text, re.MULTILINE):
            profile.pytest_evidence.append(f"{rel} imports pytest")
            break
    for rel in profile.python_test_files[:200]:
        if re.search(r"unittest\.TestCase", _read(root / rel, 200_000)):
            profile.unittest_evidence.append(f"{rel} uses unittest.TestCase")
            break

    profile.uses_pytest = bool(profile.pytest_evidence)
    if pyproject_text or profile.python_test_files or any(
        f.startswith(("requirements", "setup")) for f in profile.metadata_files
    ):
        profile.ecosystems.append("python")

    # --- Node metadata ---
    package_text = _read(root / "package.json")
    if package_text:
        profile.metadata_files.append("package.json")
        profile.ecosystems.append("node")
        try:
            scripts = json.loads(package_text).get("scripts", {})
        except (json.JSONDecodeError, AttributeError):
            scripts = {}
        if isinstance(scripts, dict):
            profile.npm_scripts = {str(k): str(v) for k, v in scripts.items()}

    # --- Makefile ---
    makefile = _read(root / "Makefile")
    if makefile:
        profile.metadata_files.append("Makefile")
        profile.make_targets = sorted(set(_MAKE_TARGET.findall(makefile)))
    return profile


def _npm_script_real(profile: ProjectProfile, name: str) -> bool:
    script = profile.npm_scripts.get(name)
    return bool(script) and _NPM_PLACEHOLDER not in str(script)


def detect_commands(repo: RepositoryContext, profile: ProjectProfile | None = None) -> list[TestCommand]:
    """Verification commands backed by repository evidence, most relevant first."""
    profile = profile or inspect_project(repo)
    commands: list[TestCommand] = []

    # Python tests
    if profile.python_test_files or profile.uses_pytest:
        if profile.uses_pytest or not profile.unittest_evidence:
            evidence = profile.pytest_evidence + (
                [f"{len(profile.python_test_files)} Python test file(s), e.g. "
                 f"{profile.python_test_files[0]}"] if profile.python_test_files else []
            )
            commands.append(
                TestCommand(f"{PYTHON} -m pytest", CheckKind.TEST, "python", tuple(evidence))
            )
        if profile.unittest_evidence:
            commands.append(
                TestCommand(
                    f"{PYTHON} -m unittest discover",
                    CheckKind.TEST,
                    "python",
                    tuple(profile.unittest_evidence),
                    # fallback only when pytest is not the declared runner
                    gating=not profile.uses_pytest,
                )
            )

    # Node scripts (declared in package.json - never assumed)
    if "node" in profile.ecosystems:
        if _npm_script_real(profile, "test"):
            commands.append(
                TestCommand("npm test", CheckKind.TEST, "node",
                            (f"package.json scripts.test = {profile.npm_scripts['test']!r}",))
            )
        if _npm_script_real(profile, "build"):
            commands.append(
                TestCommand("npm run build", CheckKind.BUILD, "node",
                            ("package.json scripts.build",))
            )
        for script, kind in (("typecheck", CheckKind.TYPECHECK), ("type-check", CheckKind.TYPECHECK),
                             ("lint", CheckKind.LINT)):
            if _npm_script_real(profile, script):
                commands.append(
                    TestCommand(f"npm run {script}", kind, "node",
                                (f"package.json scripts.{script}",), gating=False)
                )

    # Makefile test target: only when no direct runner was found (make may do setup/network).
    if not any(c.kind is CheckKind.TEST for c in commands) and "test" in profile.make_targets:
        commands.append(TestCommand("make test", CheckKind.TEST, "make", ("Makefile target 'test'",)))

    # Python static checks: advisory, only when configured.
    if profile.ruff_config:
        commands.append(
            TestCommand("ruff check .", CheckKind.LINT, "python", (profile.ruff_config,), gating=False)
        )
    if profile.mypy_config:
        commands.append(
            TestCommand("mypy .", CheckKind.TYPECHECK, "python", (profile.mypy_config,), gating=False)
        )
    return commands


def related_python_tests(
    repo: RepositoryContext, changed_files: Sequence[str], profile: ProjectProfile
) -> list[str]:
    """Test files most likely to exercise ``changed_files`` (by name and by import)."""
    related: list[str] = []
    modules: set[str] = set()
    for changed in changed_files:
        path = Path(changed)
        if path.suffix != ".py":
            continue
        if _PY_TEST_FILE.match(path.name):
            related.append(changed)
            continue
        modules.add(path.stem if path.stem != "__init__" else path.parent.name)
    if not modules:
        return [f for f in dict.fromkeys(related) if (repo.root / f).is_file()]

    names = {f"test_{m}.py" for m in modules} | {f"{m}_test.py" for m in modules}
    import_re = re.compile(
        r"^\s*(?:from\s+[\w.]*\b(" + "|".join(map(re.escape, modules)) + r")\b[\w.]*\s+import"
        r"|import\s+[\w.]*\b(" + "|".join(map(re.escape, modules)) + r")\b)",
        re.MULTILINE,
    )
    for rel in profile.python_test_files:
        if Path(rel).name in names or import_re.search(_read(repo.root / rel, 200_000)):
            related.append(rel)
    return [f for f in dict.fromkeys(related) if (repo.root / f).is_file()]
