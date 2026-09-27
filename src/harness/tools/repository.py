"""RepositoryContext: the single place that confines tools to one repository root."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from harness.tools.base import ToolError

# Directories never listed or searched, and never written into.
IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "htmlcov",
        ".next",
        ".cache",
    }
)
PROTECTED_DIRS: frozenset[str] = frozenset({".git"})


class PathOutsideRepositoryError(ToolError):
    pass


# Incremented whenever a tool may have changed files under a root; lets read-only analyses
# (e.g. project detection) cache results safely between modifications.
_GENERATIONS: dict[Path, int] = {}


def bump_generation(root: Path) -> None:
    _GENERATIONS[root] = _GENERATIONS.get(root, 0) + 1


def generation(root: Path) -> int:
    return _GENERATIONS.get(root, 0)


def is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(".egg-info")


@dataclass(frozen=True)
class RepositoryContext:
    root: Path

    def __post_init__(self) -> None:
        resolved = Path(self.root).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Repository root does not exist or is not a directory: {resolved}")
        object.__setattr__(self, "root", resolved)

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def is_git_repo(self) -> bool:
        return (self.root / ".git").exists()

    def resolve(self, path: str | os.PathLike[str] = ".") -> Path:
        """Absolute, symlink-resolved path that is guaranteed to be inside the root.

        Relative paths are taken relative to the root. Absolute paths are accepted only if
        they already point inside it. ``..`` segments and symlinks are resolved *before* the
        containment check, so neither can escape.
        """
        raw = os.fspath(path)
        if "\x00" in raw:
            raise PathOutsideRepositoryError("Path contains a NUL byte")
        if raw.startswith("~"):
            raise PathOutsideRepositoryError(f"Home-relative paths are not allowed: {raw!r}")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise PathOutsideRepositoryError(f"Path escapes the repository root: {raw!r}")
        return resolved

    def resolve_for_write(self, path: str | os.PathLike[str]) -> Path:
        resolved = self.resolve(path)
        if resolved == self.root:
            raise ToolError("Cannot write to the repository root itself")
        parts = resolved.relative_to(self.root).parts
        if any(part in PROTECTED_DIRS for part in parts):
            raise ToolError(f"Refusing to modify protected path: {self.relative(resolved)}")
        return resolved

    def relative(self, path: str | os.PathLike[str]) -> str:
        """Repository-relative POSIX path ('.' for the root). Does not follow symlinks for
        paths that are already lexically inside the root."""
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            rel = candidate.relative_to(self.root)
        except ValueError:
            rel = candidate.resolve().relative_to(self.root)
        return rel.as_posix()
