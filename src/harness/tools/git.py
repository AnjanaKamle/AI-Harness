"""Read-only Git tools: git_status, git_diff, git_log.

Destructive operations (push, reset, branch deletion...) are deliberately not implemented.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult, truncate_text
from harness.tools.policy import sanitized_environment
from harness.tools.repository import RepositoryContext

GIT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_DIFF_CHARS = 12_000
MAX_UNTRACKED_IN_DIFF = 20
MAX_UNTRACKED_BYTES = 100_000

_STATUS_NAMES = {
    "M": "modified",
    "T": "type_changed",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "U": "unmerged",
    "?": "untracked",
    "!": "ignored",
}


def run_git(
    repo: RepositoryContext, args: list[str], *, ok_codes: tuple[int, ...] = (0,)
) -> str:
    """Run a git command confined to ``repo.root``; raise ToolError on failure."""
    env = sanitized_environment()
    # Never let git discover a repository *above* the harness root.
    env["GIT_CEILING_DIRECTORIES"] = str(repo.root.parent)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    cmd = ["git", "--no-pager", "-c", "color.ui=never", "-c", "core.quotePath=false", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo.root,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        raise ToolError("git is not installed") from None
    except subprocess.TimeoutExpired:
        raise ToolError(f"git {args[0]} timed out") from None
    if proc.returncode not in ok_codes:
        message = proc.stderr.decode("utf-8", "replace").strip() or f"exit code {proc.returncode}"
        raise ToolError(f"git {args[0]} failed: {message[:500]}")
    return proc.stdout.decode("utf-8", "replace")


def ensure_git_repo(repo: RepositoryContext) -> None:
    try:
        top = run_git(repo, ["rev-parse", "--show-toplevel"]).strip()
    except ToolError:
        raise ToolError(f"Not a git repository: {repo.name}") from None
    if Path(top).resolve() != repo.root:
        raise ToolError("Repository root is not the top level of its git work tree")


def _has_head(repo: RepositoryContext) -> bool:
    try:
        run_git(repo, ["rev-parse", "--verify", "--quiet", "HEAD"])
        return True
    except ToolError:
        return False


class _GitTool(BaseTool):
    def __init__(self, repo: RepositoryContext) -> None:
        self.repo = repo

    def _pathspec(self, path: str | None) -> list[str]:
        if not path:
            return []
        return ["--", self.repo.relative(self.repo.resolve(path))]


class GitStatusTool(_GitTool):
    name = "git_status"
    description = "Show the current branch and changed/staged/untracked files (read-only)."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, **_: Any) -> ToolExecutionResult:
        ensure_git_repo(self.repo)
        raw = run_git(self.repo, ["status", "--porcelain=v1", "--branch", "-z"])
        records = raw.split("\0")
        branch: str | None = None
        files: list[dict[str, Any]] = []
        i = 0
        while i < len(records):
            record = records[i]
            i += 1
            if not record:
                continue
            if record.startswith("## "):
                branch = record[3:]
                continue
            index, worktree, path = record[0], record[1], record[3:]
            entry: dict[str, Any] = {
                "path": path,
                "index": index.strip() or None,
                "worktree": worktree.strip() or None,
                "status": _STATUS_NAMES.get(index if index not in " ?" else worktree, "changed"),
                "staged": index not in " ?!",
            }
            if index in "RC":  # rename/copy: the next record is the original path
                entry["from_path"] = records[i]
                i += 1
            files.append(entry)

        return ToolExecutionResult.success(
            self.name,
            {
                "branch": branch,
                "clean": not files,
                "files": files,
                "counts": {
                    "staged": sum(f["staged"] for f in files),
                    "unstaged": sum(1 for f in files if f["worktree"] and f["worktree"] != "?"),
                    "untracked": sum(1 for f in files if f["status"] == "untracked"),
                },
            },
        )


class GitDiffTool(_GitTool):
    name = "git_diff"
    description = (
        "Show the working-tree diff (or staged diff with staged=true) as a unified diff plus a "
        "per-file summary of added/removed lines. New untracked files are included. Use this "
        "to review your own changes before reporting success."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Limit to a file or directory."},
            "staged": {"type": "boolean"},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
            "include_untracked": {"type": "boolean"},
        },
        "additionalProperties": False,
    }

    def __init__(self, repo: RepositoryContext, max_chars: int = DEFAULT_MAX_DIFF_CHARS) -> None:
        super().__init__(repo)
        self.max_chars = max_chars

    def execute(
        self,
        path: str | None = None,
        staged: bool = False,
        context_lines: int = 3,
        include_untracked: bool = True,
        **_: Any,
    ) -> ToolExecutionResult:
        ensure_git_repo(self.repo)
        spec = self._pathspec(path)
        base = ["diff", "--no-ext-diff", "--no-color"] + (["--cached"] if staged else [])

        files: list[dict[str, Any]] = []
        for line in run_git(self.repo, [*base, "--numstat", *spec]).splitlines():
            added, deleted, name = line.split("\t", 2)
            files.append(
                {
                    "path": name,
                    "status": "modified",
                    "additions": int(added) if added.isdigit() else 0,
                    "deletions": int(deleted) if deleted.isdigit() else 0,
                    "binary": added == "-",
                }
            )
        patch = run_git(self.repo, [*base, f"-U{context_lines}", *spec])

        if include_untracked and not staged:
            untracked = run_git(
                self.repo, ["ls-files", "--others", "--exclude-standard", *spec]
            ).splitlines()
            for name in untracked[:MAX_UNTRACKED_IN_DIFF]:
                entry, text = self._untracked_diff(name, context_lines)
                files.append(entry)
                patch += text
            if len(untracked) > MAX_UNTRACKED_IN_DIFF:
                patch += f"\n[{len(untracked) - MAX_UNTRACKED_IN_DIFF} more untracked files]\n"

        diff, truncated = truncate_text(patch, self.max_chars)
        total_add = sum(f["additions"] for f in files)
        total_del = sum(f["deletions"] for f in files)
        return ToolExecutionResult.success(
            self.name,
            {
                "has_changes": bool(files),
                "summary": f"{len(files)} file(s) changed, +{total_add} -{total_del}",
                "files": files,
                "total_additions": total_add,
                "total_deletions": total_del,
                "staged": staged,
                "diff": diff,
                "truncated": truncated,
            },
        )

    def _untracked_diff(self, name: str, context_lines: int) -> tuple[dict[str, Any], str]:
        entry: dict[str, Any] = {
            "path": name, "status": "untracked", "additions": 0, "deletions": 0, "binary": False
        }
        target = self.repo.resolve(name)
        if target.stat().st_size > MAX_UNTRACKED_BYTES:
            return entry, f"\n[new file {name}: too large to show]\n"
        text = run_git(
            self.repo,
            ["diff", "--no-ext-diff", "--no-color", "--no-index", f"-U{context_lines}",
             "--", "/dev/null", name],
            ok_codes=(0, 1),
        )
        entry["additions"] = sum(
            1 for ln in text.splitlines() if ln.startswith("+") and not ln.startswith("+++")
        )
        entry["binary"] = "Binary files" in text
        return entry, text


class GitLogTool(_GitTool):
    name = "git_log"
    description = "List recent commits (hash, author, date, subject), optionally for one path."
    input_schema = {
        "type": "object",
        "properties": {
            "max_count": {"type": "integer", "minimum": 1, "maximum": 100},
            "path": {"type": "string"},
        },
        "additionalProperties": False,
    }

    def execute(self, max_count: int = 10, path: str | None = None, **_: Any) -> ToolExecutionResult:
        ensure_git_repo(self.repo)
        if not _has_head(self.repo):
            return ToolExecutionResult.success(self.name, {"commits": [], "count": 0})
        fmt = "%H%x1f%h%x1f%an%x1f%aI%x1f%s%x1e"
        raw = run_git(
            self.repo, ["log", f"-n{max_count}", f"--pretty=format:{fmt}", *self._pathspec(path)]
        )
        commits = []
        for record in raw.split("\x1e"):
            record = record.strip("\n")
            if not record:
                continue
            full, short, author, date, subject = record.split("\x1f", 4)
            commits.append(
                {"hash": full, "short": short, "author": author, "date": date, "subject": subject}
            )
        return ToolExecutionResult.success(self.name, {"commits": commits, "count": len(commits)})
