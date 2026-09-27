"""Filesystem tools: list_files, read_file, write_file, edit_file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult, truncate_text
from harness.tools.repository import RepositoryContext, bump_generation, is_ignored_dir

MAX_READ_BYTES = 2_000_000
DEFAULT_MAX_READ_CHARS = 40_000


def read_text(path: Path, display: str) -> str:
    """Read a UTF-8 text file with size and binary guards."""
    if not path.exists():
        raise ToolError(f"File not found: {display}")
    if not path.is_file():
        raise ToolError(f"Not a regular file: {display}")
    size = path.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(f"File too large to read ({size} bytes > {MAX_READ_BYTES}): {display}")
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise ToolError(f"Binary file, not readable as text: {display}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"File is not valid UTF-8: {display}") from exc


class _RepoTool(BaseTool):
    def __init__(self, repo: RepositoryContext) -> None:
        self.repo = repo


class ListFilesTool(_RepoTool):
    name = "list_files"
    description = (
        "List files under a directory of the repository (relative paths). Skips .git and "
        "generated directories such as node_modules, .venv and __pycache__. Directories are "
        "suffixed with '/' in non-recursive mode."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory relative to the repo root."},
            "recursive": {"type": "boolean", "description": "Walk subdirectories (default true)."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 2000},
        },
        "additionalProperties": False,
    }

    def execute(
        self, path: str = ".", recursive: bool = True, max_results: int = 300, **_: Any
    ) -> ToolExecutionResult:
        base = self.repo.resolve(path)
        if not base.exists():
            raise ToolError(f"Directory not found: {path}")
        if not base.is_dir():
            raise ToolError(f"Not a directory: {path}")

        entries: list[str] = []
        truncated = False
        if recursive:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(d for d in dirnames if not is_ignored_dir(d))
                for filename in sorted(filenames):
                    if len(entries) >= max_results:
                        truncated = True
                        break
                    entries.append(self.repo.relative(Path(dirpath, filename)))
                if truncated:
                    break
        else:
            for child in sorted(base.iterdir(), key=lambda p: p.name):
                if child.is_dir() and is_ignored_dir(child.name):
                    continue
                if len(entries) >= max_results:
                    truncated = True
                    break
                rel = self.repo.relative(child)
                entries.append(rel + "/" if child.is_dir() else rel)

        return ToolExecutionResult.success(
            self.name,
            {
                "path": self.repo.relative(base),
                "entries": entries,
                "count": len(entries),
                "truncated": truncated,
            },
        )


class ReadFileTool(_RepoTool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the repository. Optionally restrict to a 1-based inclusive "
        "line range. Large outputs are truncated; use start_line/end_line to page."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, repo: RepositoryContext, max_chars: int = DEFAULT_MAX_READ_CHARS) -> None:
        super().__init__(repo)
        self.max_chars = max_chars

    def execute(
        self, path: str, start_line: int | None = None, end_line: int | None = None, **_: Any
    ) -> ToolExecutionResult:
        target = self.repo.resolve(path)
        rel = self.repo.relative(target)
        text = read_text(target, rel)
        lines = text.splitlines(keepends=True)
        total = len(lines)

        start = start_line or 1
        end = min(end_line or total, total)
        if start_line is not None and total and start > total:
            raise ToolError(f"start_line {start} is beyond end of file ({total} lines)")
        if end_line is not None and start_line is not None and end_line < start_line:
            raise ToolError("end_line must be >= start_line")

        selected = "".join(lines[start - 1 : end]) if total else ""
        content, truncated = truncate_text(selected, self.max_chars)
        return ToolExecutionResult.success(
            self.name,
            {
                "path": rel,
                "content": content,
                "start_line": start if total else 0,
                "end_line": end,
                "total_lines": total,
                "size_bytes": target.stat().st_size,
                "truncated": truncated,
            },
        )


class WriteFileTool(_RepoTool):
    name = "write_file"
    description = (
        "Create or overwrite a text file in the repository with the given full content. "
        "Parent directories are created. Prefer edit_file for changes to existing files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "create_dirs": {"type": "boolean", "description": "Create parent dirs (default true)."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def execute(
        self, path: str, content: str, create_dirs: bool = True, **_: Any
    ) -> ToolExecutionResult:
        target = self.repo.resolve_for_write(path)
        rel = self.repo.relative(target)
        if target.is_dir():
            raise ToolError(f"Path is a directory: {rel}")
        if not target.parent.exists():
            if not create_dirs:
                raise ToolError(f"Parent directory does not exist: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)

        existed = target.exists()
        old_size = target.stat().st_size if existed else 0
        old_bytes = target.read_bytes() if existed else None
        new_bytes = content.encode("utf-8")
        changed = old_bytes != new_bytes
        if changed:
            target.write_bytes(new_bytes)
            bump_generation(self.repo.root)
        return ToolExecutionResult.success(
            self.name,
            {
                "path": rel,
                "created": not existed,
                "changed": changed,
                "old_size": old_size,
                "new_size": len(new_bytes),
            },
        )


class EditFileTool(_RepoTool):
    name = "edit_file"
    description = (
        "Replace an exact text snippet in an existing file. old_text must match the file "
        "exactly (including whitespace) and occur exactly expected_occurrences times "
        "(default 1); otherwise nothing is changed. Include enough surrounding lines in "
        "old_text to make it unique."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "expected_occurrences": {"type": "integer", "minimum": 1},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_occurrences: int = 1,
        **_: Any,
    ) -> ToolExecutionResult:
        target = self.repo.resolve_for_write(path)
        rel = self.repo.relative(target)
        original = read_text(target, rel)
        if not old_text:
            raise ToolError("old_text must not be empty")

        occurrences = original.count(old_text)
        base = {"path": rel, "changed": False, "occurrences": occurrences}
        if occurrences == 0:
            return ToolExecutionResult.failure(
                self.name, f"old_text not found in {rel}; re-read the file and copy it exactly", base
            )
        if occurrences != expected_occurrences:
            return ToolExecutionResult.failure(
                self.name,
                f"old_text occurs {occurrences} times in {rel}, expected {expected_occurrences}; "
                "add surrounding context to make it unique",
                base,
            )

        updated = original.replace(old_text, new_text)
        old_size = len(original.encode("utf-8"))
        new_size = len(updated.encode("utf-8"))
        changed = updated != original
        if changed:
            target.write_text(updated, encoding="utf-8")
            bump_generation(self.repo.root)
        return ToolExecutionResult.success(
            self.name,
            {**base, "changed": changed, "old_size": old_size, "new_size": new_size},
        )
