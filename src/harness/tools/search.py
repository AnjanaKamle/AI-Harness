"""search_code: find where something is defined or used before reading whole files.

Uses ripgrep when it is installed, otherwise a pure-Python walk. Both backends return
the same structure.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult
from harness.tools.policy import sanitized_environment
from harness.tools.repository import IGNORED_DIRS, RepositoryContext, is_ignored_dir

MAX_LINE_CHARS = 300
MAX_SEARCH_FILE_BYTES = 1_000_000
RIPGREP_TIMEOUT_SECONDS = 30


@dataclass
class _Hit:
    file: str
    line: int
    text: str


def _clip(line: str) -> str:
    line = line.rstrip("\r\n")
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + "…"


def _file_glob(file_type: str | None) -> str | None:
    """'py', '.py' or '*.py' -> '*.py'; anything with a wildcard is used as-is."""
    if not file_type:
        return None
    if any(ch in file_type for ch in "*?["):
        return file_type
    return "*." + file_type.lstrip(".")


class SearchCodeTool(BaseTool):
    name = "search_code"
    description = (
        "Search repository files for a literal string (or a regex when regex=true). Returns "
        "file, line number, the matching line and a few lines of context. Use this to find "
        "relevant files before reading them."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "path": {"type": "string", "description": "File or directory to search (default '.')."},
            "file_type": {"type": "string", "description": "Extension or glob, e.g. 'py' or '*.ts'."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 500},
            "context_lines": {"type": "integer", "minimum": 0, "maximum": 10},
            "regex": {"type": "boolean"},
            "ignore_case": {"type": "boolean"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        repo: RepositoryContext,
        use_ripgrep: bool | None = None,
        *,
        max_results_cap: int | None = None,
    ) -> None:
        self.repo = repo
        self.max_results_cap = max_results_cap
        self.ripgrep = shutil.which("rg") if use_ripgrep is not False else None
        if use_ripgrep and self.ripgrep is None:
            raise ValueError("ripgrep requested but 'rg' is not on PATH")

    def execute(
        self,
        query: str,
        path: str = ".",
        file_type: str | None = None,
        max_results: int = 50,
        context_lines: int = 2,
        regex: bool = False,
        ignore_case: bool = False,
        **_: Any,
    ) -> ToolExecutionResult:
        if not query:
            raise ToolError("query must not be empty")
        if self.max_results_cap is not None:
            max_results = min(max_results, self.max_results_cap)
        base = self.repo.resolve(path)
        if not base.exists():
            raise ToolError(f"Path not found: {path}")
        pattern: re.Pattern[str] | None = None
        if regex:
            try:
                pattern = re.compile(query, re.IGNORECASE if ignore_case else 0)
            except re.error as exc:
                raise ToolError(f"Invalid regex: {exc}") from exc

        glob = _file_glob(file_type)
        if self.ripgrep:
            engine = "ripgrep"
            hits, truncated = self._ripgrep(query, base, glob, max_results, regex, ignore_case)
        else:
            engine = "python"
            hits, truncated = self._python(query, pattern, base, glob, max_results, ignore_case)

        matches = self._with_context(hits, context_lines)
        return ToolExecutionResult.success(
            self.name,
            {
                "query": query,
                "engine": engine,
                "matches": matches,
                "count": len(matches),
                "files": sorted({m["file"] for m in matches}),
                "truncated": truncated,
            },
        )

    # --- backends --------------------------------------------------------------------

    def _ripgrep(
        self,
        query: str,
        base: Path,
        glob: str | None,
        max_results: int,
        regex: bool,
        ignore_case: bool,
    ) -> tuple[list[_Hit], bool]:
        assert self.ripgrep is not None
        cmd = [self.ripgrep, "--json", "--no-config", "--hidden", "--sort", "path"]
        cmd += ["--max-columns", "2000"]
        if not regex:
            cmd.append("--fixed-strings")
        if ignore_case:
            cmd.append("--ignore-case")
        if glob:
            cmd += ["--glob", glob]
        for name in sorted(IGNORED_DIRS):
            cmd += ["--glob", f"!{name}"]
        cmd += ["--", query, self.repo.relative(base)]

        cmd[-3:-3] = ["--max-count", str(max_results + 1), "--max-filesize", "1M"]
        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo.root,
                env=sanitized_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=RIPGREP_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"ripgrep timed out after {RIPGREP_TIMEOUT_SECONDS}s") from None
        if proc.returncode not in (0, 1):  # 1 = no matches
            raise ToolError(f"ripgrep failed: {proc.stderr.strip()[:500]}")
        hits: list[_Hit] = []
        truncated = False
        for raw in proc.stdout.splitlines():
            event = json.loads(raw)
            if event.get("type") != "match":
                continue
            data = event["data"]
            text = data.get("lines", {}).get("text")
            file = data.get("path", {}).get("text")
            if text is None or file is None:  # non-UTF-8 content
                continue
            if len(hits) >= max_results:
                truncated = True
                break
            hits.append(_Hit(self.repo.relative(file), data["line_number"], _clip(text)))
        return hits, truncated

    def _python(
        self,
        query: str,
        pattern: re.Pattern[str] | None,
        base: Path,
        glob: str | None,
        max_results: int,
        ignore_case: bool,
    ) -> tuple[list[_Hit], bool]:
        needle = query.lower() if ignore_case else query

        def matches(line: str) -> bool:
            if pattern is not None:
                return pattern.search(line) is not None
            return needle in (line.lower() if ignore_case else line)

        hits: list[_Hit] = []
        for file in self._iter_files(base, glob):
            lines = self._read_lines(file)
            for number, line in enumerate(lines or [], start=1):
                if matches(line):
                    if len(hits) >= max_results:
                        return hits, True
                    hits.append(_Hit(self.repo.relative(file), number, _clip(line)))
        return hits, False

    def _iter_files(self, base: Path, glob: str | None) -> list[Path]:
        if base.is_file():
            return [base]
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if not is_ignored_dir(d))
            for filename in sorted(filenames):
                if glob and not fnmatch.fnmatch(filename, glob):
                    continue
                found.append(Path(dirpath, filename))
        return found

    # --- helpers ---------------------------------------------------------------------

    def _read_lines(self, file: Path) -> list[str] | None:
        """Lines of a searchable text file, or None (outside repo, binary, huge, non-UTF-8)."""
        try:
            resolved = self.repo.resolve(file)
            if resolved.stat().st_size > MAX_SEARCH_FILE_BYTES:
                return None
            raw = resolved.read_bytes()
        except (ToolError, OSError):
            return None
        if b"\x00" in raw[:8192]:
            return None
        try:
            return raw.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            return None

    def _with_context(self, hits: list[_Hit], context_lines: int) -> list[dict[str, Any]]:
        cache: dict[str, list[str] | None] = {}
        out: list[dict[str, Any]] = []
        for hit in hits:
            entry: dict[str, Any] = {"file": hit.file, "line": hit.line, "text": hit.text}
            if context_lines:
                if hit.file not in cache:
                    cache[hit.file] = self._read_lines(self.repo.root / hit.file)
                lines = cache[hit.file] or []
                i = hit.line - 1
                entry["before"] = [_clip(x) for x in lines[max(0, i - context_lines) : i]]
                entry["after"] = [_clip(x) for x in lines[i + 1 : i + 1 + context_lines]]
            out.append(entry)
        return out
