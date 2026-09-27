"""Basic relevance filtering: which repository files belong in an agent's context.

Signals, in priority order (scores):
  1. files explicitly mentioned in the task / evidence     100
  2. files modified during the current task                  80
  3. files returned by a search for the task's identifiers   60 (+5 per extra term)
  4. relevant test files (for mentioned/changed code)        50
  5. files imported by the files above                       40
  6. documentation mentioning the task's identifiers         20

Everything else is excluded. Only paths, reasons and a short symbol outline (signatures,
never bodies) are packaged - agents read file contents on demand through tools.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from harness.tools.repository import RepositoryContext
from harness.tools.search import SearchCodeTool
from harness.verification.detection import inspect_project, related_python_tests

SCORE_MENTIONED = 100
SCORE_MODIFIED = 80
SCORE_SEARCH = 60
SCORE_TEST = 50
SCORE_IMPORTED = 40
SCORE_DOC = 20

MAX_SEARCH_TERMS = 6
MAX_OUTLINE_LINES = 12
MAX_OUTLINED_FILES = 3
MAX_OUTLINE_BYTES = 300_000

_PATH_TOKEN = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.[A-Za-z][A-Za-z0-9]{0,4})(?=[\s,;:)'\"`\]]|$)")
_BACKTICK = re.compile(r"`([A-Za-z_][\w.]*)`")
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_COMMON = {
    "the", "and", "for", "with", "that", "this", "from", "into", "should", "function", "method",
    "class", "return", "returns", "value", "values", "file", "files", "code", "test", "tests",
    "make", "sure", "using", "use", "fix", "add", "implement", "feature", "library", "package",
    "module", "error", "bug", "task", "repository", "change", "update", "support", "print",
    "assert", "true", "false", "none", "self", "import", "python", "python3", "pytest", "status",
    "failure", "failed", "passed", "attempt", "repair", "exit", "command", "result", "summary",
}
_PY_IMPORT = re.compile(r"^\s*(?:from\s+(\.*[\w.]*)\s+import|import\s+([\w.]+))", re.MULTILINE)
_JS_IMPORT = re.compile(r"""(?:require\(\s*|from\s+)['"](\.{1,2}/[^'"]+)['"]""")
_SYMBOL = re.compile(
    r"^\s*(?:async\s+def|def|class|function|export\s+(?:default\s+)?(?:function|class|const)|"
    r"interface|type)\s+[A-Za-z_]"
)


@dataclass
class RelevantFile:
    path: str
    score: int
    reasons: list[str] = field(default_factory=list)
    outline: list[str] = field(default_factory=list)

    def add(self, score: int, reason: str) -> None:
        self.score = max(self.score, score) + (5 if self.reasons and score >= SCORE_SEARCH else 0)
        if reason not in self.reasons:
            self.reasons.append(reason)


def search_terms(texts: Iterable[str], limit: int = MAX_SEARCH_TERMS) -> list[str]:
    """Code identifiers worth searching for: `backticked`, called(), snake_case, CamelCase."""
    ranked: dict[str, int] = {}
    for text in texts:
        for term in _BACKTICK.findall(text):
            ranked[term.split(".")[-1]] = ranked.get(term.split(".")[-1], 0) + 3
        for term in _CALL.findall(text):
            ranked[term] = ranked.get(term, 0) + 2
        for term in _IDENT.findall(text):
            if ("_" in term.strip("_") or re.search(r"[a-z][A-Z]", term)) and len(term) >= 4:
                ranked[term] = ranked.get(term, 0) + 1
    terms = [t for t in ranked if t.lower() not in _COMMON and len(t) >= 3]
    return sorted(terms, key=lambda t: (-ranked[t], t))[:limit]


class RelevanceRanker:
    def __init__(self, repo: RepositoryContext, *, max_files: int = 8) -> None:
        self.repo = repo
        self.max_files = max_files
        self._search = SearchCodeTool(repo, max_results_cap=40)

    # --- signals ---------------------------------------------------------------------

    def mentioned_files(self, texts: Iterable[str]) -> list[str]:
        found: list[str] = []
        index: dict[str, list[str]] | None = None
        for text in texts:
            for token in _PATH_TOKEN.findall(text):
                token = token.strip("./") if token.startswith("./") else token
                candidate = self.repo.root / token
                try:
                    if candidate.is_file() and self.repo.resolve(token):
                        found.append(self.repo.relative(candidate))
                        continue
                except Exception:  # noqa: BLE001 - outside repo / invalid: not a mention
                    continue
                if index is None:
                    index = self._basename_index()
                name = token.rsplit("/", 1)[-1]
                candidates = index.get(name, [])
                if "/" in token:  # partial path, e.g. tools/base.py -> src/harness/tools/base.py
                    candidates = [c for c in candidates if c.endswith("/" + token)]
                if len(candidates) == 1:  # only unambiguous matches count as mentions
                    found.append(candidates[0])
        return list(dict.fromkeys(found))

    def _basename_index(self) -> dict[str, list[str]]:
        profile_files = self._search._iter_files(self.repo.root, None)  # noqa: SLF001
        index: dict[str, list[str]] = {}
        for path in profile_files[:20_000]:
            index.setdefault(path.name, []).append(self.repo.relative(path))
        return index

    def searched_files(self, terms: Sequence[str]) -> dict[str, list[str]]:
        hits: dict[str, list[str]] = {}
        for term in terms:
            result = self._search.run({"query": term, "max_results": 20, "context_lines": 0})
            if not result.ok:
                continue
            for file in result.data.get("files", []):
                hits.setdefault(file, []).append(term)
        return hits

    def imported_files(self, sources: Iterable[str]) -> dict[str, str]:
        imports: dict[str, str] = {}
        for rel in sources:
            path = self.repo.root / rel
            text = _read(path)
            if not text:
                continue
            if path.suffix == ".py":
                for dotted_from, dotted in _PY_IMPORT.findall(text):
                    for target in self._resolve_python_import(dotted_from or dotted, path):
                        imports.setdefault(target, rel)
            elif path.suffix in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
                for spec in _JS_IMPORT.findall(text):
                    for ext in ("", ".js", ".ts", ".tsx", ".jsx", "/index.js", "/index.ts"):
                        candidate = (path.parent / (spec + ext)).resolve()
                        if candidate.is_file() and candidate.is_relative_to(self.repo.root):
                            imports.setdefault(self.repo.relative(candidate), rel)
                            break
        return imports

    def _resolve_python_import(self, dotted: str, importer: Path) -> list[str]:
        if not dotted:
            return []
        level = len(dotted) - len(dotted.lstrip("."))
        parts = [p for p in dotted.lstrip(".").split(".") if p]
        bases = (
            [importer.parents[level - 1]] if level else [self.repo.root, self.repo.root / "src"]
        )
        out = []
        for base in bases:
            stem = base.joinpath(*parts) if parts else base
            for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
                if candidate.is_file() and candidate.resolve().is_relative_to(self.repo.root):
                    out.append(self.repo.relative(candidate))
                    break
        return out

    # --- ranking -----------------------------------------------------------------------

    def rank(
        self,
        texts: Sequence[str],
        *,
        changed_files: Sequence[str] = (),
        extra_terms: Sequence[str] = (),
    ) -> list[RelevantFile]:
        files: dict[str, RelevantFile] = {}

        def add(path: str, score: int, reason: str) -> None:
            files.setdefault(path, RelevantFile(path, 0)).add(score, reason)

        for path in self.mentioned_files(texts):
            add(path, SCORE_MENTIONED, "explicitly mentioned")
        for path in changed_files:
            if (self.repo.root / path).is_file():
                add(path, SCORE_MODIFIED, "modified during this task")
        terms = list(dict.fromkeys([*extra_terms, *search_terms(texts)]))[:MAX_SEARCH_TERMS]
        for path, hit_terms in self.searched_files(terms).items():
            if Path(path).suffix.lower() in (".md", ".rst", ".txt"):
                add(path, SCORE_DOC, f"documentation mentioning {', '.join(hit_terms[:3])}")
            else:
                add(path, SCORE_SEARCH, f"search match: {', '.join(hit_terms[:3])}")

        core = [f.path for f in files.values() if f.score >= SCORE_SEARCH]
        profile = inspect_project(self.repo)
        for test in related_python_tests(self.repo, core, profile):
            add(test, SCORE_TEST, "test for relevant code")
        for path, importer in self.imported_files(core).items():
            add(path, SCORE_IMPORTED, f"imported by {importer}")

        ranked = sorted(files.values(), key=lambda f: (-f.score, f.path))[: self.max_files]
        for item in ranked[:MAX_OUTLINED_FILES]:
            item.outline = outline(self.repo.root / item.path)
        return ranked


def _read(path: Path, limit: int = MAX_OUTLINE_BYTES) -> str:
    try:
        if path.is_file() and path.stat().st_size <= limit:
            raw = path.read_bytes()
            if b"\x00" not in raw[:4096]:
                return raw.decode("utf-8", errors="replace")
    except OSError:
        pass
    return ""


def outline(path: Path) -> list[str]:
    """Signature lines (def/class/function...) - never function bodies."""
    lines = []
    for number, line in enumerate(_read(path).splitlines(), start=1):
        if _SYMBOL.match(line):
            lines.append(f"{number}: {line.strip()[:120]}")
            if len(lines) >= MAX_OUTLINE_LINES:
                lines.append("[OUTPUT TRUNCATED: more symbols]")
                break
    return lines
