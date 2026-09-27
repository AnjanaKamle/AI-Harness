"""Deterministic planning: does this task need research, and what should be asked?

Research is needed when the task names an external library/framework/SDK or asks about
documentation. It is *required* when a named library has no local evidence (not declared
in project metadata, not imported in the code, not installed) - then the Coder cannot
safely proceed if research fails.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field

from harness.orchestrator.graph import NodeKind, TaskGraph, TaskNode
from harness.orchestrator.state import PlanStep, StepStatus
from harness.tools.repository import RepositoryContext
from harness.tools.search import SearchCodeTool
from harness.verification.detection import inspect_project

_KEYWORDS = r"(?:library|package|framework|sdk|module|client|api|toolkit)"
_NAME = r"[`'\"]?([A-Za-z][A-Za-z0-9_.-]*[A-Za-z0-9])[`'\"]?"
_LIB_PATTERNS = (
    re.compile(rf"\b{_NAME}\s+{_KEYWORDS}\b", re.IGNORECASE),  # "the requests library"
    re.compile(rf"\b{_KEYWORDS}\s+{_NAME}", re.IGNORECASE),  # "library X"
    re.compile(r"\b(?:use|using|with|via|adopt|integrate|migrate to|switch to)\s+(?:the\s+)?"
               r"[`'\"]([A-Za-z][A-Za-z0-9_.-]*)[`'\"]", re.IGNORECASE),  # use `X`
)
_DOC_HINT = re.compile(
    r"\b(documentation|docs|official api|latest version|deprecated|deprecation|changelog|"
    r"release notes|migration guide)\b",
    re.IGNORECASE,
)
_NOT_LIBRARIES = {
    "a", "an", "the", "this", "that", "our", "your", "new", "same", "standard", "external",
    "third-party", "python", "javascript", "typescript", "node", "rest", "public", "internal",
    "http", "web", "test", "tests", "existing", "following", "given", "its", "their", "json",
    "which", "some", "any", "each", "own", "client", "api", "sdk", "library", "package",
    "module", "framework", "toolkit", "code", "function", "method", "class",
    # common English words that precede/follow "library" etc. in task prose
    "to", "for", "in", "on", "of", "and", "or", "with", "from", "by", "as", "at", "into",
    "use", "using", "used", "uses", "via", "adopt", "integrate", "migrate", "switch", "add",
    "implement", "build", "create", "call", "import", "install", "update", "upgrade", "fix",
    "is", "are", "be", "it", "not", "no", "instead", "then", "so", "if", "when", "all",
    "helper", "wrapper", "utility", "shared", "local", "custom", "default",
    # generic references ("the relevant library", "a suitable package")
    "relevant", "appropriate", "correct", "right", "proper", "suitable", "necessary",
    "needed", "required", "specific", "particular", "underlying", "current", "latest",
    "other", "another", "such", "same", "used", "chosen", "preferred", "recommended",
}
MAX_LIBRARIES = 2
# "research ... if needed", "use the correct library API": research is wanted, but only if the
# project actually depends on an external library.
_CONDITIONAL_RESEARCH = re.compile(
    r"\b(research\b|look up\b|(?:correct|right|proper)\s+(?:library|api|usage)|library\s+api)",
    re.IGNORECASE,
)
_TEST_TOOLING = frozenset({"__future__", "pytest", "unittest", "mock", "hypothesis", "nose"})
_PY_IMPORT_TOP = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)[\w.]*\s+import|import\s+([A-Za-z_]\w*))",
                            re.MULTILINE)


@dataclass
class ResearchPlan:
    needed: bool
    required: bool
    reason: str
    libraries: list[str] = field(default_factory=list)
    api_questions: list[str] = field(default_factory=list)
    project_questions: list[str] = field(default_factory=list)
    local_evidence: dict[str, list[str]] = field(default_factory=dict)

    @property
    def questions(self) -> list[str]:
        return [*self.api_questions, *self.project_questions]

    @property
    def can_proceed_without_research(self) -> bool:
        """True when every named library has local evidence the Coder can inspect."""
        return not self.required or all(self.local_evidence.get(lib) for lib in self.libraries)

    def to_dict(self) -> dict[str, object]:
        return {
            "needed": self.needed,
            "required": self.required,
            "reason": self.reason,
            "libraries": self.libraries,
            "questions": self.questions,
            "local_evidence": self.local_evidence,
            "can_proceed_without_research": self.can_proceed_without_research,
        }


def extract_libraries(task: str) -> list[str]:
    found: list[str] = []
    for pattern in _LIB_PATTERNS:
        for name in pattern.findall(task):
            if name.lower() not in _NOT_LIBRARIES and name not in found:
                found.append(name)
    return found[:MAX_LIBRARIES]


def local_evidence(repo: RepositoryContext, library: str) -> list[str]:
    """Where the library already appears locally: metadata, imports, installed package."""
    evidence: list[str] = []
    profile = inspect_project(repo)
    needle = library.lower()
    for name in profile.metadata_files:
        try:
            if needle in (repo.root / name).read_text(encoding="utf-8", errors="replace").lower():
                evidence.append(f"declared in {name}")
        except OSError:
            continue
    module = library.replace("-", "_")
    pattern = (
        rf"^\s*(import|from)\s+{re.escape(module)}\b|require\(['\"]{re.escape(library)}['\"]\)"
        rf"|from\s+['\"]{re.escape(library)}['\"]"
    )
    search = SearchCodeTool(repo, max_results_cap=5).run(
        {"query": pattern, "regex": True, "ignore_case": True, "context_lines": 0}
    )
    if search.ok and search.data.get("files"):
        evidence.append("imported in " + ", ".join(search.data["files"][:3]))
    try:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module) and importlib.util.find_spec(module):
            evidence.append("installed in the Python environment")
    except (ImportError, ValueError):
        pass
    return evidence


def third_party_libraries(repo: RepositoryContext, limit: int = MAX_LIBRARIES) -> list[str]:
    """External libraries the project actually uses (imports / package.json dependencies),
    excluding the standard library and the project's own modules."""
    import json
    import os
    import sys

    from harness.tools.repository import is_ignored_dir

    counts: dict[str, int] = {}
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(repo.root):
        dirnames[:] = [d for d in dirnames if not is_ignored_dir(d) and d not in ("tests", "test")]
        for name in filenames:
            if (not name.endswith(".py") or name.startswith("test_") or name.endswith("_test.py")
                    or name == "conftest.py" or scanned >= 300):
                continue
            scanned += 1
            try:
                text = (repo.root / dirpath / name).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for a, b in _PY_IMPORT_TOP.findall(text):
                module = a or b
                if (module in sys.stdlib_module_names or module in _TEST_TOOLING
                        or (repo.root / f"{module}.py").exists() or (repo.root / module).is_dir()
                        or (repo.root / "src" / module).is_dir()):
                    continue
                counts[module] = counts.get(module, 0) + 1
    package = repo.root / "package.json"
    if package.is_file():
        try:
            deps = json.loads(package.read_text(encoding="utf-8")).get("dependencies", {})
            for dep in deps if isinstance(deps, dict) else {}:
                counts[dep] = counts.get(dep, 0) + 1
        except (OSError, json.JSONDecodeError):
            pass
    return sorted(counts, key=lambda m: (-counts[m], m))[:limit]


def plan_research(
    task: str, repo: RepositoryContext, *, force: bool | None = None
) -> ResearchPlan:
    if force is False:
        return ResearchPlan(False, False, "research disabled by caller")
    libraries = extract_libraries(task)
    doc_hint = bool(_DOC_HINT.search(task))
    if not libraries and not doc_hint and not force and _CONDITIONAL_RESEARCH.search(task):
        used = third_party_libraries(repo)
        if not used:
            return ResearchPlan(
                False, False,
                "research was requested if needed, but the project uses no external library",
            )
        plan = _plan_for_libraries(task, repo, used, doc_hint=False)
        plan.required = False  # conditional research never blocks the task
        plan.reason = f"research requested if needed; the project uses {', '.join(used)}"
        return plan
    if not libraries and not doc_hint and not force:
        return ResearchPlan(False, False, "task names no external library or documentation")
    return _plan_for_libraries(task, repo, libraries, doc_hint=doc_hint)


def _plan_for_libraries(
    task: str, repo: RepositoryContext, libraries: list[str], *, doc_hint: bool
) -> ResearchPlan:
    short_task = task.strip().splitlines()[0][:200]
    plan = ResearchPlan(
        needed=True,
        required=bool(libraries),
        reason=(
            f"task depends on external librar{'ies' if len(libraries) > 1 else 'y'} "
            f"{', '.join(libraries)}" if libraries else "task asks about documentation"
        ),
        libraries=libraries,
    )
    for lib in libraries:
        plan.api_questions.append(
            f"What is the supported {lib} API for this task: \"{short_task}\"? Identify the exact "
            "functions/classes, signatures, required parameters and version constraints."
        )
        plan.project_questions.append(
            f"How is {lib} used or declared in this repository (dependencies, existing imports, "
            "configuration), and how should the change fit the existing code?"
        )
        plan.local_evidence[lib] = local_evidence(repo, lib)
    if not libraries:
        plan.api_questions.append(f"What does the relevant documentation say about: \"{short_task}\"?")
    return plan


def build_plan(research: ResearchPlan) -> list[PlanStep]:
    """The executable plan for a task (Researcher -> Coder -> Researcher -> Coder -> Tester)."""
    steps: list[PlanStep] = []

    def add(description: str, agent: str, status: StepStatus = StepStatus.PENDING) -> None:
        steps.append(PlanStep(f"step-{len(steps) + 1}", description, agent, status))

    if not research.needed:
        add(f"Research not needed ({research.reason})", "researcher", StepStatus.SKIPPED)
    else:
        add("Determine the supported API: " + " | ".join(research.api_questions), "researcher")
    add("Inspect the repository", "coder")
    if research.project_questions:
        add("Clarify the project-specific implementation: "
            + " | ".join(research.project_questions), "researcher")
    add("Implement the required code changes", "coder")
    add("Run tests and verify the result", "tester")
    return steps


# --- Phase 5: task-graph planner ---------------------------------------------------------

_MUSIC_VERB = r"(?:play|put on|queue up|queue|listen to)"
# "No. 5" / "Op. 67" are not sentence ends inside a music query
_QUERY = r"(?P<query>(?:(?:No|Op|Nr|BWV|K)\.\s*\d+|[^.,;!?])+?)"
_MUSIC_REQUEST = re.compile(
    rf"(?:^|(?<=[.;!?])\s*|\b(?:and|please|also|then)\s+){_MUSIC_VERB}\s+{_QUERY}"
    r"(?=\s+(?:while|and then|then|as|during|before|after|and)\b|[.,;!?](?!\s*\d)|$)",
    re.IGNORECASE,
)
_MUSIC_CONTROL = re.compile(
    r"(?:^|(?<=[.;!?])\s*|\b(?:and|please|also|then)\s+)"
    r"(?P<command>(?:stop|pause|resume|unpause)\s+(?:the\s+)?music"
    r"|(?:set|turn)\s+(?:the\s+)?(?:music\s+)?volume\s+(?:up\s+|down\s+)?to\s+\d{1,3}\s*%?"
    r"|(?:set\s+)?(?:the\s+)?music\s+volume\s+to\s+\d{1,3}\s*%?)",
    re.IGNORECASE,
)
_NOT_MUSIC = re.compile(
    r"\b(button|function|method|handler|test|tests|bug|file|class|endpoint|route|api|sound effect|"
    r"animation|role|game|store)\b",
    re.IGNORECASE,
)
_CONNECTOR = re.compile(r"^\s*(?:while|and then|then|as|during|and)\s+(?:you\s+)?", re.IGNORECASE)
# "... while you work" / "as you code" left behind once the music clause is removed
_WORK_CLAUSE = re.compile(
    r"\s*,?\s*\b(?:and\s+)?(?:while|as|during)\s+(?:you|we|I)?\s*(?:'re\s+|are\s+)?"
    r"(?:work|working|code|coding|do (?:that|this|it)|go|that|this)\b",
    re.IGNORECASE,
)
_CODING_INTENT = re.compile(
    r"\b(fix|implement|add|update|refactor|change|write|create|remove|delete|rename|debug|bug|"
    r"test|tests|inspect|build|make|support|migrate|use|improve|correct|repair|resolve|handle)\w*\b",
    re.IGNORECASE,
)


@dataclass
class PlannedTask:
    graph: TaskGraph
    original_task: str
    coding_task: str | None
    music_command: str | None
    research: ResearchPlan

    def to_dict(self) -> dict[str, object]:
        return {
            "coding_task": self.coding_task,
            "music_command": self.music_command,
            "research": self.research.to_dict(),
            "graph": self.graph.to_dict(),
        }


def split_music_request(task: str) -> tuple[str | None, str]:
    """Separate a music request from the coding work: ('play Beethoven', 'Inspect and fix...').

    Returns (music command or None, remaining coding text)."""
    control = _MUSIC_CONTROL.search(task)
    if control:
        command = control.group("command").strip().lower()
        command = re.sub(r"\s+(?:the\s+)?music$", " music", command, flags=re.IGNORECASE)
        rest = task[: control.start()] + task[control.end():]
        return command, _tidy(rest)
    match = _MUSIC_REQUEST.search(task)
    if not match or _NOT_MUSIC.search(match.group("query")):
        return None, task.strip()
    command = f"play {match.group('query').strip()}"
    rest = task[: match.start()] + task[match.end():]
    return command, _tidy(rest)


def _tidy(text: str) -> str:
    text = _WORK_CLAUSE.sub("", text)
    text = _CONNECTOR.sub("", text.strip(" ,;")).strip()
    text = re.sub(r"\s+(?:and|while|as)\s*([.!?]?)$", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([.,;!?])", r"\1", re.sub(r"\s{2,}", " ", text)).strip(" ,;")
    return text[:1].upper() + text[1:] if text else ""


class Planner:
    """Turns a task + repository into a dependency-aware TaskGraph.

    Research is included only when the task needs external knowledge; music (if requested)
    is an independent optional node; coding work flows Inspect (+Research) -> Implement ->
    Test -> Verify. Repairs are added to the graph at run time by the controller.
    """

    def plan(
        self, task: str, repo: RepositoryContext, *, research: bool | None = None
    ) -> PlannedTask:
        music_command, coding_text = split_music_request(task)
        coding_task = coding_text if coding_text and _CODING_INTENT.search(coding_text) else None
        research_plan = (
            plan_research(coding_task, repo, force=research) if coding_task
            else ResearchPlan(False, False, "no coding work requested")
        )
        graph = TaskGraph()
        if music_command:
            graph.add(TaskNode(
                "music-1", f"Music: {music_command}", "music", NodeKind.MUSIC,
                priority=1, required=coding_task is None, payload={"command": music_command},
            ))
        if coding_task:
            inspect = graph.add(TaskNode(
                "inspect-1", "Inspect the repository", "coder", NodeKind.INSPECT, priority=5,
            ))
            baseline = graph.add(TaskNode(
                "baseline-1", "Run the existing tests before any change (baseline)", "tester",
                NodeKind.BASELINE, priority=5, required=False,
            ))
            implement_deps = [inspect.id, baseline.id]
            if research_plan.needed:
                research_required = research_plan.required and not research_plan.can_proceed_without_research
                api = graph.add(TaskNode(
                    "research-1", "Research: " + " | ".join(research_plan.api_questions)[:300],
                    "researcher", NodeKind.RESEARCH, priority=5, required=research_required,
                    payload={"questions": research_plan.api_questions, "phase": "api"},
                ))
                implement_deps.append(api.id)
                if research_plan.project_questions:
                    project = graph.add(TaskNode(
                        "research-2",
                        "Research (project-specific): "
                        + " | ".join(research_plan.project_questions)[:300],
                        "researcher", NodeKind.RESEARCH, dependencies=[api.id, inspect.id],
                        priority=4, required=research_required,
                        payload={"questions": research_plan.project_questions, "phase": "project"},
                    ))
                    implement_deps.append(project.id)
            implement = graph.add(TaskNode(
                "implement-1", "Implement the required code changes", "coder", NodeKind.IMPLEMENT,
                dependencies=implement_deps, priority=3, payload={"task": coding_task},
            ))
            test = graph.add(TaskNode(
                "test-1", "Run tests and verify the change", "tester", NodeKind.TEST,
                dependencies=[implement.id], priority=2,
            ))
            graph.add(TaskNode(
                "verify-1", "Verify the final state (verification gate)", "orchestrator",
                NodeKind.VERIFY, dependencies=[test.id], priority=0,
            ))
        elif music_command:
            graph.add(TaskNode(
                "verify-1", "Verify the final state (verification gate)", "orchestrator",
                NodeKind.VERIFY, dependencies=["music-1"], priority=0,
            ))
        graph.validate()
        return PlannedTask(graph, task, coding_task, music_command, research_plan)
