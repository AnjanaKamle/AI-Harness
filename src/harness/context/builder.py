"""ContextBuilder: a small, targeted context package per agent.

Nothing is sent wholesale. Each agent gets only what its job needs, within a character
budget; sections are added by priority and anything cut is marked [OUTPUT TRUNCATED].

  Coder       task, plan, relevant repository info & files, relevant research, current failures
  Tester      task, changed files, test configuration, previous failure, requirements
  Researcher  research questions, project metadata, only the code that mentions the topic
  Music       playback command and music parameters - nothing else
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from harness.context.manager import ContextCategory, ContextEntry, ContextManager
from harness.context.relevance import RelevanceRanker, RelevantFile
from harness.context.summaries import (
    summarize_code_changes,
    summarize_failures,
    summarize_plan,
    summarize_research,
    summarize_tests,
)
from harness.context.usage import estimated_tokens
from harness.orchestrator.state import AgentState
from harness.tools.base import TRUNCATION_MARKER, truncate_text
from harness.tools.repository import RepositoryContext
from harness.verification.detection import detect_commands, inspect_project

DEFAULT_MAX_CONTEXT_CHARS = 16_000
MIN_SECTION_CHARS = 300

VERIFICATION_REQUIREMENTS = (
    "VERIFIED_SUCCESS requires: (1) code changes exist, (2) relevant verification actually "
    "executed, (3) verification passed, (4) no unresolved critical error. Report evidence "
    "only from commands that ran; never modify source files."
)


@dataclass
class ContextSection:
    title: str
    content: str
    priority: int  # lower = more important
    max_chars: int | None = None


@dataclass
class ContextPackage:
    agent: str
    max_chars: int
    sections: list[ContextSection] = field(default_factory=list)
    relevant_files: list[RelevantFile] = field(default_factory=list)
    _rendered: str | None = None
    _stats: dict[str, Any] = field(default_factory=dict)

    def add(self, title: str, content: str, priority: int, max_chars: int | None = None) -> None:
        if content and content.strip():
            self.sections.append(ContextSection(title, content.strip(), priority, max_chars))
            self._rendered = None

    def render(self) -> str:
        if self._rendered is not None:
            return self._rendered
        budget = self.max_chars
        included: list[tuple[int, str, str]] = []
        dropped: list[str] = []
        truncated: list[str] = []
        for index, section in sorted(
            enumerate(self.sections), key=lambda item: (item[1].priority, item[0])
        ):
            header = f"## {section.title}\n"
            available = budget - len(header) - 2
            limit = min(section.max_chars or available, available)
            if limit < MIN_SECTION_CHARS and len(section.content) > max(limit, 0):
                dropped.append(section.title)
                continue
            body, cut = truncate_text(section.content, limit)
            if cut:
                truncated.append(section.title)
            included.append((index, section.title, header + body))
            budget -= len(header) + len(body) + 2
        # Keep the author's section order in the output, regardless of priority.
        parts = [text for _, _, text in sorted(included)]
        if dropped:
            parts.append(f"{TRUNCATION_MARKER}: sections omitted to fit the context budget: "
                         f"{', '.join(dropped)}]")
        self._rendered = "\n\n".join(parts)
        self._stats = {
            "agent": self.agent,
            "context_chars": len(self._rendered),
            "estimated_context_tokens": estimated_tokens(len(self._rendered)),
            "max_context_chars": self.max_chars,
            "sections": [t for _, t, _ in sorted(included)],
            "truncated_sections": truncated,
            "dropped_sections": dropped,
            "relevant_files": [f.path for f in self.relevant_files],
        }
        return self._rendered

    @property
    def stats(self) -> dict[str, Any]:
        self.render()
        return dict(self._stats)


def _relevant_files_text(files: Sequence[RelevantFile]) -> str:
    if not files:
        return "No specific files identified yet - use search_code to locate relevant code."
    lines = []
    for f in files:
        lines.append(f"- {f.path}  ({'; '.join(f.reasons)})")
        lines.extend(f"    {o}" for o in f.outline)
    return "\n".join(lines)


def _changed_files(state: AgentState) -> list[str]:
    files: list[str] = []
    for change in state.code_changes:
        files.extend(change.get("files_changed", []))
    return list(dict.fromkeys(files))


class ContextBuilder:
    def __init__(
        self,
        repo: RepositoryContext,
        context: ContextManager | None = None,
        *,
        max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        max_files: int = 8,
    ) -> None:
        self.repo = repo
        self.context = context
        self.max_chars = max_chars
        self.ranker = RelevanceRanker(repo, max_files=max_files)

    # --- shared pieces -------------------------------------------------------------------

    def repository_facts(self, *, include_commands: bool = True) -> str:
        profile = inspect_project(self.repo)
        facts = {
            "name": self.repo.name,
            "git_repository": self.repo.is_git_repo,
            "ecosystems": profile.ecosystems or ["unknown"],
            "metadata_files": profile.metadata_files,
        }
        if include_commands:
            facts["verification_commands"] = [c.command for c in detect_commands(self.repo, profile)]
        return "\n".join(f"{k}: {v}" for k, v in facts.items())

    def relevant_files(
        self, texts: Sequence[str], state: AgentState | None, extra_terms: Sequence[str] = ()
    ) -> list[RelevantFile]:
        changed = _changed_files(state) if state else []
        files = self.ranker.rank(texts, changed_files=changed, extra_terms=extra_terms)
        if self.context is not None and state is not None:
            for f in files:
                self.context.put(
                    ContextEntry(
                        ContextCategory.RELEVANT_FILE,
                        f"{state.task_id}:{f.path}",
                        f"{f.path}: {'; '.join(f.reasons)}",
                        source="context_builder",
                        metadata={"score": f.score, "reasons": f.reasons},
                        task_id=state.task_id,
                    )
                )
        return files

    @staticmethod
    def research_text(state: AgentState) -> str:
        usable = [f for f in state.research_findings if f.get("kind") != "UNCERTAINTY"]
        uncertain = [f for f in state.research_findings if f.get("kind") == "UNCERTAINTY"]
        text = summarize_research(usable)
        if uncertain:
            text += "\nOpen uncertainties (do not treat as facts):\n" + "\n".join(
                f"- {f.get('finding')}" for f in uncertain[:5]
            )
        return text.strip()

    # --- per agent ---------------------------------------------------------------------

    def for_coder(
        self, task: str, state: AgentState, available_tools: Sequence[str] = ()
    ) -> ContextPackage:
        package = ContextPackage("coder", self.max_chars)
        package.add("TASK", task, priority=0, max_chars=9_000)
        if state.task != task:
            package.add("ORIGINAL TASK", state.task, priority=0, max_chars=2_000)
        package.add("PLAN", summarize_plan(state.plan), priority=3, max_chars=1_500)
        if self.context is not None:
            inspection = self.context.get(ContextCategory.REPOSITORY, f"{state.task_id}:inspection")
            if inspection is not None:
                package.add("REPOSITORY INSPECTION (from the inspection step)",
                            inspection.content, priority=2, max_chars=1_500)
        package.add("REPOSITORY", self.repository_facts(), priority=2, max_chars=1_200)

        research = self.research_text(state)
        terms = [f.get("technical_details", "") for f in state.research_findings][:3]
        files = self.relevant_files([task, state.task, research, *terms], state)
        package.relevant_files = files
        package.add("RELEVANT FILES (paths, reasons, signatures - read them with tools)",
                    _relevant_files_text(files), priority=1, max_chars=3_500)
        package.add("RESEARCH FINDINGS (verified by the Researcher)", research, priority=1,
                    max_chars=4_000)

        current = []
        if state.failure_history:
            current.append(summarize_failures(state.failure_history))
        if state.failures:
            latest = state.failures[-1]
            current.append(
                f"Latest agent failure ({latest.get('agent')}): "
                + "; ".join(map(str, latest.get("errors", [])[:5]))
            )
        package.add("CURRENT FAILURES", "\n".join(current), priority=1, max_chars=4_000)
        if state.baseline and state.baseline.get("available"):
            b = state.baseline
            failing = b.get("failed_tests") or []
            package.add(
                "BASELINE (tests before any change)",
                f"{b.get('command', '').split(' -rfE')[0]} -> {b.get('status')} "
                f"(passed={b.get('passed')}, failed={b.get('failed')})"
                + ("\nalready failing: " + ", ".join(failing[:10]) if failing else ""),
                priority=2, max_chars=1_200,
            )
        package.add("PREVIOUS CHANGES", summarize_code_changes(state.code_changes), priority=4,
                    max_chars=800)
        package.add("TEST HISTORY", summarize_tests(state.test_results), priority=4,
                    max_chars=800)
        if available_tools:
            package.add("AVAILABLE TOOLS", ", ".join(available_tools), priority=0)
        return package

    def for_tester(self, state: AgentState) -> ContextPackage:
        package = ContextPackage("tester", self.max_chars)
        package.add("TASK", state.task, priority=0, max_chars=2_000)
        changed = _changed_files(state)
        package.add("CHANGED FILES", "\n".join(changed) or "none reported", priority=0)
        profile = inspect_project(self.repo)
        commands = detect_commands(self.repo, profile)
        package.add(
            "TEST CONFIGURATION",
            "\n".join(
                f"- {c.command} [{c.kind.value}, {'gating' if c.gating else 'advisory'}] "
                f"evidence: {'; '.join(c.evidence)[:200]}"
                for c in commands
            ) or "No verification command detected from repository evidence.",
            priority=1,
        )
        if state.failure_history:
            package.add("PREVIOUS FAILURE", summarize_failures(state.failure_history[-1:]),
                        priority=2, max_chars=3_000)
        package.add("VERIFICATION REQUIREMENTS", VERIFICATION_REQUIREMENTS, priority=0)
        return package

    def for_researcher(self, questions: Sequence[str], state: AgentState | None = None,
                       topics: Sequence[str] = ()) -> ContextPackage:
        package = ContextPackage("researcher", self.max_chars)
        package.add("RESEARCH QUESTIONS", "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)),
                    priority=0, max_chars=3_000)
        profile = inspect_project(self.repo)
        package.add("PROJECT METADATA", self._dependency_text(profile.metadata_files),
                    priority=1, max_chars=2_500)
        package.add("REPOSITORY", self.repository_facts(include_commands=False), priority=1,
                    max_chars=600)
        if state is not None and state.research_findings:
            package.add("FINDINGS SO FAR (build on these, do not repeat)",
                        summarize_research(state.research_findings), priority=1, max_chars=3_000)
        # Only code that mentions the research topics - no general repository context.
        if topics:
            hits = self.ranker.searched_files(list(topics)[:4])
            lines = [f"- {path} (mentions {', '.join(t)})" for path, t in list(hits.items())[:6]]
            package.add("CODE MENTIONING THE TOPIC", "\n".join(lines) or "none found", priority=2,
                        max_chars=1_200)
        return package

    @staticmethod
    def for_music(command: str, parameters: Mapping[str, Any] | None = None) -> ContextPackage:
        """Music gets only the playback command and its parameters - no task/repo/history."""
        package = ContextPackage("music", 2_000)
        package.add("PLAYBACK COMMAND", command, priority=0, max_chars=500)
        if parameters:
            package.add("MUSIC PARAMETERS", json.dumps(dict(parameters), sort_keys=True),
                        priority=0, max_chars=1_000)
        return package

    def _dependency_text(self, metadata_files: Sequence[str]) -> str:
        """Declared dependencies only (bounded), not whole files."""
        out = []
        for name in metadata_files:
            path = self.repo.root / name
            if name == "package.json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for key in ("dependencies", "devDependencies", "engines"):
                    if data.get(key):
                        out.append(f"package.json {key}: {json.dumps(data[key])[:600]}")
            elif name == "pyproject.toml":
                import tomllib

                try:
                    project = tomllib.loads(path.read_text(encoding="utf-8")).get("project", {})
                except (OSError, tomllib.TOMLDecodeError):
                    continue
                if project.get("requires-python"):
                    out.append(f"requires-python: {project['requires-python']}")
                if project.get("dependencies"):
                    out.append(f"pyproject dependencies: {project['dependencies']}")
                for extra, deps in (project.get("optional-dependencies") or {}).items():
                    out.append(f"pyproject optional[{extra}]: {deps}")
            elif name.startswith("requirements"):
                try:
                    reqs = [
                        ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                        if ln.strip() and not ln.startswith("#")
                    ]
                except OSError:
                    continue
                out.append(f"{name}: {', '.join(reqs[:40])}")
        return "\n".join(out) or "No dependency declarations found."
