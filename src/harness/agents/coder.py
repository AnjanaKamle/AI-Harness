"""CoderAgent: inspects, edits and reviews code in one repository through structured tools.

SUCCESS here means only: the requested modification was made AND the resulting diff was
inspected after the last change. It does not mean tests pass (that is the Tester's job).
The harness verifies this from the actual tool-call record, not from the model's claims.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from harness.agents.base import AgentResult, AgentStatus, BaseAgent
from harness.agents.tool_loop import LoopOutcome, ToolCallingLoop, ToolCallRecord
from harness.config.settings import DEFAULT_MAX_TOOL_CALLS, Settings
from harness.context.builder import ContextBuilder
from harness.context.manager import ContextCategory, ContextEntry, ContextManager
from harness.llm.client import LLMClient, LLMError
from harness.llm.models import Message
from harness.orchestrator.state import AgentState
from harness.tools.base import BaseTool, ToolStatus, truncate_text
from harness.tools.registry import (
    CODER_TOOL_NAMES,
    ToolRegistry,
    build_default_tools,
    registry_limits,
)
from harness.tools.repository import RepositoryContext

MODIFYING_TOOLS = frozenset({"write_file", "edit_file"})
ARTIFACT_DIFF_CHARS = 8_000

CODER_SYSTEM_PROMPT = """\
You are the Coder agent of an AI coding harness. You change code in ONE repository, using \
only the tools provided. All paths are relative to the repository root.

Rules - follow them strictly:
1. Inspect before modifying: look at the repository layout (list_files / git_status) first.
2. Search before reading: use search_code to locate relevant code instead of reading large \
parts of the repository.
3. Read every file you intend to change (read_file) before editing it.
4. Make the minimal change that accomplishes the task. Prefer edit_file with an exact, \
unique old_text over rewriting whole files.
5. Preserve the existing architecture, style and public interfaces.
6. Never claim success without calling git_diff AFTER your last modification and checking it.
7. Do not modify files unrelated to the task.
8. Never reveal, print or write secrets (API keys, tokens, passwords, .env contents).
9. Never attempt destructive commands (rm, sudo, git push/reset/clean, etc.); use terminal \
only for safe development commands.
10. Finish with a structured result.
11. If no existing test exercises the behaviour you change, add a focused test for it.

When you are done (or cannot proceed), reply WITHOUT calling any tool, with ONLY this JSON \
object:
{
  "status": "SUCCESS" | "FAILURE" | "BLOCKED",
  "summary": "<what you changed and why, or why you could not>",
  "files_changed": ["<relative path>", ...],
  "next_action": "<recommended next step, e.g. run the tests>",
  "errors": ["<problem>", ...]
}
Use SUCCESS only if the change is made and you inspected the diff. Use BLOCKED when you \
need information or permissions you do not have. Use FAILURE otherwise.\
"""

REPAIR_PROMPT = (
    "Your last reply was not the required JSON object. Reply with ONLY the JSON object "
    '{"status", "summary", "files_changed", "next_action", "errors"} and nothing else.'
)

DEFAULT_NEXT_ACTION = {
    AgentStatus.SUCCESS: "Run the test suite to verify the change.",
    AgentStatus.FAILURE: "Review the errors and retry the task with more specific guidance.",
    AgentStatus.BLOCKED: "Provide the missing information or permissions, then retry.",
}


@dataclass
class CoderResult(AgentResult):
    files_inspected: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    diff_summary: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Evidence:
    files_inspected: list[str]
    files_changed: list[str]
    diff_after_last_change: dict[str, Any] | None


def parse_final_response(text: str | None) -> dict[str, Any] | None:
    """Extract the final JSON object from the model's reply (tolerates code fences/prose)."""
    if not text:
        return None
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and str(data.get("status", "")).upper() in AgentStatus.__members__:
            return data
    return None


def _unique(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def collect_evidence(records: Sequence[ToolCallRecord]) -> _Evidence:
    inspected: list[str] = []
    changed: list[str] = []
    last_change = 0
    last_diff: tuple[int, dict[str, Any]] | None = None
    for record in records:
        if record.status is not ToolStatus.SUCCESS:
            continue
        if record.name == "read_file" and "path" in record.data:
            inspected.append(record.data["path"])
        elif record.name in MODIFYING_TOOLS and record.data.get("changed"):
            changed.append(record.data["path"])
            last_change = record.index
        elif record.name == "git_diff":
            last_diff = (record.index, record.data)
    diff = last_diff[1] if last_diff and last_diff[0] > last_change else None
    return _Evidence(_unique(inspected), _unique(changed), diff)


@dataclass
class InspectionResult(AgentResult):
    """Read-only repository inspection (no model call, no writes)."""

    relevant_files: list[dict[str, Any]] = field(default_factory=list)
    repository: str = ""
    git_status: dict[str, Any] | None = None


class CoderAgent(BaseAgent):
    name = "coder"
    description = (
        "Implements code changes: inspects the repository, searches and reads relevant code, "
        "makes minimal edits and reviews the resulting diff."
    )

    def __init__(
        self,
        llm: LLMClient,
        context: ContextManager,
        repo: RepositoryContext,
        tools: Sequence[BaseTool] | None = None,
        *,
        settings: Settings | None = None,
        max_tool_calls: int | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        if tools is None:
            tools = [t for t in build_default_tools(repo, settings) if t.name in CODER_TOOL_NAMES]
        super().__init__(llm, context, tools)
        self.repo = repo
        self.max_tool_calls = max_tool_calls or (
            settings.max_tool_calls if settings else DEFAULT_MAX_TOOL_CALLS
        )
        self.registry = ToolRegistry(self.tools.values(), **registry_limits(settings))
        self.builder = context_builder or ContextBuilder(
            repo, context, max_chars=settings.max_context_chars if settings else 16_000
        )
        self.last_context_stats: dict[str, Any] = {}

    # --- prompt construction -----------------------------------------------------------

    def build_messages(self, task: str, state: AgentState) -> list[Message]:
        """System rules + a targeted context package (never the whole repository/history)."""
        package = self.builder.for_coder(task, state, self.available_tools)
        self.last_context_stats = package.stats
        user = package.render() + "\n\nStart by inspecting the repository."
        return [Message.system(CODER_SYSTEM_PROMPT), Message.user(user)]

    # --- inspection (read-only phase, schedulable separately) -------------------------

    def inspect(self, task: str, state: AgentState) -> InspectionResult:
        """Inspect the repository for ``task`` using only read tools: git state, repository
        facts and relevance-ranked files. Deterministic; never writes."""
        git_status = None
        if "git_status" in self.tools and self.repo.is_git_repo:
            outcome = self.registry.execute("git_status", {})
            git_status = outcome.data if outcome.ok else {"error": outcome.error}
        files = self.builder.relevant_files([task, state.task], state)
        repository = self.builder.repository_facts()
        dirty = [f["path"] for f in (git_status or {}).get("files", [])] if git_status else []
        summary = (
            f"Inspected {self.repo.name}: {len(files)} relevant file(s)"
            + (f"; uncommitted changes in {', '.join(dirty[:5])}" if dirty else "")
        )
        lines = [f"- {f.path} ({'; '.join(f.reasons)})" for f in files]
        if dirty:
            lines.append("Uncommitted changes present before this task: " + ", ".join(dirty[:10]))
        self.context.put(
            ContextEntry(
                ContextCategory.REPOSITORY,
                f"{state.task_id}:inspection",
                "\n".join(lines) or "No specific files identified.",
                source=self.name,
                metadata={"relevant_files": [f.path for f in files]},
                task_id=state.task_id,
            )
        )
        return InspectionResult(
            self.name,
            AgentStatus.SUCCESS,
            summary,
            relevant_files=[{"path": f.path, "score": f.score, "reasons": f.reasons} for f in files],
            repository=repository,
            git_status=git_status,
        )

    # --- execution ----------------------------------------------------------------------

    def execute(self, task: str, state: AgentState) -> CoderResult:
        loop = ToolCallingLoop(self.llm, self.registry, max_tool_calls=self.max_tool_calls)
        messages = self.build_messages(task, state)
        try:
            outcome = loop.run(messages)
            parsed = parse_final_response(outcome.final_text)
            if parsed is None and outcome.final_text is not None:
                repaired = loop.finalize(outcome.messages, REPAIR_PROMPT)
                parsed = parse_final_response(None if repaired.tool_calls else repaired.content)
        except LLMError as exc:
            result = CoderResult(
                self.name,
                AgentStatus.FAILURE,
                summary="The language model call failed",
                errors=[str(exc)],
                next_action="Check the LLM configuration/connectivity and retry.",
                metadata={"retryable": exc.retryable, "llm_error": True,
                          "error_code": exc.code.value},
            )
            self._record(state, result)
            return result

        result = self._build_result(outcome, parsed)
        result.metadata["usage"] = {**loop.usage.to_dict(), **self.last_context_stats}
        self._record(state, result)
        return result

    def _build_result(self, outcome: LoopOutcome, parsed: dict[str, Any] | None) -> CoderResult:
        evidence = collect_evidence(outcome.records)
        errors: list[str] = []
        summary = ""
        next_action = ""

        if parsed is None:
            status = AgentStatus.FAILURE
            errors.append(
                "Tool call limit reached before a final result"
                if outcome.limit_reached
                else "Model did not return a structured final result"
            )
        else:
            status = AgentStatus(str(parsed["status"]).upper())
            summary = str(parsed.get("summary", ""))
            next_action = str(parsed.get("next_action", ""))
            raw_errors = parsed.get("errors") or []
            errors.extend(str(e) for e in (raw_errors if isinstance(raw_errors, list) else [raw_errors]))
            claimed = parsed.get("files_changed") or []
            if isinstance(claimed, list):
                unverified = sorted(set(map(str, claimed)) - set(evidence.files_changed))
                if unverified:
                    errors.append(f"Model reported changes not made through tools: {unverified}")

        # SUCCESS is decided by evidence, not by the model's claim.
        if status is AgentStatus.SUCCESS:
            if not evidence.files_changed:
                status = AgentStatus.FAILURE
                errors.append("Reported SUCCESS but no file was modified")
            elif evidence.diff_after_last_change is None:
                status = AgentStatus.FAILURE
                errors.append("Reported SUCCESS without inspecting git_diff after the last change")

        diff = evidence.diff_after_last_change
        diff_summary = None
        artifacts: dict[str, Any] = {}
        if diff is not None:
            diff_summary = {
                "summary": diff.get("summary"),
                "files": [
                    {k: f.get(k) for k in ("path", "status", "additions", "deletions")}
                    for f in diff.get("files", [])
                ],
                "total_additions": diff.get("total_additions", 0),
                "total_deletions": diff.get("total_deletions", 0),
            }
            artifacts["diff"] = truncate_text(diff.get("diff", ""), ARTIFACT_DIFF_CHARS)[0]

        return CoderResult(
            agent_name=self.name,
            status=status,
            summary=summary or f"Coder finished with status {status.value}",
            artifacts=artifacts,
            errors=errors,
            metadata={
                "tool_call_count": len(outcome.records),
                "tool_call_limit": self.max_tool_calls,
                "limit_reached": outcome.limit_reached,
                "iterations": outcome.iterations,
            },
            files_inspected=evidence.files_inspected,
            files_changed=evidence.files_changed,
            diff_summary=diff_summary,
            tool_calls=[r.to_dict() for r in outcome.records],
            next_action=next_action or DEFAULT_NEXT_ACTION[status],
        )

    def _record(self, state: AgentState, result: CoderResult) -> None:
        """Publish the outcome to shared state and the context manager."""
        if result.files_changed:
            state.code_changes.append(
                {
                    "agent": self.name,
                    "status": result.status.value,
                    "files_changed": result.files_changed,
                    "diff_summary": result.diff_summary,
                }
            )
        if result.status is not AgentStatus.SUCCESS:
            state.failures.append(
                {"agent": self.name, "status": result.status.value, "errors": result.errors}
            )
        self.context.put(
            ContextEntry(
                ContextCategory.AGENT_OUTPUT,
                f"{self.name}:{state.task_id}:{len(state.code_changes) + len(state.failures)}",
                result.summary,
                source=self.name,
                metadata={"status": result.status.value, "files_changed": result.files_changed},
                task_id=state.task_id,
            )
        )
