"""ResearcherAgent: answers technical questions with evidence-backed, structured findings.

It never modifies the repository: its tools are read-only repository inspection
(list_files, read_file, search_code) plus research tools (web_search, fetch_documentation,
package_info, lookup_python_api).

Every cited source is checked against what the tools actually returned in this session
(SourceLedger). Findings citing anything else are rejected as fabricated. If no usable
finding survives, the result is BLOCKED with an explanation - findings are never invented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from harness.agents.base import AgentResult, AgentStatus, BaseAgent
from harness.agents.coder import parse_final_response
from harness.agents.tool_loop import ToolCallingLoop, ToolCallRecord
from harness.config.settings import DEFAULT_MAX_TOOL_CALLS, Settings
from harness.context.builder import ContextBuilder
from harness.context.manager import ContextCategory, ContextEntry, ContextManager
from harness.llm.client import LLMClient, LLMError
from harness.llm.models import Message
from harness.orchestrator.state import AgentState
from harness.research.models import (
    FindingError,
    FindingKind,
    ResearchFinding,
    SourceLedger,
    verify_finding,
)
from harness.tools.base import BaseTool, ToolStatus
from harness.tools.registry import ToolRegistry, build_default_tools, registry_limits
from harness.tools.repository import RepositoryContext
from harness.tools.research import RESEARCH_TOOL_NAMES, build_research_tools

READ_ONLY_REPO_TOOLS = ("list_files", "read_file", "search_code")
FORBIDDEN_RESEARCHER_TOOLS = frozenset({"write_file", "edit_file", "terminal", "run_tests"})
DEFAULT_RESEARCH_TOOL_CALLS = 15

RESEARCHER_SYSTEM_PROMPT = """\
You are the Researcher agent of an AI coding harness. You answer technical questions for \
the Coder with evidence. You cannot and must not modify the repository.

How to research:
1. Prefer authoritative, local evidence first: lookup_python_api (installed or project \
modules), package_info (official registry), and the repository itself (search_code, \
read_file) for project-specific questions.
2. Use web_search only to find documentation URLs; search snippets are leads, NOT evidence. \
Read a page with fetch_documentation before relying on it. If web_search is unavailable, \
package_info returns official documentation/homepage URLs you can fetch instead.
3. Keep queries short and generic. Never put secrets, credentials or private code in a query.
4. Stop as soon as the questions are answered.

Classify every finding:
- FACT: stated by a source you retrieved in this session (cite it exactly: the URL/final_url \
from fetch_documentation, the 'source' from package_info or lookup_python_api, or a \
repository file path you read).
- INFERENCE: your reasoning from the evidence (cite the source it is based on, if any).
- UNCERTAINTY: something you could not establish.
NEVER cite a source you did not retrieve - the harness verifies every source and rejects \
fabricated ones. If you cannot establish an answer, return status BLOCKED and explain why.

When done, reply WITHOUT calling tools, with ONLY this JSON object:
{
  "status": "SUCCESS" | "BLOCKED" | "FAILURE",
  "summary": "<answer in 1-3 sentences, for the Coder>",
  "findings": [
    {"question": "...", "kind": "FACT" | "INFERENCE" | "UNCERTAINTY", "finding": "...",
     "source": "<retrieved source or null>", "relevance": "<why it matters for the task>",
     "confidence": "HIGH" | "MEDIUM" | "LOW", "technical_details": "<exact API names, \
signatures, versions, config keys>", "recommended_action": "<what the Coder should do>"}
  ],
  "open_questions": ["..."]
}\
"""

REPAIR_PROMPT = (
    "Your last reply was not the required JSON object. Reply with ONLY the JSON object "
    '{"status", "summary", "findings", "open_questions"} and nothing else.'
)


@dataclass
class ResearchResult(AgentResult):
    questions: list[str] = field(default_factory=list)
    findings: list[ResearchFinding] = field(default_factory=list)
    rejected_findings: list[dict[str, Any]] = field(default_factory=list)
    sources_consulted: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def usable_findings(self) -> list[ResearchFinding]:
        return [f for f in self.findings if f.usable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "summary": self.summary,
            "questions": self.questions,
            "findings": [f.to_dict() for f in self.findings],
            "rejected_findings": self.rejected_findings,
            "sources_consulted": self.sources_consulted,
            "open_questions": self.open_questions,
            "errors": self.errors,
            "metadata": self.metadata,
        }


def build_ledger(records: Sequence[ToolCallRecord], repo: RepositoryContext) -> SourceLedger:
    """Collect every source the tools actually returned (successful calls only)."""
    ledger = SourceLedger.empty()
    for r in records:
        if r.status is not ToolStatus.SUCCESS:
            continue
        data = r.data
        if r.name == "fetch_documentation":
            ledger.add_evidence([data.get("url", ""), data.get("final_url", "")])
        elif r.name in ("package_info", "lookup_python_api"):
            ledger.add_evidence([data.get("source", "")])
            if r.name == "package_info":
                ledger.add_leads(v for v in (data.get("urls") or {}).values() if isinstance(v, str))
        elif r.name == "read_file":
            ledger.add_evidence([data.get("path", "")])
        elif r.name == "search_code":
            ledger.add_evidence(data.get("files", []))
        elif r.name == "web_search":
            ledger.add_leads(h.get("url", "") for h in data.get("results", []))
    return ledger


class ResearcherAgent(BaseAgent):
    name = "researcher"
    description = (
        "Researches libraries, APIs and documentation (and project-specific usage) and "
        "returns evidence-backed findings classified as FACT, INFERENCE or UNCERTAINTY."
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
            repo_tools = [t for t in build_default_tools(repo, settings) if t.name in READ_ONLY_REPO_TOOLS]
            research_tools = build_research_tools(
                repo,
                backend=settings.research_backend if settings else "duckduckgo",
                timeout_seconds=settings.research_timeout_seconds if settings else 15.0,
                max_results=settings.max_research_results if settings else 5,
            )
            tools = [*repo_tools, *research_tools]
        forbidden = FORBIDDEN_RESEARCHER_TOOLS.intersection(t.name for t in tools)
        if forbidden:
            raise ValueError(f"ResearcherAgent must not have modifying tools: {sorted(forbidden)}")
        super().__init__(llm, context, tools)
        self.repo = repo
        self.max_tool_calls = max_tool_calls or min(
            settings.max_tool_calls if settings else DEFAULT_MAX_TOOL_CALLS,
            DEFAULT_RESEARCH_TOOL_CALLS,
        )
        self.registry = ToolRegistry(self.tools.values(), **registry_limits(settings))
        self.builder = context_builder or ContextBuilder(
            repo, context, max_chars=settings.max_context_chars if settings else 16_000
        )
        self.topics: list[str] = []

    def build_messages(self, questions: Sequence[str], state: AgentState | None) -> list[Message]:
        package = self.builder.for_researcher(questions, state, topics=self.topics)
        self.last_context_stats = package.stats
        user = package.render() + (
            f"\n\nAVAILABLE TOOLS: {', '.join(self.available_tools)}\n"
            "Research the questions, then return the JSON result."
        )
        return [Message.system(RESEARCHER_SYSTEM_PROMPT), Message.user(user)]

    def execute(self, task: str, state: AgentState) -> ResearchResult:
        questions = [q.strip() for q in task.split("\n") if q.strip()] or [task]
        loop = ToolCallingLoop(self.llm, self.registry, max_tool_calls=self.max_tool_calls)
        try:
            outcome = loop.run(self.build_messages(questions, state))
            parsed = _parse(outcome.final_text)
            if parsed is None and outcome.final_text is not None:
                repaired = loop.finalize(outcome.messages, REPAIR_PROMPT)
                parsed = _parse(None if repaired.tool_calls else repaired.content)
        except LLMError as exc:
            result = ResearchResult(
                self.name, AgentStatus.FAILURE, "The language model call failed during research",
                errors=[str(exc)], questions=questions,
                metadata={"llm_error": True, "retryable": exc.retryable,
                          "error_code": exc.code.value},
            )
            self._record(state, result)
            return result

        usage = {**loop.usage.to_dict(), **getattr(self, "last_context_stats", {})}
        result = self._build_result(questions, outcome.records, parsed, outcome.limit_reached)
        result.metadata["usage"] = usage
        self._record(state, result)
        return result

    def _build_result(
        self,
        questions: list[str],
        records: Sequence[ToolCallRecord],
        parsed: dict[str, Any] | None,
        limit_reached: bool,
    ) -> ResearchResult:
        ledger = build_ledger(records, self.repo)
        tool_calls = [r.to_dict() for r in records]
        base: dict[str, Any] = {
            "questions": questions,
            "sources_consulted": ledger.sorted_sources(),
            "tool_calls": tool_calls,
        }
        if parsed is None:
            reason = (
                "Tool call limit reached before a research result"
                if limit_reached else "Researcher did not return a structured result"
            )
            return ResearchResult(self.name, AgentStatus.FAILURE, reason, errors=[reason], **base)

        accepted: list[ResearchFinding] = []
        rejected: list[dict[str, Any]] = []
        for raw in parsed.get("findings") or []:
            try:
                accepted.append(verify_finding(ResearchFinding.from_dict(raw), ledger))
            except FindingError as exc:
                rejected.append({"finding": raw, "reason": str(exc)})

        claimed = AgentStatus(str(parsed["status"]).upper())
        summary = str(parsed.get("summary", ""))
        errors = [f"Rejected finding: {r['reason']}" for r in rejected]
        usable = [f for f in accepted if f.usable]
        status = claimed
        if claimed is AgentStatus.SUCCESS and not usable:
            status = AgentStatus.BLOCKED
            errors.append(
                "No verifiable FACT or confident INFERENCE was established; refusing to invent "
                "findings"
            )
        if status is AgentStatus.BLOCKED and not summary:
            summary = "Research could not establish an answer."
        return ResearchResult(
            self.name,
            status,
            summary,
            errors=errors,
            findings=accepted,
            rejected_findings=rejected,
            open_questions=[str(q) for q in parsed.get("open_questions") or []],
            **base,
        )

    def _record(self, state: AgentState, result: ResearchResult) -> None:
        """Findings go to shared state and the context manager; failures are recorded too."""
        for i, finding in enumerate(result.findings, start=len(state.research_findings) + 1):
            record = {**finding.to_dict(), "research_status": result.status.value}
            state.research_findings.append(record)
            self.context.put(
                ContextEntry(
                    ContextCategory.RESEARCH,
                    f"{state.task_id}:finding:{i}",
                    finding.one_line(),
                    source=self.name,
                    metadata=record,
                    task_id=state.task_id,
                )
            )
        if result.status is not AgentStatus.SUCCESS:
            state.failures.append(
                {"agent": self.name, "status": result.status.value,
                 "errors": result.errors or [result.summary]}
            )
        self.context.put(
            ContextEntry(
                ContextCategory.AGENT_OUTPUT,
                f"{state.task_id}:researcher:{len(state.research_findings)}",
                result.summary,
                source=self.name,
                metadata={"status": result.status.value, "findings": len(result.findings),
                          "rejected": len(result.rejected_findings)},
                task_id=state.task_id,
            )
        )


def _parse(text: str | None) -> dict[str, Any] | None:
    parsed = parse_final_response(text)
    if parsed is not None and not isinstance(parsed.get("findings", []), list):
        return None
    return parsed


__all__ = [
    "RESEARCHER_SYSTEM_PROMPT",
    "RESEARCH_TOOL_NAMES",
    "ResearchResult",
    "ResearcherAgent",
    "build_ledger",
]
