"""Research findings, research tools (mocked network) and the Researcher agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from harness.agents import AgentStatus, ResearcherAgent, ResearchResult
from harness.context import InMemoryContextManager
from harness.context.manager import ContextCategory
from harness.llm import LLMError, LLMResponse, Message, StopReason, ToolCall
from harness.orchestrator import AgentState
from harness.research import (
    Confidence,
    FindingError,
    FindingKind,
    ResearchFinding,
    SourceLedger,
    verify_finding,
)
from harness.tools import ReadFileTool, RepositoryContext, SearchCodeTool, WriteFileTool
from harness.tools.research import (
    DisabledSearchBackend,
    DuckDuckGoBackend,
    FetchDocumentationTool,
    HttpResponse,
    LookupPythonApiTool,
    PackageInfoTool,
    SearchBackend,
    SearchHit,
    WebSearchTool,
    validate_public_url,
)
from harness.tools.research.text import extract_relevant, html_to_text

from .conftest import FAKE_KEY, MockLLMClient

# --- fakes ----------------------------------------------------------------------------------


class FakeHttp:
    """Serves canned responses by URL prefix; records requests; never touches the network."""

    def __init__(self, pages: dict[str, tuple[str, str]]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def get(self, url: str) -> HttpResponse:
        self.requested.append(url)
        for prefix, (content_type, body) in self.pages.items():
            if url.startswith(prefix):
                return HttpResponse(url, url, 200, content_type, body.encode())
        from harness.tools.base import ToolError

        raise ToolError(f"HTTP 404 fetching {url}")


class FakeSearch(SearchBackend):
    name = "fake"

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        return self.hits[:max_results]


DOC_URL = "https://docs.example.org/widgets/api"
DOC_HTML = """<html><head><title>Widgets API</title><script>var tracking = 1;</script></head>
<body><nav>Home | Blog | Pricing</nav>
<h1>Widgets</h1><p>Widgets is a library for building widgets.</p>
<h2>Installation</h2><p>Install with pip install widgets.</p>
<h2>Rendering widgets</h2><p>Call <code>render_widget(spec, *, theme="light")</code> to render.
The theme parameter accepts light or dark.</p><pre>from widgets import render_widget
render_widget({"kind": "button"}, theme="dark")</pre>
<h2>Changelog</h2><p>""" + ("Version history entry. " * 400) + """</p>
<footer>Copyright</footer></body></html>"""


@pytest.fixture
def repo(tmp_path: Path) -> RepositoryContext:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "app.py").write_text("def build():\n    return None\n")
    return RepositoryContext(root)


# --- ResearchFinding -----------------------------------------------------------------------


def finding(**overrides: Any) -> ResearchFinding:
    data = {
        "question": "Which function renders a widget?",
        "kind": "FACT",
        "finding": "render_widget(spec, *, theme) renders a widget",
        "source": DOC_URL,
        "relevance": "the task must render widgets",
        "confidence": "HIGH",
        "technical_details": "render_widget(spec, *, theme='light')",
        "recommended_action": "call render_widget with theme='dark'",
        **overrides,
    }
    return ResearchFinding.from_dict(data)


def test_research_finding_fields_and_roundtrip() -> None:
    f = finding()
    assert (f.kind, f.confidence, f.source) == (FindingKind.FACT, Confidence.HIGH, DOC_URL)
    assert f.usable and "render_widget" in f.one_line()
    assert ResearchFinding.from_dict(json.loads(json.dumps(f.to_dict()))) == f
    for field_name in ("question", "finding", "source", "relevance", "confidence",
                       "technical_details", "recommended_action", "kind"):
        assert field_name in f.to_dict()


def test_fact_requires_source() -> None:
    with pytest.raises(FindingError, match="FACT must cite"):
        finding(source=None)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [({"kind": "OPINION"}, "'kind' must be one of"), ({"confidence": "SURE"}, "'confidence'"),
     ({"finding": ""}, "non-empty"), ({"question": " "}, "non-empty")],
)
def test_finding_validation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(FindingError, match=message):
        finding(**overrides)


def test_fact_inference_uncertainty_are_distinct() -> None:
    inference = finding(kind="INFERENCE", source=None, confidence="MEDIUM")
    uncertainty = finding(kind="UNCERTAINTY", source=None, confidence="HIGH")
    assert inference.usable and not uncertainty.usable
    assert uncertainty.confidence is Confidence.LOW  # an uncertainty cannot be high-confidence
    assert not finding(kind="INFERENCE", source=None, confidence="LOW").usable


def test_source_verification() -> None:
    ledger = SourceLedger.empty()
    ledger.add_evidence([DOC_URL + "/", "app.py", "python:json.dumps@stdlib-3.13"])
    ledger.add_leads(["https://blog.example.org/post"])

    assert verify_finding(finding(), ledger).source_verified  # trailing slash normalized
    assert verify_finding(finding(source="repo:app.py"), ledger).source_verified
    assert verify_finding(finding(source="python:json.dumps"), ledger).source_verified
    with pytest.raises(FindingError, match="never retrieved"):
        verify_finding(finding(source="https://made-up.example.com/docs"), ledger)
    downgraded = verify_finding(finding(source="https://blog.example.org/post"), ledger)
    assert downgraded.kind is FindingKind.INFERENCE and downgraded.confidence is Confidence.LOW
    assert not downgraded.source_verified


# --- research tools ------------------------------------------------------------------------


def test_web_search_bounded_and_validated() -> None:
    hits = [SearchHit(f"t{i}", f"https://e.org/{i}", "s" * 1_000) for i in range(20)]
    backend = FakeSearch(hits)
    tool = WebSearchTool(backend, max_results_cap=3)
    result = tool.run({"query": "widgets render api", "max_results": 10})
    assert result.ok and result.data["count"] == 3 and result.data["truncated"]
    assert all(len(r["snippet"]) <= 300 for r in result.data["results"])

    assert not tool.run({}).ok  # missing query
    assert not tool.run({"query": "x", "max_results": 50}).ok  # schema maximum
    assert not tool.run({"query": "   "}).ok
    assert not tool.run({"query": "q" * 400}).ok
    leaked = tool.run({"query": "why does sk-FAKEFAKEFAKEFAKEFAKEFAKE fail"})
    assert not leaked.ok and "secret" in (leaked.error or "")
    assert backend.queries == ["widgets render api"]  # nothing else left the machine


def test_web_search_refuses_configured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_API_KEY", "super-secret-value-123")
    result = WebSearchTool(FakeSearch([])).run({"query": "error super-secret-value-123"})
    assert not result.ok and "secret" in (result.error or "")


def test_disabled_search_backend_fails_clearly() -> None:
    result = WebSearchTool(DisabledSearchBackend()).run({"query": "anything"})
    assert not result.ok and "disabled" in (result.error or "")


def test_duckduckgo_backend_parses_results() -> None:
    html = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.example.org%2Fa'
        '&rut=x">Doc A</a><a class="result__snippet" href="#">About <b>A</b></a>'
        '<a class="result__a" href="https://docs.example.org/b">Doc B</a>'
    )
    http = FakeHttp({"https://html.duckduckgo.com/": ("text/html", html)})
    hits = DuckDuckGoBackend(http).search("widgets", 5)
    assert [(h.title, h.url) for h in hits] == [
        ("Doc A", "https://docs.example.org/a"), ("Doc B", "https://docs.example.org/b")
    ]
    assert hits[0].snippet == "About A" and "q=widgets" in http.requested[0]


def test_duckduckgo_challenge_is_a_failure_not_empty_results() -> None:
    challenge = "<html><body>Unfortunately, bots use DuckDuckGo too. anomaly</body></html>"

    class ChallengeHttp(FakeHttp):
        def get(self, url: str) -> HttpResponse:
            return HttpResponse(url, url, 202, "text/html", challenge.encode())

    result = WebSearchTool(DuckDuckGoBackend(ChallengeHttp({}))).run({"query": "widgets"})
    assert not result.ok and "bot-detection" in (result.error or "")
    odd = FakeHttp({"https://html.duckduckgo.com/": ("text/html", "<html>maintenance</html>")})
    assert "unrecognized page" in (WebSearchTool(DuckDuckGoBackend(odd)).run({"query": "x"}).error or "")


def test_html_to_text_strips_noise() -> None:
    title, text = html_to_text(DOC_HTML)
    assert title == "Widgets API"
    assert "tracking" not in text and "Pricing" not in text and "Copyright" not in text
    assert "## Rendering widgets" in text and "```" in text


def test_extract_relevant_returns_focused_bounded_excerpts() -> None:
    _, text = html_to_text(DOC_HTML)
    excerpts, truncated = extract_relevant(text, "render_widget theme", max_chars=1_500)
    assert excerpts[0].heading == "Rendering widgets"
    assert sum(len(e.text) for e in excerpts) <= 1_500
    assert truncated


def test_fetch_documentation_extracts_not_dumps() -> None:
    http = FakeHttp({DOC_URL: ("text/html; charset=utf-8", DOC_HTML)})
    result = FetchDocumentationTool(http).run(
        {"url": DOC_URL, "focus": "render_widget theme", "max_chars": 1_200}
    )
    assert result.ok, result.error
    data = result.data
    assert data["title"] == "Widgets API" and data["final_url"] == DOC_URL
    assert any("render_widget" in e["text"] for e in data["excerpts"])
    assert data["returned_chars"] <= 1_200 < data["page_chars"]
    assert data["truncated"] is True
    assert "Version history entry. Version history entry. Version history" not in json.dumps(
        data["excerpts"][:1]
    )


def test_fetch_documentation_errors() -> None:
    http = FakeHttp({"https://e.org/img": ("image/png", "PNG")})
    tool = FetchDocumentationTool(http)
    assert "Unsupported content type" in (tool.run({"url": "https://e.org/img"}).error or "")
    assert "404" in (tool.run({"url": "https://e.org/missing"}).error or "")
    assert not tool.run({"url": DOC_URL, "max_chars": 50}).ok  # below schema minimum


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.org/x", "http://localhost:8080/", "http://127.0.0.1/",
     "http://10.0.0.5/admin", "http://169.254.169.254/latest/meta-data",
     "http://[::1]/", "https://user:pass@example.org/", "http://printer.local/"],
)
def test_validate_public_url_blocks_non_public_targets(url: str) -> None:
    from harness.tools.base import ToolError

    with pytest.raises(ToolError):
        validate_public_url(url)


def test_package_info_pypi_and_npm() -> None:
    pypi = json.dumps({"info": {"name": "widgets", "version": "2.1.0", "summary": "Widgets",
                                "requires_python": ">=3.9",
                                "project_urls": {"Documentation": DOC_URL}}})
    npm = json.dumps({"name": "left-pad", "dist-tags": {"latest": "1.3.0"},
                      "versions": {"1.3.0": {"engines": {"node": ">=4"}}},
                      "description": "pad", "homepage": "https://e.org/lp"})
    http = FakeHttp({"https://pypi.org/pypi/widgets/json": ("application/json", pypi),
                     "https://registry.npmjs.org/left-pad": ("application/json", npm)})
    tool = PackageInfoTool(http)
    py = tool.run({"name": "widgets", "ecosystem": "python"})
    assert py.ok and py.data["version"] == "2.1.0" and py.data["urls"]["Documentation"] == DOC_URL
    assert py.data["source"] == "https://pypi.org/pypi/widgets/json"
    js = tool.run({"name": "left-pad", "ecosystem": "node"})
    assert js.ok and js.data["version"] == "1.3.0" and js.data["engines"] == {"node": ">=4"}
    assert not tool.run({"name": "bad name!", "ecosystem": "python"}).ok
    assert not tool.run({"name": "x", "ecosystem": "ruby"}).ok


def test_lookup_python_api(repo: RepositoryContext) -> None:
    tool = LookupPythonApiTool(repo)
    stdlib = tool.run({"target": "json.dumps"})
    assert stdlib.ok and stdlib.data["kind"] == "function"
    assert stdlib.data["source"].startswith("python:json.dumps@stdlib-")
    assert "skipkeys" in stdlib.data["signature"]
    project = tool.run({"target": "app.build"})
    assert project.ok and project.data["project_module"] and project.data["file"] == "app.py"
    assert not tool.run({"target": "json; import os"}).ok
    assert "not importable" in (tool.run({"target": "no_such_pkg_xyz.func"}).error or "")
    assert not tool.run({"target": "json.no_such_attr"}).ok


# --- Researcher agent -----------------------------------------------------------------------


def tools(*calls: ToolCall) -> LLMResponse:
    return LLMResponse("", StopReason.TOOL_USE, tool_calls=calls)


def reply(status: str, findings: list[dict[str, Any]], summary: str = "summary") -> LLMResponse:
    return LLMResponse(
        json.dumps({"status": status, "summary": summary, "findings": findings,
                    "open_questions": []}),
        StopReason.END_TURN,
    )


def make_researcher(repo: RepositoryContext, script: list[Any]) -> tuple[ResearcherAgent, MockLLMClient, InMemoryContextManager]:
    http = FakeHttp({DOC_URL: ("text/html", DOC_HTML)})
    research_tools = [
        WebSearchTool(FakeSearch([SearchHit("Widgets API", DOC_URL, "render widgets")])),
        FetchDocumentationTool(http),
        PackageInfoTool(http),
        LookupPythonApiTool(repo),
    ]
    llm = MockLLMClient(script)
    ctx = InMemoryContextManager()
    agent = ResearcherAgent(llm, ctx, repo, tools=[ReadFileTool(repo), SearchCodeTool(repo), *research_tools])
    return agent, llm, ctx


def fact(source: str, **extra: Any) -> dict[str, Any]:
    return {"question": "q", "kind": "FACT", "finding": "render_widget renders widgets",
            "source": source, "relevance": "needed", "confidence": "HIGH",
            "technical_details": "render_widget(spec, *, theme)", "recommended_action": "use it",
            **extra}


def test_researcher_success_with_verified_sources(repo: RepositoryContext) -> None:
    script = [
        tools(ToolCall("1", "web_search", {"query": "widgets render api"})),
        tools(ToolCall("2", "fetch_documentation", {"url": DOC_URL, "focus": "render_widget"})),
        reply("SUCCESS", [fact(DOC_URL), {"question": "q", "kind": "UNCERTAINTY",
                                          "finding": "theme list may change", "source": None,
                                          "relevance": "minor", "confidence": "LOW"}]),
    ]
    agent, llm, ctx = make_researcher(repo, script)
    state = AgentState(task="Use library widgets to render a button")
    result = agent.run("Which widgets API renders a button?", state)

    assert isinstance(result, ResearchResult) and result.status is AgentStatus.SUCCESS
    assert [f.kind for f in result.findings] == [FindingKind.FACT, FindingKind.UNCERTAINTY]
    assert result.findings[0].source_verified and result.rejected_findings == []
    assert DOC_URL in result.sources_consulted
    assert [c["name"] for c in result.tool_calls] == ["web_search", "fetch_documentation"]
    assert result.metadata["usage"]["llm_turns"] == 3
    assert result.metadata["usage"]["estimated_input_tokens"] > 0
    # handed off through shared state and the context manager
    assert state.research_findings[0]["source"] == DOC_URL
    assert len(ctx.entries(ContextCategory.RESEARCH)) == 2
    assert state.usage["researcher"]["llm_turns"] == 3
    # the model was told the questions, with no repository dump
    first_user = llm.calls[0][0][1].content
    assert "Which widgets API renders a button?" in first_user
    assert "def build" not in first_user


def test_researcher_result_format(repo: RepositoryContext) -> None:
    agent, _, _ = make_researcher(
        repo, [tools(ToolCall("1", "fetch_documentation", {"url": DOC_URL})), reply("SUCCESS", [fact(DOC_URL)])]
    )
    data = agent.run("q", AgentState(task="t")).to_dict()  # type: ignore[attr-defined]
    assert set(data) >= {"status", "summary", "questions", "findings", "rejected_findings",
                         "sources_consulted", "open_questions", "errors"}
    json.dumps(data)  # serializable


def test_fabricated_source_is_rejected_and_blocks(repo: RepositoryContext) -> None:
    script = [reply("SUCCESS", [fact("https://docs.example.org/never-fetched")])]
    agent, _, _ = make_researcher(repo, script)
    state = AgentState(task="t")
    result = agent.run("q", state)
    assert isinstance(result, ResearchResult)
    assert result.status is AgentStatus.BLOCKED and result.findings == []
    assert "never retrieved" in result.rejected_findings[0]["reason"]
    assert any("refusing to invent" in e for e in result.errors)
    assert state.research_findings == []  # nothing invented reaches the Coder
    assert state.failures[-1]["agent"] == "researcher"


def test_missing_source_on_fact_is_rejected(repo: RepositoryContext) -> None:
    agent, _, _ = make_researcher(repo, [reply("SUCCESS", [fact(None)])])  # type: ignore[arg-type]
    result = agent.run("q", AgentState(task="t"))
    assert result.status is AgentStatus.BLOCKED
    assert "FACT must cite" in result.rejected_findings[0]["reason"]  # type: ignore[attr-defined]


def test_search_snippet_is_not_enough_for_a_fact(repo: RepositoryContext) -> None:
    script = [tools(ToolCall("1", "web_search", {"query": "widgets"})),
              reply("SUCCESS", [fact(DOC_URL)])]  # cited, but never fetched
    agent, _, _ = make_researcher(repo, script)
    result = agent.run("q", AgentState(task="t"))
    assert result.status is AgentStatus.BLOCKED  # downgraded to LOW inference -> not usable
    assert result.findings[0].kind is FindingKind.INFERENCE  # type: ignore[attr-defined]


def test_researcher_blocked_passthrough(repo: RepositoryContext) -> None:
    agent, _, _ = make_researcher(repo, [reply("BLOCKED", [], summary="No docs reachable")])
    result = agent.run("q", AgentState(task="t"))
    assert result.status is AgentStatus.BLOCKED and result.summary == "No docs reachable"


def test_researcher_failures_do_not_raise(repo: RepositoryContext) -> None:
    def boom(messages: list[Message]) -> LLMResponse:
        raise LLMError("provider down")

    agent, _, _ = make_researcher(repo, [boom])
    result = agent.run("q", AgentState(task="t"))
    assert result.status is AgentStatus.FAILURE and result.metadata["llm_error"]

    agent, _, _ = make_researcher(repo, [LLMResponse("not json", StopReason.END_TURN),
                                         LLMResponse("still not", StopReason.END_TURN)])
    result = agent.run("q", AgentState(task="t"))
    assert result.status is AgentStatus.FAILURE


def test_researcher_cannot_modify_repository(repo: RepositoryContext) -> None:
    agent, _, _ = make_researcher(repo, [])
    assert not {"write_file", "edit_file", "terminal", "run_tests"} & set(agent.available_tools)
    with pytest.raises(ValueError, match="must not have modifying tools"):
        ResearcherAgent(MockLLMClient(), InMemoryContextManager(), repo,
                        tools=[ReadFileTool(repo), WriteFileTool(repo)])
    # an attempt to call a write tool is reported back as an unknown tool
    script = [tools(ToolCall("1", "write_file", {"path": "app.py", "content": "x"})),
              reply("BLOCKED", [], summary="cannot write")]
    agent, _, _ = make_researcher(repo, script)
    result = agent.run("q", AgentState(task="t"))
    assert result.tool_calls[0]["status"] == "FAILURE"  # type: ignore[attr-defined]
    assert (repo.root / "app.py").read_text() == "def build():\n    return None\n"


def test_default_researcher_tools(repo: RepositoryContext) -> None:
    from harness.config.settings import Settings

    agent = ResearcherAgent(MockLLMClient(), InMemoryContextManager(), repo,
                            settings=Settings(api_key=FAKE_KEY, research_backend="none"))
    assert agent.available_tools == sorted(
        ["list_files", "read_file", "search_code", "web_search", "fetch_documentation",
         "package_info", "lookup_python_api"]
    )
    assert agent.max_tool_calls <= 15
