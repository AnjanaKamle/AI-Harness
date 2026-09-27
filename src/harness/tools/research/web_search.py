"""web_search: find candidate documentation pages. Results are titles, URLs and short
snippets only - pages must be read with fetch_documentation before being cited as facts."""

from __future__ import annotations

import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult
from harness.tools.research.common import check_outbound_text
from harness.tools.research.http import HttpClient

MAX_SNIPPET_CHARS = 300


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


class SearchBackend(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Return up to ``max_results`` hits. Raise ToolError on failure."""


class DisabledSearchBackend(SearchBackend):
    name = "none"

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        raise ToolError("Web search is disabled (AI_RESEARCH_BACKEND=none)")


class _DuckDuckGoParser(HTMLParser):
    """Parses html.duckduckgo.com result markup (result__a / result__snippet links)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[dict[str, str]] = []
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr = {k: v or "" for k, v in attrs}
        classes = attr.get("class", "").split()
        if "result__a" in classes:
            self.hits.append({"title": "", "url": _unwrap(attr.get("href", "")), "snippet": ""})
            self._field = "title"
        elif "result__snippet" in classes and self.hits:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field and self.hits:
            self.hits[-1][self._field] += data


def _unwrap(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<target>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlsplit(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        return target or href
    return href


class DuckDuckGoBackend(SearchBackend):
    name = "duckduckgo"
    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def search(self, query: str, max_results: int) -> list[SearchHit]:
        url = self.endpoint + "?" + urllib.parse.urlencode({"q": query})
        response = self.http.get(url)
        page = response.text()
        # A bot-detection page is not "no results": report it instead of hiding it. The
        # harness does not try to get around such challenges.
        if response.status == 202 or "anomaly" in page.lower() and "result__a" not in page:
            raise ToolError(
                "Search provider refused the request (bot-detection challenge); web search is "
                "unavailable right now - use package_info / lookup_python_api / "
                "fetch_documentation with known documentation URLs instead"
            )
        parser = _DuckDuckGoParser()
        parser.feed(page)
        if not parser.hits and "result__" not in page and "No results" not in page:
            raise ToolError("Search provider returned an unrecognized page (no result markup)")
        hits = [
            SearchHit(h["title"].strip(), h["url"], " ".join(h["snippet"].split()))
            for h in parser.hits
            if h["url"].startswith(("http://", "https://")) and "duckduckgo.com" not in h["url"]
        ]
        return hits[:max_results]


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web for documentation. Returns titles, URLs and short snippets. Snippets "
        "are leads, not evidence: fetch a page with fetch_documentation before citing it. Never "
        "put code, secrets or private project details in the query."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, backend: SearchBackend, *, max_results_cap: int = 5) -> None:
        self.backend = backend
        self.max_results_cap = max_results_cap

    def execute(self, query: str, max_results: int = 5, **_: Any) -> ToolExecutionResult:
        query = check_outbound_text(query)
        limit = min(max_results, self.max_results_cap)
        hits = self.backend.search(query, limit + 1)
        results = [
            {**asdict(h), "snippet": h.snippet[:MAX_SNIPPET_CHARS]} for h in hits[:limit]
        ]
        return ToolExecutionResult.success(
            self.name,
            {
                "query": query,
                "backend": self.backend.name,
                "results": results,
                "count": len(results),
                "truncated": len(hits) > limit,
            },
        )
