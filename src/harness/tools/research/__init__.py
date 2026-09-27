"""Research tools: web_search, fetch_documentation, package_info, lookup_python_api.

All are read-only, bounded, and refuse to send secrets or reach non-public addresses.
"""

from __future__ import annotations

from harness.tools.base import BaseTool
from harness.tools.repository import RepositoryContext
from harness.tools.research.fetch import FetchDocumentationTool
from harness.tools.research.http import HttpClient, HttpResponse, UrllibHttpClient, validate_public_url
from harness.tools.research.packages import PackageInfoTool
from harness.tools.research.python_api import LookupPythonApiTool
from harness.tools.research.web_search import (
    DisabledSearchBackend,
    DuckDuckGoBackend,
    SearchBackend,
    SearchHit,
    WebSearchTool,
)

RESEARCH_TOOL_NAMES: tuple[str, ...] = (
    "web_search",
    "fetch_documentation",
    "package_info",
    "lookup_python_api",
)


def build_research_tools(
    repo: RepositoryContext,
    *,
    backend: str = "duckduckgo",
    timeout_seconds: float = 15.0,
    max_results: int = 5,
    http: HttpClient | None = None,
    search_backend: SearchBackend | None = None,
) -> list[BaseTool]:
    client = http or UrllibHttpClient(timeout=timeout_seconds)
    if search_backend is None:
        search_backend = (
            DuckDuckGoBackend(client) if backend == "duckduckgo" else DisabledSearchBackend()
        )
    return [
        WebSearchTool(search_backend, max_results_cap=max_results),
        FetchDocumentationTool(client),
        PackageInfoTool(client),
        LookupPythonApiTool(repo),
    ]


__all__ = [
    "RESEARCH_TOOL_NAMES",
    "DisabledSearchBackend",
    "DuckDuckGoBackend",
    "FetchDocumentationTool",
    "HttpClient",
    "HttpResponse",
    "LookupPythonApiTool",
    "PackageInfoTool",
    "SearchBackend",
    "SearchHit",
    "UrllibHttpClient",
    "WebSearchTool",
    "build_research_tools",
    "validate_public_url",
]
