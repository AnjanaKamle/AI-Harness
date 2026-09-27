"""fetch_documentation: read a documentation page and return only the relevant excerpts."""

from __future__ import annotations

import json
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult
from harness.tools.research.common import check_outbound_text
from harness.tools.research.http import HttpClient
from harness.tools.research.text import extract_relevant, html_to_text

DEFAULT_MAX_CHARS = 6_000
MAX_CHARS_CAP = 12_000
_TEXT_TYPES = ("text/html", "text/plain", "text/markdown", "text/x-rst", "application/json",
               "application/xhtml", "text/x-markdown")


class FetchDocumentationTool(BaseTool):
    name = "fetch_documentation"
    description = (
        "Fetch a public documentation page (http/https) and return the passages most relevant "
        "to 'focus', not the whole page. The returned final_url is a citable source."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "focus": {"type": "string", "description": "What to look for on the page."},
            "max_chars": {"type": "integer", "minimum": 500, "maximum": MAX_CHARS_CAP},
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, http: HttpClient, *, default_max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.http = http
        self.default_max_chars = default_max_chars

    def execute(
        self, url: str, focus: str | None = None, max_chars: int | None = None, **_: Any
    ) -> ToolExecutionResult:
        if focus:
            focus = check_outbound_text(focus, "focus")
        budget = min(max_chars or self.default_max_chars, MAX_CHARS_CAP)
        response = self.http.get(url)
        if response.status >= 400:
            raise ToolError(f"HTTP {response.status} fetching {url}")
        content_type = response.content_type.split(";")[0].strip().lower()
        if content_type and not content_type.startswith(_TEXT_TYPES):
            raise ToolError(f"Unsupported content type {content_type!r} (text/HTML only)")

        raw = response.text()
        title = ""
        if "html" in content_type or raw.lstrip().lower().startswith(("<!doctype html", "<html")):
            title, text = html_to_text(raw)
        elif "json" in content_type:
            try:
                text = json.dumps(json.loads(raw), indent=1)
            except json.JSONDecodeError:
                text = raw
        else:
            text = raw
        if not text.strip():
            raise ToolError(f"No readable text at {url}")

        excerpts, truncated = extract_relevant(text, focus, max_chars=budget)
        return ToolExecutionResult.success(
            self.name,
            {
                "url": response.url,
                "final_url": response.final_url,
                "title": title,
                "focus": focus,
                "excerpts": [
                    {"heading": e.heading, "text": e.text, "score": e.score} for e in excerpts
                ],
                "page_chars": len(text),
                "returned_chars": sum(len(e.text) for e in excerpts),
                "truncated": truncated,
            },
        )
