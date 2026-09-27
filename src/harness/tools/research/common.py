"""Shared guards for research tools."""

from __future__ import annotations

import os
import re

from harness.tools.base import ToolError

MAX_QUERY_CHARS = 300

_SECRET_PATTERNS = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})"
)


def check_outbound_text(text: str, what: str = "query") -> str:
    """Refuse to send secrets (or the configured API key) to external services."""
    text = text.strip()
    if not text:
        raise ToolError(f"{what} must not be empty")
    if len(text) > MAX_QUERY_CHARS:
        raise ToolError(f"{what} is too long ({len(text)} > {MAX_QUERY_CHARS} chars)")
    api_key = os.environ.get("AI_API_KEY", "")
    if _SECRET_PATTERNS.search(text) or (len(api_key) >= 8 and api_key in text):
        raise ToolError(f"{what} appears to contain a secret; refusing to send it externally")
    return text
