"""package_info: official package-registry metadata (PyPI JSON API, npm registry)."""

from __future__ import annotations

import json
import re
import urllib.parse
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult
from harness.tools.research.http import HttpClient

_NAME = re.compile(r"^(@[a-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]*$")
MAX_SUMMARY_CHARS = 500


class PackageInfoTool(BaseTool):
    name = "package_info"
    description = (
        "Look up a package in its official registry (PyPI for python, npm for node): latest "
        "version, summary, documentation/homepage/repository URLs and runtime requirements."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "ecosystem": {"type": "string", "enum": ["python", "node"]},
        },
        "required": ["name", "ecosystem"],
        "additionalProperties": False,
    }

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def execute(self, name: str, ecosystem: str, **_: Any) -> ToolExecutionResult:
        name = name.strip()
        if not _NAME.match(name):
            raise ToolError(f"Invalid package name {name!r}")
        if ecosystem == "python":
            url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json"
            info = json.loads(self.http.get(url).text()).get("info", {})
            data = {
                "name": info.get("name", name),
                "version": info.get("version"),
                "summary": (info.get("summary") or "")[:MAX_SUMMARY_CHARS],
                "requires_python": info.get("requires_python"),
                "urls": {k: v for k, v in (info.get("project_urls") or {}).items() if v},
                "homepage": info.get("home_page") or None,
            }
        else:
            url = "https://registry.npmjs.org/" + urllib.parse.quote(name, safe="@")
            doc = json.loads(self.http.get(url).text())
            latest = (doc.get("dist-tags") or {}).get("latest")
            version_info = (doc.get("versions") or {}).get(latest, {})
            repo = doc.get("repository")
            data = {
                "name": doc.get("name", name),
                "version": latest,
                "summary": (doc.get("description") or "")[:MAX_SUMMARY_CHARS],
                "engines": version_info.get("engines"),
                "urls": {
                    "homepage": doc.get("homepage"),
                    "repository": repo.get("url") if isinstance(repo, dict) else repo,
                },
            }
        data["registry_url"] = url
        data["source"] = url
        return ToolExecutionResult.success(self.name, data)
