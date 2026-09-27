"""lookup_python_api: authoritative, offline API facts from the installed environment.

Runs ``inspect`` in a separate process (so imports cannot affect the harness) and returns
the object's signature, docstring, public members and the distribution version. The
repository root (and ./src) are on sys.path, so project modules can be inspected too.
The source string ("python:<target>@<version>") is citable evidence.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from harness.tools.base import BaseTool, ToolError, ToolExecutionResult
from harness.tools.policy import sanitized_environment
from harness.tools.repository import RepositoryContext, is_ignored_dir

_TARGET = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
TIMEOUT_SECONDS = 20

_SCRIPT = r"""
import importlib, importlib.metadata, inspect, json, sys
sys.path[:0] = [".", "src"]
target = sys.argv[1]
parts = target.split(".")
obj = module = None
for i in range(len(parts), 0, -1):
    try:
        module = importlib.import_module(".".join(parts[:i]))
    except ImportError:
        continue
    obj = module
    try:
        for attr in parts[i:]:
            obj = getattr(obj, attr)
    except AttributeError as exc:
        print(json.dumps({"error": f"{target}: {exc}"})); sys.exit(0)
    break
if module is None:
    print(json.dumps({"error": f"module for {target!r} is not importable here"})); sys.exit(0)
top = module.__name__.split(".")[0]
version = None
stdlib = top in sys.stdlib_module_names
if stdlib:
    version = "stdlib-%d.%d" % sys.version_info[:2]
else:
    try:
        dists = importlib.metadata.packages_distributions().get(top, [])
        version = importlib.metadata.version(dists[0] if dists else top)
    except Exception:
        version = getattr(sys.modules.get(top), "__version__", None)
try:
    signature = str(inspect.signature(obj))
except (TypeError, ValueError):
    signature = None
kind = ("module" if inspect.ismodule(obj) else "class" if inspect.isclass(obj)
        else "function" if callable(obj) else type(obj).__name__)
members = []
if inspect.ismodule(obj) or inspect.isclass(obj):
    members = sorted(n for n in dir(obj) if not n.startswith("_"))
origin = "stdlib" if stdlib else (getattr(module, "__file__", None) or "built-in")
print(json.dumps({"target": target, "module": module.__name__, "version": version, "kind": kind,
                  "signature": signature, "doc": inspect.getdoc(obj) or "", "members": members,
                  "file": origin}))
"""


class LookupPythonApiTool(BaseTool):
    name = "lookup_python_api"
    description = (
        "Inspect a Python module/class/function available in the project environment (e.g. "
        "'json.dumps', 'requests.Session.get', or a project module like 'app.models'). Returns "
        "signature, docstring, public members and installed version. The returned 'source' is "
        "citable evidence."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "Dotted path, e.g. 'package.module.Name'"},
        },
        "required": ["target"],
        "additionalProperties": False,
    }

    def __init__(self, repo: RepositoryContext, *, max_doc_chars: int = 3_000) -> None:
        self.repo = repo
        self.max_doc_chars = max_doc_chars

    def execute(self, target: str, **_: Any) -> ToolExecutionResult:
        target = target.strip()
        if not _TARGET.match(target):
            raise ToolError(f"Invalid Python target {target!r}; use a dotted name like 'json.dumps'")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _SCRIPT, target],
                cwd=self.repo.root,
                env=sanitized_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"Inspecting {target} timed out") from None
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        if not lines:
            raise ToolError(f"Could not inspect {target}: {proc.stderr.strip()[-500:] or 'no output'}")
        info = json.loads(lines[-1])
        if "error" in info:
            raise ToolError(info["error"])

        doc = info.get("doc", "")
        truncated = len(doc) > self.max_doc_chars
        if truncated:
            doc = doc[: self.max_doc_chars] + "\n[OUTPUT TRUNCATED: docstring cut]"
        members = info.get("members", [])
        file = info.get("file", "")
        in_repo = (
            file.startswith(str(self.repo.root))
            and "site-packages" not in file
            and not any(is_ignored_dir(p) for p in Path(file).relative_to(self.repo.root).parts[:-1])
        )
        info.update(
            doc=doc,
            members=members[:80],
            members_total=len(members),
            file=self.repo.relative(file) if in_repo else (
                file if file in ("stdlib", "built-in") else "site-packages"
            ),
            project_module=in_repo,
            source=f"python:{target}@{info.get('version') or ('project' if in_repo else 'stdlib')}",
            truncated=truncated or len(members) > 80,
        )
        return ToolExecutionResult.success(self.name, info)
