"""Dashboard view-model: a read-only projection of orchestration events.

The UI never drives the core. It subscribes to the Orchestrator's event log and derives
what to show (agent statuses, current activity, progress, metrics, a readable event stream)
from those events alone. Pure Python - no terminal needed, so it is unit-testable.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from harness.orchestrator.events import OrchestrationEvent

AGENTS = ("coder", "researcher", "tester", "music")
LABELS = {"coder": "Coder", "researcher": "Researcher", "tester": "Tester", "music": "Music",
          "orchestrator": "Orchestrator"}

IDLE, WAITING, RUNNING, SUCCESS, FAILED, BLOCKED, DISABLED = (
    "IDLE", "WAITING", "RUNNING", "SUCCESS", "FAILED", "BLOCKED", "DISABLED"
)
MAX_EVENT_LINES = 300


@dataclass
class NodeView:
    id: str
    agent: str
    kind: str
    description: str
    required: bool = True
    status: str = "PENDING"


@dataclass
class AgentView:
    name: str
    activity: str = ""
    disabled_reason: str = ""


@dataclass
class DashboardModel:
    task: str
    mode: str = ""  # e.g. "DEMO MODE - scripted responses", "PLAN ONLY"
    started: float = field(default_factory=time.monotonic)
    coding_task: str | None = None
    music_command: str | None = None
    orchestrator_state: str = "STARTING"
    nodes: dict[str, NodeView] = field(default_factory=dict)
    agents: dict[str, AgentView] = field(default_factory=lambda: {a: AgentView(a) for a in AGENTS})
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_EVENT_LINES))
    tool_calls: int = 0
    agent_turns: int = 0
    estimated_tokens: int = 0
    reported_tokens: int = 0
    retries: int = 0
    repairs: int = 0
    test_status: str = "not run"
    final_status: str | None = None
    provider_status: str = ""  # "Connected" once a model call has succeeded
    finished_at: float | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _last_line: str = ""

    # --- queries (thread-safe snapshots for the renderer) ---------------------------------

    def elapsed(self) -> float:
        return (self.finished_at or time.monotonic()) - self.started

    def agent_status(self, agent: str) -> str:
        with self._lock:
            view = self.agents[agent]
            nodes = [n for n in self.nodes.values() if n.agent == agent]
            if not nodes:
                return DISABLED if view.disabled_reason else IDLE
            if any(n.status == "RUNNING" for n in nodes):
                return RUNNING
            if any(n.status in ("PENDING", "READY") for n in nodes):
                return WAITING
            last = [n for n in nodes if n.status not in ("SKIPPED",)]
            if not last:
                return IDLE
            final = last[-1].status
            return {"SUCCESS": SUCCESS, "FAILED": FAILED, "BLOCKED": BLOCKED}.get(final, IDLE)

    def progress(self) -> dict[str, int]:
        with self._lock:
            work = [n for n in self.nodes.values() if n.kind != "VERIFY"]
            done = sum(1 for n in work if n.status == "SUCCESS")
            pending = sum(1 for n in work if n.status in ("PENDING", "READY", "RUNNING"))
            return {"completed": done, "pending": pending, "total": len(work)}

    def event_lines(self, limit: int | None = None) -> list[str]:
        with self._lock:
            items = list(self.lines)
        return items[-limit:] if limit else items

    # --- event ingestion -------------------------------------------------------------------

    def apply(self, event: OrchestrationEvent) -> None:
        with self._lock:
            self._apply(event)
            line = humanize(event)
            if line and line != self._last_line:
                self._last_line = line
                self.lines.append(f"{_clock(self.elapsed())} {line}")

    def _node(self, event: OrchestrationEvent) -> NodeView | None:
        return self.nodes.get(event.node or "")

    def _apply(self, e: OrchestrationEvent) -> None:  # noqa: C901 - a flat event switch
        kind = e.event.value
        data = e.data
        node = self._node(e)
        if kind == "TASK_CREATED":
            self.orchestrator_state = "PLANNING"
        elif kind == "PLAN_CREATED":
            self.coding_task = data.get("coding_task")
            self.music_command = data.get("music_command")
            for n in data.get("nodes", []):
                self.nodes[n["id"]] = NodeView(n["id"], n["agent"], n["kind"], n["description"],
                                               n.get("required", True))
            for name, status in (data.get("agents") or {}).items():
                if name in self.agents and str(status).startswith("DISABLED"):
                    self.agents[name].disabled_reason = str(status)[len("DISABLED"):].strip(" ()")
            self.orchestrator_state = "SCHEDULING"
        elif kind == "AGENT_STARTED" and node:
            node.status = "RUNNING"
            if node.agent in self.agents:
                self.agents[node.agent].activity = _start_activity(node)
            self.orchestrator_state = "VERIFYING" if node.kind == "VERIFY" else "RUNNING"
        elif kind == "TOOL_STARTED" and e.agent in self.agents:
            self.agents[e.agent].activity = tool_activity(data.get("tool", ""), data.get("target", ""))
        elif kind in ("TOOL_CALLED", "TOOL_FAILED"):
            self.tool_calls += 1
            if kind == "TOOL_CALLED" and data.get("tool") == "play_music" and data.get("summary"):
                self.agents["music"].activity = f"playing {data['summary']}"
        elif kind == "AGENT_COMPLETED" and node:
            node.status = "SUCCESS"
            self._usage(data)
            if node.agent in self.agents and node.agent != "music":
                self.agents[node.agent].activity = f"done: {_short(e.details, 70)}"
            elif node.agent == "music":
                self.agents["music"].activity = _short(e.details.removeprefix("Music: "), 70)
        elif kind in ("AGENT_FAILED", "TEST_FAILED", "RESEARCH_FAILED") and node:
            node.status = "FAILED"
            self._usage(data)
            if node.agent in self.agents:
                self.agents[node.agent].activity = f"failed: {_short(e.details, 70)}"
        elif kind == "AGENT_RETRIED" and node:
            node.status = "READY"
            self.retries += 1
        elif kind == "AGENT_TIMEOUT" and node and node.agent in self.agents:
            self.agents[node.agent].activity = "timed out; terminating"
        elif kind == "NODE_BLOCKED" and node:
            node.status = "BLOCKED"
        elif kind == "NODE_SKIPPED" and node:
            node.status = "SKIPPED"
        elif kind == "REPAIR_STARTED":
            self.repairs += 1
            self.retries += 1
            for n in data.get("new_nodes", []):
                self.nodes[n["id"]] = NodeView(n["id"], n["agent"], n["kind"], n["description"])
        if kind == "TEST_PASSED":
            self.test_status = "PASS " + _counts(e.details)
        elif kind == "TEST_FAILED":
            self.test_status = "FAIL " + _counts(e.details)
        elif kind == "VERIFICATION_STARTED":
            self.orchestrator_state = "VERIFYING"
        elif kind in ("TASK_COMPLETED", "TASK_BLOCKED", "TASK_FAILED"):
            self.final_status = {"TASK_COMPLETED": "VERIFIED SUCCESS", "TASK_BLOCKED": "BLOCKED",
                                 "TASK_FAILED": "FAILED"}[kind]
            self.orchestrator_state = self.final_status
            self.finished_at = time.monotonic()
            for agent in self.agents.values():
                if agent.activity.startswith(("reading", "running", "searching", "modifying",
                                              "inspecting", "implementing", "listing", "reviewing")):
                    agent.activity = ""

    def _usage(self, data: dict[str, Any]) -> None:
        usage = data.get("usage") or {}
        if int(usage.get("llm_turns", 0)) > 0 and self.mode.startswith("provider:"):
            self.provider_status = "Connected"
        self.agent_turns += int(usage.get("llm_turns", 0))
        self.estimated_tokens += int(usage.get("estimated_input_tokens", 0)) + int(
            usage.get("estimated_output_tokens", 0))
        self.reported_tokens += int(usage.get("reported_input_tokens", 0)) + int(
            usage.get("reported_output_tokens", 0))

    def mark_plan_only(self, reason: str) -> None:
        with self._lock:
            self.orchestrator_state = "PLAN ONLY"
            self.final_status = None
            self.finished_at = time.monotonic()
            self.lines.append(f"{_clock(self.elapsed())} {reason}")


# --- human-readable text ------------------------------------------------------------------------


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


_RUNNER_FLAGS = re.compile(r"\s+-rfE --tb=short --color=no -p no:cacheprovider")


def _short(text: str, limit: int) -> str:
    text = _RUNNER_FLAGS.sub("", " ".join(str(text).split()))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def short_command(command: str) -> str:
    """'python3 -m pytest -rfE --tb=short ... tests/x.py' -> 'python3 -m pytest tests/x.py'."""
    parts = [p for p in command.split() if not p.startswith("-") and p not in ("no:cacheprovider",)]
    if parts[:1] == ["python3"] and "-m" in command.split():
        parts = ["python3", "-m", *parts[1:]]
    return " ".join(parts)


def tool_activity(tool: str, target: str) -> str:
    target = _short(target, 60)
    return {
        "list_files": "listing files",
        "read_file": f"reading {target}",
        "search_code": f"searching code for '{target}'",
        "write_file": f"modifying {target}",
        "edit_file": f"modifying {target}",
        "git_status": "checking git status",
        "git_diff": "reviewing the diff",
        "git_log": "reading git history",
        "terminal": f"running {target}",
        "detect_tests": "detecting test configuration",
        "run_tests": f"running {short_command(target)}",
        "web_search": f"searching the web for '{target}'",
        "fetch_documentation": f"reading documentation {target}",
        "package_info": f"looking up package {target}",
        "lookup_python_api": f"looking up {target}",
        "play_music": f"starting {target}",
        "pause_music": "pausing music",
        "resume_music": "resuming music",
        "stop_music": "stopping music",
        "set_volume": f"setting volume to {target}",
    }.get(tool, f"{tool} {target}".strip())


def _start_activity(node: NodeView) -> str:
    return {
        "INSPECT": "inspecting the repository",
        "IMPLEMENT": "implementing the change",
        "REPAIR": f"repairing ({node.description.split(':')[0].lower()})",
        "TEST": "running verification",
        "BASELINE": "running baseline tests (before any change)",
        "RESEARCH": "researching: " + _short(node.description.split(":", 1)[-1].strip(), 50),
        "MUSIC": _short(node.description.replace("Music: ", ""), 60),
        "VERIFY": "checking completion evidence",
    }.get(node.kind, node.description)


def _counts(details: str) -> str:
    match = re.search(r"\(([^()]*\b(?:passed|failed|errors?)\b[^()]*)\)", details)
    return f"({match.group(1)})" if match else ""


def humanize(e: OrchestrationEvent) -> str | None:  # noqa: C901 - a flat event switch
    """One readable line per meaningful event (noise such as READY/TOOL_CALLED is dropped)."""
    who = LABELS.get(e.agent, e.agent.title())
    kind = e.event.value
    data = e.data
    tool = data.get("tool", "")
    if kind == "PLAN_CREATED":
        nodes = data.get("nodes", [])
        return f"Orchestrator created plan ({len(nodes)} tasks; {e.details.split(' tasks; ', 1)[-1]})"
    if kind == "AGENT_STARTED":
        if e.agent == "orchestrator":
            return "Orchestrator verifying the final state"
        if (e.node or "").startswith("repair"):
            return f"{who} started {e.details.split(':')[0].lower()}"
        return f"{who} started: {_short(e.details, 60)}"
    if kind == "TOOL_STARTED" and tool == "run_tests":
        return f"{who} running {short_command(data.get('target', ''))}"
    if kind == "TOOL_CALLED":
        if tool in ("edit_file", "write_file") and data.get("summary") == "changed":
            return f"{who} modified {data.get('target')}"
        if tool == "run_tests":
            return f"{who}: {short_command(data.get('target', ''))} -> {data.get('summary')}"
        if tool == "fetch_documentation":
            return f"{who} read documentation {_short(data.get('target', ''), 50)}"
        if tool == "lookup_python_api":
            return f"{who} looked up {data.get('target')}"
        if tool == "play_music":
            return f"{who} playing {data.get('summary') or data.get('target')}"
        return None
    if kind == "TOOL_FAILED":
        return f"{who}: {_short(e.details, 90)}"
    if kind == "AGENT_COMPLETED":
        if e.agent == "orchestrator":
            return None
        if data.get("kind") == "INSPECT":
            return f"{who} inspected the repository"
        if data.get("kind") == "BASELINE":
            return _short(e.details, 90)
        if data.get("kind") == "RESEARCH":
            return f"Research complete: {_short(e.details, 60)}"
        if data.get("kind") == "TEST":
            return None  # TEST_PASSED carries the message
        return f"{who} finished: {_short(e.details, 60)}"
    if kind == "TEST_PASSED":
        return f"All tests passed {_counts(e.details)}".strip()
    if kind == "TEST_FAILED":
        failed = re.search(r"(\d+) failed", e.details)
        return (f"{failed.group(1)} test(s) failed" if failed else "Tests failed") + (
            f" {_counts(e.details)}" if _counts(e.details) else "")
    if kind == "REPAIR_STARTED":
        return f"Repair attempt {e.data.get('attempt', '')}".strip()
    if kind in ("AGENT_FAILED", "RESEARCH_FAILED"):
        return f"{who} failed: {_short(e.details.split(': ', 1)[-1], 70)}"
    if kind == "AGENT_RETRIED":
        return f"{who} retrying: {_short(e.details, 60)}"
    if kind == "AGENT_TIMEOUT":
        return f"{who} timed out: {_short(e.details, 60)}"
    if kind == "NODE_BLOCKED":
        return f"{e.node} blocked: {_short(e.details, 60)}"
    if kind == "STATE_REJECTED":
        return f"Orchestrator rejected a state change from {who}"
    if kind == "VERIFICATION_PASSED" and e.node is None:
        return "Verification passed"
    if kind == "VERIFICATION_FAILED" and e.node is None:
        return f"Verification failed: {_short(e.details, 70)}"
    if kind == "TASK_COMPLETED":
        return "Task complete: VERIFIED SUCCESS"
    if kind == "TASK_BLOCKED":
        return "Task BLOCKED"
    if kind == "TASK_FAILED":
        return "Task FAILED"
    return None
