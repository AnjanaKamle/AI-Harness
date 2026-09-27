"""Rich renderables for the dashboard, the final result and errors (presentation only).

Everything shown is read from the DashboardModel (events) or the structured FinalResult /
AgentState - the UI never computes or invents outcomes.
"""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from harness.orchestrator.final_result import FinalResult
from harness.ui.model import AGENTS, LABELS, DashboardModel, short_command

STATUS_STYLE = {
    "IDLE": "dim", "WAITING": "yellow", "RUNNING": "bold cyan", "SUCCESS": "bold green",
    "FAILED": "bold red", "BLOCKED": "bold magenta", "DISABLED": "dim italic",
}
FINAL_STYLE = {"VERIFIED_SUCCESS": "bold white on green", "VERIFIED SUCCESS": "bold white on green",
               "BLOCKED": "bold white on magenta", "FAILED": "bold white on red"}
EVENT_LINES = 14


def _status(text: str) -> Text:
    return Text(text, style=STATUS_STYLE.get(text, ""))


def header(model: DashboardModel) -> Panel:
    title = Text("AI CODING HARNESS", style="bold white")
    if model.mode:
        title.append(f"   [{model.mode}]", style="bold yellow")
    if model.provider_status:
        title.append(f"   Status: {model.provider_status}", style="bold green")
    return Panel(title, style="blue", padding=(0, 1))


def task_panel(model: DashboardModel) -> Panel:
    body = Text(model.task)
    if model.coding_task and model.coding_task != model.task:
        body.append(f"\ncoding: {model.coding_task}", style="dim")
    if model.music_command:
        body.append(f"\nmusic:  {model.music_command}", style="dim")
    return Panel(body, title="CURRENT TASK", title_align="left")


def agents_table(model: DashboardModel) -> Table:
    table = Table(expand=True, show_edge=False, pad_edge=False)
    table.add_column("AGENT", style="bold", width=12)
    table.add_column("STATUS", width=10)
    table.add_column("CURRENT ACTIVITY", overflow="ellipsis", no_wrap=True)
    for agent in AGENTS:
        status = model.agent_status(agent)
        view = model.agents[agent]
        activity = view.activity or (view.disabled_reason if status == "DISABLED" else "")
        table.add_row(LABELS[agent], _status(status), Text(activity, style="dim" if status in
                                                                ("IDLE", "DISABLED") else ""))
    return table


def orchestrator_panel(model: DashboardModel) -> Panel:
    state = model.orchestrator_state
    style = FINAL_STYLE.get(state, "bold cyan")
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_row(Text.assemble(("Orchestrator: ", "bold"), (state, style)))
    grid.add_row(agents_table(model))
    return Panel(grid, title="ORCHESTRATOR / AGENTS", title_align="left")


def progress_panel(model: DashboardModel) -> Panel:
    p = model.progress()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("completed", f"{p['completed']}/{p['total']} tasks")
    grid.add_row("pending", str(p["pending"]))
    grid.add_row("retries", f"{model.retries} (repairs: {model.repairs})")
    grid.add_row("tests", model.test_status)
    return Panel(grid, title="PROGRESS", title_align="left")


def metrics_panel(model: DashboardModel) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("tool calls", str(model.tool_calls))
    grid.add_row("agent turns", str(model.agent_turns))
    grid.add_row("elapsed", f"{model.elapsed():.1f}s")
    tokens = f"~{model.estimated_tokens:,} (estimated)"
    if model.reported_tokens:
        tokens += f" / {model.reported_tokens:,} reported by provider"
    grid.add_row("tokens", tokens)
    return Panel(grid, title="METRICS", title_align="left")


def events_panel(model: DashboardModel, limit: int = EVENT_LINES) -> Panel:
    lines = model.event_lines(limit) or ["waiting for events..."]
    return Panel(Text("\n".join(lines)), title="EVENTS", title_align="left")


def dashboard(model: DashboardModel) -> RenderableType:
    row = Table.grid(expand=True)
    row.add_column(ratio=1)
    row.add_column(ratio=1)
    row.add_row(progress_panel(model), metrics_panel(model))
    return Group(header(model), task_panel(model), orchestrator_panel(model), row, events_panel(model))


# --- final result --------------------------------------------------------------------------------


def _bullets(items: list[str], empty: str = "none") -> str:
    return "\n".join(f"• {i}" for i in items) if items else empty


def final_view(final: FinalResult) -> RenderableType:
    """The final report, rendered strictly from the structured FinalResult / AgentState."""
    label = final.status.value.replace("_", " ")
    banner = Panel(Text(label, justify="center", style=FINAL_STYLE.get(final.status.value, "bold")),
                   padding=(0, 1))
    state = final.state
    research: dict[str, Any] = (state.research or {}) if state else {}
    research_text = "no"
    if final.research_performed:
        research_text = (f"yes - {research.get('usable_findings', 0)} usable finding(s); "
                         f"libraries: {', '.join(research.get('libraries') or []) or '-'}")
    elif research.get("reason"):
        research_text = f"no ({research['reason']})"

    tests = [f"attempt {t['attempt']}: {short_command(str(t['command']))} -> {t['status']}"
             + (f" ({t['passed'] or 0} passed, {t['failed'] or 0} failed)" if t.get("passed") is not None
                or t.get("failed") is not None else "")
             for t in final.tests_run]
    repairs = state.repair_attempts if state else 0

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column()
    grid.add_row("Summary", final.summary)
    grid.add_row("Files changed", _bullets(final.files_changed))
    grid.add_row("Tests executed", _bullets(tests, "none"))
    grid.add_row("Test results", f"{final.tests_passed} passed, {final.tests_failed} failed"
                 if final.tests_run else "no tests were run")
    grid.add_row("Research performed", research_text)
    grid.add_row("Repair attempts", str(repairs))
    grid.add_row("Retries", str(final.retries))
    if final.music:
        music = final.music
        music_line = music.get("summary") or music.get("node_status", "")
        if music.get("errors"):
            music_line = f"{music.get('node_status')}: {music['errors'][0]}"
        grid.add_row("Music", str(music_line))
    grid.add_row("Remaining issues", _bullets(final.unresolved_issues))
    grid.add_row("Duration", f"{final.duration_seconds:.1f}s, {final.tool_calls} tool calls")
    return Group(banner, Panel(grid, title="FINAL RESULT", title_align="left"))


def plan_view(state: Any, reason: str) -> RenderableType:
    """Plan-only mode (no model provider): the planned task graph and why agents did not run."""
    table = Table(title="PLANNED TASK GRAPH", expand=True)
    table.add_column("task")
    table.add_column("agent")
    table.add_column("depends on")
    table.add_column("description", overflow="fold")
    for node in (state.task_graph or {}).get("nodes", []):
        table.add_row(node["id"], node["agent"], ", ".join(node["dependencies"]) or "-",
                      node["description"] + ("" if node["required"] else " (optional)"))
    return Group(table, Panel(Text(reason, style="yellow"), title="NOT EXECUTED", title_align="left"))


def error_view(message: str, log_path: str | None = None) -> RenderableType:
    body = Text(message, style="bold red")
    if log_path:
        body.append(f"\nDetails were written to {log_path}", style="dim")
    return Panel(body, title="ERROR", title_align="left", border_style="red")
