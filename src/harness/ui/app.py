"""Terminal UI session (presentation layer only).

The app asks for a task (or takes TASK=...), starts the Orchestrator on a background
thread, and renders a live dashboard from the Orchestrator's event stream. It never makes
orchestration decisions: the final view is rendered from the structured FinalResult.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live

from harness.config.settings import Settings
from harness.llm.client import LLMError
from harness.orchestrator.final_result import FinalResult
from harness.orchestrator.orchestrator import Orchestrator
from harness.orchestrator.verification_manager import FinalStatus
from harness.ui import render
from harness.ui.model import DashboardModel

log = logging.getLogger("harness.ui")

EXIT_OK, EXIT_CONFIG_ERROR, EXIT_BLOCKED, EXIT_FATAL, EXIT_INTERRUPTED = 0, 2, 4, 5, 130
QUIT_WORDS = {"", "q", "quit", "exit"}
REFRESH_SECONDS = 0.12
PLAN_ONLY_REASON = (
    "LIVE MODEL EXECUTION UNAVAILABLE. AI_API_KEY is read from the environment, but no model "
    "provider is configured (AI_PROVIDER / AI_MODEL): the official evaluation provider has not "
    "been specified, so none is assumed. The plan above was created; agents were not run. "
    "Configure the provider as described in README 'Model configuration', or run `make demo` "
    "for the scripted sample-repository demo."
)


def exit_code(final: FinalResult) -> int:
    return {FinalStatus.VERIFIED_SUCCESS: EXIT_OK, FinalStatus.BLOCKED: EXIT_BLOCKED,
            FinalStatus.FAILED: EXIT_FATAL}[final.status]


class HarnessApp:
    def __init__(
        self,
        settings: Settings,
        *,
        repo: str | Path,
        console: Console | None = None,
        input_fn: Callable[[str], str] | None = None,
        live: bool = True,
        agents: dict[str, Any] | None = None,
        orchestrator: Orchestrator | None = None,
        mode: str = "",
        log_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.repo = Path(repo)
        self.console = console or Console()
        self.input_fn = input_fn or (lambda prompt: self.console.input(prompt))
        self.live = live
        self.orchestrator = orchestrator or Orchestrator(settings)
        self._agents = agents
        self.mode = mode
        self.log_path = log_path
        self.last_final: FinalResult | None = None
        self.last_model: DashboardModel | None = None

    # --- session -----------------------------------------------------------------------------

    def session(self, task: str | None = None) -> int:
        """Run one task (TASK given) or an interactive loop until the user quits."""
        try:
            if task:
                return self.run_task(task)
            code = EXIT_OK
            while True:
                try:
                    entered = self.input_fn(
                        "\n[bold]Enter a task[/bold] [dim](empty or 'quit' to exit)[/dim]: "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if entered.lower() in QUIT_WORDS:
                    break
                code = self.run_task(entered)
            return code
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        music = (self._agents or {}).get("music")
        if music is not None and hasattr(music, "shutdown"):
            music.shutdown()

    # --- one task --------------------------------------------------------------------------------

    def run_task(self, task: str) -> int:
        model = DashboardModel(task, mode=self.mode)
        self.last_model = model
        if self.settings.provider is None:
            return self._plan_only(task, model)
        try:
            agents = self._ensure_agents()
        except (LLMError, ValueError) as exc:
            log.exception("could not create agents")
            self.console.print(render.error_view(f"Cannot start the agents: {exc}", self._log()))
            return EXIT_CONFIG_ERROR

        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                outcome["final"] = self.orchestrator.execute(
                    task, self.repo, agents=agents, listeners=[model.apply]
                )
            except Exception as exc:  # noqa: BLE001 - shown concisely; details go to the log
                log.exception("orchestration crashed")
                outcome["error"] = exc

        thread = threading.Thread(target=work, name="harness-orchestrator", daemon=True)
        try:
            thread.start()
            if self.live and self.console.is_terminal:
                with Live(render.dashboard(model), console=self.console, refresh_per_second=8,
                          transient=False) as live:
                    while thread.is_alive():
                        thread.join(REFRESH_SECONDS)
                        live.update(render.dashboard(model))
                    live.update(render.dashboard(model))
            else:
                thread.join()
                self.console.print(render.dashboard(model))
        except KeyboardInterrupt:
            self.console.print(render.error_view(
                "Interrupted. Running operations were not waited for; the repository may contain "
                "partial changes.", self._log()))
            return EXIT_INTERRUPTED

        if "error" in outcome:
            error = outcome["error"]
            self.console.print(render.error_view(
                f"The harness hit an unexpected error: {type(error).__name__}: {error}", self._log()))
            return EXIT_FATAL
        final: FinalResult = outcome["final"]
        self.last_final = final
        self.console.print(render.final_view(final))
        if final.unresolved_issues and self._log():
            self.console.print(f"[dim]Detailed log: {self._log()}[/dim]")
        return exit_code(final)

    def _plan_only(self, task: str, model: DashboardModel) -> int:
        state = self.orchestrator.plan_graph(task, self.repo, listeners=[model.apply])
        model.mark_plan_only("Plan created; agents not run (no AI_PROVIDER)")
        self.console.print(render.dashboard(model))
        self.console.print(render.plan_view(state, PLAN_ONLY_REASON))
        return EXIT_OK

    def _ensure_agents(self) -> dict[str, Any]:
        if self._agents is None:
            self._agents = self.orchestrator.build_agents(
                self.orchestrator_repo(), include_music=True
            )
        return self._agents

    def orchestrator_repo(self) -> Any:
        from harness.tools.repository import RepositoryContext

        return RepositoryContext(self.repo)

    def _log(self) -> str | None:
        return str(self.log_path) if self.log_path else None
