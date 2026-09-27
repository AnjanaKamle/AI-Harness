"""Application entry point: ``python -m harness.main [--task TEXT] [--repo PATH]``.

Task source, in priority order: ``--task`` argument, ``TASK`` environment variable,
interactive prompt. Repository: ``--repo``, else ``REPO``, else the current directory.

Output:
  * interactive terminal -> the TUI (live dashboard + final report); ``--json`` disables it
  * otherwise (pipes, CI, tests) -> JSON on stdout
With an LLM provider configured (``AI_PROVIDER``) the autonomous controller plans a task
graph, schedules the agents, repairs failures and verifies the result. Without one, only the
task graph is produced. ``AI_PROVIDER=demo`` runs the scripted demo on a temporary copy of
the bundled sample repository. Detailed logs always go to a log file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence

from harness import __version__
from harness.config.settings import ConfigurationError, Settings
from harness.demo import DEMO_PROVIDER, prepare_sample_repo, register_demo_provider
from harness.llm.client import LLMError, create_llm_client
from harness.llm.models import Message
from harness.llm.providers import is_scripted
from harness.orchestrator.orchestrator import Orchestrator
from harness.orchestrator.verification_manager import FinalStatus
from harness.logging_setup import configure_logging

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_NO_TASK = 3
EXIT_BLOCKED = 4
EXIT_FATAL = 5

LIVE_EXECUTION_UNAVAILABLE = (
    "No LLM provider configured. Configure AI_PROVIDER and AI_MODEL before live execution. "
    "CONFIGURATION_ERROR: Live model execution is unavailable: AI_API_KEY is set, but no "
    "provider is configured (e.g. AI_PROVIDER=deepseek or AI_PROVIDER=qwen with AI_MODEL and, "
    "for Qwen, AI_BASE_URL). The task was not executed; printing the planned task graph only."
)

log = logging.getLogger("harness")


def read_task(cli_task: str | None) -> str | None:
    for candidate in (cli_task, os.environ.get("TASK")):
        if candidate and candidate.strip():
            return candidate.strip()
    try:
        if sys.stdin.isatty():
            return input("Enter a task: ").strip() or None
        return sys.stdin.read().strip() or None
    except (EOFError, KeyboardInterrupt):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="Multi-agent AI coding harness")
    parser.add_argument("--task", "-t", help="task description (else $TASK, else prompt)")
    parser.add_argument("--repo", "-r", help="repository to work on (else $REPO, else cwd)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print JSON instead of the TUI")
    output.add_argument("--tui", action="store_true", help="force the TUI")
    parser.add_argument("--check", action="store_true",
                        help="check the configured LLM provider (one minimal request) and exit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def provider_label(settings: Settings) -> str:
    """Safe provider/model description for diagnostics and the TUI (never the key)."""
    if settings.provider is None:
        return "no provider configured"
    return f"provider: {settings.provider.strip().lower()} · model: {settings.model or '(not set)'}"


def check_provider(settings: Settings) -> int:
    """Live connectivity check with one minimal text request (``--check``, ``make check``)."""
    print(f"Provider: {settings.provider or '(not set)'}")
    print(f"Model:    {settings.model or '(not set)'}")
    print(f"API key:  {'present' if settings.api_key else 'missing'}")
    try:
        client = create_llm_client(settings)
    except LLMError as exc:
        print(f"Status:   {exc.code.value}\n          {exc}")
        return EXIT_CONFIG_ERROR
    describe = getattr(client, "describe", None)
    if callable(describe):
        print(f"Endpoint: {describe().get('endpoint', '-')}")
    try:
        response = client.generate([Message.user("Reply with the single word OK.")], max_tokens=16)
    except LLMError as exc:
        print(f"Status:   {exc.code.value}\n          {exc}")
        return EXIT_CONFIG_ERROR
    print(f"Status:   CONNECTED (model reported: {response.model or settings.model})")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"CONFIGURATION_ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.check:
        configure_logging(settings.log_level_value, log_file=settings.log_file, console=False)
        return check_provider(settings)

    use_tui = args.tui or (not args.json and sys.stdout.isatty())
    log_path = configure_logging(settings.log_level_value, log_file=settings.log_file,
                                 console=not use_tui)
    log.debug("settings: %s", settings.public_dict())
    log.info("LLM configuration: %s", provider_label(settings))

    scripted = is_scripted(settings.provider)
    if scripted:
        register_demo_provider()
    try:
        repo = args.repo or os.environ.get("REPO") or (
            str(prepare_sample_repo()) if scripted else os.getcwd()
        )
    except Exception as exc:  # noqa: BLE001 - concise message; details in the log
        log.exception("could not prepare the demo repository")
        print(f"Cannot prepare the demo repository: {exc}", file=sys.stderr)
        return EXIT_FATAL

    if use_tui:
        from harness.ui.app import HarnessApp

        task = next((t.strip() for t in (args.task, os.environ.get("TASK")) if t and t.strip()), None)
        if scripted:
            mode = "SCRIPTED MODE - demo client, sample repository"
        elif settings.provider is None:
            mode = "CONFIGURATION_ERROR - no LLM provider configured"
        else:
            mode = provider_label(settings)
        app = HarnessApp(settings, repo=repo, mode=mode, log_path=log_path)
        return app.session(task)

    task = read_task(args.task)
    if task is None:
        print("No task provided. Use --task, TASK=..., or type one when prompted.", file=sys.stderr)
        return EXIT_NO_TASK

    orchestrator = Orchestrator(settings)
    try:
        if settings.provider is None:
            print(LIVE_EXECUTION_UNAVAILABLE, file=sys.stderr)
            log.warning("CONFIGURATION_ERROR: no LLM provider configured")
            state = orchestrator.plan_graph(task, repo)
            state.outcome = {"status": "CONFIGURATION_ERROR", "reason": LIVE_EXECUTION_UNAVAILABLE}
            print(state.to_json())
            return EXIT_OK
        final = orchestrator.execute(task, repo)
    except LLMError as exc:
        print(f"{exc.code.value}: {exc}", file=sys.stderr)
        print(json.dumps({"status": exc.code.value, "error": str(exc), "task": task}, indent=2))
        return EXIT_CONFIG_ERROR
    except ValueError as exc:
        print(f"CONFIGURATION_ERROR: Cannot run the harness: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except Exception as exc:  # noqa: BLE001 - clean error for the user; details in the log
        log.exception("unexpected error")
        print(f"Harness error: {type(exc).__name__}: {exc}"
              + (f" (details: {log_path})" if log_path else ""), file=sys.stderr)
        return EXIT_FATAL
    print(final.to_json())
    return {
        FinalStatus.VERIFIED_SUCCESS: EXIT_OK,
        FinalStatus.BLOCKED: EXIT_BLOCKED,
        FinalStatus.FAILED: EXIT_FATAL,
    }[final.status]


if __name__ == "__main__":
    raise SystemExit(main())
