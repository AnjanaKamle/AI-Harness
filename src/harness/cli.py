"""Command-line entry point: ``harness <command>`` or ``python -m harness <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from harness import __version__
from harness.config import ConfigError, HarnessConfig, load_config
from harness.llm import LLMError, create_llm_client
from harness.logging_setup import setup_logging


def _cmd_config(config: HarnessConfig, args: argparse.Namespace) -> int:
    print(json.dumps(config.to_public_dict(), indent=2))
    return 0


def _cmd_ping(config: HarnessConfig, args: argparse.Namespace) -> int:
    llm = create_llm_client(config.llm)
    try:
        reply = llm.ask(args.prompt)
    except LLMError as exc:
        print(f"LLM check failed: {exc}", file=sys.stderr)
        return 1
    print(f"[{config.llm.provider}:{llm.model}] {reply}")
    return 0


def _cmd_run(config: HarnessConfig, args: argparse.Namespace) -> int:
    print(
        "The orchestrator run loop is not implemented yet (phase 1 is the foundation only).",
        file=sys.stderr,
    )
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description="Multi-agent AI coding harness")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--env-file", default=".env", help="path to .env file (default: .env)")
    parser.add_argument("--provider", help="override HARNESS_LLM_PROVIDER (anthropic|mock)")
    parser.add_argument("--log-level", help="override HARNESS_LOG_LEVEL")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="print the resolved configuration (secrets masked)").set_defaults(
        func=_cmd_config
    )

    ping = sub.add_parser("ping", help="send a one-line prompt to the configured LLM")
    ping.add_argument("prompt", nargs="?", default="Reply with exactly: pong")
    ping.set_defaults(func=_cmd_ping)

    run = sub.add_parser("run", help="run a task through the orchestrator (later phase)")
    run.add_argument("task", help="task description")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides: dict[str, str] = {}
    if args.provider:
        overrides["HARNESS_LLM_PROVIDER"] = args.provider
    if args.log_level:
        overrides["HARNESS_LOG_LEVEL"] = args.log_level

    try:
        config = load_config(args.env_file, environ={**os.environ, **overrides})
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.logging)
    return int(args.func(config, args))


if __name__ == "__main__":
    raise SystemExit(main())
