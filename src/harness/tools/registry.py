"""ToolRegistry: name -> tool lookup, schemas for the LLM, and execution by name."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

from harness.config.settings import Settings
from harness.llm.models import ToolCall, ToolDefinition, ToolResult
from harness.tools.base import BaseTool, ToolExecutionResult
from harness.tools.filesystem import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from harness.tools.git import GitDiffTool, GitLogTool, GitStatusTool
from harness.tools.policy import CommandPolicy
from harness.tools.repository import RepositoryContext
from harness.tools.search import SearchCodeTool
from harness.tools.terminal import TerminalTool
from harness.tools.testing import build_testing_tools

log = logging.getLogger("harness.tools.registry")

DEFAULT_MAX_LLM_CHARS = 12_000
DEFAULT_MAX_LLM_LINES = 400
_LOG_ARG_CHARS = 200


def redact_credential(text: str) -> str:
    """Never let the harness credential reach a model through a tool result (e.g. a file
    or command output that happens to contain it)."""
    import os

    key = os.environ.get("AI_API_KEY", "")
    return text.replace(key, "[REDACTED]") if len(key) >= 6 and key in text else text


class CancellationToken:
    """Cooperative cancellation: once cancelled, no further tool calls are executed."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    def cancel(self, reason: str) -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


# observer(tool_name, arguments, result, attempt) - called after every execution attempt
ToolObserver = Callable[[str, Any, ToolExecutionResult, int], None]
RETRY_BACKOFF_SECONDS = 0.2


class UnknownToolError(KeyError):
    def __init__(self, name: str, available: Iterable[str]) -> None:
        super().__init__(f"Unknown tool {name!r}. Available tools: {sorted(available)}")
        self.name = name

    def __str__(self) -> str:
        return str(self.args[0])


def _brief(arguments: Any) -> str:
    text = json.dumps(arguments, ensure_ascii=False, default=str)
    return text if len(text) <= _LOG_ARG_CHARS else text[:_LOG_ARG_CHARS] + "…"


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[BaseTool] = (),
        *,
        max_llm_chars: int = DEFAULT_MAX_LLM_CHARS,
        max_llm_lines: int = DEFAULT_MAX_LLM_LINES,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.max_llm_chars = max_llm_chars
        self.max_llm_lines = max_llm_lines
        self.max_tool_retries = 0
        self.observer: ToolObserver | None = None
        self.before: Callable[[str, Any], None] | None = None  # called as a tool starts
        self.cancel_token: CancellationToken | None = None
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        name = getattr(tool, "name", None)
        if not name or not isinstance(name, str):
            raise ValueError(f"Tool {type(tool).__name__} has no name")
        if name in self._tools:
            raise ValueError(f"Tool {name!r} is already registered")
        self._tools[name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(name, self._tools) from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        return [self._tools[n].definition() for n in self.names]

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": d.name, "description": d.description, "input_schema": d.input_schema}
            for d in self.definitions()
        ]

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        """A registry holding only ``names`` (each must exist)."""
        return ToolRegistry(
            (self.get(n) for n in names),
            max_llm_chars=self.max_llm_chars,
            max_llm_lines=self.max_llm_lines,
        )

    def execute(self, name: str, arguments: Any) -> ToolExecutionResult:
        """Run a tool by name. Unknown names and malformed arguments become FAILURE results.
        Failures a tool marks ``retryable`` are retried up to ``max_tool_retries`` times; each
        retry and its reason is recorded in ``result.metadata["retries"]``."""
        retries: list[dict[str, Any]] = []
        attempt = 0
        while True:
            attempt += 1
            if self.before is not None:
                try:
                    self.before(name, arguments)
                except Exception:  # noqa: BLE001 - observability must never break execution
                    log.exception("tool start observer failed")
            result = self._execute_once(name, arguments)
            if self.observer is not None:
                try:
                    self.observer(name, arguments, result, attempt)
                except Exception:  # noqa: BLE001 - observability must never break execution
                    log.exception("tool observer failed")
            transient = bool(result.metadata.get("retryable"))
            if result.ok or not transient or attempt > self.max_tool_retries:
                break
            if self.cancel_token is not None and self.cancel_token.cancelled:
                break
            retries.append({"attempt": attempt, "reason": result.error or "transient failure"})
            log.info("retrying tool %s (attempt %d): %s", name, attempt + 1, result.error)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        if retries:
            result.metadata["retries"] = retries
        return result

    def _execute_once(self, name: str, arguments: Any) -> ToolExecutionResult:
        if self.cancel_token is not None and self.cancel_token.cancelled:
            result = ToolExecutionResult.failure(
                name, f"Operation cancelled ({self.cancel_token.reason}); tool not executed"
            )
            log.info("tool call %s refused: cancelled", name)
            return result
        tool = self._tools.get(name)
        if isinstance(arguments, str):  # some providers deliver raw JSON text
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                result = ToolExecutionResult.failure(
                    name, f"MALFORMED_TOOL_CALL: Arguments for {name} are not valid JSON: {exc.msg}"
                )
                log.info("tool call %s(<malformed>) -> %s: %s", name, result.status, result.error)
                return result
        if tool is None:
            result = ToolExecutionResult.failure(name, str(UnknownToolError(name, self._tools)))
        elif not isinstance(arguments, dict):
            result = ToolExecutionResult.failure(
                name, f"Arguments for {name} must be a JSON object, got {type(arguments).__name__}"
            )
        else:
            result = tool.run(arguments)
        log.info(
            "tool call %s(%s) -> %s%s",
            name,
            _brief(arguments),
            result.status,
            f": {result.error}" if result.error else "",
        )
        return result

    def execute_call(self, call: ToolCall) -> tuple[ToolExecutionResult, ToolResult]:
        """Execute a model-issued tool call and build the ToolResult to send back."""
        result = self.execute(call.name, call.arguments)
        return result, ToolResult(
            tool_call_id=call.id,
            content=redact_credential(result.to_llm_content(self.max_llm_chars, self.max_llm_lines)),
            is_error=not result.ok,
        )


CODER_TOOL_NAMES: tuple[str, ...] = (
    "list_files",
    "read_file",
    "search_code",
    "write_file",
    "edit_file",
    "terminal",
    "git_status",
    "git_diff",
)

# The Tester gathers evidence only: no write_file, edit_file or free-form terminal.
TESTER_TOOL_NAMES: tuple[str, ...] = (
    "list_files",
    "read_file",
    "search_code",
    "git_status",
    "git_diff",
    "detect_tests",
    "run_tests",
)


def registry_limits(settings: Settings | None) -> dict[str, int]:
    """Per-result output limits for a registry, from settings."""
    if settings is None:
        return {}
    return {
        "max_llm_chars": settings.max_tool_output_chars,
        "max_llm_lines": settings.max_tool_output_lines,
    }


def build_default_tools(
    repo: RepositoryContext,
    settings: Settings | None = None,
    *,
    policy: CommandPolicy | None = None,
) -> list[BaseTool]:
    """All Phase 2 tools bound to ``repo``, sized from ``settings`` when given."""
    max_chars = settings.max_tool_output_chars if settings else DEFAULT_MAX_LLM_CHARS
    timeout = settings.command_timeout_seconds if settings else 120.0
    test_timeout = settings.test_timeout_seconds if settings else 300.0
    return [
        ListFilesTool(repo),
        ReadFileTool(repo, max_chars=max_chars),
        WriteFileTool(repo),
        EditFileTool(repo),
        SearchCodeTool(
            repo, max_results_cap=settings.max_search_results if settings else None
        ),
        TerminalTool(repo, policy, default_timeout=timeout, max_output_chars=max_chars),
        GitStatusTool(repo),
        GitDiffTool(repo, max_chars=max_chars),
        GitLogTool(repo),
        *build_testing_tools(repo, test_timeout),
    ]


def build_registry(
    repo: RepositoryContext,
    settings: Settings | None = None,
    names: Iterable[str] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(build_default_tools(repo, settings), **registry_limits(settings))
    return registry.subset(names) if names is not None else registry
