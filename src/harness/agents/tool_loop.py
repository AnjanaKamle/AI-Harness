"""Generic LLM tool-calling loop.

    while the model requests tools:
        execute each call -> append results -> ask the model again

Bounded by ``max_tool_calls``. When the budget runs out, remaining calls get an error
result and the model is asked once, without tools, for its final answer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from harness.context.usage import UsageStats
from harness.llm.client import LLMClient
from harness.llm.models import LLMResponse, Message, ToolCall, ToolResult
from harness.tools.base import ToolExecutionResult, ToolStatus
from harness.tools.registry import ToolRegistry

log = logging.getLogger("harness.agents.tool_loop")

LIMIT_REACHED_PROMPT = (
    "The tool call limit ({limit}) has been reached. No more tools are available. "
    "Reply now with your final result in the required JSON format."
)
_RECORD_ARG_CHARS = 200


class OperationCancelled(RuntimeError):
    """The orchestrator cancelled this agent run (e.g. it exceeded its time limit)."""


@dataclass
class ToolCallRecord:
    index: int
    call_id: str
    name: str
    arguments: Any
    status: ToolStatus
    error: str | None
    duration_ms: float
    data: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Compact, log-safe view (long argument values such as file contents are clipped)."""

        def clip(value: Any) -> Any:
            if isinstance(value, str) and len(value) > _RECORD_ARG_CHARS:
                return value[:_RECORD_ARG_CHARS] + f"…[{len(value)} chars]"
            return value

        args = (
            {k: clip(v) for k, v in self.arguments.items()}
            if isinstance(self.arguments, dict)
            else clip(self.arguments)
        )
        return {
            "index": self.index,
            "name": self.name,
            "arguments": args,
            "status": self.status.value,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class LoopOutcome:
    final_text: str | None
    messages: list[Message]
    records: list[ToolCallRecord]
    limit_reached: bool
    iterations: int
    last_response: LLMResponse | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class ToolCallingLoop:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        *,
        max_tool_calls: int,
        max_tokens: int | None = None,
    ) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        self.llm = llm
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.usage = UsageStats()

    def _generate(self, history: list[Message], tools: list[Any] | None) -> LLMResponse:
        response = self.llm.generate(history, tools=tools, max_tokens=self.max_tokens)
        self.usage.record_turn(history, tools, response)
        return response

    def run(self, messages: Sequence[Message]) -> LoopOutcome:
        history = list(messages)
        records: list[ToolCallRecord] = []
        tools = self.registry.definitions()
        iterations = 0
        response: LLMResponse | None = None

        # Each iteration either ends the loop or consumes >= 1 call of the budget, so this
        # bound is only a backstop against misbehaving clients.
        while iterations <= self.max_tool_calls:
            iterations += 1
            token = self.registry.cancel_token
            if token is not None and token.cancelled:
                raise OperationCancelled(token.reason)
            response = self._generate(history, tools)
            history.append(response.to_message())
            if not response.tool_calls:
                return self._outcome(response.content, history, records, False, iterations, response)

            results: list[ToolResult] = []
            exhausted = False
            for call in response.tool_calls:
                if len(records) >= self.max_tool_calls:
                    exhausted = True
                    results.append(self._budget_error(call))
                    continue
                results.append(self._execute(call, records))
            history.append(Message.tool(*results))

            if exhausted or len(records) >= self.max_tool_calls:
                log.warning("tool call limit (%d) reached", self.max_tool_calls)
                return self._finish_without_tools(history, records, iterations)

        return self._finish_without_tools(history, records, iterations)

    def finalize(self, history: list[Message], prompt: str) -> LLMResponse:
        """Ask the model a follow-up without offering tools (e.g. to repair its final format)."""
        history.append(Message.user(prompt))
        response = self._generate(history, None)
        history.append(response.to_message())
        return response

    # --- internals -----------------------------------------------------------------------

    def _execute(self, call: ToolCall, records: list[ToolCallRecord]) -> ToolResult:
        name = call.name if isinstance(call.name, str) and call.name else "<missing>"
        result: ToolExecutionResult
        result, tool_result = self.registry.execute_call(
            ToolCall(call.id, name, call.arguments)  # type: ignore[arg-type]
        )
        records.append(
            ToolCallRecord(
                index=len(records) + 1,
                call_id=call.id,
                name=name,
                arguments=call.arguments,
                status=result.status,
                error=result.error,
                duration_ms=float(result.metadata.get("duration_ms", 0.0)),
                data=result.data,
            )
        )
        return tool_result

    def _budget_error(self, call: ToolCall) -> ToolResult:
        log.warning("skipping tool call %s: budget exhausted", call.name)
        return ToolResult(
            call.id,
            f'{{"ok": false, "error": "tool call limit ({self.max_tool_calls}) reached; not executed"}}',
            is_error=True,
        )

    def _finish_without_tools(
        self, history: list[Message], records: list[ToolCallRecord], iterations: int
    ) -> LoopOutcome:
        response = self.finalize(history, LIMIT_REACHED_PROMPT.format(limit=self.max_tool_calls))
        final = None if response.tool_calls else response.content
        return self._outcome(final, history, records, True, iterations + 1, response)

    def _outcome(
        self,
        final: str | None,
        history: list[Message],
        records: list[ToolCallRecord],
        limit_reached: bool,
        iterations: int,
        response: LLMResponse,
    ) -> LoopOutcome:
        self.usage.tool_calls = len(records)
        return LoopOutcome(
            final, history, records, limit_reached, iterations, response, self.usage.to_dict()
        )
