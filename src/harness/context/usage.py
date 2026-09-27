"""Approximate context/token accounting for observability.

Exact token counts are only known when the provider reports them (``LLMResponse.usage``);
otherwise sizes are estimated from characters (``estimated_tokens`` ~= chars / 4) and are
labelled as estimates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from harness.llm.models import LLMResponse, Message, ToolDefinition

CHARS_PER_TOKEN = 4


def estimated_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN) if chars > 0 else 0


def message_chars(message: Message) -> int:
    size = len(message.content)
    for call in message.tool_calls:
        size += len(call.name) + len(json.dumps(call.arguments, default=str))
    for result in message.tool_results:
        size += len(result.content)
    return size


def request_chars(messages: Sequence[Message], tools: Sequence[ToolDefinition] | None) -> int:
    size = sum(message_chars(m) for m in messages)
    for tool in tools or ():
        size += len(tool.name) + len(tool.description) + len(json.dumps(tool.input_schema))
    return size


@dataclass
class UsageStats:
    llm_turns: int = 0
    tool_calls: int = 0
    input_chars: int = 0
    output_chars: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0

    def record_turn(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None,
        response: LLMResponse,
    ) -> None:
        self.llm_turns += 1
        self.input_chars += request_chars(messages, tools)
        self.output_chars += message_chars(response.to_message())
        self.reported_input_tokens += response.usage.input_tokens
        self.reported_output_tokens += response.usage.output_tokens

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["estimated_input_tokens"] = estimated_tokens(self.input_chars)
        data["estimated_output_tokens"] = estimated_tokens(self.output_chars)
        return data


_SUMMABLE = (
    "llm_turns", "tool_calls", "input_chars", "output_chars", "reported_input_tokens",
    "reported_output_tokens", "estimated_input_tokens", "estimated_output_tokens",
    "context_chars", "estimated_context_tokens", "runs",
)


def accumulate_usage(totals: dict[str, Any], agent: str, usage: dict[str, Any]) -> None:
    """Add one agent run's usage into ``totals`` (per agent and overall)."""
    usage = {**usage, "runs": 1}
    for bucket in (agent, "total"):
        target = totals.setdefault(bucket, {})
        for key in _SUMMABLE:
            if key in usage and isinstance(usage[key], (int, float)):
                target[key] = target.get(key, 0) + usage[key]
