"""Provider-neutral data structures exchanged with any LLM (text-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    ERROR = "error"
    OTHER = "other"


@dataclass(frozen=True)
class ToolDefinition:
    """What the model is told about a tool: name, purpose, JSON-schema of its input."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """The outcome of a tool call, sent back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    # Opaque data a provider adapter needs echoed back on later turns (e.g. a reasoning
    # trace). Only the adapter that produced it reads it; the rest of the harness ignores it.
    provider_data: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(Role.USER, content)

    @classmethod
    def assistant(
        cls,
        content: str,
        tool_calls: tuple[ToolCall, ...] = (),
        provider_data: dict[str, Any] | None = None,
    ) -> Message:
        return cls(Role.ASSISTANT, content, tool_calls=tool_calls,
                   provider_data=dict(provider_data or {}))

    @classmethod
    def tool(cls, *results: ToolResult) -> Message:
        return cls(Role.TOOL, tool_results=tuple(results))


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    content: str
    stop_reason: StopReason
    tool_calls: tuple[ToolCall, ...] = ()
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    provider_data: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> Message:
        """The assistant turn to append to the conversation history."""
        return Message.assistant(self.content, self.tool_calls, self.provider_data)
