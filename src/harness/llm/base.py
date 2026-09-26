"""Provider-agnostic LLM interface used by every agent.

Agents depend only on :class:`LLMClient`; concrete providers live in sibling modules and
are chosen by :func:`harness.llm.factory.create_llm_client`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class LLMError(RuntimeError):
    """Base class for LLM failures surfaced to the harness."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class LLMRefusalError(LLMError):
    """The model (and any configured fallback chain) declined the request."""


class LLMClient(ABC):
    """Minimal chat-completion contract. Extended in later phases (tools, streaming)."""

    model: str

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return the model's reply to ``messages``. Raises :class:`LLMError` on failure."""

    def ask(self, prompt: str, *, system: str | None = None) -> str:
        """Convenience single-turn helper."""
        return self.complete([Message("user", prompt)], system=system).text
