"""Deterministic offline LLM for tests and dry runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from harness.llm.base import LLMClient, LLMResponse, Message, Usage

Responder = Callable[[list[Message], str | None], str]


class MockLLMClient(LLMClient):
    """Returns scripted replies in order, or a responder function's output.

    Every call is recorded in ``calls`` so tests can assert on prompts.
    """

    def __init__(
        self,
        responses: Iterable[str] | None = None,
        *,
        responder: Responder | None = None,
        model: str = "mock-model",
    ) -> None:
        self.model = model
        self._responses = list(responses or [])
        self._responder = responder
        self.calls: list[tuple[list[Message], str | None]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), system))
        if self._responses:
            text = self._responses.pop(0)
        elif self._responder is not None:
            text = self._responder(messages, system)
        else:
            text = f"[mock] {messages[-1].content if messages else ''}"
        return LLMResponse(
            text=text,
            model=self.model,
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=sum(len(m.content.split()) for m in messages),
                output_tokens=len(text.split()),
            ),
        )
