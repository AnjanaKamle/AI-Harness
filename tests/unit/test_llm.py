from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from harness.config import ConfigError, LLMConfig
from harness.llm import (
    LLMError,
    LLMRefusalError,
    Message,
    MockLLMClient,
    create_llm_client,
)
from harness.llm.anthropic_client import FALLBACK_BETA, AnthropicLLMClient


def test_mock_scripted_responses_then_echo() -> None:
    llm = MockLLMClient(["first", "second"])
    assert llm.ask("a") == "first"
    assert llm.ask("b") == "second"
    assert llm.ask("c") == "[mock] c"
    assert [call[0][-1].content for call in llm.calls] == ["a", "b", "c"]


def test_mock_responder_receives_system_prompt() -> None:
    llm = MockLLMClient(responder=lambda msgs, system: f"{system}|{msgs[-1].content}")
    assert llm.ask("hi", system="sys") == "sys|hi"


def test_factory_mock() -> None:
    assert isinstance(create_llm_client(LLMConfig(provider="mock")), MockLLMClient)


def test_factory_unknown_provider() -> None:
    with pytest.raises(ConfigError):
        create_llm_client(LLMConfig(provider="nope"))


# --- Anthropic client (SDK stubbed; no network) --------------------------------------------


def _response(
    *, text: str = "hello", stop_reason: str = "end_turn", category: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        model="claude-opus-5",
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=3),
    )


class _Messages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _FakeSDK:
    def __init__(self, result: Any) -> None:
        self.messages = _Messages(result)
        self.beta = SimpleNamespace(messages=_Messages(result))


def test_anthropic_complete_with_fallback() -> None:
    sdk = _FakeSDK(_response())
    llm = AnthropicLLMClient(LLMConfig(), client=sdk)
    resp = llm.complete([Message("user", "hi")], system="be brief")

    assert resp.text == "hello"
    assert resp.usage.total_tokens == 13
    sent = sdk.beta.messages.kwargs
    assert sent["model"] == "claude-opus-5"
    assert sent["system"] == "be brief"
    assert sent["betas"] == [FALLBACK_BETA]
    assert sent["extra_body"] == {"fallbacks": "default"}
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_complete_without_fallback() -> None:
    sdk = _FakeSDK(_response())
    llm = AnthropicLLMClient(LLMConfig(refusal_fallback=False, max_tokens=123), client=sdk)
    llm.complete([Message("user", "hi")])
    assert "betas" not in sdk.messages.kwargs
    assert "system" not in sdk.messages.kwargs
    assert sdk.messages.kwargs["max_tokens"] == 123


def test_anthropic_refusal_raises() -> None:
    sdk = _FakeSDK(_response(stop_reason="refusal", category="cyber"))
    llm = AnthropicLLMClient(LLMConfig(), client=sdk)
    with pytest.raises(LLMRefusalError, match="cyber"):
        llm.ask("hi")


def _status_error(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("err", response=httpx.Response(code, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (lambda: _status_error(anthropic.RateLimitError, 429), True),
        (lambda: _status_error(anthropic.InternalServerError, 500), True),
        (lambda: _status_error(anthropic.BadRequestError, 400), False),
        (lambda: _status_error(anthropic.AuthenticationError, 401), False),
        (
            lambda: anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            True,
        ),
    ],
)
def test_anthropic_errors_are_wrapped(error: Any, retryable: bool) -> None:
    llm = AnthropicLLMClient(LLMConfig(refusal_fallback=False), client=_FakeSDK(error()))
    with pytest.raises(LLMError) as info:
        llm.ask("hi")
    assert info.value.retryable is retryable
