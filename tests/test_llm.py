from __future__ import annotations

import pytest

from harness.config.settings import Settings
from harness.llm import (
    LLMClient,
    LLMNotConfiguredError,
    LLMResponse,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolResult,
    create_llm_client,
    register_provider,
)
from harness.llm import client as client_module

from .conftest import FAKE_KEY, MockLLMClient


def test_llm_client_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]


def test_mock_client_generate_with_tools() -> None:
    tool = ToolDefinition("read_file", "Read a file", {"type": "object", "properties": {}})
    scripted = LLMResponse(
        content="",
        stop_reason=StopReason.TOOL_USE,
        tool_calls=(ToolCall("call-1", "read_file", {"path": "a.py"}),),
    )
    llm = MockLLMClient([scripted])

    response = llm.generate([Message.system("sys"), Message.user("open a.py")], tools=[tool])

    assert response.wants_tools
    assert response.tool_calls[0].arguments == {"path": "a.py"}
    messages, tools = llm.calls[0]
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    assert tools == [tool]


def test_conversation_round_trip_types() -> None:
    call = ToolCall("call-1", "read_file", {"path": "a.py"})
    response = LLMResponse(content="reading", stop_reason=StopReason.TOOL_USE, tool_calls=(call,))
    assistant = response.to_message()
    result_msg = Message.tool(ToolResult("call-1", "print('hi')"))

    assert assistant.role is Role.ASSISTANT and assistant.tool_calls == (call,)
    assert result_msg.role is Role.TOOL
    assert result_msg.tool_results[0].tool_call_id == "call-1"
    assert not result_msg.tool_results[0].is_error


def test_mock_default_response() -> None:
    response = MockLLMClient().generate([Message.user("hi")])
    assert response.content == "ok" and response.stop_reason is StopReason.END_TURN
    assert response.usage.total_tokens == 0


def test_factory_requires_provider() -> None:
    with pytest.raises(LLMNotConfiguredError, match="AI_PROVIDER"):
        create_llm_client(Settings(api_key=FAKE_KEY))


def test_factory_unknown_provider() -> None:
    with pytest.raises(LLMNotConfiguredError, match="Unknown provider"):
        create_llm_client(Settings(api_key=FAKE_KEY, provider="nope"))


def test_registered_provider_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "_PROVIDERS", {})
    seen: list[Settings] = []

    def factory(settings: Settings) -> LLMClient:
        seen.append(settings)
        return MockLLMClient()

    register_provider("Test", factory)
    settings = Settings(api_key=FAKE_KEY, provider="test")
    assert isinstance(create_llm_client(settings), MockLLMClient)
    assert seen == [settings]
