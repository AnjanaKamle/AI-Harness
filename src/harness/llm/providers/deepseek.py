"""DeepSeek adapter (OpenAI-compatible Chat Completions).

Provider-specific behaviour (per DeepSeek's API documentation):
* documented base URL ``https://api.deepseek.com`` - used only when AI_BASE_URL is unset;
* reasoning mode returns ``reasoning_content``; it is kept as opaque provider data and sent
  back on the following assistant turn, as the API requires during tool-call turns;
* ``finish_reason = "insufficient_system_resource"`` is a transient server condition.
The model name always comes from AI_MODEL - none is assumed.
"""

from __future__ import annotations

from typing import Any

from harness.llm.client import LLMError, LLMErrorCode
from harness.llm.models import LLMResponse, Message, Role
from harness.llm.providers.openai_compat import OpenAICompatibleClient

REASONING_KEY = "deepseek.reasoning_content"


class DeepSeekClient(OpenAICompatibleClient):
    provider = "deepseek"
    display_name = "DeepSeek"
    default_base_url = "https://api.deepseek.com"  # documented in DeepSeek's API docs

    def message_to_wire(self, message: Message) -> list[dict[str, Any]]:
        wire = super().message_to_wire(message)
        if message.role is Role.ASSISTANT and message.provider_data.get(REASONING_KEY):
            wire[0]["reasoning_content"] = message.provider_data[REASONING_KEY]
        return wire

    def extra_provider_data(self, message: dict[str, Any]) -> dict[str, Any]:
        reasoning = message.get("reasoning_content")
        return {REASONING_KEY: reasoning} if isinstance(reasoning, str) and reasoning else {}

    def parse_response(self, data: Any) -> LLMResponse:
        try:
            reason = data["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError):
            reason = None
        if reason == "insufficient_system_resource":
            raise LLMError("DeepSeek reported insufficient system resources", retryable=True,
                           code=LLMErrorCode.SERVER_ERROR)
        return super().parse_response(data)


def create(settings: Any) -> DeepSeekClient:
    return DeepSeekClient(settings)
