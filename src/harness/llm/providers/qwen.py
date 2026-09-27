"""Qwen adapter (OpenAI-compatible Chat Completions, e.g. Alibaba Cloud Model Studio's
compatible mode or a self-hosted Qwen server).

Provider-specific behaviour:
* no default endpoint: Model Studio's documented endpoints are region- and
  workspace-specific, so AI_BASE_URL is required (a clear CONFIGURATION_ERROR otherwise);
* some Qwen deployments return tool calls as ``<tool_call>{"name":..,"arguments":..}</tool_call>``
  blocks inside the text instead of structured ``tool_calls``; those are parsed here;
* reasoning models may inline ``<think>...</think>``; it is stripped from the content.
The model name always comes from AI_MODEL - none is assumed.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from harness.llm.models import LLMResponse, StopReason, ToolCall
from harness.llm.providers.openai_compat import OpenAICompatibleClient

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class QwenClient(OpenAICompatibleClient):
    provider = "qwen"
    display_name = "Qwen"
    default_base_url = None  # region/workspace-specific: must come from AI_BASE_URL

    def parse_response(self, data: Any) -> LLMResponse:
        response = super().parse_response(data)
        if response.tool_calls or "<tool_call>" not in response.content:
            return response
        calls: list[ToolCall] = []
        for index, block in enumerate(_TOOL_CALL_BLOCK.findall(response.content)):
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                calls.append(ToolCall(f"call_{uuid.uuid4().hex[:12]}_{index}", "<malformed>", block))  # type: ignore[arg-type]
                continue
            arguments = parsed.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            calls.append(ToolCall(f"call_{uuid.uuid4().hex[:12]}_{index}",
                                  str(parsed.get("name", "")), arguments))
        if not calls:
            return response
        text = _TOOL_CALL_BLOCK.sub("", response.content).strip()
        return LLMResponse(text, StopReason.TOOL_USE, tuple(calls), response.model, response.usage,
                           response.provider_data)


def create(settings: Any) -> QwenClient:
    return QwenClient(settings)
