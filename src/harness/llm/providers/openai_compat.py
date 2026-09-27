"""Shared base for providers that speak the OpenAI-compatible Chat Completions protocol.

Translation (internal <-> wire) lives here once; provider adapters subclass it and
override only what genuinely differs (defaults, extra response fields, quirks).

    internal Message / ToolDefinition / ToolCall / ToolResult
        -> {"model", "messages": [...], "tools": [...]}  -> POST {base_url}/chat/completions
        <- {"choices": [{"message": {...}, "finish_reason"}], "usage": {...}}
        -> LLMResponse(content, stop_reason, tool_calls, usage)

Provider objects never leave the adapter. Failures become ``LLMError`` with a
provider-neutral ``LLMErrorCode``; credentials never appear in messages.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from typing import Any

from harness.config.settings import BASE_URL_ENV, MODEL_ENV, Settings
from harness.llm.client import LLMClient, LLMError, LLMErrorCode, LLMNotConfiguredError
from harness.llm.models import (
    LLMResponse,
    Message,
    Role,
    StopReason,
    ToolCall,
    ToolDefinition,
    Usage,
)
from harness.llm.providers.transport import (
    HttpTransport,
    Transport,
    TransportNetworkError,
    TransportTimeout,
)

log = logging.getLogger("harness.llm.provider")

_FINISH = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
MAX_BACKOFF_SECONDS = 20.0
_KEY_ECHO = re.compile(r"(?i)(api[\s_-]?key\W{0,3})\S+")


def strip_reasoning_tags(text: str) -> str:
    """Remove <think>...</think> blocks some reasoning models emit inline."""
    return _THINK_BLOCK.sub("", text).strip() if "<think" in text.lower() else text


class OpenAICompatibleClient(LLMClient):
    """Base adapter. Subclasses set ``provider``/``display_name``/``default_base_url``."""

    provider = "openai-compatible"
    display_name = "OpenAI-compatible"
    default_base_url: str | None = None  # only set when the provider documents one
    requires_model = True

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Transport | None = None,
        backoff_seconds: float = 1.0,
    ) -> None:
        if not settings.model:
            raise LLMNotConfiguredError(
                f"Provider {self.provider!r} requires {MODEL_ENV}; no model name is assumed."
            )
        base_url = settings.base_url or self.default_base_url
        if not base_url:
            raise LLMNotConfiguredError(
                f"Provider {self.provider!r} requires {BASE_URL_ENV}: its endpoint depends on "
                "the account/region, so none is assumed."
            )
        self.model = settings.model
        self.base_url = base_url.rstrip("/")
        self._api_key = settings.api_key  # never logged, never put in messages
        self.timeout = settings.timeout_seconds
        self.max_retries = settings.max_retries
        self.max_output_tokens = settings.max_output_tokens
        self.transport: Transport = transport or HttpTransport()
        self.backoff_seconds = backoff_seconds

    # --- diagnostics ----------------------------------------------------------------------

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def describe(self) -> dict[str, str]:
        """Safe diagnostics (never includes the credential)."""
        return {"provider": self.provider, "model": self.model, "endpoint": self.endpoint}

    def _redact(self, text: str) -> str:
        if self._api_key and len(self._api_key) >= 4:
            text = text.replace(self._api_key, "[REDACTED]")
        # providers sometimes echo a masked key ("api key: ****3456"): drop even that
        return _KEY_ECHO.sub(r"\1[REDACTED]", text)

    # --- request translation ------------------------------------------------------------------

    def message_to_wire(self, message: Message) -> list[dict[str, Any]]:
        if message.role is Role.TOOL:
            return [{"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content}
                    for r in message.tool_results]
        if message.role is Role.ASSISTANT:
            wire: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                wire["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.name, "arguments": _arguments_json(c.arguments)}}
                    for c in message.tool_calls
                ]
            return [wire]
        return [{"role": message.role.value, "content": message.content}]

    def tool_to_wire(self, tool: ToolDefinition) -> dict[str, Any]:
        return {"type": "function",
                "function": {"name": tool.name, "description": tool.description,
                             "parameters": tool.input_schema}}

    def build_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "stream": False, "messages": [
            wire for message in messages for wire in self.message_to_wire(message)
        ]}
        if tools:
            payload["tools"] = [self.tool_to_wire(t) for t in tools]
            payload["tool_choice"] = "auto"
        tokens = max_tokens or self.max_output_tokens
        if tokens:
            payload["max_tokens"] = tokens
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    # --- response translation -------------------------------------------------------------------

    def parse_tool_calls(self, message: dict[str, Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            if not isinstance(raw, dict):
                raise LLMError("tool call is not an object", retryable=True,
                               code=LLMErrorCode.MALFORMED_TOOL_CALL)
            function = raw.get("function") or {}
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or not name.strip():
                raise LLMError("tool call without a function name", retryable=True,
                               code=LLMErrorCode.MALFORMED_TOOL_CALL)
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    # keep the raw text: the tool registry rejects it (MALFORMED_TOOL_CALL)
                    # and the model sees exactly why - the call is never executed.
                    pass
            call_id = raw.get("id") or f"call_{uuid.uuid4().hex[:12]}_{index}"
            calls.append(ToolCall(str(call_id), name.strip(), arguments))  # type: ignore[arg-type]
        return calls

    def extra_provider_data(self, message: dict[str, Any]) -> dict[str, Any]:
        """Hook: provider-specific fields to echo back on later turns."""
        return {}

    def finish_reason(self, reason: str | None) -> StopReason:
        return _FINISH.get(reason or "", StopReason.OTHER)

    def parse_response(self, data: Any) -> LLMResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.display_name} returned an unexpected response shape ({exc})",
                           retryable=True, code=LLMErrorCode.MALFORMED_RESPONSE) from None
        content = message.get("content") or ""
        if not isinstance(content, str):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        tool_calls = self.parse_tool_calls(message)
        usage = data.get("usage") or {}
        stop = self.finish_reason(choice.get("finish_reason"))
        if tool_calls and stop is not StopReason.TOOL_USE:
            stop = StopReason.TOOL_USE
        return LLMResponse(
            content=strip_reasoning_tags(content),
            stop_reason=stop,
            tool_calls=tuple(tool_calls),
            model=data.get("model") or self.model,
            usage=Usage(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)),
            provider_data=self.extra_provider_data(message),
        )

    # --- transport + error mapping -----------------------------------------------------------------

    def _error_for_status(self, status: int, body: bytes) -> LLMError:
        detail = ""
        try:
            parsed = json.loads(body.decode("utf-8", "replace"))
            error = parsed.get("error", parsed) if isinstance(parsed, dict) else {}
            detail = str(error.get("message") or error.get("msg") or "")[:300] if isinstance(error, dict) else ""
        except (json.JSONDecodeError, AttributeError):
            detail = body.decode("utf-8", "replace")[:200]
        detail = self._redact(detail)
        name = self.display_name
        if status in (401, 403):
            return LLMError(f"{name} rejected the credentials (HTTP {status}); check AI_API_KEY, "
                            f"AI_PROVIDER and AI_BASE_URL. {detail}".strip(),
                            retryable=False, code=LLMErrorCode.AUTHENTICATION_ERROR)
        if status == 429:
            return LLMError(f"{name} rate limit (HTTP 429). {detail}".strip(), retryable=True,
                            code=LLMErrorCode.RATE_LIMITED)
        if status >= 500:
            return LLMError(f"{name} server error (HTTP {status}). {detail}".strip(), retryable=True,
                            code=LLMErrorCode.SERVER_ERROR)
        if status in (400, 404, 422):
            return LLMError(f"{name} rejected the request (HTTP {status}); check AI_MODEL and "
                            f"AI_BASE_URL. {detail}".strip(), retryable=False,
                            code=LLMErrorCode.BAD_REQUEST)
        return LLMError(f"{name} returned HTTP {status}. {detail}".strip(), retryable=False,
                        code=LLMErrorCode.PROVIDER_ERROR)

    def _post(self, payload: dict[str, Any]) -> Any:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        attempt = 0
        while True:
            attempt += 1
            try:
                result = self.transport.post_json(self.endpoint, headers, payload, self.timeout)
            except TransportTimeout as exc:
                # not retried here: the agent-level recovery policy decides (bounded)
                raise LLMError(f"{self.display_name} did not respond within {self.timeout:g}s",
                               retryable=True, code=LLMErrorCode.PROVIDER_TIMEOUT) from exc
            except TransportNetworkError as exc:
                error = LLMError(f"Could not reach {self.display_name}: {self._redact(str(exc))}",
                                 retryable=True, code=LLMErrorCode.NETWORK_ERROR)
                retry_after = None
            else:
                if 200 <= result.status < 300:
                    try:
                        return result.json()
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        raise LLMError(f"{self.display_name} returned invalid JSON", retryable=True,
                                       code=LLMErrorCode.MALFORMED_RESPONSE) from None
                error = self._error_for_status(result.status, result.body)
                retry_after = result.headers.get("retry-after")
            if not error.retryable or attempt > self.max_retries:
                raise error
            delay = self.backoff_seconds * (2 ** (attempt - 1))
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    pass
            delay = min(delay, MAX_BACKOFF_SECONDS)
            log.info("%s %s; retrying in %.1fs (attempt %d of %d)", self.display_name,
                     error.code.value, delay, attempt + 1, self.max_retries + 1)
            time.sleep(delay)

    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload = self.build_payload(messages, tools, max_tokens, temperature)
        return self.parse_response(self._post(payload))


def _arguments_json(arguments: Any) -> str:
    return arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)


def create(settings: Any) -> OpenAICompatibleClient:
    """Create a generic OpenAI-compatible client."""
    return OpenAICompatibleClient(settings)


create.requires_model = True