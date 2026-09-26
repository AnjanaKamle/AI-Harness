"""Anthropic (Claude) implementation of :class:`LLMClient`."""

from __future__ import annotations

from typing import Any

from harness.config import LLMConfig
from harness.llm.base import LLMClient, LLMError, LLMRefusalError, LLMResponse, Message, Usage
from harness.logging_setup import get_logger

log = get_logger("llm.anthropic")

# Server-side refusal fallback: on a policy decline the API re-runs the request on a
# fallback model chosen by refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLMClient(LLMClient):
    def __init__(self, config: LLMConfig, client: Any | None = None) -> None:
        self.config = config
        self.model = config.model
        if client is None:
            import anthropic

            # api_key=None lets the SDK resolve credentials itself
            # (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile).
            client = anthropic.Anthropic(
                api_key=config.api_key,
                timeout=config.timeout_s,
                max_retries=config.max_retries,
            )
        self._client = client

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import anthropic

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            params["system"] = system

        try:
            if self.config.refusal_fallback:
                response = self._client.beta.messages.create(
                    **params, betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}
                )
            else:
                response = self._client.messages.create(**params)
        except anthropic.AuthenticationError as exc:
            raise LLMError(f"Anthropic authentication failed: {exc}") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Anthropic rejected the request: {exc}") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Anthropic model/endpoint not found ({self.model}): {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Anthropic rate limit: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            retryable = exc.status_code >= 500
            raise LLMError(
                f"Anthropic API error {exc.status_code}: {exc}", retryable=retryable
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach Anthropic API: {exc}", retryable=True) from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusalError(f"Model declined the request (category={category})")

        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) or 0,
        )
        log.debug(
            "llm call complete",
            extra={
                "model": response.model,
                "stop_reason": response.stop_reason,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        )
        return LLMResponse(
            text=text, model=response.model, stop_reason=response.stop_reason, usage=usage
        )
