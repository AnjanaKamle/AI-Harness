"""Built-in provider adapters behind ``LLMClient``.

    LLMClient
      ├── scripted  (ScriptedLLM: deterministic demo/test client; explicit opt-in only)
      └── real providers (OpenAI-compatible transport, shared)
            ├── deepseek  (DeepSeekClient)
            └── qwen      (QwenClient)

Selection is explicit and deterministic: AI_PROVIDER (case-insensitive) picks the adapter,
AI_MODEL the model, AI_BASE_URL the endpoint. Nothing is inferred from the API key and no
model name is assumed.
"""

from __future__ import annotations

from harness.llm.client import register_provider
from harness.llm.providers import deepseek, qwen, scripted
from harness.llm.providers.deepseek import DeepSeekClient
from harness.llm.providers.openai_compat import OpenAICompatibleClient
from harness.llm.providers.qwen import QwenClient
from harness.llm.providers.transport import HttpResult, HttpTransport, Transport

REAL_PROVIDERS = ("deepseek", "qwen")
SCRIPTED_PROVIDERS = ("scripted", "demo")


def register_builtin_providers() -> None:
    register_provider("deepseek", deepseek.create, requires_model=True)
    register_provider("qwen", qwen.create, requires_model=True)
    for name in SCRIPTED_PROVIDERS:
        register_provider(name, scripted.create, requires_model=False)


def is_scripted(provider: str | None) -> bool:
    return (provider or "").strip().lower() in SCRIPTED_PROVIDERS


register_builtin_providers()

__all__ = [
    "REAL_PROVIDERS",
    "SCRIPTED_PROVIDERS",
    "DeepSeekClient",
    "HttpResult",
    "HttpTransport",
    "OpenAICompatibleClient",
    "QwenClient",
    "Transport",
    "is_scripted",
    "register_builtin_providers",
]
