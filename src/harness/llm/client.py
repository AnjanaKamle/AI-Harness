"""LLMClient interface and provider registry.

The harness depends only on :class:`LLMClient`. Concrete providers register a factory
under a name; ``AI_PROVIDER`` selects one at runtime. No provider is registered yet because
the official evaluation model has not been specified.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from harness.config.settings import MODEL_ENV, PROVIDER_ENV, Settings
from harness.llm.models import LLMResponse, Message, ToolDefinition


class LLMErrorCode(StrEnum):
    """Provider-neutral failure categories (what went wrong, not which vendor)."""

    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    NETWORK_ERROR = "NETWORK_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    BAD_REQUEST = "BAD_REQUEST"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"
    MALFORMED_TOOL_CALL = "MALFORMED_TOOL_CALL"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class LLMError(RuntimeError):
    """A provider call failed. ``retryable`` tells callers whether a retry may help;
    ``code`` is the provider-neutral category."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        code: LLMErrorCode | str = LLMErrorCode.PROVIDER_ERROR,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.code = LLMErrorCode(code)


class LLMNotConfiguredError(LLMError):
    """No usable provider is configured."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False, code=LLMErrorCode.CONFIGURATION_ERROR)


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Send ``messages`` (and optional tool definitions) and return the model's reply.

        Implementations must raise :class:`LLMError` for provider failures.
        """


ProviderFactory = Callable[[Settings], LLMClient]
ENTRY_POINT_GROUP = "ai_coding_harness.providers"


@dataclass(frozen=True)
class _Provider:
    factory: ProviderFactory
    requires_model: bool


_PROVIDERS: dict[str, _Provider] = {}


def register_provider(name: str, factory: ProviderFactory, *, requires_model: bool = False) -> None:
    """Register an adapter under ``name`` (selected with AI_PROVIDER=name)."""
    _PROVIDERS[name.lower()] = _Provider(factory, requires_model)


def _ensure_builtins() -> None:
    import harness.llm.providers  # noqa: F401 - registers deepseek, qwen, scripted


def available_providers() -> list[str]:
    _ensure_builtins()
    return sorted(_PROVIDERS)


def _load_plugin(spec: str) -> _Provider | None:
    """Resolve AI_PROVIDER values that are not registered names:

    * ``package.module:factory`` - import an adapter from the Python path
    * an installed entry point in the ``ai_coding_harness.providers`` group

    Plugin adapters require AI_MODEL unless the factory sets ``requires_model = False``.
    No provider or model is ever assumed by the harness itself.
    """
    factory: Any = None
    if ":" in spec:
        module_name, _, attr = spec.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise LLMNotConfiguredError(
                f"{PROVIDER_ENV}={spec!r}: cannot import adapter module {module_name!r} ({exc})"
            ) from None
        factory = getattr(module, attr, None)
        if factory is None:
            raise LLMNotConfiguredError(f"{PROVIDER_ENV}={spec!r}: {module_name} has no attribute {attr!r}")
    else:
        for entry in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
            if entry.name.lower() == spec.lower():
                factory = entry.load()
                break
    if factory is None:
        return None
    if not callable(factory):
        raise LLMNotConfiguredError(f"{PROVIDER_ENV}={spec!r} does not refer to a callable factory")
    return _Provider(factory, bool(getattr(factory, "requires_model", True)))


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the client selected by ``settings.provider``."""
    if settings.provider is None:
        raise LLMNotConfiguredError(
            "No LLM provider configured. Configure AI_PROVIDER and AI_MODEL before live execution. "
            f"({PROVIDER_ENV} is not set, so live model execution is unavailable.) "
            "The official evaluation provider/model has not been configured; set "
            f"{PROVIDER_ENV} to a registered provider ({', '.join(available_providers()) or 'none'}), "
            "an adapter 'package.module:factory', or an installed provider entry point."
        )
    _ensure_builtins()
    name = settings.provider.strip()
    provider = _PROVIDERS.get(name.lower()) or _load_plugin(name)
    if provider is None:
        raise LLMNotConfiguredError(
            f"Unknown provider {settings.provider!r}. "
            f"Registered providers: {available_providers() or 'none'}; adapters can also be given "
            f"as 'package.module:factory' or installed under the {ENTRY_POINT_GROUP!r} entry point."
        )
    if provider.requires_model and not settings.model:
        raise LLMNotConfiguredError(
            f"Provider {settings.provider!r} requires {MODEL_ENV}; no model name is assumed."
        )
    client = provider.factory(settings)
    if not isinstance(client, LLMClient):
        raise LLMNotConfiguredError(
            f"Provider {settings.provider!r} returned {type(client).__name__}, not an LLMClient"
        )
    return client
