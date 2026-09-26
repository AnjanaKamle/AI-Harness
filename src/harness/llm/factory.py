from __future__ import annotations

from harness.config import ConfigError, LLMConfig
from harness.llm.base import LLMClient


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Instantiate the LLM client selected by ``config.provider``."""
    if config.provider == "mock":
        from harness.llm.mock_client import MockLLMClient

        return MockLLMClient()
    if config.provider == "anthropic":
        from harness.llm.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(config)
    raise ConfigError(f"Unknown LLM provider: {config.provider!r}")
