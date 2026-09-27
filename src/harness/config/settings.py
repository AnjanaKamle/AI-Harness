"""Centralized runtime settings, read from the process environment only.

Secrets are never read from files. Model / provider / endpoint are left unset by default
because the hackathon has not specified them; they are supplied via environment variables.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

API_KEY_ENV = "AI_API_KEY"
PROVIDER_ENV = "AI_PROVIDER"
MODEL_ENV = "AI_MODEL"
BASE_URL_ENV = "AI_BASE_URL"
TIMEOUT_ENV = "AI_TIMEOUT_SECONDS"
MAX_RETRIES_ENV = "AI_MAX_RETRIES"
LOG_LEVEL_ENV = "LOG_LEVEL"
MAX_TOOL_CALLS_ENV = "AI_MAX_TOOL_CALLS"
MAX_TOOL_OUTPUT_CHARS_ENV = "AI_MAX_TOOL_OUTPUT_CHARS"
COMMAND_TIMEOUT_ENV = "AI_COMMAND_TIMEOUT_SECONDS"
MAX_REPAIR_ATTEMPTS_ENV = "AI_MAX_REPAIR_ATTEMPTS"
TEST_TIMEOUT_ENV = "AI_TEST_TIMEOUT_SECONDS"
MAX_TOOL_OUTPUT_LINES_ENV = "AI_MAX_TOOL_OUTPUT_LINES"
MAX_SEARCH_RESULTS_ENV = "AI_MAX_SEARCH_RESULTS"
MAX_RESEARCH_RESULTS_ENV = "AI_MAX_RESEARCH_RESULTS"
MAX_CONTEXT_CHARS_ENV = "AI_MAX_CONTEXT_CHARS"
RESEARCH_BACKEND_ENV = "AI_RESEARCH_BACKEND"
RESEARCH_TIMEOUT_ENV = "AI_RESEARCH_TIMEOUT_SECONDS"
MAX_CONCURRENT_AGENTS_ENV = "AI_MAX_CONCURRENT_AGENTS"
MAX_TOOL_RETRIES_ENV = "AI_MAX_TOOL_RETRIES"
MAX_AGENT_RETRIES_ENV = "AI_MAX_AGENT_RETRIES"
AGENT_TIMEOUT_ENV = "AI_AGENT_TIMEOUT_SECONDS"
MUSIC_BACKEND_ENV = "AI_MUSIC_BACKEND"
MUSIC_DIR_ENV = "AI_MUSIC_DIR"
MAX_OUTPUT_TOKENS_ENV = "AI_MAX_OUTPUT_TOKENS"
LOG_FILE_ENV = "AI_LOG_FILE"

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 12_000
DEFAULT_COMMAND_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_REPAIR_ATTEMPTS = 5
DEFAULT_TEST_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_TOOL_OUTPUT_LINES = 400
DEFAULT_MAX_SEARCH_RESULTS = 50
DEFAULT_MAX_RESEARCH_RESULTS = 5
DEFAULT_MAX_CONTEXT_CHARS = 16_000
DEFAULT_RESEARCH_BACKEND = "duckduckgo"
DEFAULT_RESEARCH_TIMEOUT_SECONDS = 15.0
VALID_RESEARCH_BACKENDS = ("duckduckgo", "none")
DEFAULT_MAX_CONCURRENT_AGENTS = 3
DEFAULT_MAX_TOOL_RETRIES = 2
DEFAULT_MAX_AGENT_RETRIES = 2
DEFAULT_AGENT_TIMEOUT_SECONDS = 900.0
DEFAULT_MUSIC_BACKEND = "auto"
# auto: use a local system audio player if one works, otherwise music is DISABLED
VALID_MUSIC_BACKENDS = ("auto", "system", "none", "disabled")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigurationError(ValueError):
    """Raised when settings are missing or invalid."""


class MissingAPIKeyError(ConfigurationError):
    def __init__(self) -> None:
        super().__init__(
            f"{API_KEY_ENV} is not set. Export it in your environment before running, "
            f"e.g. `export {API_KEY_ENV}=...`. It is never read from files."
        )


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    log_level: str = DEFAULT_LOG_LEVEL
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
    max_repair_attempts: int = DEFAULT_MAX_REPAIR_ATTEMPTS
    test_timeout_seconds: float = DEFAULT_TEST_TIMEOUT_SECONDS
    max_tool_output_lines: int = DEFAULT_MAX_TOOL_OUTPUT_LINES
    max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS
    max_research_results: int = DEFAULT_MAX_RESEARCH_RESULTS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    research_backend: str = DEFAULT_RESEARCH_BACKEND
    research_timeout_seconds: float = DEFAULT_RESEARCH_TIMEOUT_SECONDS
    max_concurrent_agents: int = DEFAULT_MAX_CONCURRENT_AGENTS
    max_tool_retries: int = DEFAULT_MAX_TOOL_RETRIES
    max_agent_retries: int = DEFAULT_MAX_AGENT_RETRIES
    agent_timeout_seconds: float = DEFAULT_AGENT_TIMEOUT_SECONDS
    music_backend: str = DEFAULT_MUSIC_BACKEND
    music_dir: str | None = None
    max_output_tokens: int | None = None  # sent to the provider only when set
    log_file: str | None = None

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise MissingAPIKeyError()
        if self.timeout_seconds <= 0:
            raise ConfigurationError(f"{TIMEOUT_ENV} must be positive")
        if self.max_retries < 0:
            raise ConfigurationError(f"{MAX_RETRIES_ENV} must be >= 0")
        if self.log_level not in VALID_LOG_LEVELS:
            raise ConfigurationError(f"{LOG_LEVEL_ENV} must be one of {VALID_LOG_LEVELS}")
        if self.max_tool_calls < 1:
            raise ConfigurationError(f"{MAX_TOOL_CALLS_ENV} must be >= 1")
        if self.max_tool_output_chars < 500:
            raise ConfigurationError(f"{MAX_TOOL_OUTPUT_CHARS_ENV} must be >= 500")
        if self.command_timeout_seconds <= 0:
            raise ConfigurationError(f"{COMMAND_TIMEOUT_ENV} must be positive")
        if self.max_repair_attempts < 0:
            raise ConfigurationError(f"{MAX_REPAIR_ATTEMPTS_ENV} must be >= 0")
        if self.test_timeout_seconds <= 0:
            raise ConfigurationError(f"{TEST_TIMEOUT_ENV} must be positive")
        for env_name, value, minimum in (
            (MAX_TOOL_OUTPUT_LINES_ENV, self.max_tool_output_lines, 20),
            (MAX_SEARCH_RESULTS_ENV, self.max_search_results, 1),
            (MAX_RESEARCH_RESULTS_ENV, self.max_research_results, 1),
            (MAX_CONTEXT_CHARS_ENV, self.max_context_chars, 2_000),
        ):
            if value < minimum:
                raise ConfigurationError(f"{env_name} must be >= {minimum}")
        if self.research_backend not in VALID_RESEARCH_BACKENDS:
            raise ConfigurationError(f"{RESEARCH_BACKEND_ENV} must be one of {VALID_RESEARCH_BACKENDS}")
        if self.research_timeout_seconds <= 0:
            raise ConfigurationError(f"{RESEARCH_TIMEOUT_ENV} must be positive")
        if self.max_concurrent_agents < 1:
            raise ConfigurationError(f"{MAX_CONCURRENT_AGENTS_ENV} must be >= 1")
        if self.max_tool_retries < 0 or self.max_agent_retries < 0:
            raise ConfigurationError(
                f"{MAX_TOOL_RETRIES_ENV} and {MAX_AGENT_RETRIES_ENV} must be >= 0"
            )
        if self.agent_timeout_seconds <= 0:
            raise ConfigurationError(f"{AGENT_TIMEOUT_ENV} must be positive")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ConfigurationError(f"{MAX_OUTPUT_TOKENS_ENV} must be >= 1")
        if self.music_backend not in VALID_MUSIC_BACKENDS:
            raise ConfigurationError(f"{MUSIC_BACKEND_ENV} must be one of {VALID_MUSIC_BACKENDS}")

    @property
    def log_level_value(self) -> int:
        return logging.getLevelNamesMapping()[self.log_level]

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ

        def get(key: str) -> str | None:
            value = env.get(key, "").strip()
            return value or None

        def get_number(key: str, kind: type[int] | type[float], default: float) -> float:
            raw = get(key)
            if raw is None:
                return default
            try:
                return kind(raw)
            except ValueError as exc:
                raise ConfigurationError(f"{key} must be a {kind.__name__}, got {raw!r}") from exc

        api_key = get(API_KEY_ENV)
        if api_key is None:
            raise MissingAPIKeyError()

        return cls(
            api_key=api_key,
            provider=get(PROVIDER_ENV),
            model=get(MODEL_ENV),
            base_url=get(BASE_URL_ENV),
            timeout_seconds=float(get_number(TIMEOUT_ENV, float, DEFAULT_TIMEOUT_SECONDS)),
            max_retries=int(get_number(MAX_RETRIES_ENV, int, DEFAULT_MAX_RETRIES)),
            log_level=(get(LOG_LEVEL_ENV) or DEFAULT_LOG_LEVEL).upper(),
            max_tool_calls=int(get_number(MAX_TOOL_CALLS_ENV, int, DEFAULT_MAX_TOOL_CALLS)),
            max_tool_output_chars=int(
                get_number(MAX_TOOL_OUTPUT_CHARS_ENV, int, DEFAULT_MAX_TOOL_OUTPUT_CHARS)
            ),
            command_timeout_seconds=float(
                get_number(COMMAND_TIMEOUT_ENV, float, DEFAULT_COMMAND_TIMEOUT_SECONDS)
            ),
            max_repair_attempts=int(
                get_number(MAX_REPAIR_ATTEMPTS_ENV, int, DEFAULT_MAX_REPAIR_ATTEMPTS)
            ),
            test_timeout_seconds=float(
                get_number(TEST_TIMEOUT_ENV, float, DEFAULT_TEST_TIMEOUT_SECONDS)
            ),
            max_tool_output_lines=int(
                get_number(MAX_TOOL_OUTPUT_LINES_ENV, int, DEFAULT_MAX_TOOL_OUTPUT_LINES)
            ),
            max_search_results=int(
                get_number(MAX_SEARCH_RESULTS_ENV, int, DEFAULT_MAX_SEARCH_RESULTS)
            ),
            max_research_results=int(
                get_number(MAX_RESEARCH_RESULTS_ENV, int, DEFAULT_MAX_RESEARCH_RESULTS)
            ),
            max_context_chars=int(get_number(MAX_CONTEXT_CHARS_ENV, int, DEFAULT_MAX_CONTEXT_CHARS)),
            research_backend=(get(RESEARCH_BACKEND_ENV) or DEFAULT_RESEARCH_BACKEND).lower(),
            research_timeout_seconds=float(
                get_number(RESEARCH_TIMEOUT_ENV, float, DEFAULT_RESEARCH_TIMEOUT_SECONDS)
            ),
            max_concurrent_agents=int(
                get_number(MAX_CONCURRENT_AGENTS_ENV, int, DEFAULT_MAX_CONCURRENT_AGENTS)
            ),
            max_tool_retries=int(get_number(MAX_TOOL_RETRIES_ENV, int, DEFAULT_MAX_TOOL_RETRIES)),
            max_agent_retries=int(
                get_number(MAX_AGENT_RETRIES_ENV, int, DEFAULT_MAX_AGENT_RETRIES)
            ),
            agent_timeout_seconds=float(
                get_number(AGENT_TIMEOUT_ENV, float, DEFAULT_AGENT_TIMEOUT_SECONDS)
            ),
            music_backend=(get(MUSIC_BACKEND_ENV) or DEFAULT_MUSIC_BACKEND).lower(),
            music_dir=get(MUSIC_DIR_ENV),
            max_output_tokens=(
                int(get_number(MAX_OUTPUT_TOKENS_ENV, int, 0)) or None
                if get(MAX_OUTPUT_TOKENS_ENV) else None
            ),
            log_file=get(LOG_FILE_ENV),
        )

    def public_dict(self) -> dict[str, object]:
        """Settings safe to log or print: the API key is masked."""
        return {
            "api_key": "***set***",
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "log_level": self.log_level,
            "max_tool_calls": self.max_tool_calls,
            "max_tool_output_chars": self.max_tool_output_chars,
            "command_timeout_seconds": self.command_timeout_seconds,
            "max_repair_attempts": self.max_repair_attempts,
            "test_timeout_seconds": self.test_timeout_seconds,
            "max_tool_output_lines": self.max_tool_output_lines,
            "max_search_results": self.max_search_results,
            "max_research_results": self.max_research_results,
            "max_context_chars": self.max_context_chars,
            "research_backend": self.research_backend,
            "research_timeout_seconds": self.research_timeout_seconds,
            "max_concurrent_agents": self.max_concurrent_agents,
            "max_tool_retries": self.max_tool_retries,
            "max_agent_retries": self.max_agent_retries,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "music_backend": self.music_backend,
            "music_dir": self.music_dir,
            "max_output_tokens": self.max_output_tokens,
            "log_file": self.log_file,
        }
