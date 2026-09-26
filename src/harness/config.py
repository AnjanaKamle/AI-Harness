"""Configuration system.

Precedence (highest first): explicit overrides > process environment > .env file > defaults.
All settings use the ``HARNESS_`` prefix except provider credentials, which keep their
conventional names (e.g. ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

VALID_PROVIDERS = ("anthropic", "mock")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
VALID_LOG_FORMATS = ("text", "json")


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-opus-5"
    max_tokens: int = 16000
    timeout_s: float = 600.0
    max_retries: int = 2
    refusal_fallback: bool = True
    api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "text"
    file: Path | None = None


@dataclass(frozen=True)
class HarnessConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    workspace_dir: Path = Path("workspace")
    max_recovery_attempts: int = 3

    def validate(self) -> HarnessConfig:
        if self.llm.provider not in VALID_PROVIDERS:
            raise ConfigError(
                f"HARNESS_LLM_PROVIDER must be one of {VALID_PROVIDERS}, got {self.llm.provider!r}"
            )
        if self.llm.provider == "anthropic" and not self.llm.model:
            raise ConfigError("HARNESS_LLM_MODEL must be set for the anthropic provider")
        if self.llm.max_tokens <= 0:
            raise ConfigError("HARNESS_LLM_MAX_TOKENS must be positive")
        if self.llm.timeout_s <= 0:
            raise ConfigError("HARNESS_LLM_TIMEOUT_S must be positive")
        if self.llm.max_retries < 0:
            raise ConfigError("HARNESS_LLM_MAX_RETRIES must be >= 0")
        if self.logging.level not in VALID_LOG_LEVELS:
            raise ConfigError(f"HARNESS_LOG_LEVEL must be one of {VALID_LOG_LEVELS}")
        if self.logging.format not in VALID_LOG_FORMATS:
            raise ConfigError(f"HARNESS_LOG_FORMAT must be one of {VALID_LOG_FORMATS}")
        if self.max_recovery_attempts < 0:
            raise ConfigError("HARNESS_MAX_RECOVERY_ATTEMPTS must be >= 0")
        return self

    def to_public_dict(self) -> dict[str, Any]:
        """Serializable view with secrets masked (safe to log or print)."""
        data = asdict(self)
        data["llm"]["api_key"] = "***" if self.llm.api_key else None
        data["workspace_dir"] = str(self.workspace_dir)
        data["logging"]["file"] = str(self.logging.file) if self.logging.file else None
        return data


def parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE lines, '#' comments, optional quotes, optional 'export'."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value
    return values


def _to_bool(key: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{key} must be a boolean, got {value!r}")


def _to_num(key: str, value: str, kind: type[int] | type[float]) -> Any:
    try:
        return kind(value)
    except ValueError as exc:
        raise ConfigError(f"{key} must be {kind.__name__}, got {value!r}") from exc


def load_config(
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> HarnessConfig:
    """Build a validated HarnessConfig from defaults, a .env file and the environment."""
    merged: dict[str, str] = {}
    if env_file is not None:
        merged.update(parse_dotenv(Path(env_file)))
    merged.update(os.environ if environ is None else environ)

    def get(key: str) -> str | None:
        value = merged.get(key)
        return value if value not in (None, "") else None

    llm = LLMConfig()
    llm_updates: dict[str, Any] = {}
    if (v := get("HARNESS_LLM_PROVIDER")) is not None:
        llm_updates["provider"] = v.lower()
    if (v := get("HARNESS_LLM_MODEL")) is not None:
        llm_updates["model"] = v
    if (v := get("HARNESS_LLM_MAX_TOKENS")) is not None:
        llm_updates["max_tokens"] = _to_num("HARNESS_LLM_MAX_TOKENS", v, int)
    if (v := get("HARNESS_LLM_TIMEOUT_S")) is not None:
        llm_updates["timeout_s"] = _to_num("HARNESS_LLM_TIMEOUT_S", v, float)
    if (v := get("HARNESS_LLM_MAX_RETRIES")) is not None:
        llm_updates["max_retries"] = _to_num("HARNESS_LLM_MAX_RETRIES", v, int)
    if (v := get("HARNESS_LLM_REFUSAL_FALLBACK")) is not None:
        llm_updates["refusal_fallback"] = _to_bool("HARNESS_LLM_REFUSAL_FALLBACK", v)
    if (v := get("ANTHROPIC_API_KEY")) is not None:
        llm_updates["api_key"] = v
    llm = replace(llm, **llm_updates)

    log = LoggingConfig()
    log_updates: dict[str, Any] = {}
    if (v := get("HARNESS_LOG_LEVEL")) is not None:
        log_updates["level"] = v.upper()
    if (v := get("HARNESS_LOG_FORMAT")) is not None:
        log_updates["format"] = v.lower()
    if (v := get("HARNESS_LOG_FILE")) is not None:
        log_updates["file"] = Path(v)
    log = replace(log, **log_updates)

    top: dict[str, Any] = {"llm": llm, "logging": log}
    if (v := get("HARNESS_WORKSPACE_DIR")) is not None:
        top["workspace_dir"] = Path(v)
    if (v := get("HARNESS_MAX_RECOVERY_ATTEMPTS")) is not None:
        top["max_recovery_attempts"] = _to_num("HARNESS_MAX_RECOVERY_ATTEMPTS", v, int)

    return HarnessConfig(**top).validate()


__all__ = [
    "ConfigError",
    "HarnessConfig",
    "LLMConfig",
    "LoggingConfig",
    "load_config",
    "parse_dotenv",
]
