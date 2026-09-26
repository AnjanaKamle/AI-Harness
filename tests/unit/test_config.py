from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import ConfigError, load_config, parse_dotenv


def test_defaults_without_env() -> None:
    cfg = load_config(env_file=None, environ={})
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-opus-5"
    assert cfg.llm.refusal_fallback is True
    assert cfg.logging.level == "INFO"
    assert cfg.max_recovery_attempts == 3


def test_environment_overrides_dotenv(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("HARNESS_LLM_MODEL=from-file\nHARNESS_LOG_LEVEL=debug\n")
    cfg = load_config(env, environ={"HARNESS_LLM_MODEL": "from-env"})
    assert cfg.llm.model == "from-env"
    assert cfg.logging.level == "DEBUG"


def test_type_coercion() -> None:
    cfg = load_config(
        env_file=None,
        environ={
            "HARNESS_LLM_MAX_TOKENS": "2048",
            "HARNESS_LLM_TIMEOUT_S": "30.5",
            "HARNESS_LLM_REFUSAL_FALLBACK": "off",
            "HARNESS_WORKSPACE_DIR": "/tmp/ws",
        },
    )
    assert cfg.llm.max_tokens == 2048
    assert cfg.llm.timeout_s == 30.5
    assert cfg.llm.refusal_fallback is False
    assert cfg.workspace_dir == Path("/tmp/ws")


@pytest.mark.parametrize(
    "environ",
    [
        {"HARNESS_LLM_PROVIDER": "openai"},
        {"HARNESS_LLM_MAX_TOKENS": "0"},
        {"HARNESS_LLM_MAX_TOKENS": "lots"},
        {"HARNESS_LOG_LEVEL": "LOUD"},
        {"HARNESS_LOG_FORMAT": "xml"},
        {"HARNESS_LLM_REFUSAL_FALLBACK": "maybe"},
        {"HARNESS_MAX_RECOVERY_ATTEMPTS": "-1"},
    ],
)
def test_invalid_values_raise(environ: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        load_config(env_file=None, environ=environ)


def test_empty_values_fall_back_to_defaults() -> None:
    cfg = load_config(env_file=None, environ={"ANTHROPIC_API_KEY": "", "HARNESS_LOG_FILE": ""})
    assert cfg.llm.api_key is None
    assert cfg.logging.file is None


def test_public_dict_masks_secret() -> None:
    cfg = load_config(env_file=None, environ={"ANTHROPIC_API_KEY": "sk-secret"})
    public = cfg.to_public_dict()
    assert public["llm"]["api_key"] == "***"
    assert "sk-secret" not in repr(cfg)


def test_parse_dotenv(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "\n"
        "A=1\n"
        "export B=two\n"
        'C="quoted # not comment"\n'
        "D=value # trailing comment\n"
        "not a pair\n"
    )
    assert parse_dotenv(env) == {"A": "1", "B": "two", "C": "quoted # not comment", "D": "value"}


def test_parse_dotenv_missing_file(tmp_path: Path) -> None:
    assert parse_dotenv(tmp_path / "nope.env") == {}
