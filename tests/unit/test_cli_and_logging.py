from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from harness.cli import main
from harness.config import LoggingConfig
from harness.logging_setup import get_logger, setup_logging


def test_cli_config_prints_masked_json(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    assert main(["--provider", "mock", "config"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["llm"]["provider"] == "mock"
    assert out["llm"]["api_key"] == "***"


def test_cli_reads_env_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = tmp_path / "custom.env"
    env.write_text("HARNESS_LLM_PROVIDER=mock\nHARNESS_LLM_MODEL=from-file\n")
    assert main(["--env-file", str(env), "config"]) == 0
    assert json.loads(capsys.readouterr().out)["llm"]["model"] == "from-file"


def test_cli_ping_with_mock(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--provider", "mock", "ping", "hello"]) == 0
    assert "[mock:mock-model] [mock] hello" in capsys.readouterr().out


def test_cli_bad_config_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--provider", "bogus", "config"]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_cli_run_is_placeholder() -> None:
    assert main(["--provider", "mock", "run", "do something"]) == 2


def test_json_logging_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "harness.log"
    setup_logging(LoggingConfig(level="DEBUG", format="json", file=log_file))
    get_logger("test").info("hello", extra={"task_id": "abc"})
    for handler in logging.getLogger("harness").handlers:
        handler.flush()
    record = json.loads(log_file.read_text().strip().splitlines()[-1])
    assert record["msg"] == "hello"
    assert record["logger"] == "harness.test"
    assert record["task_id"] == "abc"


def test_setup_logging_is_idempotent() -> None:
    setup_logging(LoggingConfig())
    setup_logging(LoggingConfig())
    assert len(logging.getLogger("harness").handlers) == 1


def test_get_logger_namespacing() -> None:
    assert get_logger("llm").name == "harness.llm"
    assert get_logger("harness.core").name == "harness.core"
