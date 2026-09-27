"""Optional live provider tests (``make test-live``). Never part of the normal suite.

They run only when AI_LIVE_TESTS=1 and AI_API_KEY / AI_PROVIDER / AI_MODEL (and AI_BASE_URL
if the provider needs it) were set in the environment that started pytest.
"""

from __future__ import annotations

import pytest

from harness.config.settings import Settings
from harness.llm import LLMError, Message, ToolDefinition, create_llm_client

from ..conftest import ORIGINAL_ENV

pytestmark = pytest.mark.live


def live_settings() -> Settings:
    if ORIGINAL_ENV.get("AI_LIVE_TESTS") != "1":
        pytest.skip("live tests are opt-in: run `make test-live` (sets AI_LIVE_TESTS=1)")
    missing = [k for k in ("AI_API_KEY", "AI_PROVIDER", "AI_MODEL") if not ORIGINAL_ENV.get(k)]
    if missing:
        pytest.skip(f"live provider not configured: missing {', '.join(missing)}")
    return Settings.from_env(ORIGINAL_ENV)


def test_live_text_round_trip() -> None:
    settings = live_settings()
    try:
        reply = create_llm_client(settings).generate([Message.user("Reply with the single word OK.")],
                                                     max_tokens=32)
    except LLMError as exc:
        pytest.fail(f"{exc.code.value}: {exc}")
    assert reply.content.strip()


def test_live_tool_call() -> None:
    settings = live_settings()
    tool = ToolDefinition("read_file", "Read a file from the repository.",
                          {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"]})
    reply = create_llm_client(settings).generate(
        [Message.system("Use the read_file tool to read calc.py. Do not answer in text."),
         Message.user("Open calc.py.")], tools=[tool], max_tokens=256)
    assert reply.tool_calls and reply.tool_calls[0].name == "read_file"
    assert isinstance(reply.tool_calls[0].arguments, dict)
