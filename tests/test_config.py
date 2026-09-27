from __future__ import annotations

import pytest

from harness.config import API_KEY_ENV, ConfigurationError, MissingAPIKeyError, Settings

from .conftest import FAKE_KEY


def test_reads_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    assert Settings.from_env().api_key == FAKE_KEY


def test_defaults_do_not_invent_provider_or_model() -> None:
    s = Settings.from_env({API_KEY_ENV: FAKE_KEY})
    assert s.provider is None and s.model is None and s.base_url is None
    assert s.timeout_seconds == 60.0 and s.max_retries == 2 and s.log_level == "INFO"


def test_all_fields_from_environment() -> None:
    s = Settings.from_env(
        {
            API_KEY_ENV: FAKE_KEY,
            "AI_PROVIDER": "some-provider",
            "AI_MODEL": "some-model",
            "AI_BASE_URL": "https://example.invalid/v1",
            "AI_TIMEOUT_SECONDS": "12.5",
            "AI_MAX_RETRIES": "5",
            "LOG_LEVEL": "debug",
        }
    )
    assert (s.provider, s.model, s.base_url) == (
        "some-provider",
        "some-model",
        "https://example.invalid/v1",
    )
    assert s.timeout_seconds == 12.5 and s.max_retries == 5 and s.log_level == "DEBUG"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_api_key_is_a_clear_error(value: str | None) -> None:
    env = {} if value is None else {API_KEY_ENV: value}
    with pytest.raises(MissingAPIKeyError, match=API_KEY_ENV):
        Settings.from_env(env)


def test_missing_api_key_on_direct_construction() -> None:
    with pytest.raises(MissingAPIKeyError):
        Settings(api_key="")


@pytest.mark.parametrize(
    "extra",
    [
        {"AI_TIMEOUT_SECONDS": "soon"},
        {"AI_TIMEOUT_SECONDS": "0"},
        {"AI_MAX_RETRIES": "-1"},
        {"AI_MAX_RETRIES": "1.5"},
        {"LOG_LEVEL": "LOUD"},
    ],
)
def test_invalid_values_raise(extra: dict[str, str]) -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env({API_KEY_ENV: FAKE_KEY, **extra})


def test_api_key_never_exposed() -> None:
    s = Settings(api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(s)
    assert FAKE_KEY not in str(s.public_dict())
