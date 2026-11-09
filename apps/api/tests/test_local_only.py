"""Local-first guards: nothing reaches a hosted service unless explicitly allowed."""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings, settings
from app.services import llm
from app.services.llm import ExternalServiceBlockedError, generate_response

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "היי"}]


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_provider": "anthropic"},
        {"llm_provider": "openai"},
        {"llm_provider": "ollama", "llm_model": "gpt-oss:120b-cloud"},
        {"llm_provider": "ollama", "llm_model": "qwen3.5:397b:cloud"},
        {"tts_provider": "elevenlabs"},
        {"tts_provider": "gtts"},
        {"embedding_provider": "openai"},
    ],
)
def test_settings_reject_external_services(overrides: dict) -> None:
    with pytest.raises(ValueError, match="ALLOW_EXTERNAL_SERVICES"):
        Settings(_env_file=None, **overrides)


def test_elevenlabs_flag_allows_only_elevenlabs() -> None:
    cfg = Settings(_env_file=None, allow_elevenlabs_tts=True, tts_provider="elevenlabs")
    assert cfg.tts_allowed
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings(_env_file=None, allow_elevenlabs_tts=True, tts_provider="elevenlabs", llm_provider="anthropic")
    with pytest.raises(ValueError, match="TTS_PROVIDER"):
        Settings(_env_file=None, allow_elevenlabs_tts=True, tts_provider="gtts")


def test_settings_accept_external_with_consent() -> None:
    cfg = Settings(_env_file=None, allow_external_services=True, llm_provider="anthropic", tts_provider="elevenlabs")
    assert cfg.llm_provider == "anthropic"


def test_defaults_are_local() -> None:
    cfg = Settings(_env_file=None)
    assert (cfg.llm_provider, cfg.tts_provider, cfg.save_conversation_log) == ("ollama", "none", False)
    assert cfg.ollama_base_url.startswith("http://127.0.0.1")


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_provider": "anthropic"},
        {"llm_model": "gpt-oss:120b-cloud"},
        {"ollama_base_url": "https://ollama.example.com"},
    ],
)
def test_llm_blocks_external_at_call_time(monkeypatch: pytest.MonkeyPatch, overrides: dict) -> None:
    for key, value in overrides.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("must not send"))

    with pytest.raises(ExternalServiceBlockedError):
        generate_response(MESSAGES)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_ollama_native_request(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict = {}

    def fake_post(url: str, json: dict, timeout: float) -> _FakeResponse:
        sent.update(url=url, body=json)
        return _FakeResponse({"message": {"content": "<think>hmm</think>\nנכון....."}, "prompt_eval_count": 10})

    monkeypatch.setattr(llm.httpx, "post", fake_post)

    assert generate_response(MESSAGES) == "נכון....."
    assert sent["url"] == "http://127.0.0.1:11434/api/chat"
    assert sent["body"]["think"] is False
    assert sent["body"]["stream"] is False
    assert sent["body"]["options"]["num_ctx"] == settings.ollama_num_ctx
