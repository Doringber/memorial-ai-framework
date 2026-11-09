"""Tests for the Telegram Bot router and service."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import telegram_bot
from tests.conftest import FakeLLM


@pytest.fixture
def tg_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple[str, int, str]]:
    sent: list[tuple[str, int, str]] = []
    monkeypatch.setattr(settings, "telegram_bot_token", "123456:TEST_TOKEN")
    monkeypatch.setattr(settings, "telegram_webhook_secret", "tg-secret")
    monkeypatch.setattr(settings, "family_access_key", "family-pass")
    monkeypatch.setattr(settings, "admin_access_key", "admin-pass")
    monkeypatch.setattr(telegram_bot, "_chat_sessions", {})
    monkeypatch.setattr(telegram_bot, "_voice_mode_chats", set())
    monkeypatch.setattr(telegram_bot, "_last_bot_reply", {})

    def fake_send_msg(chat_id: int, text: str, *, with_speak_button: bool = False) -> None:
        sent.append(("text", chat_id, text))

    def fake_send_voice(chat_id: int, audio_path: Path, caption: str = "") -> None:
        sent.append(("voice", chat_id, audio_path.name))

    monkeypatch.setattr(telegram_bot, "send_telegram_message", fake_send_msg)
    monkeypatch.setattr(telegram_bot, "send_telegram_voice", fake_send_voice)
    monkeypatch.setattr(telegram_bot, "_answer_callback_query", lambda *a, **kw: None)
    monkeypatch.setattr(telegram_bot, "_clear_inline_keyboard", lambda *a, **kw: None)
    return sent


def test_webhook_rejects_missing_or_wrong_secret(tg_env: list) -> None:
    client = TestClient(app)
    res = client.post("/api/telegram/webhook", json={"message": {"chat": {"id": 100}, "text": "היי"}})
    assert res.status_code == 401


def test_stranger_is_blocked_until_family_password_sent(
    tg_env: list[tuple[str, int, str]], fake_llm: FakeLLM
) -> None:
    client = TestClient(app)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "tg-secret"}

    # 1. Stranger sends a message without authenticating -> blocked
    r1 = client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 555}, "text": "מה נשמע?"}},
        headers=headers,
    )
    assert r1.json()["status"] == "unauthorized"
    assert tg_env[-1][0] == "text"
    assert "/start" in tg_env[-1][2]
    assert fake_llm.calls == []

    # 2. Wrong password on /start -> still blocked
    r2 = client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 555}, "text": "/start wrong-key"}},
        headers=headers,
    )
    assert r2.json()["status"] == "unauthorized"

    # 3. Correct family password on /start -> authorized and persisted
    r3 = client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 555}, "text": "/start family-pass"}},
        headers=headers,
    )
    assert r3.json()["status"] == "authorized"
    assert "סימולציה" in tg_env[-1][2]

    # 4. Subsequent text message works and replies in text only (0 voice credits)
    r4 = client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 555}, "text": "מה שלומך?"}},
        headers=headers,
    )
    assert r4.json() == {"status": "ok", "voice_sent": False}
    assert tg_env[-1] == ("text", 555, fake_llm.reply)


def test_voice_note_transcribes_and_replies_with_voice(
    tg_env: list[tuple[str, int, str]],
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", "777")

    fake_ogg = tmp_path / "in.ogg"
    fake_ogg.write_bytes(b"OggS")
    monkeypatch.setattr(telegram_bot, "download_telegram_file", lambda file_id, suffix=".ogg": fake_ogg)
    monkeypatch.setattr(
        telegram_bot,
        "transcribe_audio",
        lambda path: {"text": "מה אתה חושב על האוטו שלך?", "language": "he"},
    )

    out_mp3 = settings.audio_output_dir / "00000000-0000-0000-0000-000000000099.mp3"
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    out_mp3.write_bytes(b"mp3")
    monkeypatch.setattr(
        telegram_bot,
        "synthesize_speech",
        lambda text: {"audio_filename": out_mp3.name, "provider": "elevenlabs"},
    )

    client = TestClient(app)
    res = client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 777}, "voice": {"file_id": "file-1"}}},
        headers={"X-Telegram-Bot-Api-Secret-Token": "tg-secret"},
    )
    assert res.json() == {"status": "ok", "voice_sent": True}
    assert ("text", 777, fake_llm.reply) in tg_env
    assert ("voice", 777, out_mp3.name) in tg_env
    assert not fake_ogg.exists()


def test_standalone_voice_command_and_inline_speak_button(
    tg_env: list[tuple[str, int, str]],
    fake_llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", "888")
    out_mp3 = settings.audio_output_dir / "00000000-0000-0000-0000-000000000088.mp3"
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    out_mp3.write_bytes(b"mp3")
    monkeypatch.setattr(
        telegram_bot,
        "synthesize_speech",
        lambda text: {"audio_filename": out_mp3.name, "provider": "elevenlabs"},
    )

    client = TestClient(app)
    headers = {"X-Telegram-Bot-Api-Secret-Token": "tg-secret"}

    # 1. Send normal text question ("אתה אוהב מנגו?")
    client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 888}, "text": "אתה אוהב מנגו?"}},
        headers=headers,
    )

    # 2. Tap inline "🔊 השמע בקול" button on that message -> sends voice note
    cb = client.post(
        "/api/telegram/webhook",
        json={
            "callback_query": {
                "id": "cb-1",
                "data": "speak",
                "message": {"message_id": 10, "chat": {"id": 888}, "text": fake_llm.reply},
            }
        },
        headers=headers,
    )
    assert cb.json() == {"status": "ok", "voice_sent": True}
    assert tg_env[-1] == ("voice", 888, out_mp3.name)

    # 3. Send standalone "/voice" -> speaks last reply and turns on auto voice mode
    v_on = client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 888}, "text": "/voice"}},
        headers=headers,
    )
    assert v_on.json() == {"status": "voice_mode_on", "voice_sent": True}

    # 4. Next regular text message now automatically includes a voice note
    r_next = client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 888}, "text": "ועוד שאלה"}},
        headers=headers,
    )
    assert r_next.json() == {"status": "ok", "voice_sent": True}


def test_register_webhook_requires_admin_and_https(tg_env: list) -> None:
    client = TestClient(app)
    assert (
        client.post(
            "/api/telegram/register-webhook",
            json={"public_url": "https://ronny.example.com"},
            headers={"X-Access-Key": "family-pass"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/telegram/register-webhook",
            json={"public_url": "http://insecure.example.com"},
            headers={"X-Access-Key": "admin-pass"},
        ).status_code
        == 422
    )
