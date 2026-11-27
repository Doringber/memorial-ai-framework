"""Tests for the Telegram Bot router and service."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.services import telegram_bot
from app.services.avatar_video import VideoInfo, VideoUnavailableError
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
    monkeypatch.setattr(telegram_bot, "_video_mode_chats", set())
    monkeypatch.setattr(telegram_bot, "_last_bot_reply", {})

    def fake_send_msg(
        chat_id: int, text: str, *, with_speak_button: bool = False, with_video_button: bool = True
    ) -> None:
        sent.append(("text", chat_id, text))

    def fake_send_voice(chat_id: int, audio_path: Path, caption: str = "") -> None:
        sent.append(("voice", chat_id, audio_path.name))

    def fake_send_video(chat_id: int, video_path: Path, caption: str = telegram_bot.VIDEO_CAPTION) -> None:
        sent.append(("video", chat_id, caption))

    monkeypatch.setattr(telegram_bot, "send_telegram_message", fake_send_msg)
    monkeypatch.setattr(telegram_bot, "send_telegram_voice", fake_send_voice)
    monkeypatch.setattr(telegram_bot, "send_telegram_video", fake_send_video)
    # Render inline so tests see the video synchronously.
    monkeypatch.setattr(telegram_bot, "_run_in_background", lambda fn: fn())
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


@pytest.fixture
def video_env(tg_env: list, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Video switched on, TTS and the avatar sidecar stubbed. Returns rendered mp4 paths."""
    monkeypatch.setattr(settings, "avatar_video_enabled", True)
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", "999")
    out_mp3 = settings.audio_output_dir / "00000000-0000-0000-0000-000000000077.mp3"
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    out_mp3.write_bytes(b"mp3")
    monkeypatch.setattr(
        telegram_bot,
        "synthesize_speech",
        lambda text: {"audio_filename": out_mp3.name, "provider": "local"},
    )
    rendered: list[Path] = []

    def fake_render(audio_path: Path) -> Path:
        assert audio_path == out_mp3
        settings.video_output_dir.mkdir(parents=True, exist_ok=True)
        mp4 = settings.video_output_dir / f"render-{len(rendered)}.mp4"
        mp4.write_bytes(b"mp4")
        rendered.append(mp4)
        return mp4

    monkeypatch.setattr(telegram_bot, "render_avatar_video", fake_render)
    monkeypatch.setattr(telegram_bot, "video_info", lambda: VideoInfo(enabled=True, available=True))
    return rendered


HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "tg-secret"}


def test_video_button_renders_captioned_video_and_cleans_up(
    tg_env: list[tuple[str, int, str]], video_env: list[Path], fake_llm: FakeLLM
) -> None:
    client = TestClient(app)
    res = client.post(
        "/api/telegram/webhook",
        json={
            "callback_query": {
                "id": "cb-v",
                "data": "video",
                "message": {"message_id": 11, "chat": {"id": 999}, "text": fake_llm.reply},
            }
        },
        headers=HEADERS,
    )
    assert res.json() == {"status": "ok", "video_queued": True}
    assert tg_env[-1] == ("video", 999, telegram_bot.VIDEO_CAPTION)
    assert len(video_env) == 1 and not video_env[0].exists()


def test_video_command_toggles_video_mode(
    tg_env: list[tuple[str, int, str]], video_env: list[Path], fake_llm: FakeLLM
) -> None:
    client = TestClient(app)
    first = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": "היי"}}, headers=HEADERS
    )
    assert first.json() == {"status": "ok", "voice_sent": False}

    on = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": "/video"}}, headers=HEADERS
    )
    assert on.json() == {"status": "video_mode_on", "video_queued": True}  # last reply as video

    nxt = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": "ועוד"}}, headers=HEADERS
    )
    # Video replaces the voice note: same speech, one message.
    assert nxt.json() == {"status": "ok", "voice_sent": False, "video_queued": True}
    assert [kind for kind, _, _ in tg_env].count("video") == 2
    assert not any(kind == "voice" for kind, _, _ in tg_env)

    off = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": "/video"}}, headers=HEADERS
    )
    assert off.json() == {"status": "video_mode_off"}


def test_inline_video_command_answers_once_as_video(
    tg_env: list[tuple[str, int, str]], video_env: list[Path], fake_llm: FakeLLM
) -> None:
    client = TestClient(app)
    res = client.post(
        "/api/telegram/webhook",
        json={"message": {"chat": {"id": 999}, "text": "/video מה שלומך?"}},
        headers=HEADERS,
    )
    assert res.json() == {"status": "ok", "voice_sent": False, "video_queued": True}
    assert ("text", 999, fake_llm.reply) in tg_env
    assert tg_env[-1][0] == "video"
    assert 999 not in telegram_bot._video_mode_chats


def test_video_disabled_refuses_without_rendering(
    tg_env: list[tuple[str, int, str]], fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "telegram_allowed_chat_ids", "999")
    monkeypatch.setattr(
        telegram_bot, "render_avatar_video", lambda p: pytest.fail("must not render when video is off")
    )
    client = TestClient(app)
    cmd = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": "/video"}}, headers=HEADERS
    )
    assert cmd.json() == {"status": "video_unavailable"}
    cb = client.post(
        "/api/telegram/webhook",
        json={"callback_query": {"id": "c", "data": "video", "message": {"chat": {"id": 999}, "text": "x"}}},
        headers=HEADERS,
    )
    assert cb.json() == {"status": "video_unavailable"}
    assert tg_env[-1] == ("text", 999, "וידאו לא מופעל בשרת הזה.")


def test_video_failure_keeps_text_reply(
    tg_env: list[tuple[str, int, str]], video_env: list[Path], fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sidecar_down(audio_path: Path) -> Path:
        raise VideoUnavailableError("down", "Avatar sidecar is not running")

    monkeypatch.setattr(telegram_bot, "render_avatar_video", sidecar_down)
    telegram_bot._video_mode_chats.add(999)
    client = TestClient(app)
    res = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": "היי"}}, headers=HEADERS
    )
    assert res.status_code == 200
    assert ("text", 999, fake_llm.reply) in tg_env
    assert tg_env[-1] == ("text", 999, telegram_bot.VIDEO_FAILED_MESSAGE)
    assert not any(kind == "video" for kind, _, _ in tg_env)


def test_video_mode_refuses_to_start_when_sidecar_is_down(
    tg_env: list[tuple[str, int, str]], video_env: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(telegram_bot, "video_info", lambda: VideoInfo(enabled=True, available=False))
    client = TestClient(app)
    res = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": "/video"}}, headers=HEADERS
    )
    assert res.json() == {"status": "video_unavailable"}
    assert 999 not in telegram_bot._video_mode_chats


@pytest.mark.parametrize("text", ["חשבתי להתאבד", "/video חשבתי להתאבד"])
def test_crisis_reply_is_never_rendered_as_video(
    tg_env: list[tuple[str, int, str]], video_env: list[Path], fake_llm: FakeLLM, text: str
) -> None:
    """Safety-system replies (helplines) must not be spoken by the person's face."""
    telegram_bot._video_mode_chats.add(999)
    client = TestClient(app)
    res = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": text}}, headers=HEADERS
    )
    assert "video_queued" not in res.json()
    assert video_env == []
    assert "1201" in tg_env[-1][2]
    assert fake_llm.calls == []

    # Turning video mode on again must not replay the crisis reply either.
    telegram_bot._video_mode_chats.discard(999)
    again = client.post(
        "/api/telegram/webhook", json={"message": {"chat": {"id": 999}, "text": "/video"}}, headers=HEADERS
    )
    assert again.json() == {"status": "video_mode_on", "video_queued": False}
    assert video_env == []


def test_speak_markup_includes_video_button_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict] = []

    class FakeClient:
        def __init__(self, *a: object, **kw: object) -> None: ...
        def __enter__(self) -> FakeClient:
            return self
        def __exit__(self, *a: object) -> None: ...
        def post(self, url: str, json: dict) -> object:
            payloads.append(json)
            return type("R", (), {"raise_for_status": lambda self: None})()

    monkeypatch.setattr(settings, "telegram_bot_token", "123456:TEST_TOKEN")
    monkeypatch.setattr(telegram_bot.httpx, "Client", FakeClient)

    telegram_bot.send_telegram_message(1, "x", with_speak_button=True)
    monkeypatch.setattr(settings, "avatar_video_enabled", True)
    telegram_bot.send_telegram_message(1, "x", with_speak_button=True)

    assert payloads[0]["reply_markup"] == telegram_bot.SPEAK_BUTTON_MARKUP
    assert payloads[1]["reply_markup"] == telegram_bot.SPEAK_AND_VIDEO_MARKUP
