from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from app.core.config import settings
from app.services import tts
from app.services.tts import VoiceUnavailableError, prepare_text_for_speech, synthesize_speech


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("הנשר נחת..... 😜😂", "הנשר נחת..."),
        ("יפה מאוד, בשעה טובה 👍", "יפה מאוד, בשעה טובה"),
        ("אוהב אתכם 🥰\u200d  מאוד", "אוהב אתכם מאוד"),
        ("נכון", "נכון"),
    ],
)
def test_prepare_text_for_speech(text: str, expected: str) -> None:
    assert prepare_text_for_speech(text) == expected


def test_emoji_only_reply_is_rejected() -> None:
    with pytest.raises(ValueError):
        synthesize_speech("👍😂")


def test_voice_off_by_default() -> None:
    with pytest.raises(VoiceUnavailableError) as exc:
        synthesize_speech("שלום")
    assert exc.value.reason == "off"


@pytest.fixture
def provider_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def fake_elevenlabs(text: str, output_path: str) -> None:
        calls.append("elevenlabs")
        Path(output_path).write_bytes(b"mp3")

    def fake_gtts(text: str, output_path: str) -> None:
        calls.append("gtts")
        Path(output_path).write_bytes(b"mp3")

    monkeypatch.setattr(tts, "_synthesize_elevenlabs", fake_elevenlabs)
    monkeypatch.setattr(tts, "_synthesize_gtts", fake_gtts)
    return calls


@pytest.fixture
def elevenlabs_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "tts_provider", "elevenlabs")
    monkeypatch.setattr(settings, "allow_elevenlabs_tts", True)
    monkeypatch.setattr(settings, "elevenlabs_api_key", "key")
    monkeypatch.setattr(settings, "elevenlabs_voice_id", "voice")


@pytest.mark.parametrize("provider", ["elevenlabs", "gtts"])
def test_external_tts_blocked_without_consent(
    monkeypatch: pytest.MonkeyPatch, provider_calls: list[str], provider: str
) -> None:
    monkeypatch.setattr(settings, "tts_provider", provider)

    with pytest.raises(VoiceUnavailableError) as exc:
        synthesize_speech("שלום")
    assert exc.value.reason == "blocked"
    assert provider_calls == []


def test_elevenlabs_flag_does_not_unlock_gtts(monkeypatch: pytest.MonkeyPatch, provider_calls: list[str]) -> None:
    monkeypatch.setattr(settings, "tts_provider", "gtts")
    monkeypatch.setattr(settings, "allow_elevenlabs_tts", True)

    with pytest.raises(VoiceUnavailableError):
        synthesize_speech("שלום")
    assert provider_calls == []


@pytest.mark.usefixtures("elevenlabs_on")
def test_elevenlabs_with_consent(provider_calls: list[str]) -> None:
    result = synthesize_speech("שלום")

    assert result["provider"] == "elevenlabs"
    assert provider_calls == ["elevenlabs"]
    assert tts.elevenlabs_chars_used() == len("שלום")


@pytest.mark.usefixtures("elevenlabs_on")
def test_elevenlabs_failure_stays_text_only(monkeypatch: pytest.MonkeyPatch, provider_calls: list[str]) -> None:
    def failing(text: str, output_path: str) -> None:
        Path(output_path).write_bytes(b"partial")
        raise RuntimeError("quota_exceeded")

    monkeypatch.setattr(tts, "_synthesize_elevenlabs", failing)

    with pytest.raises(VoiceUnavailableError) as exc:
        synthesize_speech("שלום")
    assert exc.value.reason == "failed"
    assert "gtts" not in provider_calls
    assert list(settings.audio_output_dir.glob("*.mp3")) == []


@pytest.mark.usefixtures("elevenlabs_on")
def test_elevenlabs_budget_is_enforced(monkeypatch: pytest.MonkeyPatch, provider_calls: list[str]) -> None:
    monkeypatch.setattr(settings, "elevenlabs_monthly_char_limit", 6)
    synthesize_speech("שלום")

    with pytest.raises(VoiceUnavailableError) as exc:
        synthesize_speech("שלום")
    assert exc.value.reason == "budget"
    assert provider_calls == ["elevenlabs"]


@pytest.mark.usefixtures("elevenlabs_on")
def test_elevenlabs_request_has_no_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sent: dict = {}

    class FakeTTS:
        def convert(self, voice_id: str, **kwargs: object):
            sent.update(voice_id=voice_id, **kwargs)
            return iter([b"mp3"])

    class FakeClient:
        def __init__(self, api_key: str, timeout: float) -> None:
            self.text_to_speech = FakeTTS()

    fake_module = types.SimpleNamespace(ElevenLabs=FakeClient, VoiceSettings=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "elevenlabs", fake_module)

    tts._synthesize_elevenlabs("שלום", str(tmp_path / "out.mp3"))

    assert sent["voice_id"] == "voice"
    assert sent["model_id"] == "eleven_v4"
    assert sent["language_code"] == "he"
    assert sent["request_options"] == {"max_retries": 0}
    assert sent["voice_settings"]["stability"] == pytest.approx(0.70)
    assert sent["voice_settings"]["similarity_boost"] == pytest.approx(0.99)
    assert sent["voice_settings"]["use_speaker_boost"] is True


def test_local_tts_provider_requires_no_external_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    monkeypatch.setattr(settings, "tts_provider", "local")
    monkeypatch.setattr(settings, "local_voice_reference_wav", str(ref))
    monkeypatch.setattr(tts, "_local_clone_available", lambda: True)
    monkeypatch.setattr(
        tts,
        "_synthesize_local_clone",
        lambda text, out: Path(out).write_bytes(b"local-mp3"),
    )

    info = tts.voice_info()
    assert (info.provider, info.external, info.available) == ("local", False, True)
    result = synthesize_speech("שלום")
    assert result["provider"] == "local"
    assert tts.elevenlabs_chars_used() == 0


def test_voice_info_reports_external_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tts.voice_info().available is False
    monkeypatch.setattr(settings, "tts_provider", "elevenlabs")
    monkeypatch.setattr(settings, "allow_elevenlabs_tts", True)
    monkeypatch.setattr(settings, "elevenlabs_api_key", "key")
    monkeypatch.setattr(settings, "elevenlabs_voice_id", "voice")

    info = tts.voice_info()
    assert (info.provider, info.external, info.available) == ("elevenlabs", True, True)
