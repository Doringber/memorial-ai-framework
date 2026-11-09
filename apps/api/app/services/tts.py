"""Text-to-Speech service.

Providers:
- "none"       — voice output off (default until a local engine is chosen)
- "elevenlabs" — the cloned voice (ELEVENLABS_VOICE_ID). External and paid:
                 needs ALLOW_ELEVENLABS_TTS (or ALLOW_EXTERNAL_SERVICES) and
                 is capped by ELEVENLABS_MONTHLY_CHAR_LIMIT.
- "gtts"       — Google TTS, generic voice. External: needs ALLOW_EXTERNAL_SERVICES.

A failed request is never retried and never falls back to another provider:
the reply simply stays text-only.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from loguru import logger

from app.core.config import EXTERNAL_TTS_PROVIDERS, settings

VoiceUnavailableReason = Literal["off", "blocked", "budget", "failed"]

# Emoji, pictographs, variation selectors and joiners — spoken aloud otherwise.
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D\U00002B00-\U00002BFF]+"
)
_ELLIPSIS = re.compile(r"\.{2,}")
_SPACES = re.compile(r"\s{2,}")
_usage_lock = threading.Lock()


class VoiceUnavailableError(RuntimeError):
    """Voice output is off, not allowed, over budget, or the engine failed."""

    def __init__(self, reason: VoiceUnavailableReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(frozen=True)
class VoiceInfo:
    provider: str
    external: bool
    available: bool
    chars_used: int
    chars_limit: int


def prepare_text_for_speech(text: str) -> str:
    """Strip characters the voice engine would read literally or stumble on."""
    text = _EMOJI.sub("", text)
    text = _ELLIPSIS.sub("...", text)
    return _SPACES.sub(" ", text).strip()


def _usage_file() -> Path:
    return settings.usage_dir / f"elevenlabs-{date.today():%Y-%m}.json"


def elevenlabs_chars_used() -> int:
    path = _usage_file()
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("chars", 0))
    except (ValueError, OSError):
        logger.warning("Unreadable ElevenLabs usage file {}; treating budget as spent", path.name)
        return settings.elevenlabs_monthly_char_limit


def _record_elevenlabs_chars(count: int) -> None:
    settings.usage_dir.mkdir(parents=True, exist_ok=True)
    path = _usage_file()
    used = elevenlabs_chars_used() + count
    path.write_text(json.dumps({"chars": used}), encoding="utf-8")


def _local_clone_available() -> bool:
    if not settings.local_voice_reference_wav:
        return False
    if not Path(settings.local_voice_reference_wav).is_file():
        return False
    try:
        import importlib.util

        return importlib.util.find_spec("chatterbox") is not None
    except Exception:
        return False


def voice_info() -> VoiceInfo:
    provider = settings.tts_provider
    if provider == "local":
        return VoiceInfo(
            provider="local",
            external=False,
            available=_local_clone_available(),
            chars_used=0,
            chars_limit=0,
        )
    available = provider in EXTERNAL_TTS_PROVIDERS and settings.tts_allowed
    if provider == "elevenlabs":
        available = available and bool(settings.elevenlabs_api_key and settings.elevenlabs_voice_id)
    return VoiceInfo(
        provider=provider,
        external=provider in EXTERNAL_TTS_PROVIDERS,
        available=available,
        chars_used=elevenlabs_chars_used() if provider == "elevenlabs" else 0,
        chars_limit=settings.elevenlabs_monthly_char_limit if provider == "elevenlabs" else 0,
    )


def _prune_old_audio() -> None:
    cutoff = time.time() - settings.audio_retention_hours * 3600
    for path in settings.audio_output_dir.glob("*.mp3"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def synthesize_speech(text: str) -> dict:
    """Convert text to an mp3 file. Returns 'audio_filename' and 'provider'."""
    spoken = prepare_text_for_speech(text)
    if not spoken:
        raise ValueError("Nothing to speak after removing emoji/punctuation")

    provider = settings.tts_provider
    if provider == "local":
        if not _local_clone_available():
            raise VoiceUnavailableError("off", "Local voice clone model or reference WAV is not configured")
    elif provider not in EXTERNAL_TTS_PROVIDERS:
        raise VoiceUnavailableError("off", f"No voice engine configured (TTS_PROVIDER={provider})")
    elif not settings.tts_allowed:
        raise VoiceUnavailableError("blocked", f"TTS provider '{provider}' is external and not allowed")

    settings.audio_output_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_audio()
    output_file = settings.audio_output_dir / f"{uuid.uuid4()}.mp3"

    try:
        if provider == "local":
            _synthesize_local_clone(spoken, str(output_file))
        elif provider == "elevenlabs":
            _synthesize_elevenlabs_within_budget(spoken, output_file)
        else:
            _synthesize_gtts(spoken, str(output_file))
    except VoiceUnavailableError:
        output_file.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001 — any provider failure leaves the reply text-only
        output_file.unlink(missing_ok=True)
        status = getattr(e, "status_code", None)
        logger.error("TTS {} failed ({}{})", provider, type(e).__name__, f", HTTP {status}" if status else "")
        raise VoiceUnavailableError("failed", f"TTS provider '{provider}' failed") from e

    logger.info("TTS generated: {} ({} bytes, provider={})", output_file.name, output_file.stat().st_size, provider)
    return {"audio_filename": output_file.name, "provider": provider}


def _synthesize_elevenlabs_within_budget(text: str, output_file: Path) -> None:
    if not (settings.elevenlabs_api_key and settings.elevenlabs_voice_id):
        raise VoiceUnavailableError("off", "ELEVENLABS_API_KEY or ELEVENLABS_VOICE_ID is not set")
    with _usage_lock:
        if elevenlabs_chars_used() + len(text) > settings.elevenlabs_monthly_char_limit:
            raise VoiceUnavailableError("budget", "Monthly ElevenLabs character limit reached")
        # Counted before the call: a request that fails mid-stream may still be billed.
        _record_elevenlabs_chars(len(text))
    _synthesize_elevenlabs(text, str(output_file))


def _synthesize_elevenlabs(text: str, output_path: str) -> None:
    from elevenlabs import ElevenLabs, VoiceSettings

    client = ElevenLabs(api_key=settings.elevenlabs_api_key, timeout=60)
    logger.info("TTS to ElevenLabs {} ({} chars)", settings.elevenlabs_model_id, len(text))

    audio_stream = client.text_to_speech.convert(
        voice_id=settings.elevenlabs_voice_id,
        text=text,
        model_id=settings.elevenlabs_model_id,
        language_code="he",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=settings.elevenlabs_stability,
            similarity_boost=settings.elevenlabs_similarity_boost,
            use_speaker_boost=settings.elevenlabs_speaker_boost,
        ),
        request_options={"max_retries": 0},
    )
    with open(output_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)


_local_tts_model: object | None = None
_local_tts_lock = threading.Lock()


def _synthesize_local_clone(text: str, output_path: str) -> None:
    """Zero-shot local Hebrew voice cloning via Chatterbox Multilingual on Apple Silicon / CPU."""
    import shutil
    import subprocess
    import tempfile

    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    global _local_tts_model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with _local_tts_lock:
        if _local_tts_model is None:
            logger.info("Loading local ChatterboxMultilingualTTS on {}", device)
            _local_tts_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    wav_tensor = _local_tts_model.generate(  # type: ignore[attr-defined]
        text,
        language_id="he",
        audio_prompt_path=settings.local_voice_reference_wav,
    )
    wav_np = wav_tensor.squeeze().detach().cpu().numpy()
    sr = getattr(_local_tts_model, "sr", 24000)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = Path(tmp_wav.name)
    try:
        sf.write(str(tmp_wav_path), wav_np, sr)
        ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp_wav_path), "-b:a", "128k", output_path],
            check=True,
        )
    finally:
        tmp_wav_path.unlink(missing_ok=True)


def _synthesize_gtts(text: str, output_path: str) -> None:
    from gtts import gTTS

    has_hebrew = any("\u0590" <= c <= "\u05FF" for c in text)
    gTTS(text=text, lang="iw" if has_hebrew else "en", slow=False).save(output_path)
