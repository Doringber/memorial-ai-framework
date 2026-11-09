"""Speech-to-Text with local Whisper or ivrit.ai Hebrew Whisper (`ivrit-ai/whisper-large-v3-turbo`)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from loguru import logger

from app.core.config import settings

_whisper_model = None
_ivrit_pipe = None
_load_lock = threading.Lock()


class SpeechModelMissingError(RuntimeError):
    """The speech recognition weights aren't on disk; downloading requires explicit approval."""


def _model_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "whisper" / f"{settings.whisper_model_size}.pt"


def _get_model():
    global _whisper_model
    with _load_lock:
        if _whisper_model is None:
            import whisper

            path = _model_path()
            if not path.is_file():
                raise SpeechModelMissingError(f"Whisper model not found at {path}")
            logger.info("Loading Whisper '{}' model...", settings.whisper_model_size)
            _whisper_model = whisper.load_model(str(path))
    return _whisper_model


def _get_ivrit_pipe():
    """Load `ivrit-ai/whisper-large-v3-turbo` from local HF cache (on Apple Silicon MPS if available)."""
    global _ivrit_pipe
    with _load_lock:
        if _ivrit_pipe is None:
            import torch
            from transformers import pipeline

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device == "mps" else torch.float32
            try:
                logger.info("Loading ivrit.ai STT '{}' on {}...", settings.ivrit_whisper_model, device)
                _ivrit_pipe = pipeline(
                    "automatic-speech-recognition",
                    model=settings.ivrit_whisper_model,
                    device=device,
                    dtype=dtype,
                )
            except Exception as e:
                raise SpeechModelMissingError(
                    f"ivrit.ai model '{settings.ivrit_whisper_model}' is not cached locally yet"
                ) from e
    return _ivrit_pipe


def transcribe_audio(file_path: Path) -> dict:
    """Transcribe an audio file. Returns 'text' and 'language'."""
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if settings.stt_provider == "ivrit":
        import whisper

        audio_array = whisper.load_audio(str(file_path))
        out = _get_ivrit_pipe()(
            {"raw": audio_array, "sampling_rate": 16000},
            generate_kwargs={"language": "hebrew", "task": "transcribe"},
        )
        text = (out.get("text", "") if isinstance(out, dict) else str(out)).strip()
        logger.info("Transcribed {} chars (ivrit.ai)", len(text))
        return {"text": text, "language": "he"}

    result = _get_model().transcribe(str(file_path), fp16=False, language="he")
    text = result.get("text", "").strip()
    logger.info("Transcribed {} chars", len(text))
    return {"text": text, "language": result.get("language", "he")}
