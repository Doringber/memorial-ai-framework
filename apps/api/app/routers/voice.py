"""Voice endpoints — speech in, reply text + David's voice out."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import require_family
from app.core.uploads import save_upload
from app.models.schemas import Citation
from app.services.chat import chat_text, stream_chat_text
from app.services.stt import SpeechModelMissingError, transcribe_audio
from app.services.tts import VoiceUnavailableError, synthesize_speech, voice_info

router = APIRouter(prefix="/api/chat", tags=["voice"])

_AUDIO_SUFFIXES = {".webm", ".mp4", ".m4a", ".ogg", ".opus", ".wav", ".mp3"}
# Filenames are random UUIDs, so the URL itself is the capability; <audio>
# elements can't send auth headers.
_AUDIO_NAME = re.compile(r"^[0-9a-f-]{36}\.mp3$")


class VoiceChatResponse(BaseModel):
    transcription: str
    language: str
    reply: str
    audio_url: str
    # Why audio_url is empty: "off", "blocked", "budget" or "failed".
    voice_unavailable: str = ""
    citations: list[Citation] = []
    safety_notes: list[str] = []
    session_id: str
    stt_ms: float = 0.0
    tts_ms: float = 0.0


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class SpeakResponse(BaseModel):
    audio_url: str
    tts_ms: float = 0.0


class VoiceInfoResponse(BaseModel):
    provider: str
    external: bool
    available: bool
    chars_used: int
    chars_limit: int


def _audio_url(filename: str) -> str:
    return f"/api/chat/audio/{filename}"


def _transcribe_upload(audio: UploadFile) -> tuple[dict, float]:
    audio_path = save_upload(
        audio, settings.voice_uploads_dir, settings.max_voice_upload_mb, _AUDIO_SUFFIXES, ".webm"
    )
    stt_start = time.perf_counter()
    try:
        transcription = transcribe_audio(audio_path)
    except SpeechModelMissingError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Speech recognition is not installed") from e
    finally:
        audio_path.unlink(missing_ok=True)
    stt_ms = round((time.perf_counter() - stt_start) * 1000, 1)
    return transcription, stt_ms


@router.post("/voice", response_model=VoiceChatResponse, dependencies=[Depends(require_family)])
def voice_chat(
    audio: UploadFile = File(...),
    person_id: str = Form(...),
    session_id: str = Form(""),
) -> VoiceChatResponse:
    """Transcribe → chat → synthesize. The uploaded recording is deleted after transcription."""
    transcription, stt_ms = _transcribe_upload(audio)
    user_text = transcription["text"]
    if not user_text:
        return VoiceChatResponse(
            transcription="",
            language=transcription["language"],
            reply="לא הצלחתי לשמוע. תנסה שוב?",
            audio_url="",
            session_id=session_id or str(uuid.uuid4()),
            stt_ms=stt_ms,
        )

    chat_result = chat_text(person_id=person_id, message=user_text, session_id=session_id or None)
    audio_url, unavailable, tts_ms = "", "", 0.0
    tts_start = time.perf_counter()
    try:
        audio_url = _audio_url(synthesize_speech(chat_result["reply"])["audio_filename"])
        tts_ms = round((time.perf_counter() - tts_start) * 1000, 1)
    except VoiceUnavailableError as e:
        unavailable = e.reason
    except ValueError:
        unavailable = "off"

    return VoiceChatResponse(
        transcription=user_text,
        language=transcription["language"],
        audio_url=audio_url,
        voice_unavailable=unavailable,
        stt_ms=stt_ms,
        tts_ms=tts_ms,
        **chat_result,
    )


@router.post("/voice-stream", dependencies=[Depends(require_family)])
def voice_chat_stream(
    audio: UploadFile = File(...),
    person_id: str = Form(...),
    session_id: str = Form(""),
) -> StreamingResponse:
    """Transcribe, then stream reply sentences as NDJSON with deferred history commit."""
    transcription, stt_ms = _transcribe_upload(audio)
    user_text = transcription["text"]
    sid = session_id or str(uuid.uuid4())

    def events() -> Iterator[bytes]:
        yield (
            json.dumps(
                {
                    "type": "transcription",
                    "transcription": user_text,
                    "language": transcription["language"],
                    "stt_ms": stt_ms,
                    "session_id": sid,
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        if not user_text:
            empty_reply = "לא הצלחתי לשמוע. תנסה שוב?"
            yield (
                json.dumps(
                    {
                        "type": "sentence",
                        "index": 0,
                        "text": empty_reply,
                        "llm_ms": 0.0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            yield (
                json.dumps(
                    {
                        "type": "done",
                        "session_id": sid,
                        "reply": empty_reply,
                        "rag_ms": 0.0,
                        "llm_first_sentence_ms": 0.0,
                        "llm_total_ms": 0.0,
                        "sentence_count": 1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            return

        for event in stream_chat_text(
            person_id=person_id,
            message=user_text,
            session_id=sid,
            defer_history=True,
        ):
            yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    return StreamingResponse(events(), media_type="application/x-ndjson")


@router.get("/voice-info", response_model=VoiceInfoResponse, dependencies=[Depends(require_family)])
def get_voice_info() -> VoiceInfoResponse:
    """Which voice engine is active, whether it is external, and the monthly budget."""
    info = voice_info()
    return VoiceInfoResponse(**info.__dict__)


@router.post("/speak", response_model=SpeakResponse, dependencies=[Depends(require_family)])
def speak(req: SpeakRequest) -> SpeakResponse:
    """Synthesize an existing text reply in David's voice."""
    tts_start = time.perf_counter()
    try:
        result = synthesize_speech(req.text)
    except VoiceUnavailableError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, e.reason) from e
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    tts_ms = round((time.perf_counter() - tts_start) * 1000, 1)
    return SpeakResponse(audio_url=_audio_url(result["audio_filename"]), tts_ms=tts_ms)


@router.get("/audio/{filename}")
def serve_audio(filename: str) -> FileResponse:
    if not _AUDIO_NAME.match(filename):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audio not found")
    audio_path = settings.audio_output_dir / filename
    if not audio_path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Audio not found")
    return FileResponse(audio_path, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})
