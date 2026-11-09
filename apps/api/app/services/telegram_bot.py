"""Telegram Bot service — family authentication, text & voice note handling, webhook, and long-polling."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
from loguru import logger

from app.core.config import settings
from app.core.security import resolve_role
from app.services.chat import chat_text
from app.services.stt import SpeechModelMissingError, transcribe_audio
from app.services.tts import VoiceUnavailableError, synthesize_speech

_auth_lock = threading.Lock()
_sessions_lock = threading.Lock()
_chat_sessions: dict[int, str] = {}
_voice_mode_chats: set[int] = set()
_last_bot_reply: dict[int, str] = {}
_polling_stop = threading.Event()
_polling_thread: threading.Thread | None = None

_VOICE_CMD = re.compile(r"^/(?:voice|speak)(?:@\w+)?(?:\s+(.*))?$", re.DOTALL | re.IGNORECASE)
_START_CMD = re.compile(r"^/start(?:@\w+)?(?:\s+(.*))?$", re.DOTALL | re.IGNORECASE)
_NEW_CMD = re.compile(r"^/(?:new|reset)(?:@\w+)?\s*$", re.IGNORECASE)

SPEAK_BUTTON_MARKUP = {
    "inline_keyboard": [[{"text": "🔊 השמע בקול", "callback_data": "speak"}]]
}

WELCOME_MESSAGE = (
    "שלום למשפחה.\n"
    "⚠️ סימולציה שנוצרה בבינה מלאכותית מתוך הודעות והקלטות בארכיון המשפחתי. זה לא האדם עצמו.\n\n"
    "איך מדברים כאן:\n"
    "• כותבים הודעה רגילה — ומקבלים תשובה בטקסט (עם כפתור 🔊 השמע בקול מתחת לכל תשובה).\n"
    "• מקליטים הודעה קולית במיקרופון — ומקבלים תשובה אוטומטית גם בקול סינתטי.\n"
    "• שולחים /voice כדי להפעיל או לכבות מענה קולי אוטומטי לכל הודעה.\n"
    "• שולחים /new כדי להתחיל שיחה חדשה."
)

AUTH_REQUIRED_MESSAGE = (
    "הבוט הזה מיועד למשפחה בלבד.\n"
    "כדי להתחבר, שלחו:\n"
    "/start <סיסמה משפחתית>"
)


def _authorized_file() -> Path:
    return settings.data_dir / "telegram_authorized.json"


def _configured_allowed_ids() -> set[int]:
    out: set[int] = set()
    for part in settings.telegram_allowed_chat_ids.split(","):
        cleaned = part.strip()
        if cleaned.lstrip("-").isdigit():
            out.add(int(cleaned))
    return out


def _load_authorized_ids() -> set[int]:
    ids = _configured_allowed_ids()
    path = _authorized_file()
    if not path.is_file():
        return ids
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(saved, list):
            ids.update(int(x) for x in saved)
    except (ValueError, OSError):
        logger.warning("Could not read telegram_authorized.json")
    return ids


def _save_authorized_id(chat_id: int) -> None:
    with _auth_lock:
        ids = _load_authorized_ids()
        ids.add(chat_id)
        path = _authorized_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(ids)), encoding="utf-8")
        path.chmod(0o600)


def is_chat_authorized(chat_id: int) -> bool:
    """Check if `chat_id` is authorized to chat with David."""
    if chat_id in _load_authorized_ids():
        return True
    no_keys_configured = not (
        settings.family_access_key
        or settings.admin_access_key
        or settings.telegram_allowed_chat_ids.strip()
    )
    return not settings.is_production and no_keys_configured


def try_authorize_chat(chat_id: int, provided_key: str) -> bool:
    """Validate `provided_key` against family/admin keys and persist `chat_id` on success."""
    if is_chat_authorized(chat_id):
        _save_authorized_id(chat_id)
        return True
    if resolve_role(provided_key.strip()) is not None:
        _save_authorized_id(chat_id)
        logger.info("Authorized Telegram chat_id={}", chat_id)
        return True
    return False


def _resolve_person_id() -> str:
    if settings.persons_file.is_file():
        try:
            persons = json.loads(settings.persons_file.read_text(encoding="utf-8"))
            if persons and isinstance(persons, list) and persons[0].get("id"):
                return str(persons[0]["id"])
        except (ValueError, OSError):
            pass
    return settings.telegram_default_person_id


def _get_session_id(chat_id: int, *, reset: bool = False) -> str:
    with _sessions_lock:
        if reset or chat_id not in _chat_sessions:
            _chat_sessions[chat_id] = str(uuid.uuid4())
        return _chat_sessions[chat_id]


def _tg_api_url(method: str) -> str:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    return f"https://api.telegram.org/bot{settings.telegram_bot_token}/{method}"


def send_chat_action(chat_id: int, action: str = "typing") -> None:
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(_tg_api_url("sendChatAction"), json={"chat_id": chat_id, "action": action})
    except Exception:
        pass


def send_telegram_message(
    chat_id: int,
    text: str,
    *,
    with_speak_button: bool = False,
) -> None:
    payload: dict[str, object] = {"chat_id": chat_id, "text": text}
    if with_speak_button:
        payload["reply_markup"] = SPEAK_BUTTON_MARKUP
    with httpx.Client(timeout=20.0) as client:
        res = client.post(_tg_api_url("sendMessage"), json=payload)
        res.raise_for_status()


def _answer_callback_query(callback_id: str, text: str = "") -> None:
    payload: dict[str, object] = {"callback_query_id": callback_id}
    if text:
        payload["text"] = text
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(_tg_api_url("answerCallbackQuery"), json=payload)
    except Exception:
        pass


def _clear_inline_keyboard(chat_id: int, message_id: int) -> None:
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(
                _tg_api_url("editMessageReplyMarkup"),
                json={"chat_id": chat_id, "message_id": message_id, "reply_markup": {"inline_keyboard": []}},
            )
    except Exception:
        pass


def _to_ogg_opus(audio_path: Path) -> Path | None:
    """Convert MP3 to Ogg Opus so Telegram renders a native voice-note waveform bubble."""
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).is_file():
        return None
    ogg_path = audio_path.with_suffix(".ogg")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(audio_path), "-c:a", "libopus", "-b:a", "48k", str(ogg_path)],
            check=True,
            timeout=20,
        )
        return ogg_path if ogg_path.is_file() else None
    except Exception:
        return None


def send_telegram_voice(chat_id: int, audio_path: Path, caption: str = "") -> None:
    ogg_path = _to_ogg_opus(audio_path)
    send_path = ogg_path or audio_path
    mime = "audio/ogg" if ogg_path else "audio/mpeg"
    try:
        with httpx.Client(timeout=45.0) as client, send_path.open("rb") as f:
            data: dict[str, str | int] = {"chat_id": chat_id}
            if caption:
                data["caption"] = caption[:1000]
            res = client.post(
                _tg_api_url("sendVoice"),
                data=data,
                files={"voice": (send_path.name, f, mime)},
            )
            res.raise_for_status()
    finally:
        if ogg_path is not None:
            ogg_path.unlink(missing_ok=True)


def download_telegram_file(file_id: str, suffix: str = ".ogg") -> Path:
    """Download a Telegram voice/audio file into `voice_uploads_dir` for local transcription."""
    settings.voice_uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.voice_uploads_dir / f"tg_{uuid.uuid4()}{suffix}"
    with httpx.Client(timeout=30.0) as client:
        meta = client.get(_tg_api_url("getFile"), params={"file_id": file_id})
        meta.raise_for_status()
        file_path = meta.json().get("result", {}).get("file_path")
        if not file_path:
            raise RuntimeError("Telegram getFile returned no file_path")
        dl_url = f"https://api.telegram.org/file/bot{settings.telegram_bot_token}/{file_path}"
        audio_res = client.get(dl_url)
        audio_res.raise_for_status()
        dest.write_bytes(audio_res.content)
        dest.chmod(0o600)
    return dest


def _speak_text_to_chat(chat_id: int, text: str) -> bool:
    """Synthesize `text` and send it as a Telegram voice note. Returns True if sent."""
    if settings.telegram_voice_replies == "off":
        return False
    try:
        send_chat_action(chat_id, "record_voice")
        tts_res = synthesize_speech(text)
        audio_file = settings.audio_output_dir / tts_res["audio_filename"]
        send_telegram_voice(chat_id, audio_file)
        return True
    except (VoiceUnavailableError, ValueError) as e:
        logger.info("Telegram voice reply skipped ({})", e)
    except Exception as e:
        logger.warning("Failed to send Telegram voice reply ({})", type(e).__name__)
    return False


def _reply_and_maybe_speak(chat_id: int, person_id: str, user_text: str, *, want_voice: bool) -> dict:
    session_id = _get_session_id(chat_id)
    send_chat_action(chat_id, "typing")
    result = chat_text(person_id=person_id, message=user_text, session_id=session_id)
    reply_text = result["reply"]
    _last_bot_reply[chat_id] = reply_text

    auto_voice = want_voice or (chat_id in _voice_mode_chats) or (settings.telegram_voice_replies == "always")
    show_button = not auto_voice and settings.telegram_voice_replies != "off"
    send_telegram_message(chat_id, reply_text, with_speak_button=show_button)

    voice_sent = _speak_text_to_chat(chat_id, reply_text) if auto_voice else False
    return {"status": "ok", "voice_sent": voice_sent}


def _handle_callback_query(cb: dict) -> dict:
    cb_id = str(cb.get("id") or "")
    data = str(cb.get("data") or "")
    msg = cb.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    msg_id = msg.get("message_id")
    text = (msg.get("text") or "").strip()

    if not isinstance(chat_id, int) or data != "speak" or not text:
        if cb_id:
            _answer_callback_query(cb_id)
        return {"status": "ignored"}

    if not is_chat_authorized(chat_id):
        if cb_id:
            _answer_callback_query(cb_id, "נדרשת סיסמה משפחתית")
        return {"status": "unauthorized"}

    if cb_id:
        _answer_callback_query(cb_id, "מכין הקלטה קולית...")
    if isinstance(msg_id, int):
        _clear_inline_keyboard(chat_id, msg_id)

    sent = _speak_text_to_chat(chat_id, text)
    return {"status": "ok", "voice_sent": sent}


def handle_telegram_update(update: dict) -> dict:
    """Process one incoming Telegram Update payload (message or callback_query)."""
    if isinstance(update.get("callback_query"), dict):
        return _handle_callback_query(update["callback_query"])

    message = update.get("message") or update.get("edited_message")
    if not isinstance(message, dict):
        return {"status": "ignored"}

    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if not isinstance(chat_id, int):
        return {"status": "ignored"}

    text = (message.get("text") or "").strip()
    start_match = _START_CMD.match(text)
    if start_match:
        payload = (start_match.group(1) or "").strip()
        if try_authorize_chat(chat_id, payload):
            _get_session_id(chat_id, reset=True)
            send_telegram_message(chat_id, WELCOME_MESSAGE)
            return {"status": "authorized"}
        send_telegram_message(chat_id, AUTH_REQUIRED_MESSAGE)
        return {"status": "unauthorized"}

    if not is_chat_authorized(chat_id):
        send_telegram_message(chat_id, AUTH_REQUIRED_MESSAGE)
        return {"status": "unauthorized"}

    if _NEW_CMD.match(text):
        _get_session_id(chat_id, reset=True)
        send_telegram_message(chat_id, "התחלנו שיחה חדשה. על מה נדבר?")
        return {"status": "reset"}

    person_id = _resolve_person_id()

    # 1. Incoming voice note / audio file
    voice_obj = message.get("voice") or message.get("audio")
    if isinstance(voice_obj, dict) and voice_obj.get("file_id"):
        send_chat_action(chat_id, "typing")
        audio_path = download_telegram_file(str(voice_obj["file_id"]), suffix=".ogg")
        try:
            transcription = transcribe_audio(audio_path)
        except SpeechModelMissingError:
            send_telegram_message(chat_id, "מודל התמלול אינו מותקן כרגע. אפשר לשלוח הודעת טקסט.")
            return {"status": "stt_unavailable"}
        finally:
            audio_path.unlink(missing_ok=True)

        user_text = transcription.get("text", "").strip()
        if not user_text:
            send_telegram_message(chat_id, "לא הצלחתי לשמוע. תנסה להקליט שוב?")
            return {"status": "empty_voice"}

        return _reply_and_maybe_speak(
            chat_id,
            person_id,
            user_text,
            want_voice=settings.telegram_voice_replies in ("voice_only", "always"),
        )

    # 2. Incoming text message or /voice command
    if text:
        voice_cmd = _VOICE_CMD.match(text)
        if voice_cmd:
            inline_question = (voice_cmd.group(1) or "").strip()
            if inline_question:
                return _reply_and_maybe_speak(chat_id, person_id, inline_question, want_voice=True)

            # Standalone `/voice`: speak the last reply if available, and toggle voice mode.
            if chat_id in _voice_mode_chats:
                _voice_mode_chats.discard(chat_id)
                send_telegram_message(
                    chat_id,
                    "מצב קול אוטומטי כובה — תשובות יישלחו בטקסט (אפשר ללחוץ על 🔊 השמע בקול מתחת לכל הודעה).",
                )
                return {"status": "voice_mode_off"}

            _voice_mode_chats.add(chat_id)
            last_reply = _last_bot_reply.get(chat_id, "")
            voice_sent = _speak_text_to_chat(chat_id, last_reply) if last_reply else False
            send_telegram_message(
                chat_id,
                "מצב קול אוטומטי הופעל 🔊 — מעכשיו כל תשובה תישלח גם בהודעה קולית (שלחו /voice שוב כדי לחזור לטקסט בלבד).",
            )
            return {"status": "voice_mode_on", "voice_sent": voice_sent}

        return _reply_and_maybe_speak(
            chat_id,
            person_id,
            text,
            want_voice=settings.telegram_voice_replies == "always",
        )

    return {"status": "ignored"}


def register_telegram_webhook(public_url: str) -> dict:
    """Register `public_url/api/telegram/webhook` with Telegram and return the bot's t.me link."""
    base = public_url.strip().rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("Telegram requires an https:// public URL for webhooks")
    webhook_url = f"{base}/api/telegram/webhook"

    payload: dict[str, object] = {
        "url": webhook_url,
        "allowed_updates": ["message", "callback_query"],
    }
    if settings.telegram_webhook_secret:
        payload["secret_token"] = settings.telegram_webhook_secret

    with httpx.Client(timeout=20.0) as client:
        me_res = client.get(_tg_api_url("getMe"))
        me_res.raise_for_status()
        username = me_res.json().get("result", {}).get("username", "")

        wh_res = client.post(_tg_api_url("setWebhook"), json=payload)
        wh_res.raise_for_status()

    return {
        "status": "ok",
        "webhook_url": webhook_url,
        "bot_username": username,
        "chat_link": f"https://t.me/{username}" if username else "",
    }


def _configure_bot_profile() -> str:
    """Clear any stale webhook so polling works, set bot menu commands, and return @username."""
    with httpx.Client(timeout=15.0) as client:
        me = client.get(_tg_api_url("getMe")).json().get("result", {})
        username = str(me.get("username", ""))
        client.post(_tg_api_url("deleteWebhook"), json={"drop_pending_updates": False})
        client.post(
            _tg_api_url("setMyCommands"),
            json={
                "commands": [
                    {"command": "start", "description": "התחלת שיחה (סימולציית זיכרון)"},
                    {"command": "voice", "description": "השמעת התשובה והפעלה/כיבוי של קול אוטומטי"},
                    {"command": "new", "description": "פתיחת שיחה חדשה ונקייה"},
                ]
            },
        )
        return username


def _polling_loop() -> None:
    try:
        username = _configure_bot_profile()
        logger.info("Telegram polling active for @{} (https://t.me/{})", username, username)
    except Exception as e:
        logger.error("Failed to initialize Telegram bot polling ({})", type(e).__name__)
        return

    offset = 0
    while not _polling_stop.is_set():
        try:
            with httpx.Client(timeout=35.0) as client:
                res = client.get(
                    _tg_api_url("getUpdates"),
                    params={
                        "offset": offset,
                        "timeout": 25,
                        "allowed_updates": json.dumps(["message", "callback_query"]),
                    },
                )
                res.raise_for_status()
                updates = res.json().get("result") or []
            for upd in updates:
                if _polling_stop.is_set():
                    break
                upd_id = upd.get("update_id")
                if isinstance(upd_id, int):
                    offset = upd_id + 1
                try:
                    handle_telegram_update(upd)
                except Exception as e:
                    logger.error("Error handling Telegram update ({}): {}", type(e).__name__, e)
        except Exception as e:
            if not _polling_stop.is_set():
                logger.debug("Telegram poll retry after {}: {}", type(e).__name__, e)
                time.sleep(3)


def start_telegram_polling() -> None:
    """Start background long-polling if `TELEGRAM_BOT_TOKEN` is configured."""
    global _polling_thread
    if not settings.telegram_bot_token or not settings.telegram_polling_enabled:
        return
    if settings.app_env == "test":
        return
    if _polling_thread is not None and _polling_thread.is_alive():
        return
    _polling_stop.clear()
    _polling_thread = threading.Thread(target=_polling_loop, name="telegram-polling", daemon=True)
    _polling_thread.start()


def stop_telegram_polling() -> None:
    _polling_stop.set()
