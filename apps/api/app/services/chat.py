"""Chat service — orchestrates commands, safety, retrieval, prompt building, and the LLM."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from datetime import datetime

from loguru import logger

from app.core.config import settings
from app.services.embeddings import embed_query
from app.services.llm import clean_reply_emojis, generate_response, iter_sentences, stream_response
from app.services.memory import forget_facts, load_explicit_facts, save_conversation_turn, save_memory
from app.services.persona import load_persona
from app.services.prompt_builder import build_dynamic_prompt, build_messages, build_persona_prompt
from app.services.retrieval import expand_query, is_small_talk, retrieve_context
from app.services.safety import check_message
from app.services.style_bank import retrieve_style_examples
from app.services.web_search import fetch_web_context, format_web_context

_default_generate_response = generate_response
_latency_lock = threading.Lock()
_LATENCY_KEYS = (
    "mode",
    "stt_ms",
    "rag_ms",
    "llm_first_sentence_ms",
    "llm_total_ms",
    "tts_first_audio_ms",
    "client_e2e_ms",
    "interrupted",
    "sentence_count",
    "spoken_sentences",
    "reply_chars",
    "spoken_chars",
)

AI_DISCLAIMER = [
    "This is an AI-generated simulation based on archived materials.",
    "This response may be inaccurate and does not represent the real person.",
]

MAX_SESSIONS = 500

# Commands must start the message and the trigger must be a whole word, so
# "שכחתי..." or "זכורה לי..." are ordinary messages. Payload follows the trigger.
_SEPARATOR = r"(?:[\s:,\-.'\"]+|$)"
_SAVE_TRIGGER = re.compile(rf"^\s*(?:/save|/remember|זכור(?:ש)?|תזכור(?:ש)?){_SEPARATOR}", re.IGNORECASE)
_FORGET_TRIGGER = re.compile(rf"^\s*(?:/forget|שכח|תשכח){_SEPARATOR}", re.IGNORECASE)

_sessions: OrderedDict[str, list[dict]] = OrderedDict()
_sessions_lock = threading.Lock()


def _result(reply: str, session_id: str, safety_notes: list[str] | None = None,
            citations: list[dict] | None = None) -> dict:
    return {
        "reply": reply,
        "citations": citations or [],
        "safety_notes": safety_notes or [],
        "session_id": session_id,
    }


def _command_payload(trigger: re.Pattern[str], message: str) -> str | None:
    """Return the text after `trigger` ("" if empty), or None if the message isn't that command."""
    match = trigger.match(message)
    return message[match.end():].strip() if match else None


def _get_history(session_id: str) -> list[dict]:
    with _sessions_lock:
        history = _sessions.get(session_id, [])
        return list(history)


def _append_history(session_id: str, user_message: str, reply: str) -> None:
    with _sessions_lock:
        history = _sessions.pop(session_id, [])
        history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ])
        _sessions[session_id] = history
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)


def commit_spoken_reply(session_id: str, user_message: str, spoken_text: str) -> dict:
    """Save or truncate the assistant's reply in `session_id` to only what was actually played."""
    user_clean = user_message.strip()
    spoken_clean = spoken_text.strip()
    with _sessions_lock:
        history = _sessions.pop(session_id, [])
        has_matching_tail = (
            len(history) >= 2
            and history[-2]["role"] == "user"
            and history[-2]["content"] == user_clean
            and history[-1]["role"] == "assistant"
        )
        if has_matching_tail:
            if spoken_clean:
                history[-1]["content"] = spoken_clean
            else:
                history.pop()
                history.pop()
        elif user_clean and spoken_clean:
            history.extend([
                {"role": "user", "content": user_clean},
                {"role": "assistant", "content": spoken_clean},
            ])
        _sessions[session_id] = history
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    return {
        "session_id": session_id,
        "committed_chars": len(spoken_clean),
    }


def log_latency_metrics(metrics: dict) -> None:
    """Append privacy-safe turn latency numbers (never text) to $MEMORIAL_DATA_HOME/eval/latency.jsonl."""
    record = {"timestamp": datetime.now().isoformat(timespec="seconds")}
    for key in _LATENCY_KEYS:
        if key in metrics and metrics[key] is not None:
            record[key] = metrics[key]
    eval_dir = settings.data_dir.parent / "eval"
    try:
        eval_dir.mkdir(parents=True, exist_ok=True)
        path = eval_dir / "latency.jsonl"
        with _latency_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            path.chmod(0o600)
    except OSError as e:
        logger.warning("Failed to write latency log: {}", e)


def _handle_command(person_id: str, message: str, session_id: str) -> dict | None:
    forget = _command_payload(_FORGET_TRIGGER, message)
    if forget == "":
        return _result("מה לשכוח? כתוב מילה מהעובדה. לדוגמה: שכח נועה", session_id)
    if forget:
        removed = forget_facts(person_id, forget)
        reply = f"מחקתי {removed} עובדות שהזכירו '{forget}'." if removed else "לא מצאתי עובדה שמתאימה."
        return _result(reply, session_id)

    save = _command_payload(_SAVE_TRIGGER, message)
    if save == "":
        return _result("מה לזכור? כתוב אחרי המילה. לדוגמה: זכור: נועה גרה ברעות", session_id)
    if save:
        save_memory(person_id=person_id, text=save, source="explicit_save", contributor="family_member")
        return _result(
            f"נרשם: {save} 👍",
            session_id,
            ["This memory was saved by the family and will be used in future chats."],
        )
    return None


def chat_text(person_id: str, message: str, session_id: str | None = None) -> dict:
    """Answer one chat message.

    Order: save/forget commands → safety → retrieval (archive + style bank)
    → prompt → LLM → conversation log.
    """
    session_id = session_id or str(uuid.uuid4())

    command_result = _handle_command(person_id, message, session_id)
    if command_result:
        return command_result

    safety = check_message(message)
    if safety:
        logger.warning("Safety '{}' triggered for session {}", safety.category, session_id)
        return _result(safety.reply, session_id, ["This request was handled by the safety system."])

    messages, citations, _ = _prepare_turn(person_id, message, session_id)

    logger.info("Generating response for session {}", session_id)
    reply = clean_reply_emojis(generate_response(messages).strip())
    _append_history(session_id, message, reply)

    if settings.save_conversation_log:
        try:
            save_conversation_turn(person_id, message, reply, session_id)
        except OSError as e:
            logger.warning("Failed to log conversation turn: {}", e)

    return _result(reply, session_id, AI_DISCLAIMER, citations)


def _prepare_turn(person_id: str, message: str, session_id: str) -> tuple[list[dict], list[dict], float]:
    """Run retrieval + prompt assembly and return (messages, citations, rag_ms)."""
    rag_start = time.perf_counter()
    persona = load_persona(person_id) or {}
    search_query = expand_query(message, persona.get("retrieval_aliases", {}))
    query_embedding = embed_query(search_query)
    small_talk = is_small_talk(message)
    subject = persona.get("display_name", "")
    retrieved = [] if small_talk else retrieve_context(person_id, search_query, subject=subject)
    style_examples = retrieve_style_examples(
        person_id,
        query_embedding,
        k=4 if small_talk else settings.style_examples_k,
        query=search_query,
    )
    web_headlines = [] if small_talk else fetch_web_context(message)
    messages = build_messages(
        persona_prompt=build_persona_prompt(persona or None),
        dynamic_prompt=build_dynamic_prompt(
            load_explicit_facts(person_id),
            style_examples,
            retrieved,
            web_context=format_web_context(web_headlines),
        ),
        conversation_history=_get_history(session_id),
        user_message=message,
    )
    citations = [
        {
            "document_title": chunk.get("metadata", {}).get("document_title", "unknown"),
            "chunk_id": chunk.get("id", ""),
        }
        for chunk in retrieved
    ]
    rag_ms = round((time.perf_counter() - rag_start) * 1000, 1)
    return messages, citations, rag_ms


def stream_chat_text(
    person_id: str,
    message: str,
    session_id: str | None = None,
    *,
    defer_history: bool = False,
) -> Iterator[dict]:
    """Yield `meta`, per-`sentence`, and `done` events for a chat turn.

    When `defer_history=True`, the assistant reply is not added to session
    history until the client confirms which sentences were actually played
    via `commit_spoken_reply`.
    """
    session_id = session_id or str(uuid.uuid4())

    command_result = _handle_command(person_id, message, session_id)
    if command_result:
        yield {
            "type": "meta",
            "session_id": session_id,
            "rag_ms": 0.0,
            "citations": [],
            "safety_notes": command_result["safety_notes"],
        }
        yield {"type": "sentence", "index": 0, "text": command_result["reply"], "llm_ms": 0.0}
        yield {
            "type": "done",
            "session_id": session_id,
            "reply": command_result["reply"],
            "rag_ms": 0.0,
            "llm_first_sentence_ms": 0.0,
            "llm_total_ms": 0.0,
            "sentence_count": 1,
        }
        return

    safety = check_message(message)
    if safety:
        logger.warning("Safety '{}' triggered for session {}", safety.category, session_id)
        notes = ["This request was handled by the safety system."]
        yield {"type": "meta", "session_id": session_id, "rag_ms": 0.0, "citations": [], "safety_notes": notes}
        yield {"type": "sentence", "index": 0, "text": safety.reply, "llm_ms": 0.0}
        yield {
            "type": "done",
            "session_id": session_id,
            "reply": safety.reply,
            "rag_ms": 0.0,
            "llm_first_sentence_ms": 0.0,
            "llm_total_ms": 0.0,
            "sentence_count": 1,
        }
        return

    messages, citations, rag_ms = _prepare_turn(person_id, message, session_id)
    yield {
        "type": "meta",
        "session_id": session_id,
        "rag_ms": rag_ms,
        "citations": citations,
        "safety_notes": AI_DISCLAIMER,
    }

    llm_start = time.perf_counter()
    first_sentence_ms: float | None = None
    sentences: list[str] = []
    tokens = (
        iter([generate_response(messages)])
        if generate_response is not _default_generate_response
        else stream_response(messages)
    )

    emoji_budget = 1
    for idx, raw_sentence in enumerate(iter_sentences(tokens)):
        sentence = clean_reply_emojis(raw_sentence, max_emojis=emoji_budget)
        if any(ord(c) >= 0x2600 for c in sentence):
            emoji_budget = 0
        if not sentence:
            continue
        elapsed_ms = round((time.perf_counter() - llm_start) * 1000, 1)
        if first_sentence_ms is None:
            first_sentence_ms = elapsed_ms
        sentences.append(sentence)
        yield {"type": "sentence", "index": idx, "text": sentence, "llm_ms": elapsed_ms}

    full_reply = " ".join(sentences).strip()
    total_llm_ms = round((time.perf_counter() - llm_start) * 1000, 1)

    if not defer_history and full_reply:
        _append_history(session_id, message, full_reply)

    if settings.save_conversation_log and full_reply:
        try:
            save_conversation_turn(person_id, message, full_reply, session_id)
        except OSError as e:
            logger.warning("Failed to log conversation turn: {}", e)

    yield {
        "type": "done",
        "session_id": session_id,
        "reply": full_reply,
        "rag_ms": rag_ms,
        "llm_first_sentence_ms": first_sentence_ms or total_llm_ms,
        "llm_total_ms": total_llm_ms,
        "sentence_count": len(sentences),
    }
