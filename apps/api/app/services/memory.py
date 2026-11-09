"""Memory service — facts and stories the family adds about the person.

Every memory lives in two places: an append-only JSONL log (source of truth,
used to rebuild the index) and the person's archive collection (for retrieval).
Facts saved from chat ("explicit_save") are also injected into every prompt.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.services.embeddings import embed_passages
from app.services.vector_store import delete_chunks, store_chunks

# Sources a family member may remove from chat with "שכח". Admin-added
# ("manual") memories can only be removed by editing the log.
FORGETTABLE_SOURCES = frozenset({"explicit_save", "chat_conversation"})
# Old builds indexed raw chat turns as memories. They hold the bot's own
# replies, so they must never come back as "facts".
UNINDEXED_SOURCES = frozenset({"chat_conversation"})

_log_lock = threading.Lock()


def _memories_file(person_id: str) -> Path:
    return settings.memories_dir / f"{person_id}_memories.jsonl"


def _conversations_file(person_id: str) -> Path:
    return settings.memories_dir / f"{person_id}_conversations.jsonl"


def memory_chunk(person_id: str, entry: dict) -> tuple[str, str, dict]:
    """Return (chunk_id, indexed_text, metadata) for a memory log entry."""
    contributor = entry.get("contributor") or "family member"
    timestamp = entry.get("timestamp", "")
    text = f"[Memory shared by {contributor} on {timestamp}]: {entry['text']}"
    metadata = {
        "person_id": person_id,
        "document_id": entry["id"],
        "document_title": f"family_memory_{timestamp}",
        "source_type": entry.get("source", "family_conversation"),
        "contributor": contributor,
        "timestamp": timestamp,
    }
    return f"memory_{entry['id']}", text, metadata


def iter_memories(person_id: str, include_forgotten: bool = False) -> Iterator[dict]:
    """Yield memory log entries in the order they were saved."""
    log_file = _memories_file(person_id)
    if not log_file.exists():
        return
    with log_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed memory line in {}", log_file.name)
                continue
            if entry.get("text") and (include_forgotten or not entry.get("forgotten")):
                yield entry


def save_memory(
    person_id: str,
    text: str,
    source: str = "family_conversation",
    contributor: str = "",
) -> dict:
    """Store a memory in the log and the vector index."""
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "text": text.strip(),
        "source": source,
        "contributor": contributor,
    }

    chunk_id, indexed_text, metadata = memory_chunk(person_id, entry)
    store_chunks(
        person_id=person_id,
        chunk_ids=[chunk_id],
        texts=[indexed_text],
        embeddings=embed_passages([indexed_text]),
        metadatas=[metadata],
    )

    settings.memories_dir.mkdir(parents=True, exist_ok=True)
    with _log_lock, _memories_file(person_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Saved {} memory for person {} ({} chars)", source, person_id, len(text))
    return {"id": entry["id"], "status": "saved", "text": entry["text"]}


def load_explicit_facts(person_id: str, limit: int = 30) -> list[str]:
    """Return facts saved from chat, newest first."""
    facts = [e["text"] for e in iter_memories(person_id) if e.get("source") == "explicit_save"]
    facts.reverse()
    return facts[:limit]


def forget_facts(person_id: str, query: str) -> int:
    """Forget every chat-saved memory containing `query`. Returns how many."""
    needle = query.strip().lower()
    log_file = _memories_file(person_id)
    if not needle or not log_file.exists():
        return 0

    forgotten_ids: list[str] = []
    with _log_lock:
        out_lines: list[str] = []
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            if (
                entry.get("source") in FORGETTABLE_SOURCES
                and not entry.get("forgotten")
                and needle in (entry.get("text") or "").lower()
            ):
                entry["forgotten"] = True
                forgotten_ids.append(entry["id"])
            out_lines.append(json.dumps(entry, ensure_ascii=False))

        if forgotten_ids:
            log_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    if forgotten_ids:
        delete_chunks(person_id, [f"memory_{mid}" for mid in forgotten_ids])
        logger.info("Forgot {} memories for person {}", len(forgotten_ids), person_id)
    return len(forgotten_ids)


def save_conversation_turn(
    person_id: str,
    user_message: str,
    bot_reply: str,
    session_id: str,
) -> None:
    """Append a chat turn to the person's conversation log (not used for retrieval)."""
    settings.memories_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user": user_message,
        "bot": bot_reply,
    }
    with _log_lock, _conversations_file(person_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
