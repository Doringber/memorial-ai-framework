"""Style bank — David's own WhatsApp messages, indexed one message per entry.

Each message is stored with the family message he was answering (when the
export has it), and embedded together with it. At chat time the exchanges
closest to the user's message are shown to the model, so it sees how he
actually replied to similar things — not just how he wrote.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config import settings
from app.services.embeddings import embed_passages
from app.services.reranker import rerank
from app.services.vector_store import delete_collection, search_chunks, store_chunks

_LINE = re.compile(
    r"^\u200e?\[(?P<date>[\d.]+), (?P<time>[\d:]+)\] (?P<sender>[^:]+): ?(?P<text>.*)$"
)
# Some exports glue several messages onto one line; split before each header.
_INLINE_HEADER = re.compile(r"(?<=[^\n\u200e])(?=\u200e?\[\d{2}\.\d{2}\.\d{2}, [\d:]+\] )")
_ATTACHMENT = re.compile(r"\u200e?<attached: (?P<name>[^>]+)>")
_URL = re.compile(r"https?://\S+")
_SYSTEM_NOTICES = (
    "Waiting for this message",
    "security code",
    "This message was deleted",
    "message was edited",
    "omitted",
)
# A family message only counts as the one he answered if it came shortly before.
_REPLY_WINDOW = timedelta(hours=3)
_BURST_WINDOW = timedelta(minutes=3)
_MEDIA_PLACEHOLDERS = {"PHOTO": "[תמונה]", "VIDEO": "[סרטון]"}
_OFF_TOPIC_STYLE_WORDS = ("קומפוט", "רמון", "רימון", "מעשנת", "לכולם", "תמתינו", "תחיו")
MIN_CHARS = 4
MAX_CHARS = 350
MAX_PROMPT_CHARS = 300
_EMBED_BATCH = 128


@dataclass(frozen=True)
class StyleExample:
    reply: str
    prompt: str = ""


@dataclass(frozen=True)
class _Message:
    sent_at: Optional[datetime]
    sender: str
    text: str


def _iter_messages(text: str) -> Iterator[_Message]:
    """Yield raw messages (multi-line bodies joined) from a WhatsApp export."""
    header: Optional[re.Match[str]] = None
    body: list[str] = []

    def build() -> _Message:
        assert header is not None
        try:
            sent_at = datetime.strptime(f"{header['date']} {header['time']}", "%d.%m.%y %H:%M:%S")
        except ValueError:
            sent_at = None
        return _Message(sent_at, header["sender"].strip(), " ".join(body))

    for raw_line in _INLINE_HEADER.sub("\n", text).splitlines():
        match = _LINE.match(raw_line.strip())
        if match:
            if header is not None:
                yield build()
            header, body = match, [match["text"]]
        elif header is not None:
            body.append(raw_line.strip())
    if header is not None:
        yield build()


def _clean(message: str, max_chars: int = MAX_CHARS) -> str:
    message = _ATTACHMENT.sub("", message)
    message = _URL.sub("", message).replace("\u200e", "").strip()
    if not (MIN_CHARS <= len(message) <= max_chars):
        return ""
    if any(notice in message for notice in _SYSTEM_NOTICES):
        return ""
    return message


def _clean_prompt(message: str) -> str:
    """Clean a family message; photos and videos become a short placeholder."""
    cleaned = _clean(message, MAX_PROMPT_CHARS)
    if cleaned:
        return cleaned
    attachment = _ATTACHMENT.search(message)
    if attachment:
        for marker, placeholder in _MEDIA_PLACEHOLDERS.items():
            if marker in attachment["name"]:
                return placeholder
    return ""


def parse_exchanges(text: str, sender: str) -> list[StyleExample]:
    """Extract `sender`'s cleaned messages, each with the message it answered."""
    examples: dict[str, str] = {}
    last_other: Optional[_Message] = None

    for message in _iter_messages(text):
        if message.sender != sender:
            last_other = message
            continue
        reply = _clean(message.text)
        if not reply:
            continue
        prompt = ""
        if (
            last_other is not None
            and message.sent_at is not None
            and last_other.sent_at is not None
            and message.sent_at - last_other.sent_at <= _REPLY_WINDOW
        ):
            prompt = _clean_prompt(last_other.text)
        if not examples.get(reply):
            examples[reply] = prompt

    return [StyleExample(reply, prompt) for reply, prompt in examples.items()]


def parse_messages(text: str, sender: str) -> list[str]:
    """Extract `sender`'s cleaned messages from a WhatsApp export, deduplicated."""
    return [example.reply for example in parse_exchanges(text, sender)]


def merge_exports(export_paths: Sequence[Path], sender: str) -> list[StyleExample]:
    """Parse several exports; a reply seen with its context beats one without."""
    merged: dict[str, str] = {}
    for path in export_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for example in parse_exchanges(text, sender):
            if not merged.get(example.reply):
                merged[example.reply] = example.prompt
    return [StyleExample(reply, prompt) for reply, prompt in merged.items()]


def _embedding_text(example: StyleExample) -> str:
    return f"{example.prompt}\n{example.reply}" if example.prompt else example.reply


def build_style_bank(person_id: str, export_paths: Sequence[Path], sender: str) -> int:
    """Rebuild the person's style collection from WhatsApp exports. Returns count."""
    examples = merge_exports(export_paths, sender)
    delete_collection(person_id, "style")
    if not examples:
        logger.warning("No messages from '{}' found in {} export(s)", sender, len(export_paths))
        return 0

    for start in range(0, len(examples), _EMBED_BATCH):
        batch = examples[start : start + _EMBED_BATCH]
        store_chunks(
            person_id=person_id,
            chunk_ids=[f"style_{hashlib.sha1(e.reply.encode()).hexdigest()[:16]}" for e in batch],
            texts=[e.reply for e in batch],
            embeddings=embed_passages([_embedding_text(e) for e in batch]),
            metadatas=[{"person_id": person_id, "prompt": e.prompt} for e in batch],
            kind="style",
        )
    with_context = sum(1 for e in examples if e.prompt)
    logger.info(
        "Style bank for person {}: {} messages ({} with the message they answered)",
        person_id,
        len(examples),
        with_context,
    )
    return len(examples)


def retrieve_style_examples(
    person_id: str,
    query_embedding: list[float],
    k: Optional[int] = None,
    query: str = "",
) -> list[StyleExample]:
    """Return the person's real exchanges most similar to the query.

    With `query`, a wider pool is reranked and only exchanges on the same
    topic are kept: embedding distances between short chat lines are too
    close together to tell an on-topic reply from a random one.
    """
    k = k or settings.style_examples_k
    pool_size = k * 3 if query else k
    results = search_chunks(person_id, query_embedding, top_k=pool_size, kind="style")
    if not (query and results):
        return [_to_example(r) for r in results]

    # The reranker reads "family message + his reply", the same text that was embedded.
    examples = {id(r): _to_example(r) for r in results}
    candidates = [{**r, "text": _embedding_text(examples[id(r)]), "source": id(r)} for r in results]
    ranked = rerank(query, candidates, top_k=k)
    if not any("rerank_score" in c for c in ranked):
        return [examples[c["source"]] for c in ranked]
    above_threshold = [
        examples[c["source"]] for c in ranked if c["rerank_score"] >= settings.rerank_min_score
    ]
    if above_threshold:
        return above_threshold

    # Cross-encoder scores short WhatsApp style lines near 0 when they aren't
    # literal QA answers; keep the top ranked exchanges so David's real voice
    # from _chat.txt is always present in the prompt.
    kept: list[StyleExample] = list(above_threshold)
    for c in ranked:
        ex = examples[c["source"]]
        if ex in kept or _is_off_topic_style(ex, query):
            continue
        kept.append(ex)
        if len(kept) >= k:
            break
    return kept or [examples[c["source"]] for c in ranked[:k]]


def _is_off_topic_style(example: StyleExample, query: str) -> bool:
    return any(word in example.reply and word not in query for word in _OFF_TOPIC_STYLE_WORDS)


def list_whatsapp_senders(text: str) -> list[dict[str, int | str]]:
    """Return all senders in a WhatsApp export sorted by clean message count descending."""
    counts: dict[str, int] = {}
    for msg in _iter_messages(text):
        if _clean(msg.text, max_chars=450):
            counts[msg.sender] = counts.get(msg.sender, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [{"sender": sender, "messages": count} for sender, count in ranked]


def extract_conversation_windows(text: str, sender: str) -> list[dict]:
    """Build self-contained multi-turn WhatsApp exchange chunks around `sender`'s replies."""
    messages = list(_iter_messages(text))
    windows: list[dict] = []
    recent_context: list[_Message] = []
    i = 0

    while i < len(messages):
        msg = messages[i]
        if msg.sender != sender:
            cleaned_other = _clean_prompt(msg.text)
            if cleaned_other:
                recent_context.append(_Message(msg.sent_at, msg.sender, cleaned_other))
                recent_context = recent_context[-3:]
            i += 1
            continue

        roni_lines: list[str] = []
        first_at = msg.sent_at
        last_at = msg.sent_at
        while i < len(messages) and messages[i].sender == sender:
            cur = messages[i]
            if last_at and cur.sent_at and (cur.sent_at - last_at) > _BURST_WINDOW and roni_lines:
                break
            cleaned_roni = _clean(cur.text, max_chars=450)
            if cleaned_roni:
                roni_lines.append(cleaned_roni)
            if cur.sent_at:
                last_at = cur.sent_at
            i += 1

        if not roni_lines:
            continue

        valid_ctx = [
            f"{m.sender}: {m.text}"
            for m in recent_context
            if first_at and m.sent_at and (first_at - m.sent_at) <= _REPLY_WINDOW
        ]
        date_str = first_at.strftime("%Y-%m-%d") if first_at else ""
        header = f"[שיחת וואטסאפ משפחתית ({date_str})]" if date_str else "[שיחת וואטסאפ משפחתית]"
        roni_joined = "\n".join(f"{sender}: {line}" for line in roni_lines)
        body = "\n".join([header, *valid_ctx[-2:], roni_joined])
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]
        windows.append(
            {
                "id": f"wa_win_{digest}",
                "text": body,
                "metadata": {
                    "document_id": "whatsapp_dir_balak",
                    "document_title": "whatsapp_dir_balak_chat",
                    "source_type": "whatsapp_exchange",
                    "timestamp": date_str,
                },
            }
        )

    return windows


def _to_example(result: dict) -> StyleExample:
    return StyleExample(result["text"], (result.get("metadata") or {}).get("prompt", ""))
