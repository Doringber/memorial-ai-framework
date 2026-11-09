from pathlib import Path

import pytest

from app.services import style_bank
from app.services.embeddings import embed_query
from app.services.style_bank import (
    StyleExample,
    build_style_bank,
    extract_conversation_windows,
    merge_exports,
    parse_exchanges,
    parse_messages,
    retrieve_style_examples,
)
from tests.conftest import PERSON_ID

EXPORT = (
    "[05.08.15, 23:56:54] דוד כהן: סתם מסעדה מקומית\n"
    "[05.08.15, 23:57:10] נועה: אבא איפה אתם?\n"
    "\u200e[06.08.15, 00:16:13] דוד כהן: נועה אל תתחכמי, בשביל זה\n"
    "המציאו את המנגל\n"
    "[06.08.15, 01:04:55] דוד כהן: הנשר נחת... \u200e<attached: 0001-PHOTO.jpg>\n"
    "[06.08.15, 01:05:00] דוד כהן: \u200e<attached: 0002-PHOTO.jpg>\n"
    "[13.05.25, 12:41:27] דוד כהן: על האש.\n"
    "[15.05.25, 23:14:24] דוד כהן:\n"
    "[16.05.25, 10:00:00] דוד כהן: https://example.com/x\n"
    "[16.05.25, 10:00:01] דוד כהן: \u200eThis message was deleted\n"
    "[16.05.25, 10:00:02] דוד כהן: סתם מסעדה מקומית\n"
)


def test_parse_keeps_only_sender_clean_unique_messages() -> None:
    assert parse_messages(EXPORT, "דוד כהן") == [
        "סתם מסעדה מקומית",
        "נועה אל תתחכמי, בשביל זה המציאו את המנגל",
        "הנשר נחת...",
        "על האש.",
    ]


def test_parse_splits_headers_glued_onto_one_line() -> None:
    glued = "[01.01.20, 10:00:00] דוד כהן: נכון[01.01.20, 10:00:05] דוד כהן: אכן כן"
    assert parse_messages(glued, "דוד כהן") == ["נכון", "אכן כן"]


def test_replies_carry_the_recent_family_message_they_answered() -> None:
    exchanges = {e.reply: e.prompt for e in parse_exchanges(EXPORT, "דוד כהן")}

    assert exchanges["סתם מסעדה מקומית"] == ""
    assert exchanges["נועה אל תתחכמי, בשביל זה המציאו את המנגל"] == "אבא איפה אתם?"
    assert exchanges["הנשר נחת..."] == "אבא איפה אתם?"  # same burst, within the window
    assert exchanges["על האש."] == ""  # ten years later is not a reply


def test_photo_becomes_placeholder_prompt() -> None:
    export = (
        "[01.01.20, 10:00:00] נטע: \u200e<attached: 0003-PHOTO-2020.jpg>\n"
        "[01.01.20, 10:01:00] דוד כהן: יפה מאוד\n"
    )
    assert parse_exchanges(export, "דוד כהן") == [StyleExample("יפה מאוד", "[תמונה]")]


def test_merge_prefers_the_version_with_context(tmp_path: Path) -> None:
    roni_only = tmp_path / "a.txt"
    roni_only.write_text("[01.01.20, 10:01:00] דוד כהן: יפה מאוד\n", encoding="utf-8")
    group = tmp_path / "b.txt"
    group.write_text(
        "[01.01.20, 10:00:00] נטע: קנינו מעשנת\n[01.01.20, 10:01:00] דוד כהן: יפה מאוד\n",
        encoding="utf-8",
    )

    assert merge_exports([roni_only, group], "דוד כהן") == [StyleExample("יפה מאוד", "קנינו מעשנת")]


def test_build_and_retrieve(tmp_path: Path) -> None:
    export = tmp_path / "chat.txt"
    export.write_text(EXPORT, encoding="utf-8")

    assert build_style_bank(PERSON_ID, [export], "דוד כהן") == 4
    assert build_style_bank(PERSON_ID, [export], "דוד כהן") == 4  # idempotent

    examples = retrieve_style_examples(PERSON_ID, embed_query("בשביל מה המציאו את המנגל"), k=1)
    assert examples == [StyleExample("נועה אל תתחכמי, בשביל זה המציאו את המנגל", "אבא איפה אתם?")]


def test_reranked_retrieval_keeps_only_on_topic_exchanges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "chat.txt"
    export.write_text(EXPORT, encoding="utf-8")
    build_style_bank(PERSON_ID, [export], "דוד כהן")
    seen: list[str] = []

    def fake_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
        seen.extend(c["text"] for c in candidates)
        scored = [{**c, "rerank_score": 0.9 if "מנגל" in c["text"] else 0.01} for c in candidates]
        return sorted(scored, key=lambda c: -c["rerank_score"])[:top_k]

    monkeypatch.setattr(style_bank, "rerank", fake_rerank)
    query = "בשביל מה המציאו את המנגל"
    examples = retrieve_style_examples(PERSON_ID, embed_query(query), k=2, query=query)

    assert examples == [StyleExample("נועה אל תתחכמי, בשביל זה המציאו את המנגל", "אבא איפה אתם?")]
    assert "אבא איפה אתם?\nנועה אל תתחכמי, בשביל זה המציאו את המנגל" in seen


def test_reranked_retrieval_falls_back_when_all_scores_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "chat.txt"
    export.write_text(EXPORT, encoding="utf-8")
    build_style_bank(PERSON_ID, [export], "דוד כהן")

    def fake_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
        return [{**c, "rerank_score": 0.01} for c in candidates[:top_k]]

    monkeypatch.setattr(style_bank, "rerank", fake_rerank)
    examples = retrieve_style_examples(PERSON_ID, embed_query("שאלה כללית"), k=2, query="שאלה כללית")
    assert len(examples) == 2


def test_extract_conversation_windows_preserves_family_context_and_bursts() -> None:
    windows = extract_conversation_windows(EXPORT, "דוד כהן")
    assert len(windows) >= 2
    joined = "\n".join(w["text"] for w in windows)
    assert "נועה: אבא איפה אתם?" in joined
    assert "דוד כהן: נועה אל תתחכמי, בשביל זה המציאו את המנגל" in joined
