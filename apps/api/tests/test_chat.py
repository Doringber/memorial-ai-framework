import json

import pytest

from app.core.config import settings
from app.services.chat import chat_text, commit_spoken_reply, log_latency_metrics, stream_chat_text
from app.services.llm import _strip_streaming_think, clean_reply_emojis, iter_sentences
from app.services.memory import iter_memories
from app.services.vector_store import count_chunks
from tests.conftest import PERSON_ID, FakeLLM


def test_regular_message_uses_persona_and_is_not_saved_as_memory(fake_llm: FakeLLM, write_persona) -> None:
    write_persona()
    result = chat_text(PERSON_ID, "אבא, עשינו על האש בשבת והיה מעולה")

    assert result["reply"] == fake_llm.reply
    assert "אתה דוד כהן." in fake_llm.last_system
    assert list(iter_memories(PERSON_ID)) == []
    assert count_chunks(PERSON_ID) == 0


def test_session_history_is_sent_on_next_turn(fake_llm: FakeLLM) -> None:
    first = chat_text(PERSON_ID, "שלום אבא")
    chat_text(PERSON_ID, "מה שלומך?", first["session_id"])

    contents = [m["content"] for m in fake_llm.calls[-1] if m["role"] != "system"]
    assert contents == ["שלום אבא", fake_llm.reply, "מה שלומך?"]


def test_save_then_fact_is_in_every_prompt(fake_llm: FakeLLM) -> None:
    saved = chat_text(PERSON_ID, "זכור: נועה גרה ברעות")
    assert saved["reply"] == "נרשם: נועה גרה ברעות 👍"
    assert fake_llm.calls == []

    chat_text(PERSON_ID, "איפה נועה גרה?")
    assert "- נועה גרה ברעות" in fake_llm.last_system


def test_forget_removes_fact_from_prompt_and_index(fake_llm: FakeLLM) -> None:
    chat_text(PERSON_ID, "זכור: לנועה יש שתי חתולות")
    assert count_chunks(PERSON_ID) == 1

    result = chat_text(PERSON_ID, "שכח חתולות")
    assert result["reply"].startswith("מחקתי 1")
    assert count_chunks(PERSON_ID) == 0

    chat_text(PERSON_ID, "מה עם החתולות?")
    assert "שתי חתולות" not in fake_llm.last_system


@pytest.mark.parametrize("message", ["שכח", "זכור", "  /forget  "])
def test_empty_commands_ask_for_payload(fake_llm: FakeLLM, message: str) -> None:
    assert "?" in chat_text(PERSON_ID, message)["reply"]
    assert fake_llm.calls == []


@pytest.mark.parametrize("message", ["שכחתי לספר לך שנטע חזרה", "זכורה לי הנסיעה לרומניה"])
def test_words_starting_with_trigger_are_normal_messages(fake_llm: FakeLLM, message: str) -> None:
    chat_text(PERSON_ID, message)
    assert len(fake_llm.calls) == 1
    assert list(iter_memories(PERSON_ID)) == []


def test_safety_short_circuits_llm(fake_llm: FakeLLM) -> None:
    result = chat_text(PERSON_ID, "אני רוצה למות")
    assert "1201" in result["reply"]
    assert fake_llm.calls == []


def test_iter_sentences_splits_hebrew_whatsapp_sentences() -> None:
    chunks = ["בסדר גמור, כלום ", "מיוחד.... 🥰 ", "מה אצלכם בבית היום? ", "תעדכנו כשתגיעו 👍"]
    assert list(iter_sentences(chunks)) == [
        "בסדר גמור, כלום מיוחד.... 🥰",
        "מה אצלכם בבית היום?",
        "תעדכנו כשתגיעו 👍",
    ]


def test_clean_reply_emojis_caps_emoji_chains() -> None:
    assert clean_reply_emojis("כלום מיוחד.... 😜😂👍", max_emojis=1) == "כלום מיוחד.... 😜"
    assert clean_reply_emojis("בסדר גמור 🥰😄", max_emojis=0) == "בסדר גמור"


def test_strip_streaming_think_across_chunk_boundaries() -> None:
    pieces = ["<thi", "nk>מחשבה פנימית</th", "ink>תשובה נקייה"]
    in_think = False
    carry = ""
    out: list[str] = []
    for piece in pieces:
        carry, in_think, clean = _strip_streaming_think(carry + piece, in_think)
        out.append(clean)
    if carry and not in_think:
        out.append(carry)
    assert "".join(out) == "תשובה נקייה"


def test_deferred_stream_commits_only_spoken_sentences_to_history(fake_llm: FakeLLM) -> None:
    fake_llm.reply = "משפט ראשון שהושמע במלואו.... 🥰 משפט שני שנקטע ולא הושמע בכלל."
    events = list(stream_chat_text(PERSON_ID, "מה נשמע?", defer_history=True))

    sentences = [e["text"] for e in events if e["type"] == "sentence"]
    session_id = events[0]["session_id"]
    assert len(sentences) == 2

    # User interrupts after sentence 0 finishes playing; sentence 1 was never heard.
    commit_spoken_reply(session_id, "מה נשמע?", sentences[0])

    fake_llm.reply = "המשך"
    chat_text(PERSON_ID, "ומה עוד?", session_id)
    contents = [m["content"] for m in fake_llm.calls[-1] if m["role"] != "system"]
    assert contents == ["מה נשמע?", sentences[0], "ומה עוד?"]


def test_barge_in_truncates_already_saved_reply_or_drops_if_unspoken(fake_llm: FakeLLM) -> None:
    fake_llm.reply = "משפט ראשון ארוך מספיק. משפט שני שלא הושמע."
    first = chat_text(PERSON_ID, "שאלה ראשונה")
    sid = first["session_id"]

    commit_spoken_reply(sid, "שאלה ראשונה", "משפט ראשון ארוך מספיק.")
    chat_text(PERSON_ID, "שאלה שנייה", sid)
    contents = [m["content"] for m in fake_llm.calls[-1] if m["role"] != "system"]
    assert contents == ["שאלה ראשונה", "משפט ראשון ארוך מספיק.", "שאלה שנייה"]

    # Interrupted before a single word was played on turn 2 -> turn 2 is dropped cleanly.
    commit_spoken_reply(sid, "שאלה שנייה", "")
    chat_text(PERSON_ID, "שאלה שלישית", sid)
    contents = [m["content"] for m in fake_llm.calls[-1] if m["role"] != "system"]
    assert contents == ["שאלה ראשונה", "משפט ראשון ארוך מספיק.", "שאלה שלישית"]


def test_latency_log_strips_any_text_fields() -> None:
    log_latency_metrics(
        {
            "mode": "voice",
            "stt_ms": 320.5,
            "rag_ms": 140.2,
            "llm_first_sentence_ms": 810.0,
            "tts_first_audio_ms": 620.0,
            "client_e2e_ms": 1890.7,
            "interrupted": True,
            "sentence_count": 2,
            "spoken_sentences": 1,
            "reply": "אסור לשמור טקסט כאן",
            "user_message": "גם לא כאן",
        }
    )
    path = settings.data_dir.parent / "eval" / "latency.jsonl"
    saved = json.loads(path.read_text(encoding="utf-8").strip())
    assert saved["client_e2e_ms"] == 1890.7
    assert saved["interrupted"] is True
    assert "reply" not in saved
    assert "user_message" not in saved
