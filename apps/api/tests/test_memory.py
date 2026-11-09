from app.services.memory import forget_facts, iter_memories, load_explicit_facts, save_memory
from app.services.vector_store import count_chunks
from tests.conftest import PERSON_ID


def test_explicit_facts_newest_first() -> None:
    save_memory(PERSON_ID, "ישן", source="explicit_save")
    save_memory(PERSON_ID, "חדש", source="explicit_save")
    save_memory(PERSON_ID, "ידני", source="manual")

    assert load_explicit_facts(PERSON_ID) == ["חדש", "ישן"]


def test_forget_only_touches_chat_sources() -> None:
    save_memory(PERSON_ID, "דוד אהב חמוצים", source="explicit_save")
    save_memory(PERSON_ID, "דוד אהב חמוצים עם שמיר", source="manual")

    assert forget_facts(PERSON_ID, "חמוצים") == 1
    remaining = [m["text"] for m in iter_memories(PERSON_ID)]
    assert remaining == ["דוד אהב חמוצים עם שמיר"]
    assert count_chunks(PERSON_ID) == 1
    assert len(list(iter_memories(PERSON_ID, include_forgotten=True))) == 2


def test_forget_empty_query_is_noop() -> None:
    save_memory(PERSON_ID, "משהו", source="explicit_save")
    assert forget_facts(PERSON_ID, "  ") == 0
