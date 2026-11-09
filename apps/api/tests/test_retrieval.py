import pytest

from app.core.config import settings
from app.services import retrieval
from app.services.memory import save_memory
from app.services.retrieval import expand_query, is_small_talk, retrieve_context
from app.services.web_search import fetch_web_context, format_web_context, needs_web_grounding
from tests.conftest import PERSON_ID


@pytest.mark.parametrize(
    "message",
    ["היוש", "מה נשמע?", "היי אבא!", "מה אתה עושה היום", "בוקר טוב דוד 🥰", "שבת שלום", "What's up"],
)
def test_small_talk_is_detected(message: str) -> None:
    assert is_small_talk(message)


@pytest.mark.parametrize(
    "message",
    ["איפה גדלת?", "מה אתה עושה בשבת בבוקר עם הנכדים?", "היי, איך הכרת את אמא?", "מה עשית בצבא?"],
)
def test_real_questions_are_not_small_talk(message: str) -> None:
    assert not is_small_talk(message)


def test_subject_turns_the_question_into_one_about_them(monkeypatch: pytest.MonkeyPatch) -> None:
    save_memory(PERSON_ID, "דוד גדל ברומניה", source="manual")
    seen: list[str] = []

    def fake_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
        seen.append(query)
        return [{**c, "rerank_score": 0.9} for c in candidates][:top_k]

    monkeypatch.setattr(retrieval, "rerank", fake_rerank)

    assert retrieve_context(PERSON_ID, "איפה גדלת?", subject="דוד כהן")
    assert seen == ["איפה גדלת? (על דוד כהן)"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("איפה הכרת את אמא?", "איפה הכרת את אמא? רחל"),
        ("איפה הכרת את רחל?", "איפה הכרת את רחל?"),
        ("מה נשמע?", "מה נשמע?"),
        ("מה אתה חושב על האוטו שלך?", "מה אתה חושב על האוטו שלך? רכב"),
    ],
)
def test_expand_query(query: str, expected: str) -> None:
    assert expand_query(query, {"אמא": "רחל"}) == expected


def test_question_only_memories_and_duplicates_are_filtered() -> None:
    save_memory(PERSON_ID, "היי איפה דוד הכיר את רחל", source="chat_conversation")
    save_memory(PERSON_ID, "דוד הכיר את רחל כשהיו שכנים בבת ים", source="manual")

    results = retrieve_context(PERSON_ID, "איפה דוד הכיר את רחל", top_k=5)
    texts = [r["text"] for r in results]

    assert len(texts) == 1
    assert "שכנים בבת ים" in texts[0]


def test_old_chat_turns_are_never_retrieved() -> None:
    save_memory(PERSON_ID, "הכנתי היום קומפוט עם רימון", source="chat_conversation")

    assert retrieve_context(PERSON_ID, "הכנתי היום קומפוט עם רימון") == []


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.9, 0.01], ["דוד גדל ברומניה"]),
        ([0.01, 0.01], []),
        ([0.15, 0.149], ["דוד גדל ברומניה"]),
    ],
)
def test_low_relevance_passages_are_dropped(
    monkeypatch: pytest.MonkeyPatch, scores: list[float], expected: list[str]
) -> None:
    save_memory(PERSON_ID, "דוד גדל ברומניה", source="manual")
    save_memory(PERSON_ID, "הגינה בבית ברעות", source="manual")

    def fake_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
        ordered = sorted(candidates, key=lambda c: "רומניה" not in c["text"])
        return [{**c, "rerank_score": s} for c, s in zip(ordered, scores)][:top_k]

    monkeypatch.setattr(settings, "rerank_min_score", 0.15)
    monkeypatch.setattr(retrieval, "rerank", fake_rerank)

    texts = [r["text"].split("]: ", 1)[1] for r in retrieve_context(PERSON_ID, "איפה גדלת", top_k=2)]
    assert texts == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("מה קורה עכשיו בחדשות ובפוליטיקה בישראל?", True),
        ("מה המצב מול איראן וארה\"ב?", True),
        ("מה הריבית והאינפלציה עכשיו?", True),
        ("איפה הכרת את אמא?", False),
        ("היוש", False),
    ],
)
def test_needs_web_grounding(query: str, expected: bool) -> None:
    assert needs_web_grounding(query) is expected


def test_web_grounding_is_blocked_when_external_services_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "allow_external_services", False)
    assert fetch_web_context("מה קורה עכשיו בחדשות ובפוליטיקה בישראל?") == []
    assert format_web_context([]) == ""
