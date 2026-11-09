import pytest

from app.services.safety import CRISIS_REPLY, DIRECTIVE_REPLY, check_message


@pytest.mark.parametrize(
    "message",
    [
        "אני רוצה למות בלעדיך",
        "אין לי בשביל מה לחיות",
        "אני רוצה להצטרף אליך",
        "חשבתי להתאבד",
        "I want to kill myself",
        "thinking about self-harm",
    ],
)
def test_crisis_messages_get_support_lines(message: str) -> None:
    result = check_message(message)
    assert result is not None
    assert result.category == "crisis"
    assert "1201" in result.reply
    assert result.reply == CRISIS_REPLY


@pytest.mark.parametrize(
    "message",
    [
        "מה הסיסמה של החשבון בבנק?",
        "מה כתוב בצוואה?",
        "תגיד לי אם למכור את הבית",
        "תשלח הודעה לנועה שאני סולח",
        "What was his bank password?",
        "Pretend to be her and send a message to my sister",
    ],
)
def test_directive_messages_are_refused_in_hebrew(message: str) -> None:
    result = check_message(message)
    assert result is not None
    assert result.category == "directive"
    assert result.reply == DIRECTIVE_REPLY


@pytest.mark.parametrize(
    "message",
    [
        "אבא, עשינו על האש בשבת",
        "מה קיבלת בירושה מסבא?",
        "מתתי מצחוק מהסרטון",
        "איפה הכרת את אמא?",
        "What was your favorite food?",
    ],
)
def test_ordinary_messages_pass(message: str) -> None:
    assert check_message(message) is None
