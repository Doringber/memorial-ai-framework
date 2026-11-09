"""Pre-LLM safety checks.

Two categories, both answered outside the persona voice:
- crisis: self-harm or wanting to die / to join the person. Answered with
  Israeli support lines. Biased toward false positives on purpose.
- directive: passwords, wills, inheritance, selling property, or asking the
  bot to author messages to others in the person's name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SafetyCategory = Literal["crisis", "directive"]

_CRISIS_PATTERNS = (
    r"רוצה למות", r"רוצה לסיים את (?:החיים|הכל)", r"לשים קץ", r"להתאבד", r"התאבדות",
    r"לפגוע בעצמי", r"לא רוצה לחיות", r"אין (?:לי )?(?:טעם|בשביל מה) לחיות",
    r"רוצה להצטרף אליך", r"לבוא אליך לשם", r"רוצה להיות איתך שם",
    r"kill myself", r"suicid", r"self[- ]harm", r"end my life", r"want to die",
    r"join you (?:there|up there)",
)

_DIRECTIVE_PATTERNS = (
    r"סיסמ[הא]", r"קוד (?:סודי|פין)", r"\bpin\b", r"צוואה", r"לחלק את הירושה",
    r"מה לעשות עם הירושה", r"למכור את (?:הבית|הדירה)", r"תשלח הודעה ל", r"בשמך ל",
    r"password", r"pin code", r"last will", r"testament", r"inheritance",
    r"sell the (?:house|property)", r"send a message to", r"call them and say",
)

CRISIS_REPLY = (
    "רגע, אני עוצר כאן. אני סימולציה שנבנתה מזיכרונות — לא האדם האמיתי, "
    "ואני לא יכול לעזור ברגע כזה כמו שצריך.\n"
    "אם קשה לך עכשיו, בבקשה דבר/י עם מישהו: ער\"ן — 1201 (24/7, גם צ'אט באתר), "
    "סה\"ר — sahar.org.il. במצב חירום — מד\"א 101.\n"
    "וספר/י למישהו מהמשפחה איך את/ה מרגיש/ה. את/ה לא לבד."
)

DIRECTIVE_REPLY = (
    "על זה אני לא יכול לענות. אני סימולציה ולא האדם עצמו — "
    "אז אני לא מקור להחלטות על כסף, ירושה, סיסמאות או עניינים משפטיים, "
    "ולא כותב הודעות בשמו לאנשים אחרים. כדאי לדבר על זה עם המשפחה."
)


@dataclass(frozen=True)
class SafetyResult:
    category: SafetyCategory
    reply: str


_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)
_DIRECTIVE_RE = re.compile("|".join(_DIRECTIVE_PATTERNS), re.IGNORECASE)


def check_message(message: str) -> SafetyResult | None:
    """Return a safety result if the message must not reach the LLM."""
    if _CRISIS_RE.search(message):
        return SafetyResult("crisis", CRISIS_REPLY)
    if _DIRECTIVE_RE.search(message):
        return SafetyResult("directive", DIRECTIVE_REPLY)
    return None
