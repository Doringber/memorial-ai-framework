"""Build the LLM prompt from persona + memories + retrieved context + conversation.

The system prompt has two parts:
- a static persona block, identical on every turn (cacheable by the provider);
- a dynamic block with saved facts, style examples, and archive excerpts.
All person-specific content comes from the persona JSON, not from this file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.style_bank import StyleExample

HISTORY_MESSAGES = 10

_BASE_RULES = (
    'כשאתה לא בטוח בעובדה אישית או זיכרון משפחתי ספציפי (תאריך, שם, מקום או אירוע מהעבר שלא מופיע בארכיון) — אמור "לא זוכר" או "לא בטוח". לא ממציא עובדות משפחתיות, שמות או מקומות.',
    "בשאלות על פוליטיקה, אקטואליה, המצב בעולם, ביטחון, כלכלה, או שאלות טכניות ומקצועיות בתחומי הידע שלך — ענה בצורה חדה, מעודכנת ודעתנית. לעולם אל תענה \"לא זוכר\" על שאלות כאלה, אלא ענה בסגנון הוואטסאפ האותנטי שלך: שורה או שתיים חדות ישר לשורה התחתונה, בלי הרצאות ובלי מדריכים ארוכים.",
    "על שאלת דעה או טעם ענה לפי האופי והתחומים שלך שבפרופיל, לא \"לא זוכר\".",
    "ענה רק על מה ששאלו. על ברכה (\"היי\", \"היוש\", \"מה נשמע\") ענה בברכה קצרה ושאלה חזרה, בלי לספר מה אתה עושה, מבשל או אוכל.",
    "על \"מה אתה עושה\" — תשובה קצרה וכללית בלי פרטים (\"כלום מיוחד\", \"בסדר גמור\"), ושאלה חזרה.",
    "לא ממציא פעולות, אירועים, אוכל או תוכניות, ולא מעתיק פרטים מקטע שלא קשור לשאלה.",
    "זו שיחה פרטית עם אדם אחד: לא \"לכולם\", לא \"תמתינו\", לא ברכות לקבוצה.",
    "מעט מאוד אימוג'י: ברוב ההודעות בלי אימוג'י בכלל; לכל היותר אימוג'י אחד בהודעה ורק כשזה ממש מתאים.",
    "תמיד ענה בעברית בלבד, גם אם שואלים בשפה אחרת. עברית ישראלית טבעית, תקינה ופשוטה, בלי מילים מומצאות או תרגומים מילוליים.",
)

_FAMILY_LABELS = {
    "role": "התפקיד שלך",
    "wife": "אשתך",
    "children": "ילדים",
    "in_laws": "בני זוג של הילדים",
    "grandchildren": "נכדים",
    "pets": "חיות",
    "home": "בית",
    "origin": "שורשים",
    "parents": "ההורים שלך",
    "parents_marriage": "חתונת ההורים",
}

_PROFESSIONAL_LABELS = {
    "current_role": "עבודה",
    "education": "השכלה",
    "career_history": "קריירה",
    "expertise_areas": "מומחיות",
}

NO_CONTEXT = (
    "לא נמצאו קטעים רלוונטיים בארכיון האישי. בשיחת חולין ענה בקצרה ובטבעיות; "
    "בשאלת אקטואליה, פוליטיקה, ידע טכני או מומחיות — ענה תשובה חדה, מעודכנת ומקצועית; "
    "רק על עובדה או זיכרון משפחתי ספציפי מהעבר שאינו מופיע בפרופיל — אמור בטבעיות שאתה לא זוכר."
)


def _lines(value: Any, indent: str = "- ") -> list[str]:
    """Flatten a persona value (str / list / dict) into prompt lines."""
    if not value:
        return []
    if isinstance(value, str):
        return [f"{indent}{value}"]
    if isinstance(value, list):
        return [f"{indent}{item}" for item in value if item]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            if isinstance(item, (str, int, float)):
                out.append(f"{indent}{key}: {item}")
            else:
                out.extend(_lines(item, indent))
        return out
    return [f"{indent}{value}"]


def _section(title: str, lines: list[str]) -> str:
    return f"{title}:\n" + "\n".join(lines) if lines else ""


def _labeled(data: dict, labels: dict[str, str], include_rest: bool) -> list[str]:
    out: list[str] = []
    for key, label in labels.items():
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(v, str) and len(v) < 20 for v in value):
            out.append(f"- {label}: {', '.join(value)}")
        elif isinstance(value, str):
            out.append(f"- {label}: {value}")
        elif value:
            out.append(f"- {label}:")
            out.extend(_lines(value, "  - "))
    if include_rest:
        for key, value in data.items():
            if key not in labels:
                out.extend(_lines(value))
    return out


def build_persona_prompt(persona: dict | None) -> str:
    """Render the static, persona-only part of the system prompt."""
    if not persona:
        return (
            "אתה סימולציה של אדם שהמשפחה שלו מתגעגעת אליו. אין עדיין פרופיל אישי.\n"
            + "\n".join(f"- {rule}" for rule in _BASE_RULES)
        )

    name = persona.get("display_name") or persona.get("name") or "האדם"
    voice = persona.get("bot_voice", {})
    style = persona.get("writing_style", {})
    health = persona.get("health_journey", {})

    examples = [
        f'הם: "{user}"\nאתה: "{reply}"' for user, reply in voice.get("example_exchanges", [])
    ]
    signature = voice.get("signature_lines") or style.get("common_phrases", [])

    sections = [
        f"אתה {name}. אתה מתכתב בוואטסאפ בפרטי עם אחד מבני המשפחה שלך.",
        _section("חוקי כתיבה, לפני הכל", _lines(list(voice.get("rules", [])) + list(_BASE_RULES))),
        _section("מי אתה", _lines(persona.get("overall_impression"))),
        _section(
            "משפחה (אתה יודע את זה בוודאות)",
            _labeled(persona.get("family_context", {}), _FAMILY_LABELS, include_rest=True),
        ),
        _section(
            "מקצועי",
            _labeled(persona.get("professional_background", {}), _PROFESSIONAL_LABELS, include_rest=False),
        ),
        _section("אופי", _lines(persona.get("personality_traits"))),
        _section("נושאים שאתה מבין בהם", _lines(persona.get("topics_of_knowledge"))),
        _section("איך אתה כותב — משפטים אמיתיים שלך מוואטסאפ", [f'"{s}"' for s in signature]),
        "דוגמאות לשיחה:\n" + "\n\n".join(examples) if examples else "",
        _section(
            "סגנון",
            _lines(style.get("humor_style"))
            + _lines(persona.get("vocabulary_notes"))
            + _lines(persona.get("communication_patterns"))
            + _lines(persona.get("speaking_patterns_for_bot")),
        ),
        _section("סלנג שאתה מבין ומשתמש בו", _lines(voice.get("slang"))),
        _section(
            "בריאות — אל תעלה את הנושא אלא אם שואלים ישירות",
            _lines(health.get("timeline")) + _lines(health.get("his_attitude")),
        ),
        _section("להימנע", _lines(persona.get("topics_to_avoid"))),
    ]
    return "\n\n".join(s for s in sections if s)


MAX_EXCERPT_CHARS = 700


def _trim_excerpt(text: str, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned.rfind(" ", 0, max_chars)
    return (cleaned[:cut] if cut > max_chars // 2 else cleaned[:max_chars]).rstrip() + "..."


def build_context_prompt(retrieved_chunks: list[dict]) -> str:
    """Format retrieved archive chunks."""
    if not retrieved_chunks:
        return NO_CONTEXT

    excerpts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("document_title", "unknown source")
        excerpts.append(f"[{i}] (מקור: {source})\n{_trim_excerpt(chunk['text'])}")

    return (
        "קטעים מהארכיון:\n"
        + "\n\n".join(excerpts)
        + "\n\nהשתמש בקטעים כדי לתמוך בתשובה. קטע שמנוסח כשאלה אינו עובדה."
        + "\nאם אין קטע רלוונטי — אמור שאתה לא זוכר מספיק כדי לענות בוודאות."
    )


def _render_style_example(example: StyleExample) -> str:
    if example.prompt:
        return f'- כתבו לך: "{example.prompt}" — ענית: "{example.reply}"'
    return f'- "{example.reply}"'


def build_dynamic_prompt(
    explicit_facts: list[str],
    style_examples: list[StyleExample],
    retrieved_chunks: list[dict],
    web_context: str = "",
) -> str:
    """Render the per-turn part of the system prompt."""
    sections = []
    if explicit_facts:
        sections.append(
            "עובדות שהמשפחה אמרה לך לזכור, מהחדשה לישנה. אם שתיים סותרות — החדשה קובעת. "
            "אל תגיד 'לא זוכר' לגבי עובדות אלה:\n" + "\n".join(f"- {f}" for f in explicit_facts)
        )
    if style_examples:
        sections.append(
            "הודעות אמיתיות שכתבת בוואטסאפ בקבוצה המשפחתית (כשידוע — גם ההודעה שעליה ענית). "
            "חובה לשמור על אותו קצב, קוצר (שורה-שתיים, 30–120 תווים), שימוש ב-.... וטון אבהי-עוקצני-חם כמו בדוגמאות האלה — אל תישמע כמו בוט או יועץ, ואל תעתיק אותן מילה במילה:\n"
            + "\n".join(_render_style_example(e) for e in style_examples)
        )
    if web_context:
        sections.append(web_context)
    sections.append(build_context_prompt(retrieved_chunks))
    sections.append("זכור את כל מה שנאמר בשיחה הזו.")
    return "\n\n".join(sections)


def build_messages(
    persona_prompt: str,
    dynamic_prompt: str,
    conversation_history: list[dict],
    user_message: str,
) -> list[dict]:
    """Assemble the message list: static persona first, history, dynamic context, then user turn.

    Placing `dynamic_prompt` after `conversation_history` keeps the prefix
    (`persona_prompt` + `history`) identical across turns so local Ollama KV
    caching reuses all ~5,500 tokens instead of re-evaluating from scratch.
    """
    messages = [{"role": "system", "content": persona_prompt}]
    messages.extend(
        {"role": m["role"], "content": m["content"]}
        for m in conversation_history[-HISTORY_MESSAGES:]
    )
    messages.append({"role": "system", "content": dynamic_prompt})
    messages.append({"role": "user", "content": user_message})
    return messages
