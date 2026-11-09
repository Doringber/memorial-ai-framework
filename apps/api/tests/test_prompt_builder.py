from app.services.prompt_builder import (
    HISTORY_MESSAGES,
    NO_CONTEXT,
    build_context_prompt,
    build_dynamic_prompt,
    build_messages,
    build_persona_prompt,
)
from app.services.style_bank import StyleExample


def test_persona_prompt_renders_curated_sections(persona: dict) -> None:
    prompt = build_persona_prompt(persona)

    assert prompt.startswith("אתה דוד כהן.")
    for expected in ("תשובה קצרה.", "רחל", "יונתן — הבכור", "גל, תמר", "BSc אלקטדודקה",
                     '"הנשר נחת...."', 'הם: "חזרנו מהטיול"', "סבבה", "טיפולים בשיבא",
                     "אל תמציא שמות", "לא ממציא"):
        assert expected in prompt
    assert "internal" not in prompt


def test_persona_prompt_handles_generated_shape() -> None:
    generated = {
        "name": "Unknown",
        "writing_style": {"common_phrases": ["נו באמת"], "humor_style": "dry"},
        "communication_patterns": {"advice_style": "direct"},
    }
    prompt = build_persona_prompt(generated)
    assert "אתה Unknown." in prompt
    assert '"נו באמת"' in prompt
    assert "dry" in prompt


def test_persona_prompt_without_persona_keeps_honesty_rules() -> None:
    prompt = build_persona_prompt(None)
    assert "אין עדיין פרופיל" in prompt
    assert "לא ממציא" in prompt
    assert "ענה רק על מה ששאלו" in prompt


def test_context_prompt_empty_and_filled() -> None:
    assert build_context_prompt([]) == NO_CONTEXT
    prompt = build_context_prompt([{"text": "קטע", "metadata": {"document_title": "chat"}}])
    assert "[1] (מקור: chat)" in prompt


def test_dynamic_prompt_includes_facts_style_and_context() -> None:
    style = [StyleExample("ולקנוח...."), StyleExample("יפה מאוד", "[תמונה]")]
    prompt = build_dynamic_prompt(["נועה גרה ברעות"], style, [])
    assert "- נועה גרה ברעות" in prompt
    assert '- "ולקנוח...."' in prompt
    assert '- כתבו לך: "[תמונה]" — ענית: "יפה מאוד"' in prompt
    assert NO_CONTEXT in prompt


def test_build_messages_order_and_history_cap() -> None:
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(30)]
    msgs = build_messages("persona", "dynamic", history, "שאלה")

    assert msgs[0] == {"role": "system", "content": "persona"}
    assert len(msgs) == 2 + HISTORY_MESSAGES + 1
    assert msgs[1]["content"] == str(30 - HISTORY_MESSAGES)
    assert msgs[-2] == {"role": "system", "content": "dynamic"}
    assert msgs[-1] == {"role": "user", "content": "שאלה"}
