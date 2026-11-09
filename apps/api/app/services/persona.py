"""Persona storage and draft generation.

`<person_id>.json` is the curated persona the bot speaks from. It is edited by
hand and never overwritten by the app. `generate_persona_draft` writes an
LLM-generated profile to `<person_id>.generated.json` for review, so a rebuild
can't wipe the curated file.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.services.llm import generate_response
from app.services.vector_store import get_all_chunks

_cache: dict[str, tuple[float, dict]] = {}

PERSONA_GENERATION_PROMPT = """You are analyzing archived writings, messages, and transcripts of a specific person.
Your goal is to build a detailed style and personality profile that can guide an AI to write in a style INSPIRED by this person.

Analyze the following text samples and produce a JSON persona profile.

IMPORTANT:
- Base your analysis ONLY on evidence in the provided texts
- If evidence is insufficient for a field, say "insufficient data"
- Do NOT invent characteristics not present in the texts
- Focus on communication style, not biographical facts

Produce a JSON object with this structure:
{
  "name": "inferred or 'Unknown'",
  "writing_style": {
    "tone": ["list of tone descriptors, e.g. warm, direct, humorous"],
    "common_phrases": ["actual phrases or expressions the person uses repeatedly"],
    "humor_style": "description of humor if present",
    "sentence_length": "short/medium/long/varied",
    "formality": "casual/formal/mixed",
    "values": ["values expressed in the texts"]
  },
  "communication_patterns": {
    "greeting_style": "how they typically start messages",
    "closing_style": "how they typically end messages",
    "emotional_expression": "how they express emotions",
    "advice_style": "how they give advice or opinions"
  },
  "topics_of_knowledge": ["subjects they discuss with authority"],
  "vocabulary_notes": "notable vocabulary patterns, language preferences, bilingual patterns",
  "topics_to_avoid": ["sensitive topics to handle carefully"],
  "overall_impression": "2-3 sentence summary of this person's communication character"
}

TEXT SAMPLES:
"""


def persona_path(person_id: str) -> Path:
    return settings.personas_dir / f"{person_id}.json"


def draft_path(person_id: str) -> Path:
    return settings.personas_dir / f"{person_id}.generated.json"


def load_persona(person_id: str) -> dict | None:
    """Load the curated persona, re-reading the file whenever it changes on disk."""
    path = persona_path(person_id)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _cache.get(person_id)
    if cached and cached[0] == mtime:
        return cached[1]
    persona = json.loads(path.read_text(encoding="utf-8"))
    _cache[person_id] = (mtime, persona)
    return persona


def generate_persona_draft(person_id: str) -> dict:
    """Generate a style profile from ingested chunks and save it as a draft."""
    chunks = get_all_chunks(person_id)
    if not chunks:
        logger.warning("No chunks found for person {}, cannot generate persona", person_id)
        return {"error": "No materials found. Please ingest documents first."}

    # Up to 60 chunks / ~12k words keeps the request inside the context window.
    words = "\n\n---\n\n".join(c["text"] for c in chunks[:60]).split()
    combined = " ".join(words[:12000])
    logger.info("Generating persona draft from {} words", min(len(words), 12000))

    text = generate_response([
        {"role": "system", "content": "You are a writing style analyst. Respond ONLY with valid JSON."},
        {"role": "user", "content": PERSONA_GENERATION_PROMPT + combined},
    ])

    start, end = text.find("{"), text.rfind("}") + 1
    draft = json.loads(text[start:end] if start >= 0 and end > start else text)

    settings.personas_dir.mkdir(parents=True, exist_ok=True)
    draft_path(person_id).write_text(
        json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Persona draft saved to {}", draft_path(person_id).name)
    return draft


def activate_persona_draft(person_id: str, display_name: str = "") -> dict:
    """Promote the generated draft (or generate one now) to the active `<person_id>.json`."""
    dpath = draft_path(person_id)
    if dpath.is_file():
        draft = json.loads(dpath.read_text(encoding="utf-8"))
    else:
        draft = generate_persona_draft(person_id)
        if "error" in draft:
            return draft

    if display_name and not draft.get("display_name"):
        draft["display_name"] = display_name
    settings.personas_dir.mkdir(parents=True, exist_ok=True)
    persona_path(person_id).write_text(
        json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _cache.pop(person_id, None)
    return draft
