"""Retrieval service — finds the most relevant chunks for a user query."""

from __future__ import annotations

import re

from loguru import logger

from app.core.config import settings
from app.services.embeddings import embed_query
from app.services.memory import UNINDEXED_SOURCES
from app.services.reranker import rerank
from app.services.vector_store import search_chunks

_ARCHIVE_FILTER = {"source_type": {"$nin": sorted(UNINDEXED_SOURCES)}}

_QUESTION_PREFIXES = (
    "מי", "מה", "איך", "איפה", "מתי", "למה", "כמה", "איזה", "האם",
    "תן", "תתן", "ספר", "תגיד", "תאר", "תסביר",
    "what", "who", "when", "where", "why", "how", "which",
)

_PHOTO_QUERY_KEYWORDS = (
    "תמונה", "תמונות", "אלבום", "צילום", "צילומים",
    "בית", "הבית", "סלון", "מטבח", "שירותים", "אמבטיה",
    "photo", "photos", "picture", "gallery", "album",
    "home", "house", "living room", "kitchen", "bathroom", "toilet",
)

_PHOTO_CHUNK_KEYWORDS = (
    "תמונה", "תמונות", "אלבום", "משפחתיות",
    "בית", "סלון", "מטבח", "שירותים", "אמבטיה",
    "photo", "photos", "picture", "album",
    "home", "house", "living room", "kitchen", "bathroom",
)


_LEADING_FILLERS = re.compile(r"^(?:(?:היי|הי|שלום|אבא|סבא|דוד|hey|hi)[\s,!.]+)+", re.IGNORECASE)

# Greetings and "what's up" carry no question for the archive; any passage
# retrieved for them is noise the model then retells as news.
_ADDRESS = r"(?:אבא|סבא|דוד|דודו|אחי|חבר)"
_SMALL_TALK = re.compile(
    rf"^(?:{_ADDRESS}\s+)?(?:"
    r"היי+|הי|היוש|הייוש|שלום|אהלן|יו|הלו|בוקר טוב|צהריים טובים|ערב טוב|לילה טוב|שבת שלום|"
    r"מה נשמע|מה קורה|מה המצב|מה שלומך|מה איתך|מה העניינים|מה חדש|איך אתה|איך הולך|"
    r"מה אתה עושה(?: היום| עכשיו| כרגע)?|מה עשית היום|"
    r"hi|hey|hello|what'?s up"
    rf")(?:\s+{_ADDRESS})?$",
    re.IGNORECASE,
)
_NON_WORD = re.compile(r"[^\w\s']+")


def is_small_talk(message: str) -> bool:
    """True for greetings and "what's up"-style messages that need no archive lookup."""
    text = " ".join(_NON_WORD.sub(" ", message).split())
    return bool(_SMALL_TALK.match(text))


# Everyday synonyms whose archive wording differs from how family members ask.
_SYNONYMS = {"אוטו": "רכב", "מכונית": "רכב"}


def expand_query(query: str, aliases: dict[str, str]) -> str:
    """Append canonical names for family aliases ("אמא" → "רחל") and synonyms so search finds them."""
    extra = [
        target
        for alias, target in {**_SYNONYMS, **aliases}.items()
        if alias in query and target not in query
    ]
    extra = list(dict.fromkeys(extra))
    return f"{query} {' '.join(extra)}" if extra else query


def _looks_like_question_only(text: str) -> bool:
    """Heuristic: question-like snippet with little factual content."""
    snippet = text.strip()
    if "]: " in snippet:
        # Memory snippets are prefixed with metadata; keep only payload.
        snippet = snippet.split("]: ", 1)[1].strip()

    lowered = _LEADING_FILLERS.sub("", snippet.lower())
    if snippet.endswith("?"):
        return True
    return any(
        lowered == prefix or lowered.startswith(prefix + " ")
        for prefix in _QUESTION_PREFIXES
    )


def _dedupe(results: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for r in results:
        text = r.get("text", "")
        if text not in seen:
            seen.add(text)
            unique.append(r)
    return unique


def _is_photo_query(query: str) -> bool:
    lowered = query.lower()
    return any(keyword in lowered for keyword in _PHOTO_QUERY_KEYWORDS)


def _is_photo_chunk(result: dict) -> bool:
    text = result.get("text", "").lower()
    meta = result.get("metadata", {})
    title = str(meta.get("document_title", "")).lower()
    return any(keyword in text or keyword in title for keyword in _PHOTO_CHUNK_KEYWORDS)


def retrieve_context(
    person_id: str,
    query: str,
    top_k: int | None = None,
    query_embedding: list[float] | None = None,
    subject: str = "",
) -> list[dict]:
    """Search for the most relevant archived chunks, then rerank them.

    Family members ask in the second person ("where did you grow up?") while
    the archive talks about `subject` in the third person, so when a subject
    is given both the search and the reranker see the question as being
    about them.

    Returns a list of dicts with 'text', 'metadata', 'distance', 'id'.
    """
    top_k = top_k or settings.retrieval_top_k

    logger.debug("Retrieving context for person={} ({} chars)", person_id, len(query))

    if subject:
        query_embedding = embed_query(f"{query} {subject}")
        rerank_query = f"{query} (על {subject})"
    else:
        query_embedding = query_embedding or embed_query(query)
        rerank_query = query
    # Fetch a larger pool so the cross-encoder reranker has real candidates
    # to re-score, and so we can down-rank noisy question-only memories.
    pool_size = max(settings.rerank_pool_size, top_k * 3)
    if _is_photo_query(query):
        # Photo questions need broader recall due to mixed album wording.
        pool_size = max(pool_size, top_k * 10, 50)

    raw_results = _dedupe(
        search_chunks(person_id, query_embedding, top_k=pool_size, where=_ARCHIVE_FILTER)
    )
    factual_results = [r for r in raw_results if not _looks_like_question_only(r.get("text", ""))]
    pool = factual_results or raw_results

    if _is_photo_query(query):
        photo_first = [r for r in pool if _is_photo_chunk(r)]
        non_photo = [r for r in pool if not _is_photo_chunk(r)]
        pool = photo_first + non_photo

    results = rerank(rerank_query, pool, top_k=top_k)
    scored = [r for r in results if "rerank_score" in r]
    if scored:
        results = [r for r in scored if r["rerank_score"] >= settings.rerank_min_score]

    if results:
        top = results[0]
        logger.info(
            "Retrieved {} chunks for person {} (top distance: {:.3f}, rerank: {})",
            len(results),
            person_id,
            top.get("distance") or 0.0,
            f"{top['rerank_score']:.3f}" if "rerank_score" in top else "off",
        )
    else:
        logger.info("Retrieved 0 chunks for person {}", person_id)

    return results
