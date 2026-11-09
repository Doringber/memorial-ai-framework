"""Cross-encoder reranker — re-scores candidate chunks for precision.

Vector search (bi-encoder) is fast but coarse. A cross-encoder reads
query+chunk together and produces a much sharper relevance score.
Use it to re-rank a small pool (e.g. top 30) down to top_k.
"""

from __future__ import annotations

from loguru import logger

from app.core.config import settings

_model = None
_load_failed = False


def _get_model():
    """Lazy-load the cross-encoder. If unavailable, rerank silently disables."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model

    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading reranker model: {}", settings.rerank_model)
        _model = CrossEncoder(settings.rerank_model, max_length=512)
        logger.info("Reranker loaded")
    except Exception as e:
        _load_failed = True
        logger.warning("Reranker unavailable ({}). Falling back to vector distance.", e)
        _model = None
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates with cross-encoder. Returns top_k sorted by score.

    Each candidate dict must have a 'text' field. Adds 'rerank_score' to each
    returned dict. If the model fails to load, returns candidates[:top_k]
    unchanged so the pipeline keeps working.
    """
    if not settings.rerank_enabled or not candidates:
        return candidates[:top_k]

    model = _get_model()
    if model is None:
        return candidates[:top_k]

    pairs = [(query, c.get("text", "")[:1200]) for c in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]
