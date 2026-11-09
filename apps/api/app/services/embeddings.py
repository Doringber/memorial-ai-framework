"""Embedding provider abstraction.

Supports:
- "local" — sentence-transformers (free, runs on CPU, no API key needed)
- "openai" — OpenAI embeddings API

E5-family models (intfloat/*-e5-*) require instruction prefixes:
  "query: ..."   for search queries
  "passage: ..." for stored documents
Without the prefix, recall drops sharply. We apply them automatically
when the configured model name contains "e5".
"""

from __future__ import annotations

from loguru import logger

from app.core.config import settings

_local_model = None
_openai_client = None


def _uses_e5_prefix() -> bool:
    return "e5" in settings.embedding_model.lower()


def _apply_prefix(texts: list[str], kind: str) -> list[str]:
    if not _uses_e5_prefix():
        return texts
    prefix = f"{kind}: "
    return [prefix + t for t in texts]


def embed_texts(texts: list[str], kind: str = "passage") -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    kind: "passage" for stored documents, "query" for search queries.
    Only affects E5 models; ignored otherwise.
    """
    if not texts:
        return []

    prepared = _apply_prefix(texts, kind)

    if settings.embedding_provider == "local":
        return _embed_local(prepared)
    else:
        return _embed_openai(prepared)


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text], kind="query")[0]


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed stored documents / chunks."""
    return embed_texts(texts, kind="passage")


def _embed_local(texts: list[str]) -> list[list[float]]:
    """Use sentence-transformers for free local embeddings."""
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: {}", settings.embedding_model)
        _local_model = SentenceTransformer(settings.embedding_model)
        logger.info("Local embedding model loaded")

    logger.debug("Embedding {} texts locally", len(texts))
    embeddings = _local_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return [e.tolist() for e in embeddings]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Use OpenAI embeddings API."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Either set it or switch to "
                "EMBEDDING_PROVIDER=local for free local embeddings."
            )
        _openai_client = OpenAI(api_key=settings.openai_api_key)

    logger.debug("Embedding {} texts with OpenAI {}", len(texts), settings.embedding_model)

    response = _openai_client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    return [item.embedding for item in response.data]
