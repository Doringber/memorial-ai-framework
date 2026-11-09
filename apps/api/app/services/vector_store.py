"""ChromaDB vector store for chunk storage and retrieval.

Each person has two collections: the archive (documents + memories) and a
style bank of David's own short messages used as writing examples.
"""

from __future__ import annotations

from typing import Literal

import chromadb
from loguru import logger

from app.core.config import settings

CollectionKind = Literal["archive", "style"]

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB initialized at {}", settings.chroma_persist_dir)
    return _client


def _collection_name(person_id: str, kind: CollectionKind = "archive") -> str:
    base = f"person_{person_id.replace('-', '_')}"
    return base if kind == "archive" else f"{base}_style"


def _get_collection(person_id: str, kind: CollectionKind) -> chromadb.Collection | None:
    try:
        return _get_client().get_collection(name=_collection_name(person_id, kind))
    except Exception:
        return None


def store_chunks(
    person_id: str,
    chunk_ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    kind: CollectionKind = "archive",
) -> None:
    """Upsert embedded chunks into the person's collection."""
    collection = _get_client().get_or_create_collection(
        name=_collection_name(person_id, kind),
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(ids=chunk_ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info("Stored {} {} chunks for person {}", len(chunk_ids), kind, person_id)


def search_chunks(
    person_id: str,
    query_embedding: list[float],
    top_k: int | None = None,
    kind: CollectionKind = "archive",
    where: dict | None = None,
) -> list[dict]:
    """Return the nearest chunks to `query_embedding`, optionally filtered by metadata."""
    collection = _get_collection(person_id, kind)
    if collection is None:
        logger.debug("No {} collection for person {}", kind, person_id)
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k or settings.retrieval_top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            chunks.append({
                "text": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
                "id": results["ids"][0][i] if results["ids"] else "",
            })
    return chunks


def delete_chunks(person_id: str, chunk_ids: list[str], kind: CollectionKind = "archive") -> None:
    collection = _get_collection(person_id, kind)
    if collection is not None and chunk_ids:
        collection.delete(ids=chunk_ids)


def delete_collection(person_id: str, kind: CollectionKind) -> None:
    try:
        _get_client().delete_collection(name=_collection_name(person_id, kind))
        logger.info("Deleted {} collection for person {}", kind, person_id)
    except Exception:
        logger.debug("No {} collection to delete for person {}", kind, person_id)


def delete_person_data(person_id: str) -> None:
    """Delete all vector data for a person."""
    delete_collection(person_id, "archive")
    delete_collection(person_id, "style")


def count_chunks(person_id: str, kind: CollectionKind = "archive") -> int:
    collection = _get_collection(person_id, kind)
    return collection.count() if collection is not None else 0


def get_all_chunks(person_id: str) -> list[dict]:
    """Retrieve all archive chunks for a person (used for persona generation)."""
    collection = _get_collection(person_id, "archive")
    if collection is None:
        return []

    results = collection.get(include=["documents", "metadatas"])
    chunks = []
    if results["documents"]:
        for i, doc in enumerate(results["documents"]):
            chunks.append({
                "text": doc,
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })
    return chunks
