"""Orchestrates the full ingestion pipeline: extract → chunk → embed → store."""

from __future__ import annotations

import uuid
from pathlib import Path

from loguru import logger

from app.services.text_extractor import extract_text
from app.services.chunker import chunk_text
from app.services.embeddings import embed_passages
from app.services.vector_store import store_chunks


def ingest_document(
    person_id: str,
    file_path: str,
    title: str = "",
    source_type: str = "",
    document_id: str | None = None,
) -> dict:
    """Full pipeline: file → text → chunks → embeddings → vector store.

    Pass a stable `document_id` to make re-ingesting the same file an upsert.
    Returns a summary of what was ingested.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not title:
        title = path.name

    # Step 1: Extract text
    logger.info("Extracting text from {}", file_path)
    raw_text = extract_text(file_path, source_type)

    if not raw_text.strip():
        logger.warning("No text extracted from {}", file_path)
        return {"status": "empty", "chunks": 0, "title": title}

    # Step 2: Chunk
    logger.info("Chunking text ({} chars)", len(raw_text))
    doc_id = document_id or str(uuid.uuid4())
    chunks = chunk_text(
        raw_text,
        metadata={
            "person_id": person_id,
            "document_id": doc_id,
            "document_title": title,
            "source_type": source_type or path.suffix.lstrip("."),
        },
    )

    if not chunks:
        return {"status": "empty", "chunks": 0, "title": title}

    # Step 3: Embed
    logger.info("Embedding {} chunks", len(chunks))
    texts = [c.text for c in chunks]
    embeddings = embed_passages(texts)

    # Step 4: Store in vector DB
    chunk_ids = [f"{doc_id}_chunk_{c.index}" for c in chunks]
    metadatas = [c.metadata for c in chunks]

    store_chunks(
        person_id=person_id,
        chunk_ids=chunk_ids,
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info("Ingested '{}': {} chunks stored", title, len(chunks))

    return {
        "status": "ok",
        "document_id": doc_id,
        "title": title,
        "chunks": len(chunks),
        "characters": len(raw_text),
    }
