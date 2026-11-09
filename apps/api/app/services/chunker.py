"""Split text into overlapping chunks for embedding."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping word-based chunks.

    Why word-based instead of character-based: preserves word boundaries,
    more predictable for embeddings, and simpler to reason about.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    base_meta = metadata or {}

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        if chunk_text_str.strip():
            chunks.append(Chunk(
                text=chunk_text_str,
                index=idx,
                metadata={**base_meta, "word_start": start, "word_end": min(end, len(words))},
            ))
            idx += 1

        if end >= len(words):
            break
        start = end - overlap

    return chunks
