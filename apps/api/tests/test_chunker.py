"""Tests for the text chunking service."""

from app.services.chunker import chunk_text


def test_basic_chunking():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert all(c.text for c in chunks)
    assert chunks[0].index == 0
    assert chunks[1].index == 1


def test_empty_text():
    chunks = chunk_text("", chunk_size=20, overlap=5)
    assert chunks == []


def test_whitespace_only():
    chunks = chunk_text("   \n\n  ", chunk_size=20, overlap=5)
    assert chunks == []


def test_short_text_single_chunk():
    text = "Hello world this is short"
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_overlap_works():
    words = [f"w{i}" for i in range(30)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=10, overlap=3)

    # Second chunk should start at word 7 (10 - 3 overlap)
    assert "w7" in chunks[1].text
    # And also contain the overlapping words
    assert "w7" in chunks[1].text


def test_metadata_passed_through():
    text = "hello world foo bar baz"
    meta = {"person_id": "123", "doc": "test"}
    chunks = chunk_text(text, chunk_size=100, overlap=0, metadata=meta)

    assert chunks[0].metadata["person_id"] == "123"
    assert chunks[0].metadata["doc"] == "test"
    assert "word_start" in chunks[0].metadata


def test_chunk_metadata_has_positions():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert chunks[0].metadata["word_start"] == 0
    assert chunks[0].metadata["word_end"] == 20
    assert chunks[1].metadata["word_start"] == 15  # 20 - 5 overlap
