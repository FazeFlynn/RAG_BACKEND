"""Tests for the chunker service."""

from app.services.chunker import chunk_documents


def test_chunk_short_document():
    docs = [{"content": "Short text", "metadata": {"source": "test.txt"}}]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Short text"
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_chunk_long_document():
    long_text = "This is a sentence. " * 100  # ~2000 chars
    docs = [{"content": long_text, "metadata": {"source": "test.txt"}}]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1

    # All chunks should have metadata
    for chunk in chunks:
        assert "chunk_index" in chunk["metadata"]
        assert "source" in chunk["metadata"]


def test_chunk_preserves_metadata():
    docs = [{"content": "Some text here", "metadata": {"source": "doc.pdf", "page": 1}}]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=0)
    assert chunks[0]["metadata"]["source"] == "doc.pdf"
    assert chunks[0]["metadata"]["page"] == 1


def test_chunk_multiple_documents():
    docs = [
        {"content": "First document content", "metadata": {"source": "a.txt"}},
        {"content": "Second document content", "metadata": {"source": "b.txt"}},
    ]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 2
    sources = {c["metadata"]["source"] for c in chunks}
    assert sources == {"a.txt", "b.txt"}


def test_chunk_empty_input():
    chunks = chunk_documents([], chunk_size=512, chunk_overlap=100)
    assert len(chunks) == 0
