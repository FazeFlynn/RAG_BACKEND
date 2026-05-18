"""
Text chunking service - splits documents into overlapping chunks for embedding.
"""

from typing import Any
from loguru import logger


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
) -> list[dict[str, Any]]:
    """Split documents into chunks with overlap."""
    all_chunks = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        chunks = _recursive_split(content, chunk_size, chunk_overlap)

        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}
            all_chunks.append({
                "content": chunk_text,
                "metadata": chunk_metadata,
            })

    logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Recursively split text using multiple separators."""
    separators = ["\n\n", "\n", ". ", " ", ""]

    return _split_with_separators(text, separators, chunk_size, chunk_overlap)


def _split_with_separators(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text trying separators in order of preference."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try each separator
    for i, sep in enumerate(separators):
        if sep == "":
            # Last resort: split by character
            return _split_by_chars(text, chunk_size, chunk_overlap)

        if sep in text:
            parts = text.split(sep)
            chunks = []
            current_chunk = ""

            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part

                if len(candidate) <= chunk_size:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # If this single part is too large, recurse with finer separator
                    if len(part) > chunk_size:
                        sub_chunks = _split_with_separators(
                            part, separators[i + 1:], chunk_size, chunk_overlap
                        )
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Add overlap between chunks
            if chunk_overlap > 0 and len(chunks) > 1:
                chunks = _add_overlap(chunks, chunk_overlap)

            return [c for c in chunks if c.strip()]

    return [text.strip()] if text.strip() else []


def _split_by_chars(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text by character count with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap from the end of previous chunk to start of next chunk."""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
        result.append(overlap_text + " " + chunks[i])
    return result
