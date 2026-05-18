"""
Vector store service - ChromaDB for persistent vector storage.
"""
from __future__ import annotations

import uuid
from typing import Any
import chromadb
from loguru import logger

from app.core.config import settings
from app.services.embeddings import embed_texts, embed_query

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(chunks: list[dict[str, Any]]) -> int:
    """Add document chunks to the vector store. Returns count of added chunks."""
    collection = get_collection()

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Embed in batches to avoid memory issues
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        embeddings = embed_texts(batch_texts)

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )

    logger.info(f"Added {len(chunks)} chunks to vector store")
    return len(chunks)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Search the vector store for relevant documents."""
    if top_k is None:
        top_k = settings.top_k

    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = []
    for i in range(len(results["ids"][0])):
        documents.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # Convert distance to similarity
        })

    return documents


def get_document_count() -> int:
    """Get the total number of documents in the store."""
    return get_collection().count()


def delete_by_source(source_name: str) -> int:
    """Delete all chunks from a specific source file."""
    collection = get_collection()
    results = collection.get(where={"source": source_name})
    if results["ids"]:
        collection.delete(ids=results["ids"])
        logger.info(f"Deleted {len(results['ids'])} chunks for source: {source_name}")
        return len(results["ids"])
    return 0


def list_sources() -> list[str]:
    """List all unique source files in the store."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    sources = set()
    for meta in results["metadatas"]:
        if "source" in meta:
            sources.add(meta["source"])
    return sorted(sources)
