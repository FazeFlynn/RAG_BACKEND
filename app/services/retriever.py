"""
Retriever service - combines vector search with optional reranking.
"""

from typing import Any
from loguru import logger

from app.core.config import settings
from app.services import vector_store
from app.services.reranker import rerank


def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve the most relevant document chunks for a query."""
    if top_k is None:
        top_k = settings.top_k

    # Fetch more candidates if we're going to rerank
    search_k = top_k * 3 if settings.use_reranker else top_k

    # Vector search
    candidates = vector_store.search(query, top_k=search_k)

    if not candidates:
        logger.debug("No documents found in vector store")
        return []

    # Rerank for better precision
    if settings.use_reranker and len(candidates) > top_k:
        results = rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    logger.info(f"Retrieved {len(results)} relevant chunks for query")
    return results
