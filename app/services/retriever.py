"""
Retriever service - combines vector search with optional reranking.
"""

from typing import Any
from loguru import logger

from app.core.config import settings
from app.services import vector_store


def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Retrieve the most relevant document chunks for a query."""
    if top_k is None:
        top_k = settings.top_k

    # Fetch more candidates if reranking is enabled
    search_k = top_k * 3 if settings.use_reranker else top_k

    # Vector search
    candidates = vector_store.search(query, top_k=search_k)

    if not candidates:
        logger.debug("No documents found in vector store")
        return []

    # Rerank only if enabled — import lazily so model never loads when disabled
    if settings.use_reranker and len(candidates) > top_k:
        from app.services.reranker import rerank   # lazy import
        results = rerank(query, candidates, top_k=top_k)
        logger.info(f"Reranked to {len(results)} chunks")
    else:
        results = candidates[:top_k]

    logger.info(f"Retrieved {len(results)} relevant chunks for query")
    return results