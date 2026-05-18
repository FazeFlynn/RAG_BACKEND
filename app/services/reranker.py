"""
Reranker service - re-scores retrieved documents for better precision.
"""

from typing import Any
from loguru import logger

from app.core.config import settings

_reranker = None


def get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker model: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
        logger.info("Reranker model loaded")
    return _reranker


def rerank(query: str, documents: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
    """Rerank documents using cross-encoder. Returns top_k best documents."""
    if not documents:
        return []

    if not settings.use_reranker:
        return documents[:top_k] if top_k else documents

    if top_k is None:
        top_k = settings.top_k

    reranker = get_reranker()

    # Create query-document pairs for scoring
    pairs = [(query, doc["content"]) for doc in documents]
    scores = reranker.predict(pairs)

    # Attach reranker scores and sort
    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)

    logger.debug(f"Reranked {len(documents)} docs, returning top {top_k}")
    return reranked[:top_k]
