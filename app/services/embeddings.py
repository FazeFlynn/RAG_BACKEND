"""
Embedding service - generates embeddings using sentence-transformers.
"""
from __future__ import annotations

from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np

from app.core.config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
        logger.info("Embedding model loaded successfully")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Generate embedding for a single query."""
    model = get_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()


def is_model_loaded() -> bool:
    return _model is not None
