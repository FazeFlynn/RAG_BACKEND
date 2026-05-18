"""Health check endpoint."""

from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services import llm, vector_store, embeddings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    ollama_ok = await llm.check_connection()
    return HealthResponse(
        status="healthy" if ollama_ok else "degraded",
        ollama_connected=ollama_ok,
        embedding_model_loaded=embeddings.is_model_loaded(),
        documents_indexed=vector_store.get_document_count(),
    )
