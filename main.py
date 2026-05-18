"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.api.routes import health, documents, chat


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description="RAG System - Document Q&A and Web Search",
        version="1.0.0",
    )

    # CORS — allow frontend origin (set via env var in production)
    origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.on_event("startup")
    async def startup():
        logger.info(f"Starting {settings.app_name}")
        logger.info(f"Embedding model: {settings.embedding_model}")
        logger.info(f"LLM: {settings.groq_model}")
        logger.info(f"Vector store: {settings.chroma_persist_dir}")
        logger.info(f"Allowed origins: {origins}")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )