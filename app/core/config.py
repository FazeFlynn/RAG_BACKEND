import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "RAG System"
    app_host: str = "0.0.0.0"
    app_port: int = 8011
    debug: bool = False

    # CORS — comma-separated list of allowed frontend origins
    # Local dev:      http://localhost:3000
    # Production:     https://your-app.vercel.app
    # Multiple:       http://localhost:3000,https://your-app.vercel.app
    allowed_origins: str = "http://localhost:3000"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # LLM (Groq)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Vector Store
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "rag_documents"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # Web Search (Tavily)
    tavily_api_key: str = ""
    web_search_max_results: int = 10
    web_scrape_timeout: int = 15

    # Upload
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 50

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def ensure_dirs(self):
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()