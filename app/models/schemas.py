from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class QueryType(str, Enum):
    DOCUMENT_QA = "document_qa"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"
    GENERAL = "general"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    query_type: Optional[QueryType] = None  # None = auto-detect
    conversation_id: Optional[str] = None


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    score: Optional[float] = None


class WebSource(BaseModel):
    url: str
    title: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    query_type: QueryType
    sources: list[SourceDocument] = []
    web_sources: list[WebSource] = []
    conversation_id: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    filename: str
    num_chunks: int
    message: str


class DocumentListResponse(BaseModel):
    documents: list[str]
    total: int


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    embedding_model_loaded: bool
    documents_indexed: int
