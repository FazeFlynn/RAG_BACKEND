"""Chat endpoint - main query interface."""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_pipeline import process_query, process_query_stream

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a query and get a response."""
    response = await process_query(request)
    return response


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Send a query and get a streaming response (SSE) - token by token."""
    return StreamingResponse(
        process_query_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
