"""Tests for the query router."""

import pytest
from app.services.router import classify_query
from app.models.schemas import QueryType


@pytest.mark.asyncio
async def test_route_no_documents():
    result = await classify_query("What is Python?", has_documents=False)
    assert result == QueryType.WEB_SEARCH


@pytest.mark.asyncio
async def test_route_document_keywords():
    result = await classify_query("Summarize the uploaded document", has_documents=True)
    assert result == QueryType.DOCUMENT_QA


@pytest.mark.asyncio
async def test_route_web_keywords():
    result = await classify_query("What is the latest news about AI in 2026?", has_documents=True)
    assert result == QueryType.WEB_SEARCH


@pytest.mark.asyncio
async def test_route_ambiguous_with_docs():
    result = await classify_query("What is machine learning?", has_documents=True)
    assert result == QueryType.DOCUMENT_QA  # defaults to docs when available
