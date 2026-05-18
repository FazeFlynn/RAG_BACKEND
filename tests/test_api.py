"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama_connected" in data
    assert "documents_indexed" in data


def test_list_documents_empty():
    response = client.get("/documents/")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_upload_unsupported_file():
    response = client.post(
        "/documents/upload",
        files={"file": ("test.xyz", b"content", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_chat_empty_query():
    response = client.post("/chat/", json={"query": ""})
    assert response.status_code == 422  # validation error


def test_delete_nonexistent_document():
    response = client.delete("/documents/nonexistent.pdf")
    assert response.status_code == 404
