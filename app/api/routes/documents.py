"""Document upload and management endpoints."""

import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.models.schemas import DocumentUploadResponse, DocumentListResponse
from app.services.file_loader import load_file, get_supported_extensions
from app.services.chunker import chunk_documents
from app.services import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for indexing."""
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in get_supported_extensions():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {get_supported_extensions()}",
        )

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Max: {settings.max_file_size_mb}MB",
        )

    # Save file to disk
    file_path = os.path.join(settings.upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # Load and chunk the document
        documents = load_file(file_path)
        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # Delete existing chunks for this file (re-upload support)
        vector_store.delete_by_source(file.filename)

        # Add to vector store
        num_chunks = vector_store.add_documents(chunks)

        return DocumentUploadResponse(
            filename=file.filename,
            num_chunks=num_chunks,
            message=f"Successfully indexed {file.filename} ({num_chunks} chunks)",
        )

    except Exception as e:
        # Clean up saved file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    sources = vector_store.list_sources()
    return DocumentListResponse(documents=sources, total=len(sources))


@router.delete("/{filename}")
async def delete_document(filename: str):
    """Delete a document and its chunks from the index."""
    deleted = vector_store.delete_by_source(filename)

    # Also remove the file from uploads
    file_path = os.path.join(settings.upload_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

    return {"message": f"Deleted {deleted} chunks for '{filename}'"}
