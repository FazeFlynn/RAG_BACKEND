"""Document upload and management endpoints."""

import os
import gc
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.models.schemas import DocumentUploadResponse, DocumentListResponse
from app.services.file_loader import load_file, get_supported_extensions
from app.services.chunker import chunk_documents
from app.services import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])

# How many raw documents (pages) to process at a time
# Keeps memory flat regardless of file size
PAGE_BATCH_SIZE = 5


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

    # Stream file to disk — never load entire file into RAM
    file_path = os.path.join(settings.upload_dir, file.filename)
    size_bytes = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):   # read 1MB at a time
                size_bytes += len(chunk)
                size_mb = size_bytes / (1024 * 1024)
                if size_mb > settings.max_file_size_mb:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large (>{settings.max_file_size_mb}MB limit)",
                    )
                f.write(chunk)
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    try:
        # Delete existing chunks for this file (re-upload support)
        vector_store.delete_by_source(file.filename)

        # Load all pages/sections from the file
        documents = load_file(file_path)

        total_chunks = 0

        # Process in small page batches to keep memory flat
        for batch_start in range(0, len(documents), PAGE_BATCH_SIZE):
            batch_docs = documents[batch_start: batch_start + PAGE_BATCH_SIZE]

            # Chunk this small batch
            chunks = chunk_documents(
                batch_docs,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            if chunks:
                # Embed and store — vector_store already batches internally
                added = vector_store.add_documents(chunks)
                total_chunks += added

            # Free memory before next batch
            del batch_docs, chunks
            gc.collect()

        # Free the full document list now that we're done
        del documents
        gc.collect()

        return DocumentUploadResponse(
            filename=file.filename,
            num_chunks=total_chunks,
            message=f"Successfully indexed {file.filename} ({total_chunks} chunks)",
        )

    except HTTPException:
        raise
    except Exception as e:
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

    file_path = os.path.join(settings.upload_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

    return {"message": f"Deleted {deleted} chunks for '{filename}'"}