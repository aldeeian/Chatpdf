"""
api.py — FastAPI REST backend for ChatPDF.

Run with:
    uvicorn api:app --reload --port 8000

Endpoints
---------
POST   /upload              Upload a PDF; returns a document ID.
POST   /query               Ask a question about an uploaded document.
GET    /documents           List all uploaded documents.
DELETE /documents/{doc_id}  Remove a document from memory.
"""

import os
import uuid
import tempfile
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, field_validator

from pdfquery import PDFQuery

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ChatPDF API",
    description="RAG-powered PDF question-answering via Gemini + FAISS.",
    version="1.0.0",
)

# Directory where uploaded PDFs are stored for the lifetime of the server.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "chatpdf_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory document registry.
# Key  : document ID (UUID string)
# Value: {"query": PDFQuery, "filename": str, "path": str}
_documents: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas (request / response shapes)
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    """Metadata returned for an uploaded document."""
    id: str
    filename: str


class QueryRequest(BaseModel):
    """Body expected by POST /query."""
    document_id: str
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace")
        return v.strip()

    @field_validator("document_id")
    @classmethod
    def doc_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_id must not be empty")
        return v.strip()


class SourceChunk(BaseModel):
    """A single retrieved context chunk surfaced alongside the answer."""
    content: str
    page: int | str
    source: str


class QueryResponse(BaseModel):
    """Response returned by POST /query."""
    document_id: str
    answer: str
    sources: List[SourceChunk]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/upload",
    response_model=DocumentInfo,
    summary="Upload a PDF and build its vector index",
    status_code=201,
)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    """Accept a PDF file, run it through the RAG ingestion pipeline, and
    return a unique document ID that you can pass to ``/query``.

    Raises 400 if the uploaded file is not a PDF.
    Raises 500 if ingestion fails (e.g. corrupt file, missing API key).
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a file with a .pdf extension.",
        )

    doc_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{doc_id}.pdf"

    # Persist the file so PyPDFLoader can read it from disk.
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    file_path.write_bytes(content)

    # Build the FAISS vector index.
    try:
        pq = PDFQuery()
        pq.ingest(str(file_path))
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {exc}",
        ) from exc

    _documents[doc_id] = {
        "query": pq,
        "filename": file.filename,
        "path": str(file_path),
    }

    return DocumentInfo(id=doc_id, filename=file.filename)


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about an uploaded document",
)
async def query_document(request: QueryRequest) -> QueryResponse:
    """Send a natural-language question together with a ``document_id``.

    Returns the Gemini-generated answer plus the top-k source chunks that
    were used to produce it.

    Raises 404 if the document ID is not found.
    Raises 500 if the LLM call fails.
    """
    doc = _documents.get(request.document_id)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.document_id}' not found. Upload it first via POST /upload.",
        )

    try:
        answer, raw_sources = doc["query"].ask_with_sources(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {exc}",
        ) from exc

    sources = [SourceChunk(**s) for s in raw_sources]
    return QueryResponse(
        document_id=request.document_id,
        answer=answer,
        sources=sources,
    )


@app.get(
    "/documents",
    response_model=List[DocumentInfo],
    summary="List all uploaded documents",
)
async def list_documents() -> List[DocumentInfo]:
    """Return metadata for every document currently held in memory."""
    return [
        DocumentInfo(id=doc_id, filename=meta["filename"])
        for doc_id, meta in _documents.items()
    ]


@app.delete(
    "/documents/{doc_id}",
    summary="Delete a document",
    status_code=200,
)
async def delete_document(doc_id: str) -> dict:
    """Remove the document's vector index from memory and delete the PDF file
    from the temporary upload directory.

    Raises 404 if the document ID is not found.
    """
    doc = _documents.pop(doc_id, None)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found.",
        )

    path = Path(doc["path"])
    path.unlink(missing_ok=True)

    return {"message": f"Document '{doc_id}' ({doc['filename']}) deleted successfully."}
