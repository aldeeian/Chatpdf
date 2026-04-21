"""
tests/test_api.py — pytest suite for the ChatPDF FastAPI backend.

Run with:
    pytest tests/ -v

The PDFQuery class is mocked throughout so no Gemini API key is needed and
no real FAISS indexing happens — we are testing the API layer (routing,
validation, error handling) not the RAG pipeline itself.
"""

import io
import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_ANSWER = "RBC revenue grew 15 percent in 2024."
FAKE_SOURCES = [
    {
        "content": "RBC revenue grew 15 percent in 2024.",
        "page": 0,
        "source": "test.pdf",
    }
]

# Minimal valid-looking PDF bytes.  The content is not actually parsed because
# PDFQuery.ingest() is mocked, so any bytes with a .pdf filename work here.
MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n1 0 obj\n<</Type /Catalog>>\nendobj\ntrailer\n<</Root 1 0 R>>\n%%EOF"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_document_store():
    """Clear the in-memory document registry before every test to prevent
    state leaking between tests."""
    import api
    api._documents.clear()
    yield
    api._documents.clear()


@pytest.fixture()
def mock_pdfquery():
    """Replace PDFQuery with a mock so no real model calls happen."""
    with patch("api.PDFQuery") as mock_cls:
        instance = MagicMock()
        instance.ask_with_sources.return_value = (FAKE_ANSWER, FAKE_SOURCES)
        mock_cls.return_value = instance
        yield mock_cls, instance


@pytest.fixture()
def client(mock_pdfquery):
    """Return a synchronous TestClient with PDFQuery mocked."""
    from api import app
    return TestClient(app)


@pytest.fixture()
def uploaded_doc_id(client):
    """Upload a fake PDF and return its document ID — used by query/delete tests."""
    response = client.post(
        "/upload",
        files={"file": ("annual_report.pdf", io.BytesIO(MINIMAL_PDF_BYTES), "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()["id"]


# ---------------------------------------------------------------------------
# Upload endpoint tests
# ---------------------------------------------------------------------------

class TestUpload:
    def test_upload_valid_pdf_returns_201_with_id(self, client):
        """A valid PDF upload should return 201 and a document ID."""
        response = client.post(
            "/upload",
            files={"file": ("report.pdf", io.BytesIO(MINIMAL_PDF_BYTES), "application/pdf")},
        )
        assert response.status_code == 201
        body = response.json()
        assert "id" in body
        assert body["filename"] == "report.pdf"

    def test_upload_non_pdf_returns_400(self, client):
        """Uploading a non-PDF file should be rejected with 400."""
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_upload_empty_file_returns_400(self, client):
        """An empty file body should be rejected with 400."""
        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code == 400

    def test_upload_ingestion_failure_returns_500(self, mock_pdfquery, client):
        """If FAISS ingestion raises, the endpoint should return 500."""
        _, instance = mock_pdfquery
        instance.ingest.side_effect = RuntimeError("FAISS error")
        response = client.post(
            "/upload",
            files={"file": ("broken.pdf", io.BytesIO(MINIMAL_PDF_BYTES), "application/pdf")},
        )
        assert response.status_code == 500
        assert "FAISS error" in response.json()["detail"]

    def test_upload_registers_document_in_store(self, client):
        """After a successful upload, the document should appear in GET /documents."""
        client.post(
            "/upload",
            files={"file": ("q1.pdf", io.BytesIO(MINIMAL_PDF_BYTES), "application/pdf")},
        )
        listed = client.get("/documents").json()
        assert len(listed) == 1
        assert listed[0]["filename"] == "q1.pdf"


# ---------------------------------------------------------------------------
# Query endpoint tests
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_answer_and_sources(self, client, uploaded_doc_id):
        """A valid query should return the mocked answer and source chunks."""
        response = client.post(
            "/query",
            json={"document_id": uploaded_doc_id, "question": "What was revenue growth?"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == FAKE_ANSWER
        assert len(body["sources"]) == 1
        assert body["sources"][0]["page"] == 0

    def test_query_unknown_document_returns_404(self, client):
        """Querying a document ID that was never uploaded should return 404."""
        response = client.post(
            "/query",
            json={"document_id": "non-existent-id", "question": "anything"},
        )
        assert response.status_code == 404

    def test_query_empty_question_returns_422(self, client, uploaded_doc_id):
        """An empty question string should fail Pydantic validation (422)."""
        response = client.post(
            "/query",
            json={"document_id": uploaded_doc_id, "question": "   "},
        )
        assert response.status_code == 422

    def test_query_missing_question_field_returns_422(self, client, uploaded_doc_id):
        """Omitting the question field entirely should fail with 422."""
        response = client.post(
            "/query",
            json={"document_id": uploaded_doc_id},
        )
        assert response.status_code == 422

    def test_query_llm_failure_returns_500(self, mock_pdfquery, client, uploaded_doc_id):
        """If the LLM call raises, the endpoint should return 500."""
        _, instance = mock_pdfquery
        instance.ask_with_sources.side_effect = RuntimeError("LLM timeout")
        response = client.post(
            "/query",
            json={"document_id": uploaded_doc_id, "question": "What is revenue?"},
        )
        assert response.status_code == 500
        assert "LLM timeout" in response.json()["detail"]


# ---------------------------------------------------------------------------
# List documents endpoint tests
# ---------------------------------------------------------------------------

class TestListDocuments:
    def test_list_empty_when_no_uploads(self, client):
        """Before any uploads the list should be empty."""
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_returns_all_uploaded_documents(self, client):
        """Every successfully uploaded document should appear in the list."""
        for name in ("a.pdf", "b.pdf"):
            client.post(
                "/upload",
                files={"file": (name, io.BytesIO(MINIMAL_PDF_BYTES), "application/pdf")},
            )
        docs = client.get("/documents").json()
        filenames = {d["filename"] for d in docs}
        assert filenames == {"a.pdf", "b.pdf"}


# ---------------------------------------------------------------------------
# Delete endpoint tests
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_existing_document_returns_200(self, client, uploaded_doc_id):
        """Deleting a known document should succeed."""
        response = client.delete(f"/documents/{uploaded_doc_id}")
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()

    def test_delete_removes_document_from_list(self, client, uploaded_doc_id):
        """After deletion the document should no longer appear in GET /documents."""
        client.delete(f"/documents/{uploaded_doc_id}")
        docs = client.get("/documents").json()
        ids = [d["id"] for d in docs]
        assert uploaded_doc_id not in ids

    def test_delete_unknown_document_returns_404(self, client):
        """Deleting a non-existent document ID should return 404."""
        response = client.delete("/documents/does-not-exist")
        assert response.status_code == 404

    def test_delete_twice_returns_404_on_second_call(self, client, uploaded_doc_id):
        """Deleting the same document twice should fail on the second attempt."""
        client.delete(f"/documents/{uploaded_doc_id}")
        response = client.delete(f"/documents/{uploaded_doc_id}")
        assert response.status_code == 404
