## ChatPDF (Gemini + RAG)

Turn any PDF into a chat experience. Upload one or more PDFs and ask questions — the app extracts text, builds a FAISS vector index, retrieves the most relevant chunks, and answers using a Gemini model grounded in your document.

---

### Tech stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| REST API | FastAPI + Uvicorn |
| RAG pipeline | LangChain (loader, splitter, retriever) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API cost) |
| LLM | Google Gemini via `ChatGoogleGenerativeAI` |
| Vector store | FAISS (CPU) |
| Testing | pytest + FastAPI TestClient (httpx) |
| Language | Python 3.13 |

---

### Project structure

```
chatpdf-upgraded/
├── api.py            ← FastAPI REST backend
├── pdfquery.py       ← RAG pipeline (ingest + query)
├── streamlitui.py    ← Streamlit chat UI
├── main.py           ← Original single-file script (reference)
├── requirements.txt
└── tests/
    ├── __init__.py
    └── test_api.py   ← pytest test suite
```

---

### Quick start

**1. Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set your Gemini API key**

```bash
# Windows Command Prompt
set GEMINI_API_KEY=your_key_here

# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"

# macOS / Linux
export GEMINI_API_KEY=your_key_here
```

Get a free key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

---

### Running the Streamlit UI

```bash
streamlit run streamlitui.py
```

Opens at `http://localhost:8501`. Enter your API key in the sidebar, upload PDFs, and start chatting.

**Features:**
- Multiple PDF uploads in one session
- Finance Mode — one-click prompts for financial analysis
- Source transparency — every answer shows the exact chunks retrieved
- Chat history that persists through the session

---

### Running the FastAPI server

```bash
uvicorn api:app --reload --port 8000
```

The server starts at `http://localhost:8000`.  
Interactive API docs are auto-generated at `http://localhost:8000/docs`.

---

### API endpoints

#### `POST /upload`

Upload a PDF file. Returns a `document_id` you use for all subsequent queries.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@annual_report.pdf"
```

**Response `201`**
```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename": "annual_report.pdf"
}
```

---

#### `POST /query`

Ask a question about an uploaded document.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"document_id": "<id>", "question": "What was the revenue growth?"}'
```

**Response `200`**
```json
{
  "document_id": "3fa85f64-...",
  "answer": "Revenue grew 15% year-over-year driven by...",
  "sources": [
    {
      "content": "RBC reported total revenue of $54.9 billion...",
      "page": 12,
      "source": "annual_report.pdf"
    }
  ]
}
```

---

#### `GET /documents`

List all documents currently held in memory.

```bash
curl http://localhost:8000/documents
```

**Response `200`**
```json
[
  {"id": "3fa85f64-...", "filename": "annual_report.pdf"},
  {"id": "9b2e1c77-...", "filename": "q3_results.pdf"}
]
```

---

#### `DELETE /documents/{doc_id}`

Remove a document from memory and delete its temporary file.

```bash
curl -X DELETE http://localhost:8000/documents/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

**Response `200`**
```json
{
  "message": "Document '3fa85f64-...' (annual_report.pdf) deleted successfully."
}
```

---

### Running the tests

```bash
pytest tests/ -v
```

The test suite mocks `PDFQuery` so **no Gemini API key is required** to run tests. All 17 tests cover the upload, query, list, and delete endpoints including validation failures and downstream error handling.

Expected output:
```
tests/test_api.py::TestUpload::test_upload_valid_pdf_returns_201_with_id PASSED
tests/test_api.py::TestUpload::test_upload_non_pdf_returns_400           PASSED
...
17 passed in 1.23s
```

---

### How it works

```
PDF file
   │
   ▼
PyPDFLoader → pages[]
   │
   ▼
RecursiveCharacterTextSplitter → chunks[]
   │
   ▼
HuggingFace MiniLM → embeddings[]
   │
   ▼
FAISS index  ←──── similarity_search(question, k=4)
                              │
                              ▼
                   Gemini LLM (grounded prompt)
                              │
                              ▼
                     answer + source chunks
```

1. **Ingest:** `PyPDFLoader` reads all pages and splits them into overlapping chunks.
2. **Embed:** Each chunk is converted to a vector by a local HuggingFace model (no API cost).
3. **Index:** Vectors are stored in FAISS for fast nearest-neighbour search.
4. **Retrieve:** Top-4 chunks most similar to the user's question are fetched.
5. **Answer:** A strict QA prompt is sent to Gemini with the retrieved context.

---

### Common issues

| Problem | Fix |
|---|---|
| `404 model not found` | Run `python list_gemini_models.py` and set `GEMINI_MODEL` to a model that appears and supports `generateContent` |
| Auth errors | Ensure `GEMINI_API_KEY` is set in your shell |
| `python-multipart` missing | Run `pip install python-multipart` (required for FastAPI file upload) |
| Blank / irrelevant answers | Increase `k` in `PDFQuery`, or check source chunks to verify retrieval quality |
| FAISS build errors on Windows | Use the pre-built wheel: `pip install faiss-cpu --prefer-binary` |
