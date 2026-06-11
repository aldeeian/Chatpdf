# ChatPDF — RAG-powered PDF Q&A

[![tests](https://github.com/aldeeian/Chatpdf/actions/workflows/tests.yml/badge.svg)](https://github.com/aldeeian/Chatpdf/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B%20via%20Groq-green)

Turn any PDF into a chat. Upload one or more PDFs and ask questions — the app extracts text, builds a FAISS vector index, retrieves the most relevant chunks, and answers using **LLaMA 3.3 70B (via Groq)** grounded strictly in your document. Every answer shows the exact source chunks and page numbers it was built from.

<!-- TODO: replace with a real screenshot/GIF after running locally:
![demo](docs/demo.gif) -->

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit (custom CSS chat interface) |
| REST API | FastAPI + Uvicorn |
| RAG pipeline | LangChain (loader, splitter, retriever) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (runs locally, zero API cost) |
| LLM | `llama-3.3-70b-versatile` via Groq (free tier available) |
| Vector store | FAISS (CPU) |
| Testing | pytest + FastAPI TestClient — 16 tests, runs in CI on every push |
| Language | Python 3.10+ |

---

## Project structure

```
Chatpdf/
├── api.py            ← FastAPI REST backend
├── pdfquery.py       ← RAG pipeline (ingest + query)
├── streamlitui.py    ← Streamlit chat UI
├── requirements.txt
├── .github/
│   └── workflows/
│       └── tests.yml ← CI: pytest on every push
└── tests/
    ├── __init__.py
    └── test_api.py   ← pytest test suite (16 tests)
```

---

## Quick start

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

**3. Set your Groq API key**

```bash
# Windows Command Prompt
set GROQ_API_KEY=your_key_here

# Windows PowerShell
$env:GROQ_API_KEY="your_key_here"

# macOS / Linux
export GROQ_API_KEY=your_key_here
```

Get a free key at [https://console.groq.com/keys](https://console.groq.com/keys).

---

## Running the Streamlit UI

```bash
streamlit run streamlitui.py
```

Opens at `http://localhost:8501`. Enter your API key in the sidebar (or set `GROQ_API_KEY` first), upload PDFs, and start chatting.

**Features:**
- Multiple PDF uploads in one session — query across all of them at once
- Auto-generated 3-bullet document summary on upload
- Finance Mode — one-click prompts for financial analysis
- Source transparency — every answer shows the exact chunks and pages retrieved
- Chat history that persists through the session

---

## Running the FastAPI server

```bash
uvicorn api:app --reload --port 8000
```

Interactive API docs are auto-generated at `http://localhost:8000/docs`.

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

#### `GET /documents`

List all documents currently held in memory.

#### `DELETE /documents/{doc_id}`

Remove a document from memory and delete its temporary file.

---

## Running the tests

```bash
pytest tests/ -v
```

The test suite mocks `PDFQuery`, so **no API key is required**. All **16 tests** cover the upload, query, list, and delete endpoints — including validation failures (empty question, non-PDF upload, unknown document IDs) and downstream error handling (ingestion failure → 500, LLM failure → 500). The same suite runs automatically in GitHub Actions on every push.

---

## How it works

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
                LLaMA 3.3 70B via Groq (grounded prompt)
                              │
                              ▼
                     answer + source chunks
```

1. **Ingest:** `PyPDFLoader` reads all pages and splits them into overlapping chunks.
2. **Embed:** Each chunk is converted to a vector by a local HuggingFace model (no API cost).
3. **Index:** Vectors are stored in FAISS for fast nearest-neighbour search.
4. **Retrieve:** The top-4 chunks most similar to the question are fetched.
5. **Answer:** A strict QA prompt — "if the answer is not in the context, say you don't know" — is sent to Groq with the retrieved context, which keeps hallucination low.

---

## Design decisions

- **Local embeddings, hosted LLM.** Embedding every chunk through an API gets expensive fast; MiniLM runs free on CPU. Only the final answer generation hits the Groq API.
- **Groq over OpenAI/Gemini.** Sub-second inference on a 70B model with a generous free tier — ideal for a demo anyone can run with a free key.
- **Built-in rate-limit retry.** A 429 from Groq triggers a 20-second wait and one retry instead of crashing the request.
- **API layer tested, pipeline mocked.** Tests verify routing, validation, and error handling deterministically, with no network calls and no model downloads — so CI is fast and free.

---

## Common issues

| Problem | Fix |
|---|---|
| Auth errors | Ensure `GROQ_API_KEY` is set in your shell (or entered in the Streamlit sidebar) |
| `model_decommissioned` error | Update `_GROQ_MODEL` in `pdfquery.py` to a current model from [console.groq.com/docs/models](https://console.groq.com/docs/models) |
| `python-multipart` missing | `pip install python-multipart` (required for FastAPI file upload) |
| Blank / irrelevant answers | Increase `k` in `PDFQuery`, or check the source chunks to verify retrieval quality |
| FAISS build errors on Windows | `pip install faiss-cpu --prefer-binary` |
| First run is slow | The MiniLM embedding model (~90 MB) downloads once, then is cached |
