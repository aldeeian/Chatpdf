# pdfquery.py — FAISS / HuggingFace Embeddings / Groq LLM
import os
import time
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

try:
    # Maintained package (langchain-community embeddings are deprecated)
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # fallback for older environments
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

_GROQ_MODEL  = "llama-3.3-70b-versatile"
_EMBED_MODEL = "all-MiniLM-L6-v2"


class PDFQuery:
    """RAG pipeline: ingest PDFs into FAISS, answer questions with Groq."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        k: int = 4,
    ):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if key:
            os.environ["GROQ_API_KEY"] = key

        self.llm        = ChatGroq(model=_GROQ_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model_name=_EMBED_MODEL)
        self.db: Optional[FAISS] = None
        self.k = int(k)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _invoke(self, prompt: str) -> str:
        """Call the LLM; on a 429 rate-limit error wait 20 s and retry once."""
        try:
            result = self.llm.invoke(prompt)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "quota" in msg or "rate" in msg or "resource exhausted" in msg:
                time.sleep(20)
                result = self.llm.invoke(prompt)
                return result.content if hasattr(result, "content") else str(result)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> None:
        """Load a PDF and merge its chunks into the shared FAISS index.

        Safe to call multiple times — each call adds to the existing index so
        you can query across several PDFs at once.
        """
        loader = PyPDFLoader(file_path)
        pages  = loader.load_and_split()
        if not pages:
            raise ValueError(f"No pages loaded from {file_path}.")
        if self.db is None:
            self.db = FAISS.from_documents(pages, self.embeddings)
        else:
            self.db.add_documents(pages)

    def ask(self, question: str) -> str:
        """Return a plain answer string. Kept for backward compatibility."""
        answer, _ = self.ask_with_sources(question)
        return answer

    def ask_with_sources(self, question: str) -> tuple[str, list[dict]]:
        """Return (answer, sources).

        Each source dict contains:
            content – first 400 chars of the retrieved chunk
            page    – 0-based page number from the original PDF
            source  – file path / name the chunk came from
        """
        if self.db is None:
            return "Please upload and ingest a PDF first.", []

        docs    = self.db.similarity_search(question, k=self.k)
        context = "\n\n".join(d.page_content for d in docs)
        prompt  = (
            "Use the context excerpts below to answer the question. "
            "If the answer is not found in the context, say you don't know. "
            "Be factual and concise.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        answer  = self._invoke(prompt)
        sources = [
            {
                "content": d.page_content[:400],
                "page":    d.metadata.get("page", "?"),
                "source":  d.metadata.get("source", "unknown"),
            }
            for d in docs
        ]
        return answer, sources

    def get_document_summary(self) -> str:
        """Return a 3-bullet summary of the ingested documents.

        Samples a broad cross-section of chunks so the summary covers the
        whole document, not just the first few pages.
        Returns an empty string if no documents are loaded or the call fails.
        """
        if self.db is None:
            return ""

        docs    = self.db.similarity_search(
            "main topic overview introduction purpose key findings", k=8
        )
        context = "\n\n".join(d.page_content for d in docs)
        prompt  = (
            "Based on the document excerpts below, write exactly 3 concise bullet points "
            "summarizing the main topics, key findings, or purpose of this document. "
            "Use this exact format:\n"
            "• [First key point]\n"
            "• [Second key point]\n"
            "• [Third key point]\n\n"
            f"Excerpts:\n{context}"
        )
        try:
            return self._invoke(prompt)
        except Exception:
            return ""

    def forget(self) -> None:
        """Clear the FAISS index (removes all ingested documents)."""
        self.db = None
