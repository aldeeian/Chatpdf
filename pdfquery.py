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

    def ingest(self, file_path: str, display_name: Optional[str] = None) -> None:
        """Load a PDF and merge its chunks into the shared FAISS index.

        Safe to call multiple times — each call adds to the existing index so
        you can query across several PDFs at once.

        display_name: original filename to show in sources (useful when the
        PDF was saved to a temporary path before ingestion).
        """
        loader = PyPDFLoader(file_path)
        pages  = loader.load_and_split()
        if not pages:
            raise ValueError(f"No pages loaded from {file_path}.")
        if display_name:
            for page in pages:
                page.metadata["source"] = display_name
        if self.db is None:
            self.db = FAISS.from_documents(pages, self.embeddings)
        else:
            self.db.add_documents(pages)

    def _retrieve(self, question: str) -> tuple[str, list[dict]]:
        """Retrieve top-k chunks with relevance scores.

        Returns (context_text, sources). Each source dict contains:
            content   – first 400 chars of the retrieved chunk
            page      – 0-based page number from the original PDF
            source    – file path / name the chunk came from
            relevance – 0–100 score (higher = more relevant to the question)
        """
        results = self.db.similarity_search_with_score(question, k=self.k)
        context = "\n\n".join(d.page_content for d, _ in results)
        sources = []
        for d, distance in results:
            # FAISS returns L2 distance (lower = closer). Convert to an
            # intuitive 0–100 relevance score for display.
            relevance = round(100.0 / (1.0 + float(distance)), 1)
            sources.append(
                {
                    "content":   d.page_content[:400],
                    "page":      d.metadata.get("page", "?"),
                    "source":    d.metadata.get("source", "unknown"),
                    "relevance": relevance,
                }
            )
        return context, sources

    @staticmethod
    def _build_prompt(question: str, context: str, history: Optional[list] = None) -> str:
        """Compose the grounded QA prompt, optionally with chat history so the
        model can resolve follow-up questions ("what about the second one?")."""
        history_block = ""
        if history:
            lines = []
            for msg in history[-6:]:  # last 3 exchanges keep the prompt small
                role = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{role}: {msg.get('content', '')}")
            history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

        return (
            "Use the context excerpts below to answer the user's latest question. "
            "If the answer is not found in the context, say you don't know. "
            "Be factual and concise.\n\n"
            f"{history_block}"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

    def ask(self, question: str) -> str:
        """Return a plain answer string. Kept for backward compatibility."""
        answer, _ = self.ask_with_sources(question)
        return answer

    def ask_with_sources(
        self, question: str, history: Optional[list] = None
    ) -> tuple[str, list[dict]]:
        """Return (answer, sources). Non-streaming variant."""
        if self.db is None:
            return "Please upload and ingest a PDF first.", []

        context, sources = self._retrieve(question)
        answer = self._invoke(self._build_prompt(question, context, history))
        return answer, sources

    def ask_stream(self, question: str, history: Optional[list] = None):
        """Streaming variant.

        Returns (token_generator, sources). Iterate the generator to receive
        the answer token-by-token — ideal for a live typing effect in the UI.
        """
        if self.db is None:
            def _empty():
                yield "Please upload and ingest a PDF first."
            return _empty(), []

        context, sources = self._retrieve(question)
        prompt = self._build_prompt(question, context, history)

        def _gen():
            for chunk in self.llm.stream(prompt):
                text = getattr(chunk, "content", None)
                if text:
                    yield text

        return _gen(), sources

    def suggest_questions(self, n: int = 3) -> list[str]:
        """Generate n example questions a reader might ask about the
        ingested documents. Returns [] if nothing is loaded or the call fails."""
        if self.db is None:
            return []

        docs = self.db.similarity_search(
            "main topic overview key findings important details", k=6
        )
        context = "\n\n".join(d.page_content for d in docs)
        prompt = (
            f"Based on the document excerpts below, write exactly {n} short, "
            "specific questions a reader would likely ask about this document. "
            "Each question must be answerable from the document and under 12 words. "
            "Return one question per line with no numbering, bullets, or extra text.\n\n"
            f"Excerpts:\n{context}"
        )
        try:
            raw = self._invoke(prompt)
            questions = [
                q.strip().lstrip("-•*0123456789. ").strip()
                for q in raw.splitlines()
                if q.strip()
            ]
            return [q for q in questions if q.endswith("?")][:n]
        except Exception:
            return []

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
