# pdfquery.py  (FAISS / HuggingFace / Gemini style)
import os
from typing import Optional
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import FAISS

# configure genai (safe to call even if not used elsewhere)
try:
    import google.generativeai as genai
except Exception:
    genai = None

class PDFQuery:
    def __init__(self, gemini_api_key: Optional[str] = None, gemini_model: Optional[str] = None, k: int = 4):
        key = gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if key and genai is not None:
            try:
                genai.configure(api_key=key)
            except Exception:
                pass
        if key:
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key

        # embeddings (local HF model)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # choose model id you found via list_models; default is the notebook's gemini-pro
        self.gemini_model = gemini_model or os.environ.get("GEMINI_MODEL") or "gemini-pro"
        self.llm = ChatGoogleGenerativeAI(model=self.gemini_model)
        self.db = None
        self.k = int(k)

    def ingest(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        if not pages:
            raise ValueError("No pages loaded from PDF.")
        # build FAISS DB
        self.db = FAISS.from_documents(pages, self.embeddings)

    def ask(self, question: str) -> str:
        if self.db is None:
            return "Please upload and ingest a PDF first."
        docs = self.db.similarity_search(question, k=self.k)
        content = "\n".join([d.page_content for d in docs])
        qa_prompt = (
            "Use the following pieces of context to answer the user. "
            "If you don't know, say you don't know. Be factual and concise."
        )
        input_text = qa_prompt + "\nContext:\n" + content + "\nUser question:\n" + question
        # notebook used llm.invoke
        result = self.llm.invoke(input_text)
        # ChatGoogleGenerativeAI.invoke returns an object; try to get text content
        try:
            return result.content
        except Exception:
            # fallback to string representation
            return str(result)

    def forget(self):
        self.db = None
