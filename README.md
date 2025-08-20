
## ChatPDF (Gemini + RAG)
Turn any PDF into a chat experience. Drag-and-drop a PDF and ask questions; the app extracts text, builds embeddings, retrieves the most relevant chunks and answers using a Gemini model-grounded in your document.

### Quick start
1. Create a virtual environment in python https://docs.python.org/3/library/venv.html
2. Run "pip install -r requirements.txt"
3. Set OPENAI_API_KEY environment variable with your openai key
4. Run "python main.py"
5. Change pdf file and query in code if you want to try with any other content
To run streamlit app, follow the steps run "streamlit run streamlitui.py"

## Tech stack
Frontend/UI: Streamlit
RAG: LangChain (loaders, splitters, retriever)
Embeddings: langchain-google-genai (Gemini embeddings)
LLM: ChatGoogleGenerativeAI (Gemini)
Vector store: FAISS (CPU) or in-memory fallback
Language: Python 3.10+


## Features
RAG pipeline: PDF → text → chunking → embeddings → vector index → top-k retrieval → grounded answer.

Gemini models: generation with ChatGoogleGenerativeAI; embeddings with GoogleGenerativeAIEmbeddings.

Grounded answers: strict QA prompt avoids hallucinations and generic “I can’t access files” replies.

Windows-friendly: FAISS path + in-memory fallback to avoid heavy native deps on some machines.

Streamlit UI: quick upload, chat history, and (optional) retrieved-chunk preview for debugging.

## How it works (brief)

1.Ingest: PyPDFLoader reads all pages.

2.Split: RecursiveCharacterTextSplitter (defaults: size 600–1000, overlap 150–200).

3.Embed: Each chunk → embedding via GoogleGenerativeAIEmbeddings.

4.Index: Chunks + embeddings inserted into FAISS (or in-memory index).

5.Retrieve: Top-k by cosine similarity for the user question.

6.Answer: Compose a strict QA prompt with the retrieved context → Gemini generates a concise answer.

## Common issues & fixes

1.404 model not found: your key doesn’t have that model. Run python list_gemini_models.py and set GEMINI_MODEL to one that appears and supports generateContent.

2.Auth errors: ensure GEMINI_API_KEY is set in your shell (not committed).

3.Windows build errors (FAISS): use the in-memory retriever path included in pdfquery.py (toggle via a flag), or install faiss-cpu wheels as pinned in requirements.txt.

4.Blank/irrelevant answers: increase k, reduce chunk_size, or enable retrieved-chunk preview to verify the context is relevant.