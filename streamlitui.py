"""
streamlitui.py — Professional redesign of the ChatPDF frontend.

Run with:
    streamlit run streamlitui.py
"""

import html as _html
import os
import tempfile

import streamlit as st

from pdfquery import PDFQuery

st.set_page_config(
    page_title="ChatPDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINANCE_PROMPTS = [
    "Summarize the key risks",
    "Extract all financial figures",
    "What was the revenue growth?",
    "List all executive names",
]

_TYPING_HTML = """
<div class="chat-row ai">
  <div class="avatar ai">✦</div>
  <div class="typing-bubble">
    <span></span><span></span><span></span>
  </div>
</div>
"""

# ---------------------------------------------------------------------------
# CSS — injected once at page load
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ── Global ──────────────────────────────────────────────────────────── */
#MainMenu, footer { visibility: hidden; }

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 6rem;
    max-width: 860px;
    margin: 0 auto;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a2744 100%) !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4   { color: #f1f5f9 !important; }
[data-testid="stSidebar"] hr   { border-color: #1e3a5f !important; }

[data-testid="stSidebar"] .stTextInput input {
    background: #0f2440 !important;
    border: 1px solid #2d4f7c !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.25) !important;
}

/* Sidebar action buttons (full-width) */
[data-testid="stSidebar"] .stButton > button {
    background: #1e3a8a !important;
    color: #bfdbfe !important;
    border: 1px solid #2563eb !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2563eb !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    opacity: 0.4 !important;
}

/* Finance chip buttons — buttons inside columns in sidebar */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button {
    background: rgba(30, 58, 138, 0.4) !important;
    color: #93c5fd !important;
    border: 1px solid #1d4ed8 !important;
    border-radius: 16px !important;
    font-size: 11.5px !important;
    font-weight: 500 !important;
    padding: 0.25rem 0.5rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #2563eb !important;
    color: #ffffff !important;
    border-color: #3b82f6 !important;
}

/* Sidebar toggle */
[data-testid="stSidebar"] .stToggle label { color: #cbd5e1 !important; }

/* File uploader in sidebar */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 2px dashed #2d4f7c !important;
    border-radius: 10px !important;
    background: rgba(30, 58, 138, 0.1) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #94a3b8 !important;
}

/* ── App header ───────────────────────────────────────────────────────── */
.app-header {
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #e2e8f0;
}
.app-header h1 {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.app-header .tagline {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
}

/* ── Empty states ─────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #94a3b8;
}
.empty-state .empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}
.empty-state p {
    font-size: 1rem;
    color: #64748b;
    max-width: 360px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Document summary box ─────────────────────────────────────────────── */
.summary-box {
    background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 14px 20px;
    margin: 0 0 1.25rem 0;
}
.summary-box .summary-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #1d4ed8;
    margin-bottom: 8px;
}
.summary-box ul {
    margin: 0;
    padding-left: 1.2em;
    color: #1e293b;
    font-size: 0.9rem;
    line-height: 1.7;
}
.summary-box p {
    color: #1e293b;
    font-size: 0.9rem;
    line-height: 1.7;
    margin: 0;
    white-space: pre-wrap;
}

/* ── Typing indicator row — .chat-row / .avatar used by _TYPING_HTML only */
.chat-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
    margin: 6px 0;
}
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
    font-style: normal;
}
.avatar.ai { background: #e0f2fe; color: #0369a1; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Chat bubbles (st.chat_message) ─────────────────────────────────── */
[data-testid="stChatMessage"] {
    animation: fadeUp 0.2s ease-out;
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
}
/* User: blue bubble, right-aligned */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    background: #2563eb;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    max-width: 74%;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] * {
    color: #ffffff !important;
}
/* AI: light gray bubble, left-aligned */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #1e293b;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] code {
    background: #e2e8f0;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.85em;
}

/* ── Typing indicator ─────────────────────────────────────────────────── */
.typing-bubble {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    display: flex;
    gap: 5px;
    align-items: center;
}
.typing-bubble span {
    display: inline-block;
    width: 7px;
    height: 7px;
    background: #94a3b8;
    border-radius: 50%;
    animation: bounce 1.3s infinite ease-in-out;
}
.typing-bubble span:nth-child(2) { animation-delay: 0.18s; }
.typing-bubble span:nth-child(3) { animation-delay: 0.36s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0);    background: #94a3b8; }
    30%            { transform: translateY(-7px); background: #3b82f6; }
}

/* ── Source expander ──────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    margin: 4px 0 12px 40px !important;
    background: #ffffff !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    color: #64748b !important;
    font-weight: 600 !important;
}

/* ── Model badge (sidebar) ────────────────────────────────────────────── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #052e16;
    color: #4ade80;
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 10.5px;
    font-family: "Courier New", monospace;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-top: 4px;
}

/* ── File badge (sidebar) ─────────────────────────────────────────────── */
.file-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #0f2440;
    color: #7dd3fc;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11.5px;
    margin: 3px 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _e(text: str) -> str:
    """HTML-escape user-supplied text before inserting into HTML."""
    return _html.escape(str(text))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session() -> None:
    defaults: dict = {
        "messages":          [],
        "GROQ_API_KEY":    os.environ.get("GROQ_API_KEY", ""),
        "pdfquery":          None,
        "uploaded_filenames": [],
        "finance_mode":      False,
        "doc_summary":       None,
        "suggested_questions": [],
        "pending_question":  None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _is_ready() -> bool:
    return (
        bool(st.session_state["GROQ_API_KEY"])
        and st.session_state["pdfquery"] is not None
        and st.session_state["pdfquery"].db is not None
    )


# ---------------------------------------------------------------------------
# Core actions
# ---------------------------------------------------------------------------

def _ingest_uploaded_files(files) -> None:
    """Ingest every uploaded PDF then generate the document summary."""
    pq = st.session_state["pdfquery"]
    pq.forget()
    st.session_state["messages"]           = []
    st.session_state["uploaded_filenames"] = []
    st.session_state["doc_summary"]        = None

    total    = len(files)
    progress = st.sidebar.progress(0, text="Preparing…")

    for i, file in enumerate(files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            path = tmp.name
        try:
            progress.progress((i + 0.3) / total, text=f"Embedding {file.name}…")
            pq.ingest(path)
            st.session_state["uploaded_filenames"].append(file.name)
        except Exception as exc:
            st.sidebar.error(f"{file.name}: {exc}")
        finally:
            os.unlink(path)
        progress.progress((i + 1) / total, text=f"Done {i + 1}/{total}")

    progress.empty()

    # Auto-generate 3-bullet summary
    try:
        summary = pq.get_document_summary()
        if summary:
            st.session_state["doc_summary"] = summary
    except Exception:
        pass

    # Auto-generate clickable example questions
    try:
        st.session_state["suggested_questions"] = pq.suggest_questions(3)
    except Exception:
        st.session_state["suggested_questions"] = []


def _process_question(question: str) -> None:
    """Run the RAG query with a live streaming answer, save results, rerun."""
    _render_user_bubble(question)

    # Last 6 messages give the model context for follow-up questions.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state["messages"][-6:]
    ]

    try:
        token_gen, sources = st.session_state["pdfquery"].ask_stream(
            question, history=history
        )
        with st.chat_message("assistant"):
            answer = st.write_stream(token_gen)  # live typing effect
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        return

    st.session_state["messages"].append({"role": "user",      "content": question, "sources": []})
    st.session_state["messages"].append({"role": "assistant", "content": answer,   "sources": sources})
    st.rerun()


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_user_bubble(text: str) -> None:
    with st.chat_message("user"):
        st.markdown(text)


def _render_ai_bubble(text: str) -> None:
    with st.chat_message("assistant"):
        st.markdown(text)


def _render_sources(sources: list) -> None:
    if not sources:
        return
    label = f"📚 {len(sources)} source chunk{'s' if len(sources) != 1 else ''} retrieved"
    with st.expander(label):
        for idx, src in enumerate(sources, start=1):
            fname = os.path.basename(src["source"])
            score = src.get("relevance")
            score_txt = f" · relevance {score}" if score is not None else ""
            st.caption(f"**Chunk {idx}** · {_e(fname)} · page {src['page']}{score_txt}")
            st.code(src["content"], language=None)
            if idx < len(sources):
                st.divider()


def _render_chat() -> None:
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            _render_user_bubble(msg["content"])
        else:
            _render_ai_bubble(msg["content"])
            _render_sources(msg.get("sources", []))


def _render_summary(summary: str) -> None:
    st.markdown(
        f"""
        <div class="summary-box">
          <div class="summary-label">✨ Document Overview</div>
          {_e(summary)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📄 ChatPDF")
        st.caption("RAG-powered document Q&A")
        st.divider()

        # ── API key ──────────────────────────────────────────────────────
        st.markdown("**Groq API Key**")
        api_key = st.text_input(
            "key",
            value=st.session_state["GROQ_API_KEY"],
            type="password",
            key="_api_key_input",
            label_visibility="collapsed",
            placeholder="gsk_…",
        )
        st.markdown(
            '<span class="model-badge">⚡ llama-3.3-70b-versatile</span>',
            unsafe_allow_html=True,
        )

        if api_key != st.session_state["GROQ_API_KEY"]:
            st.session_state["GROQ_API_KEY"]    = api_key
            st.session_state["pdfquery"]          = PDFQuery(api_key=api_key)
            st.session_state["messages"]          = []
            st.session_state["uploaded_filenames"] = []
            st.session_state["doc_summary"]       = None
            st.session_state["suggested_questions"] = []
            st.rerun()

        if not st.session_state["GROQ_API_KEY"]:
            st.warning("Enter your API key to get started.")
        elif st.session_state["pdfquery"] is None:
            st.session_state["pdfquery"] = PDFQuery(
                api_key=st.session_state["GROQ_API_KEY"]
            )

        st.divider()

        # ── Upload ───────────────────────────────────────────────────────
        st.markdown("**📂 Upload PDFs**")
        files = st.file_uploader(
            "pdfs",
            type=["pdf"],
            accept_multiple_files=True,
            disabled=not st.session_state["GROQ_API_KEY"],
            label_visibility="collapsed",
        )
        if st.button(
            "⚡ Process PDFs",
            disabled=not (st.session_state["GROQ_API_KEY"] and files),
            use_container_width=True,
        ):
            _ingest_uploaded_files(files)
            st.rerun()

        for fname in st.session_state["uploaded_filenames"]:
            st.markdown(
                f'<div class="file-badge">📄 {_e(fname)}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Finance Mode ─────────────────────────────────────────────────
        st.session_state["finance_mode"] = st.toggle(
            "💼 Finance Mode",
            value=st.session_state["finance_mode"],
            help="One-click prompts for financial document analysis",
        )

        if st.session_state["finance_mode"]:
            st.caption("Quick prompts — click to ask:")
            # 2-column chip grid
            cols = st.columns(2)
            for i, prompt in enumerate(FINANCE_PROMPTS):
                with cols[i % 2]:
                    if st.button(
                        prompt,
                        key=f"chip_{i}",
                        disabled=not _is_ready(),
                        use_container_width=True,
                    ):
                        # Set pending question; processed in main() after chat renders
                        st.session_state["pending_question"] = prompt

        st.divider()

        # ── Export chat ──────────────────────────────────────────────────
        if st.session_state["messages"]:
            export_lines = ["# ChatPDF conversation\n"]
            for m in st.session_state["messages"]:
                who = "**You**" if m["role"] == "user" else "**ChatPDF**"
                export_lines.append(f"{who}: {m['content']}\n")
            st.download_button(
                "⬇️ Export chat (.md)",
                data="\n".join(export_lines),
                file_name="chatpdf_conversation.md",
                mime="text/markdown",
                use_container_width=True,
            )

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["messages"]    = []
            st.session_state["doc_summary"] = None
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_session()
    _render_sidebar()

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="app-header">
          <h1>📄 ChatPDF</h1>
          <p class="tagline">
            Upload any PDF and ask questions — every answer is grounded in your document.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Guard: no API key ─────────────────────────────────────────────────
    if not st.session_state["GROQ_API_KEY"]:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">🔑</div>
              <p>Enter your <strong>Groq API key</strong> in the sidebar to begin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Guard: no documents loaded ────────────────────────────────────────
    if not _is_ready():
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">📂</div>
              <p>Upload one or more <strong>PDF files</strong> in the sidebar
                 and click <em>Process PDFs</em>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Document summary ──────────────────────────────────────────────────
    if st.session_state.get("doc_summary"):
        _render_summary(st.session_state["doc_summary"])

    # ── Suggested questions (shown until the first message is sent) ──────
    if st.session_state.get("suggested_questions") and not st.session_state["messages"]:
        st.caption("💡 Try asking:")
        for i, q in enumerate(st.session_state["suggested_questions"]):
            if st.button(q, key=f"suggested_{i}", use_container_width=True):
                st.session_state["pending_question"] = q

    # ── Chat history ──────────────────────────────────────────────────────
    _render_chat()

    # ── Process pending Finance chip question ─────────────────────────────
    # The chip sets pending_question in the same Streamlit run (sidebar renders
    # before main content), so it is available here without an extra rerun.
    if st.session_state.get("pending_question"):
        question = st.session_state["pending_question"]
        st.session_state["pending_question"] = None
        _process_question(question)
        return   # _process_question calls st.rerun() — nothing below runs

    # ── Chat input ────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask a question about your documents…"):
        _process_question(user_input)


if __name__ == "__main__":
    main()
