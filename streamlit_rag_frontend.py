"""
streamlit_rag_frontend.py
Multi-Agent RAG Chatbot — Premium Streamlit UI
"""
from __future__ import annotations

import ast
import json
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_rag_backend import (
    chatbot,
    generate_thread_title,
    get_all_thread_titles,
    get_thread_doc_metadata,
    ingest_pdf,
    retrieve_all_threads,
    save_thread_title,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark Glassmorphism Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── App background ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
        min-height: 100vh;
    }
    [data-testid="stMain"] { background: transparent; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 30, 0.97) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2) !important;
        backdrop-filter: blur(20px);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

    /* ── Sidebar text — LARGER sizes ── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { font-size: 0.92rem !important; }

    [data-testid="stSidebar"] h1 { font-size: 1.35rem !important; }
    [data-testid="stSidebar"] h2 { font-size: 1.15rem !important; }
    [data-testid="stSidebar"] h3 { font-size: 1.05rem !important; }

    /* ── Sidebar conversation buttons ── */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.18);
        border-radius: 10px;
        color: #d4c5fd;
        font-size: 0.88rem !important;
        font-weight: 500;
        text-align: left;
        width: 100%;
        padding: 0.65rem 1rem;
        margin-bottom: 5px;
        transition: all 0.2s ease;
        white-space: normal;
        word-wrap: break-word;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(139, 92, 246, 0.22);
        border-color: rgba(139, 92, 246, 0.45);
        color: #ede9fe;
        transform: translateX(3px);
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: rgba(20, 20, 40, 0.9) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 16px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: rgba(20, 20, 40, 0.55);
        border: 1px solid rgba(139, 92, 246, 0.1);
        border-radius: 14px;
        backdrop-filter: blur(10px);
        margin-bottom: 0.6rem;
    }

    /* ── Citation expanders ── */
    [data-testid="stExpander"] {
        background: rgba(10, 10, 30, 0.75);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 10px;
        margin: 4px 0 8px 0;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.85rem !important;
        color: #a78bfa !important;
    }

    /* ── Status / alert boxes ── */
    [data-testid="stAlert"] { border-radius: 10px; }

    /* ── Divider ── */
    hr { border-color: rgba(139, 92, 246, 0.15) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(139, 92, 246, 0.4); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(139, 92, 246, 0.65); }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(20, 20, 40, 0.5);
        border: 1px dashed rgba(139, 92, 246, 0.35);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* ── Tool badge styling ── */
    .tool-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.18);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        color: #c4b5fd;
        margin: 2px 4px 2px 0;
    }
    .tool-row {
        margin: 4px 0 10px 0;
        line-height: 2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tool Metadata
# ─────────────────────────────────────────────────────────────────────────────
TOOL_META = {
    "web_search":      ("🔍", "Web Search"),
    "get_stock_price": ("📈", "Stock Prices"),
    "get_weather":     ("🌤️", "Weather"),
    "calculator":      ("🧮", "Calculator"),
    "rag_tool":        ("📄", "Document RAG"),
}


def tool_label(name: str) -> str:
    emoji, label = TOOL_META.get(name, ("🔧", name))
    return f"{emoji} {label}"


def parse_tool_content(content: str) -> dict | None:
    """Robustly parse ToolMessage content (JSON or Python repr)."""
    if not isinstance(content, str):
        return content if isinstance(content, dict) else None
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(content)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def new_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(tid: str) -> None:
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)


def reset_chat() -> None:
    tid = new_thread_id()
    st.session_state["thread_id"] = tid
    st.session_state["message_history"] = []
    add_thread(tid)


def load_conversation(thread_id: str) -> list:
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ─────────────────────────────────────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = new_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = get_all_thread_titles()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = st.session_state["thread_id"]
selected_thread: str | None = None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Branding ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.2rem 0;">
            <h2 style="margin:0; font-size:1.4rem; font-weight:700;
                       background: linear-gradient(135deg,#a78bfa,#60a5fa);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                🤖 Agentic RAG
            </h2>
            <p style="margin:0.25rem 0 0 0; font-size:0.82rem; color:#6b7280;">
                GPT-4o-mini · FAISS · LangGraph
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("＋  New Chat", use_container_width=True, key="new_chat_btn"):
        reset_chat()
        st.rerun()

    st.markdown(
        f"<p style='font-size:0.72rem;color:#4b5563;margin:0.3rem 0;'>"
        f"Thread: <code style='color:#7c3aed;font-size:0.72rem;'>{thread_key[:22]}…</code></p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── PDF Upload ─────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.95rem;font-weight:600;color:#a78bfa;margin-bottom:0.5rem;'>"
        "📄 Document Context</p>",
        unsafe_allow_html=True,
    )

    existing_meta = get_thread_doc_metadata(thread_key)
    thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

    if existing_meta:
        st.success(
            f"**{existing_meta.get('filename')}**\n\n"
            f"📃 {existing_meta.get('documents')} pages  ·  "
            f"🧩 {existing_meta.get('chunks')} chunks"
        )
        uploaded_pdf = st.file_uploader(
            "Replace document", type=["pdf"], key=f"pdf_{thread_key}"
        )
    else:
        uploaded_pdf = st.file_uploader(
            "Upload a PDF", type=["pdf"], key=f"pdf_{thread_key}"
        )

    if uploaded_pdf:
        already_processed = uploaded_pdf.name in thread_docs
        if already_processed:
            st.info(f"`{uploaded_pdf.name}` already indexed for this session.")
        else:
            with st.status("⚙️ Indexing document…", expanded=True) as s:
                st.write("Chunking and embedding — this may take a moment.")
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                s.update(label="✅ Document indexed!", state="complete", expanded=False)
            st.rerun()

    st.divider()

    # ── Past Conversations ────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.95rem;font-weight:600;color:#a78bfa;margin-bottom:0.5rem;'>"
        "💬 Conversations</p>",
        unsafe_allow_html=True,
    )

    all_threads = st.session_state["chat_threads"][::-1]
    titles = st.session_state["thread_titles"]

    if not all_threads:
        st.markdown(
            "<p style='font-size:0.88rem;color:#4b5563;'>No conversations yet.</p>",
            unsafe_allow_html=True,
        )
    else:
        for tid in all_threads:
            display = titles.get(str(tid), str(tid)[:22] + "…")
            is_active = str(tid) == str(thread_key)
            label = ("▶ " if is_active else "   ") + display
            if st.button(label, key=f"thread_{tid}", use_container_width=True):
                selected_thread = str(tid)

    st.divider()

    # ── Tool Legend ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <p style='font-size:0.95rem;font-weight:600;color:#a78bfa;margin-bottom:0.4rem;'>🛠️ Available Tools</p>
        <div style='font-size:0.88rem;color:#9ca3af;line-height:2;'>
            🔍 Web Search &nbsp;·&nbsp; 📈 Stocks<br>
            🌤️ Weather &nbsp;·&nbsp; 🧮 Calculator<br>
            📄 Document RAG
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding:1rem 0 0.5rem 0;">
        <h1 style="margin:0;font-size:2rem;font-weight:700;
                   background:linear-gradient(135deg,#a78bfa 0%,#60a5fa 50%,#34d399 100%);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Multi-Agent RAG Chatbot
        </h1>
        <p style="margin:0.3rem 0 0 0;color:#6b7280;font-size:0.9rem;">
            Powered by GPT-4o-mini · FAISS Vector Search · LangGraph · 5 Specialized Tools
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Active doc badge
active_meta = get_thread_doc_metadata(thread_key)
if active_meta:
    st.markdown(
        f"<span style='background:rgba(52,211,153,0.15);border:1px solid rgba(52,211,153,0.3);"
        f"border-radius:20px;padding:4px 14px;font-size:0.82rem;color:#34d399;'>"
        f"📄 {active_meta.get('filename')} &nbsp;·&nbsp; "
        f"{active_meta.get('documents')} pages &nbsp;·&nbsp; "
        f"{active_meta.get('chunks')} chunks indexed</span>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<span style='background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);"
        "border-radius:20px;padding:4px 14px;font-size:0.82rem;color:#f59e0b;'>"
        "⚠️ No document indexed — upload a PDF in the sidebar to enable RAG</span>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Chat History Replay
# ─────────────────────────────────────────────────────────────────────────────
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        # Show tool badges (for assistant messages that used tools)
        if msg["role"] == "assistant" and msg.get("tools_used"):
            badges_html = "".join(
                f'<span class="tool-badge">{tool_label(t)} ✅</span>'
                for t in msg["tools_used"]
            )
            st.markdown(
                f'<div class="tool-row">🛠️ &nbsp;{badges_html}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

    # Citations below this message (outside the chat bubble)
    if msg.get("citations"):
        for cit in msg["citations"]:
            with st.expander(
                f"📄 Source: **{cit['source_file']}** · Page {cit['page']}",
                expanded=False,
            ):
                st.markdown(
                    f"<div style='font-size:0.85rem;color:#9ca3af;line-height:1.7;'>"
                    f"{cit['content'][:700]}{'…' if len(cit['content']) > 700 else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# Chat Input & Response Streaming
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about your document, get news, stocks, weather…")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input, "citations": [], "tools_used": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # ── Explicit streaming loop (reliable tool visibility + citations) ─────────
    with st.chat_message("assistant"):
        tool_indicator = st.empty()   # tool badges appear here
        response_area  = st.empty()   # streaming text appears here

        tools_used_this_turn: list[str] = []
        citations_this_turn: list[dict] = []
        full_response = ""

        for chunk, _ in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            # ── Tool result chunk ────────────────────────────────────────────
            if isinstance(chunk, ToolMessage):
                tname = getattr(chunk, "name", "tool")
                if tname not in tools_used_this_turn:
                    tools_used_this_turn.append(tname)

                # Update tool badge row
                badges_html = "".join(
                    f'<span class="tool-badge">{tool_label(t)} ✅</span>'
                    for t in tools_used_this_turn
                )
                tool_indicator.markdown(
                    f'<div class="tool-row">🛠️ &nbsp;{badges_html}</div>',
                    unsafe_allow_html=True,
                )

                # Parse RAG citations
                if tname == "rag_tool":
                    payload = parse_tool_content(chunk.content)
                    if payload and "chunks" in payload:
                        src = payload.get("source_file", "document")
                        for c in payload["chunks"]:
                            citations_this_turn.append(
                                {
                                    "source_file": src,
                                    "page": c.get("page", "N/A"),
                                    "content": c.get("content", ""),
                                }
                            )

            # ── AI text chunk → stream token by token ───────────────────────
            if isinstance(chunk, AIMessage) and chunk.content:
                full_response += chunk.content
                response_area.markdown(full_response + "▌")

        # Final render without cursor
        response_area.markdown(full_response)

    # ── Show citations directly below the chat bubble ────────────────────────
    for cit in citations_this_turn:
        with st.expander(
            f"📄 Source: **{cit['source_file']}** · Page {cit['page']}",
            expanded=False,
        ):
            st.markdown(
                f"<div style='font-size:0.85rem;color:#9ca3af;line-height:1.7;'>"
                f"{cit['content'][:700]}{'…' if len(cit['content']) > 700 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Save to message history ───────────────────────────────────────────────
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": full_response,
            "citations": citations_this_turn,
            "tools_used": tools_used_this_turn,
        }
    )

    # ── Generate smart title after first exchange ──────────────────────────────
    if len(st.session_state["message_history"]) == 2:
        title = generate_thread_title(thread_key)
        st.session_state["thread_titles"][str(thread_key)] = title

    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Thread Switching
# ─────────────────────────────────────────────────────────────────────────────
if selected_thread and selected_thread != str(thread_key):
    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(selected_thread)
    history = []
    for m in messages:
        if isinstance(m, HumanMessage):
            history.append({"role": "user", "content": m.content, "citations": [], "tools_used": []})
        elif isinstance(m, AIMessage) and m.content:
            history.append({"role": "assistant", "content": m.content, "citations": [], "tools_used": []})
    st.session_state["message_history"] = history
    st.session_state["ingested_docs"].setdefault(selected_thread, {})
    st.rerun()