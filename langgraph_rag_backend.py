from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LLM + Embeddings (OpenAI)
# ─────────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Persistent FAISS Vector Store
# ─────────────────────────────────────────────────────────────────────────────
VECTOR_STORE_DIR = Path("vector_stores")
VECTOR_STORE_DIR.mkdir(exist_ok=True)

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _vector_store_path(thread_id: str) -> Path:
    return VECTOR_STORE_DIR / str(thread_id)


def _load_existing_indexes() -> None:
    """Reload all saved FAISS indexes from disk on server startup."""
    for thread_dir in VECTOR_STORE_DIR.iterdir():
        if thread_dir.is_dir():
            index_file = thread_dir / "index.faiss"
            if index_file.exists():
                try:
                    vs = FAISS.load_local(
                        str(thread_dir),
                        embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    retriever = vs.as_retriever(
                        search_type="similarity", search_kwargs={"k": 4}
                    )
                    _THREAD_RETRIEVERS[thread_dir.name] = retriever
                    print(f"[INFO] Loaded FAISS index for thread: {thread_dir.name}")
                except Exception as exc:
                    print(f"[WARN] Could not load index for {thread_dir.name}: {exc}")


def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(
    file_bytes: bytes, thread_id: str, filename: Optional[str] = None
) -> dict:
    """
    Chunk → embed → persist the PDF as a FAISS index on disk.
    Metadata is stored in SQLite so it survives restarts.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)

        # ── Persist to disk ──────────────────────────────────────────────────
        save_path = _vector_store_path(thread_id)
        save_path.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(save_path))

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        meta = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _THREAD_METADATA[str(thread_id)] = meta
        _save_doc_metadata(str(thread_id), meta)

        return meta
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SQLite Setup  (chat history + document metadata + thread titles)
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = "chatbot.db"
_db_conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
_db_conn.execute("PRAGMA journal_mode=WAL")
_db_conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS document_metadata (
        thread_id   TEXT PRIMARY KEY,
        filename    TEXT,
        pages       INTEGER,
        chunks      INTEGER,
        uploaded_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS thread_titles (
        thread_id  TEXT PRIMARY KEY,
        title      TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """
)
_db_conn.commit()

checkpointer = SqliteSaver(conn=_db_conn)


def _save_doc_metadata(thread_id: str, meta: dict) -> None:
    _db_conn.execute(
        """INSERT OR REPLACE INTO document_metadata (thread_id, filename, pages, chunks)
           VALUES (?, ?, ?, ?)""",
        (thread_id, meta["filename"], meta["documents"], meta["chunks"]),
    )
    _db_conn.commit()


def get_thread_doc_metadata(thread_id: str) -> dict:
    """Return doc metadata — checks in-memory cache first, then SQLite."""
    tid = str(thread_id)
    if tid in _THREAD_METADATA:
        return _THREAD_METADATA[tid]
    row = _db_conn.execute(
        "SELECT filename, pages, chunks FROM document_metadata WHERE thread_id = ?",
        (tid,),
    ).fetchone()
    if row:
        meta = {"filename": row[0], "documents": row[1], "chunks": row[2]}
        _THREAD_METADATA[tid] = meta
        return meta
    return {}


def save_thread_title(thread_id: str, title: str) -> None:
    _db_conn.execute(
        "INSERT OR REPLACE INTO thread_titles (thread_id, title) VALUES (?, ?)",
        (str(thread_id), title),
    )
    _db_conn.commit()


def get_all_thread_titles() -> Dict[str, str]:
    rows = _db_conn.execute(
        "SELECT thread_id, title FROM thread_titles"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def generate_thread_title(thread_id: str) -> str:
    """Ask GPT to create a short title from the thread's first user message."""
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": str(thread_id)}}
        )
        messages = state.values.get("messages", [])
        first_human = next(
            (m.content for m in messages if isinstance(m, HumanMessage)), None
        )
        if not first_human:
            return "New Chat"

        resp = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Create a SHORT 3-5 word title for this chat based on the user's message. "
                        "Return ONLY the title — no quotes, no punctuation at end."
                    )
                ),
                HumanMessage(content=first_human[:500]),
            ]
        )
        title = resp.content.strip()[:60]
        save_thread_title(str(thread_id), title)
        return title
    except Exception:
        return "New Chat"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Tools
# ─────────────────────────────────────────────────────────────────────────────
_ddg = DuckDuckGoSearchRun(region="us-en")


@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time information, latest news, and current events.
    Use this for questions about recent happenings, general knowledge, or live data
    not available in an uploaded document.
    """
    return _ddg.run(query)


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price and market data for a ticker symbol.
    Examples: 'AAPL' for Apple, 'TSLA' for Tesla, 'GOOGL' for Google.
    Use this for any stock price or financial market queries.
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=DFHELFE6INYM1XFF"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json().get("Global Quote", {})
        if not data:
            return {"error": f"No market data found for symbol '{symbol}'."}
        return {
            "symbol": data.get("01. symbol"),
            "price": data.get("05. price"),
            "open": data.get("02. open"),
            "high": data.get("03. high"),
            "low": data.get("04. low"),
            "volume": data.get("06. volume"),
            "change": data.get("09. change"),
            "change_percent": data.get("10. change percent"),
            "latest_trading_day": data.get("07. latest trading day"),
        }
    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_weather(city: str) -> dict:
    """
    Get current weather conditions for any city worldwide.
    Use this for any weather-related queries.
    Example cities: 'Mumbai', 'London', 'New York'.
    """
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        c = r.json()["current_condition"][0]
        return {
            "city": city,
            "condition": c["weatherDesc"][0]["value"],
            "temperature_c": c["temp_C"],
            "temperature_f": c["temp_F"],
            "feels_like_c": c["FeelsLikeC"],
            "humidity_percent": c["humidity"],
            "wind_kmph": c["windspeedKmph"],
            "visibility_km": c["visibility"],
        }
    except Exception as exc:
        return {"error": f"Could not fetch weather for '{city}': {exc}"}


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform arithmetic on two numbers.
    Supported operations: add, sub, mul, div, pow, mod
    """
    try:
        ops = {
            "add": first_num + second_num,
            "sub": first_num - second_num,
            "mul": first_num * second_num,
            "div": first_num / second_num if second_num != 0 else None,
            "pow": first_num**second_num,
            "mod": first_num % second_num if second_num != 0 else None,
        }
        if operation not in ops:
            return {
                "error": f"Unknown operation '{operation}'. Use: add, sub, mul, div, pow, mod"
            }
        result = ops[operation]
        if result is None:
            return {"error": "Division / modulo by zero is not allowed."}
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as exc:
        return {"error": str(exc)}


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Search the user's uploaded PDF document for relevant information.
    ALWAYS use this tool when:
    - The user asks about the uploaded document, file, or PDF content
    - The user asks about an assignment, report, or any uploaded material
    - The user asks to summarize, explain, or find something in the document
    Do NOT answer from memory about document content — always use this tool.
    The thread_id is automatically provided — do NOT pass it manually.
    """
    # thread_id is INJECTED by LangGraph via RunnableConfig — the LLM does NOT need to pass it
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed for this thread. The user needs to upload a PDF first.",
            "query": query,
        }

    results = retriever.invoke(query)
    source_file = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "Unknown")

    return {
        "query": query,
        "source_file": source_file,
        "chunks": [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", ""),
            }
            for doc in results
        ],
    }


tools = [web_search, get_stock_price, get_weather, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  State + System Prompt
# ─────────────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


_SYSTEM_PROMPT = """\
You are a concise, expert AI assistant. You have access to these tools:
- rag_tool        → Search the user's uploaded PDF document
- web_search      → Real-time web search for news and current events
- get_stock_price → Live stock market prices and data
- get_weather     → Current weather for any city
- calculator      → Arithmetic operations (add, sub, mul, div, pow, mod)

CRITICAL RULES — FOLLOW THESE EXACTLY:
1. DOCUMENT QUESTIONS: If a document is shown as "AVAILABLE" below, you MUST call rag_tool for any question about that document. Do NOT answer from your training data about the document's content.
2. CURRENT DATE: Always use the exact date provided below. NEVER use your training data cutoff as today's date.
3. CONCISENESS: Keep answers SHORT and direct — 2-5 sentences unless the user explicitly requests detail.
4. FORMATTING: Use bullet points for lists. Use **bold** for key terms.
5. CITATIONS: After using rag_tool, always end your answer with: *📄 Source: [filename], Page [page number]*
6. NO FILLER: Never start with "Certainly!", "Of course!", "Great question!" or similar filler phrases.
7. LIVE DATA: For stock prices, weather, or news — always use the appropriate tool. Never guess.
"""


def chat_node(state: ChatState, config=None):
    """Core LLM node — answers directly or delegates to tools."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # ── Inject today's real date ──────────────────────────────────────────────
    current_date = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p IST")

    # ── Tell the LLM whether a document is available ──────────────────────────
    doc_meta = get_thread_doc_metadata(thread_id) if thread_id else {}
    if doc_meta:
        doc_context = (
            f"📄 DOCUMENT STATUS: AVAILABLE — '{doc_meta.get('filename')}' "
            f"({doc_meta.get('documents')} pages, {doc_meta.get('chunks')} chunks indexed). "
            f"You MUST use rag_tool to answer ANY question about this document's content."
        )
    else:
        doc_context = (
            "📄 DOCUMENT STATUS: NOT UPLOADED — No PDF indexed for this thread. "
            "If the user asks about a document, instruct them to upload a PDF using the sidebar."
        )

    system_content = (
        _SYSTEM_PROMPT
        + f"\n\n📅 TODAY'S DATE: {current_date}"
        + f"\n\n{doc_context}"
    )

    system = SystemMessage(content=system_content)
    response = llm_with_tools.invoke([system, *state["messages"]], config=config)
    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Graph
# ─────────────────────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Startup — reload persisted FAISS indexes
# ─────────────────────────────────────────────────────────────────────────────
_load_existing_indexes()

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Public helpers used by the frontend
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_all_threads() -> List[str]:
    """Return all thread IDs stored in the LangGraph checkpoint database."""
    all_threads: set[str] = set()
    for cp in checkpointer.list(None):
        all_threads.add(cp.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    """True if a FAISS index exists for this thread."""
    if str(thread_id) in _THREAD_RETRIEVERS:
        return True
    return _vector_store_path(str(thread_id)).exists()


def thread_document_metadata(thread_id: str) -> dict:
    return get_thread_doc_metadata(thread_id)