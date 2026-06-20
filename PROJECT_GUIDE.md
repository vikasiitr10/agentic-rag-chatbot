# Multi-Agent RAG Chatbot — Project Guide & Interview Prep

## 1. Project Overview

This is a **production-grade, multi-agent Retrieval-Augmented Generation (RAG) chatbot** built with:

| Component | Technology |
|---|---|
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | text-embedding-3-small (OpenAI) |
| Vector Database | FAISS (persistent to disk) |
| Agent Framework | LangGraph |
| Memory / Checkpointing | SQLite (WAL mode) |
| Frontend | Streamlit |
| Web Search Tool | DuckDuckGo Search |
| Stock Data Tool | Alpha Vantage API |
| Weather Tool | wttr.in (free, no key required) |
| Deployment | Railway (persistent disk) |

---

## 2. Architecture Diagram

```
User (Browser / Streamlit UI)
         │
         │ HTTP
         ▼
┌─────────────────────────────────────────────┐
│           Streamlit Frontend                │
│  - Session management (thread_id)           │
│  - PDF uploader                             │
│  - Chat UI with streaming                   │
│  - Citation expanders                       │
│  - Smart conversation titles                │
└──────────────────┬──────────────────────────┘
                   │  Python import
                   ▼
┌─────────────────────────────────────────────┐
│         LangGraph Backend                   │
│                                             │
│  ┌──────────────┐   ┌──────────────────┐   │
│  │  chat_node   │──▶│   ToolNode       │   │
│  │ (GPT-4o-mini)│◀──│ (tool executor) │   │
│  └──────────────┘   └──────┬───────────┘   │
│         │                  │               │
│         ▼                  ▼               │
│  ┌──────────────┐   ┌──────────────────┐   │
│  │  SQLite DB   │   │   Tools          │   │
│  │ (WAL mode)   │   │ - web_search     │   │
│  │ - chat hist  │   │ - get_stock_price│   │
│  │ - doc meta   │   │ - get_weather    │   │
│  │ - titles     │   │ - calculator     │   │
│  └──────────────┘   │ - rag_tool       │   │
│                     └──────┬───────────┘   │
│                            │               │
│                     ┌──────▼───────────┐   │
│                     │   FAISS Index    │   │
│                     │ (disk-persisted) │   │
│                     │ vector_stores/   │   │
│                     │ {thread_id}/     │   │
│                     └──────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## 3. Key Concepts — Deep Dives

### 3.1 What is RAG (Retrieval-Augmented Generation)?

RAG is a technique that enhances LLM responses with **relevant context retrieved from a knowledge base** (e.g., a PDF document).

**Without RAG:** LLM only uses its training data → can hallucinate or have outdated info.
**With RAG:** LLM receives relevant document chunks as context → grounded, accurate answers.

**RAG Pipeline in this project:**
```
PDF Upload
    ↓
PyPDFLoader (extract text, page by page)
    ↓
RecursiveCharacterTextSplitter
  chunk_size=1000, chunk_overlap=200
    ↓
OpenAI Embeddings (text-embedding-3-small)
  → converts each chunk to a 1536-dim vector
    ↓
FAISS Index (stored in vector_stores/{thread_id}/)
    ↓
User Query → embed query → cosine similarity search
    ↓
Top 4 most relevant chunks retrieved
    ↓
Chunks + user query sent to GPT-4o-mini
    ↓
Answer with source citations (page number, filename)
```

### 3.2 Why LangGraph over plain LangChain?

| Feature | LangChain Chains | LangGraph |
|---|---|---|
| Flow control | Linear (A→B→C) | Graph-based (any node → any node) |
| Loops / cycles | Not supported | ✅ Supported (agent loops back to LLM after tool use) |
| State management | Manual | ✅ Built-in `TypedDict` state |
| Memory persistence | Manual | ✅ Built-in checkpointing |
| Multi-agent | Complex | ✅ Native support |
| Conditional routing | Limited | ✅ `add_conditional_edges` |

**In this project:** After GPT-4o-mini decides to call a tool, LangGraph routes to `ToolNode`, executes the tool, then loops back to `chat_node` with the result. This cycle continues until GPT produces a final answer.

### 3.3 FAISS — How Vector Search Works

**FAISS (Facebook AI Similarity Search):**
- An open-source library for efficient nearest-neighbor search in high-dimensional spaces.
- Stores document chunk embeddings as vectors.
- At query time, converts the user's query to a vector, then finds the K closest vectors using **cosine similarity** or **L2 distance**.

**Persistence in this project:**
- `vector_store.save_local("vector_stores/{thread_id}/")` → creates `index.faiss` + `index.pkl`
- On server restart: `FAISS.load_local(...)` reloads all indexes
- This means users don't need to re-upload documents after server restarts

**Alternative vector DBs (for interview):**
| DB | Type | Best For |
|---|---|---|
| FAISS | In-process, disk-persisted | Small-medium datasets, single server |
| ChromaDB | File-based / client-server | Local persistence, easy setup |
| Pinecone | Fully managed cloud | Production at scale, no infra |
| Weaviate | Self-hosted / cloud | Hybrid search (vector + keyword) |
| pgvector | PostgreSQL extension | Teams already using Postgres |

### 3.4 SQLite Checkpointing (LangGraph Memory)

LangGraph's `SqliteSaver` stores the **entire conversation state** (all messages) for every thread in SQLite.

**What's stored:**
- `checkpoint` table: serialized `ChatState` (messages list) per thread
- `document_metadata` table: filename, pages, chunks for each thread
- `thread_titles` table: AI-generated conversation titles

**WAL Mode (`PRAGMA journal_mode=WAL`):**
- WAL = Write-Ahead Logging
- Allows concurrent reads during writes (no read lock during write)
- More crash-safe than default journal mode
- Critical for production multi-user scenarios

### 3.5 Tool Calling — How It Works

1. LLM receives the user's message + system prompt listing available tools
2. GPT-4o-mini decides if a tool is needed → outputs a JSON `tool_call` object (not text)
3. LangGraph's `tools_condition` detects `tool_call` → routes to `ToolNode`
4. `ToolNode` executes the matching Python function with the LLM-provided arguments
5. Result is appended to the message history as a `ToolMessage`
6. Execution loops back to `chat_node` → LLM sees the tool result and formulates a response

**Key design choice:** Tools are normal Python functions decorated with `@tool`. The LLM sees their name, docstring, and parameter types as its "API spec." This is why **writing good docstrings is critical** for tool reliability.

---

## 4. Interview Questions & Answers

### Q1: What is the difference between a RAG system and fine-tuning?
**A:** Fine-tuning modifies the model's weights with new data — expensive, requires retraining, static after training. RAG keeps the model frozen and retrieves relevant context at inference time — cheap, dynamic, updatable without retraining. RAG is preferred for knowledge that changes frequently (documents, news).

### Q2: How does chunking strategy affect RAG quality?
**A:** Chunk size determines the granularity of retrieved context. Too large → chunks include irrelevant text and hit token limits. Too small → chunks lack enough context to be useful. We use `chunk_size=1000, chunk_overlap=200` — overlap ensures information at chunk boundaries isn't lost. The `RecursiveCharacterTextSplitter` tries to split on natural boundaries (paragraph, sentence, word) before splitting mid-word.

### Q3: Why use cosine similarity for vector search?
**A:** Cosine similarity measures the angle between two vectors, not their magnitude. This makes it robust to text length differences — a short query can still match a long document chunk if they discuss the same topic. Euclidean (L2) distance is sensitive to vector magnitude, which can disadvantage shorter texts.

### Q4: How does LangGraph handle agent loops?
**A:** LangGraph uses a directed graph with conditional edges. `tools_condition` checks the LLM's output: if it contains a `tool_call`, route to `ToolNode`; if it's a final text response, end the graph. `ToolNode` always routes back to `chat_node`, creating a loop. The loop terminates when the LLM produces a plain text response.

### Q5: What is the role of `thread_id` in this system?
**A:** `thread_id` is the key for multi-session memory. Each conversation has a unique UUID. The SQLite checkpointer uses `thread_id` to store/retrieve conversation state. FAISS indexes are namespaced by `thread_id` in `vector_stores/{thread_id}/`. This means each user session is completely isolated — different documents, different conversation history.

### Q6: What are embeddings and why does model choice matter?
**A:** Embeddings are dense numerical vector representations of text where semantically similar texts have vectors close together in high-dimensional space. `text-embedding-3-small` produces 1536-dimensional vectors. Model choice matters because the query and document chunks must be embedded with the **same model** — mixing models breaks the similarity search since vectors live in different spaces.

### Q7: How do you handle tool failures in production?
**A:** Each tool has try/except blocks that return an error dict instead of raising exceptions. The LLM receives the error as a `ToolMessage` and can communicate it to the user or try an alternative approach. For example, if weather API is down, it returns `{"error": "Could not fetch weather..."}` and the LLM responds gracefully.

### Q8: Why SQLite instead of a cloud database for checkpointing?
**A:** SQLite with WAL mode is sufficient for single-server deployments with low-to-medium concurrency. It requires zero infrastructure, no network calls, and is deployed as a file. For high-concurrency production systems, you'd migrate to PostgreSQL using `langgraph-checkpoint-postgres`. The abstraction layer makes this a one-line change.

### Q9: What makes this "agentic" vs. a simple chatbot?
**A:** An agentic system autonomously decides *which tools to use* and *in what order* based on the user's intent, without hard-coded if/else logic. This chatbot can: search the web, look up stocks, check weather, do math, and search a document — all in a single turn if needed. The LLM orchestrates these tools dynamically. A simple chatbot just takes input → LLM → output.

### Q10: How does streaming work with tool calls?
**A:** LangGraph's `stream(stream_mode="messages")` yields message chunks as they're generated. We differentiate between `AIMessage` chunks (yield to UI for streaming text) and `ToolMessage` objects (show tool status indicator). The `st.write_stream()` call in Streamlit renders the AI text token-by-token as it arrives from OpenAI's streaming API.

### Q11: How would you scale this to multiple users?
**A:** Replace SQLite with PostgreSQL (`langgraph-checkpoint-postgres`) for concurrent writes. Replace local FAISS with a managed vector DB (Pinecone or pgvector) for shared, scalable retrieval. Deploy with multiple Gunicorn workers behind an nginx proxy. Add Redis for session caching. Use S3/GCS for raw PDF storage instead of the local filesystem.

### Q12: What are the limitations of FAISS persistence to disk?
**A:** (1) Single-server only — FAISS indexes can't be shared across multiple instances without copying files. (2) If the server's disk is wiped (e.g., Streamlit Community Cloud), indexes are lost. (3) Large numbers of threads create many small directories. Production fix: store FAISS indexes in S3 or switch to Pinecone.

---

## 5. Future Improvements (Roadmap)

| Improvement | Technical Approach |
|---|---|
| Multi-user authentication | Add Streamlit authenticator or Auth0 |
| Multiple PDFs per thread | Merge FAISS indexes with `FAISS.merge_from()` |
| Hybrid search (keyword + vector) | Use Weaviate or add BM25 re-ranking |
| Token usage / cost tracking | Use LangSmith or parse `response.usage` from OpenAI |
| PostgreSQL for production | Swap `SqliteSaver` → `PostgresSaver` |
| Cloud vector DB | Swap `FAISS` → `PineconeVectorStore` |
| Export chat as PDF | Use `reportlab` or `weasyprint` |
| Multi-modal: image Q&A | Use GPT-4o Vision with `PIL` image extraction |
| Voice input | Integrate Whisper API via `st.audio_input` |
| Evaluation pipeline | LangSmith traces + RAGAS metrics for RAG quality |

---

## 6. Running the Project

### Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
echo 'OPENAI_API_KEY=sk-proj-...' > .env

# 3. Run
streamlit run streamlit_rag_frontend.py
```

### Production (Railway)
```bash
# 1. Push to GitHub (vector_stores/ and .env are gitignored)
git add . && git commit -m "deploy" && git push

# 2. On railway.app → New Project → Deploy from GitHub
# 3. Set OPENAI_API_KEY in Variables tab
# 4. Add a Volume mounted at /app for persistence
# 5. Railway auto-runs: Procfile → web command
```

---

## 7. Key Files Reference

| File | Purpose |
|---|---|
| `langgraph_rag_backend.py` | LLM, embeddings, FAISS persistence, all tools, LangGraph graph, SQLite tables |
| `streamlit_rag_frontend.py` | Streamlit UI, session management, streaming, citations |
| `requirements.txt` | Python dependencies for deployment |
| `Procfile` | Railway/Heroku startup command |
| `.gitignore` | Excludes secrets, DB, vectors from git |
| `vector_stores/` | FAISS index files (one folder per thread) — NOT in git |
| `chatbot.db` | SQLite database — NOT in git |
| `.env` | API keys — NEVER in git |
