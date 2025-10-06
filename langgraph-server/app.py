#!/usr/bin/env python3

import asyncio
import json
import os
import pathlib
import sqlite3
import time
import uuid
import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Environment / model setup
# ---------------------------------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY environment variable must be set before starting the server."
    )

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MODEL_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DOCS_PATH = pathlib.Path(os.getenv("DOCS_PATH", "data/critical_role"))
DB_PATH = pathlib.Path(os.getenv("CHAT_DB_PATH", "langgraph.db"))
DB_TIMEOUT = float(os.getenv("CHAT_DB_TIMEOUT", "30"))
DB_BUSY_TIMEOUT_MS = int(os.getenv("CHAT_DB_BUSY_TIMEOUT_MS", "5000"))
DB_LOCK_RETRY_ATTEMPTS = int(os.getenv("CHAT_DB_LOCK_RETRY_ATTEMPTS", "5"))
DB_LOCK_RETRY_DELAY = float(os.getenv("CHAT_DB_LOCK_RETRY_DELAY", "0.1"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "50"))

DEBUG_LOGGING_ENABLED = os.getenv("CHAT_DEBUG_LOG", "false").lower() == "true"

logger = logging.getLogger("langgraph-server")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if DEBUG_LOGGING_ENABLED else logging.INFO)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=MODEL_TEMPERATURE)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a thorough and knowledgeable assistant with access to the full conversation history. "
            "When answering questions:\n"
            "- Reference previous messages and topics discussed in the conversation\n"
            "- Build upon prior context and information shared\n"
            "- Use the supplied reference context to provide detailed, accurate responses\n"
            "- Cite concrete details from the context where relevant\n"
            "- Acknowledge uncertainty instead of guessing\n"
            "- Maintain continuity across the conversation thread\n\n"
            "Reference Context:\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chat_chain = prompt | llm


def call_model(state: MessagesState) -> Dict[str, List[AIMessage]]:
    """Augment the latest user question with retrieved context, then answer."""
    messages = state["messages"]
    query_text = extract_last_user_question(messages)
    retrieved_docs = retriever.search(query_text) if query_text else []
    context = build_context_string(retrieved_docs)
    ai_msg = chat_chain.invoke({"messages": messages, "context": context})
    return {"messages": [ai_msg]}

# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, vectorstore: Optional[FAISS]):
        self.vectorstore = vectorstore

    def search(self, query: str) -> List[Document]:
        if not self.vectorstore or not query.strip():
            return []
        try:
            return self.vectorstore.similarity_search(query, k=4)
        except Exception as exc:  # pragma: no cover
            print(f"[Retriever] similarity_search failed: {exc}")
            return []


def load_documents() -> List[Document]:
    if not DOCS_PATH.exists():
        print(f"[Retriever] Skipping load; path not found: {DOCS_PATH}")
        return []
    if DOCS_PATH.is_file():
        return TextLoader(str(DOCS_PATH), encoding="utf-8").load()
    loader = DirectoryLoader(
        str(DOCS_PATH),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        silent_errors=True,
    )
    docs = loader.load()
    if not docs:
        print(f"[Retriever] No documents found under {DOCS_PATH}")
    return docs


def build_vectorstore(docs: List[Document]) -> Optional[FAISS]:
    if not docs:
        return None
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        print(f"[Retriever] Indexed {len(docs)} documents into FAISS store.")
        return vectorstore
    except Exception as exc:
        print(f"[Retriever] Failed to build FAISS index: {exc}")
        return None


def extract_last_user_question(messages: List[Any]) -> str:
    for message in reversed(messages or []):
        if isinstance(message, dict):
            msg_type = message.get("type") or message.get("role")
            if msg_type in {"human", "user"}:
                return render_text(message.get("content", ""))
        elif isinstance(message, HumanMessage):
            return message.content
    return ""


def build_context_string(docs: List[Document]) -> str:
    if not docs:
        return "No additional reference documents were retrieved for this question."
    formatted = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", f"doc_{idx}")
        formatted.append(f"[{source}] {doc.page_content.strip()}")
    return "\n\n".join(formatted)


retriever = Retriever(build_vectorstore(load_documents()))

# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------

LOCKED_ERROR_SUBSTRINGS = ("database is locked", "database is busy")
T = TypeVar("T")


def run_with_lock_retry(operation: Callable[[], T]) -> T:
    op_name = getattr(operation, "__name__", repr(operation))
    for attempt in range(DB_LOCK_RETRY_ATTEMPTS):
        try:
            result = operation()
            if DEBUG_LOGGING_ENABLED:
                logger.debug("run_with_lock_retry succeeded (attempt %s) for %s", attempt + 1, op_name)
            return result
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            logger.warning("run_with_lock_retry hit sqlite error on attempt %s for %s: %s", attempt + 1, op_name, exc)
            if (
                not any(token in message for token in LOCKED_ERROR_SUBSTRINGS)
                or attempt == DB_LOCK_RETRY_ATTEMPTS - 1
            ):
                raise
            delay = DB_LOCK_RETRY_DELAY * (attempt + 1)
            logger.debug("Retrying %s after %.2fs delay due to lock", op_name, delay)
            time.sleep(delay)
    raise RuntimeError("Database operation retry exhausted")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_PATH,
        timeout=DB_TIMEOUT,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {DB_BUSY_TIMEOUT_MS}")
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL").fetchone()
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                message_json TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_thread_created
                ON messages(thread_id, created_at);
            """
        )


def thread_from_row(
    row: sqlite3.Row,
    include_messages: bool = True,
    message_limit: Optional[int] = None,
) -> Dict[str, Any]:
    thread = {
        "thread_id": row["thread_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "metadata": json.loads(row["metadata"] or "{}"),
        "status": "idle",
    }
    if include_messages:
        thread["values"] = {
            "messages": fetch_messages(row["thread_id"], limit=message_limit)
        }
    return thread


def get_thread_row(
    thread_id: str,
    *,
    include_messages: bool = True,
    message_limit: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    def load():
        with get_db_connection() as conn:
            return conn.execute(
                "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()

    row = run_with_lock_retry(load)
    if DEBUG_LOGGING_ENABLED:
        logger.debug("get_thread_row(%s) -> %s", thread_id, "hit" if row else "miss")
    if not row:
        return None
    return thread_from_row(
        row, include_messages=include_messages, message_limit=message_limit
    )


def upsert_thread(
    thread_id: str, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    existing = get_thread_row(thread_id, include_messages=False)
    now = utc_now_iso()
    effective_metadata = metadata or (existing["metadata"] if existing else {})
    metadata_json = json.dumps(effective_metadata)
    if existing:
        def do_update():
            with get_db_connection() as conn:
                conn.execute(
                    "UPDATE threads SET updated_at = ?, metadata = ? WHERE thread_id = ?",
                    (now, metadata_json, thread_id),
                )

        run_with_lock_retry(do_update)
    else:
        def do_insert():
            with get_db_connection() as conn:
                conn.execute(
                    (
                        "INSERT INTO threads(thread_id, created_at, updated_at, metadata) "
                        "VALUES (?, ?, ?, ?)"
                    ),
                    (thread_id, now, now, metadata_json),
                )

        run_with_lock_retry(do_insert)
    if DEBUG_LOGGING_ENABLED:
        logger.debug("upsert_thread(%s) -> existing=%s metadata_keys=%s", thread_id, bool(existing), sorted((effective_metadata or {}).keys()))
    return get_thread_row(thread_id)


def merge_thread_metadata(thread_id: str, metadata: Optional[Dict[str, Any]]):
    if not metadata:
        return
    existing = get_thread_row(thread_id, include_messages=False)
    base = existing["metadata"] if existing else {}
    base.update(metadata)
    upsert_thread(thread_id, base)


def list_threads_db(
    limit: int,
    offset: int,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    def run_query(filter_obj: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        where_clauses: List[str] = []
        params: List[Any] = []
        query = "SELECT * FROM threads"
        if filter_obj:
            for key, value in filter_obj.items():
                where_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY datetime(updated_at) DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        def query_db():
            with get_db_connection() as conn:
                return conn.execute(query, params).fetchall()

        rows = run_with_lock_retry(query_db)
        if DEBUG_LOGGING_ENABLED:
            logger.debug("list_threads_db(%s) returned %s rows", filter_obj, len(rows))
        return [
            thread_from_row(row, include_messages=True, message_limit=MAX_CONTEXT_MESSAGES)
            for row in rows
        ]

    threads = run_query(metadata_filter)
    if metadata_filter and not threads:
        threads = run_query(None)
    return threads


def delete_thread_db(thread_id: str):
    def delete():
        with get_db_connection() as conn:
            conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

    run_with_lock_retry(delete)


def persist_messages(thread_id: str, messages: List[Dict[str, Any]]):
    if not messages:
        return
    rows = []
    for message in messages:
        message_id = message.get("id") or str(uuid.uuid4())
        message["id"] = message_id
        created_at = message.get("created_at", utc_now_iso())
        message["created_at"] = created_at
        rows.append((message_id, thread_id, created_at, json.dumps(message)))
    def write():
        with get_db_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO messages(id, thread_id, created_at, message_json)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    if DEBUG_LOGGING_ENABLED:
        logger.debug("persist_messages(%s) -> %s messages", thread_id, len(rows))
    run_with_lock_retry(write)


def fetch_messages(thread_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = (
        "SELECT message_json FROM messages "
        "WHERE thread_id = ? ORDER BY datetime(created_at) ASC"
    )
    params: List[Any] = [thread_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    def query_db():
        with get_db_connection() as conn:
            return conn.execute(query, params).fetchall()

    rows = run_with_lock_retry(query_db)
    if DEBUG_LOGGING_ENABLED:
        logger.debug("fetch_messages(%s, limit=%s) -> %s rows", thread_id, limit, len(rows))
    return [json.loads(row["message_json"]) for row in rows]


def touch_thread(thread_id: str):
    now = utc_now_iso()
    def update_thread():
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id)
            )

    run_with_lock_retry(update_thread)


init_db()

# ---------------------------------------------------------------------------
# FastAPI + LangGraph wiring
# ---------------------------------------------------------------------------

app = FastAPI(title="LangGraph Chatbot Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_graph = StateGraph(MessagesState)
agent_graph.add_node("chat_model", call_model)
agent_graph.add_edge(START, "chat_model")
agent_graph.add_edge("chat_model", END)
agent = agent_graph.compile()


def utc_now_iso() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def render_text(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n\n".join(filter(None, parts))
    if isinstance(content, str):
        return content
    return str(content)


def to_lc_message(message: Dict[str, Any]):
    msg_type = message.get("type") or message.get("role") or "human"
    content = render_text(message.get("content", ""))
    if msg_type in {"system"}:
        return SystemMessage(content=content)
    if msg_type in {"ai", "assistant"}:
        return AIMessage(content=content)
    return HumanMessage(content=content)


def to_ui_message(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        raw = message.model_dump()
    elif hasattr(message, "dict"):
        raw = message.dict()
    elif isinstance(message, dict):
        raw = dict(message)
    else:
        raw = {"type": "ai", "content": [{"type": "text", "text": str(message)}]}

    content = raw.get("content", "")
    if isinstance(content, str):
        content_blocks = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                content_blocks.append({"type": "text", "text": block.get("text", "")})
            elif isinstance(block, dict):
                content_blocks.append(block)
            else:
                content_blocks.append({"type": "text", "text": str(block)})
    else:
        content_blocks = [{"type": "text", "text": str(content)}]

    msg_type = raw.get("type") or ("ai" if raw.get("role") == "assistant" else "human")
    if msg_type == "assistant":
        msg_type = "ai"
    role = raw.get("role")
    if not role:
        role = "assistant" if msg_type == "ai" else "user" if msg_type == "human" else msg_type

    raw.update(
        {
            "id": raw.get("id", str(uuid.uuid4())),
            "type": msg_type,
            "role": role,
            "content": content_blocks,
            "created_at": raw.get("created_at", utc_now_iso()),
        }
    )
    return raw


def extract_metadata_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if isinstance(payload.get("metadata"), dict):
        metadata.update(payload["metadata"])

    # Try to extract assistant_id/graph_id from root level
    assistant_id = payload.get("assistant_id")
    graph_id = payload.get("graph_id")
    
    # Also check in config.configurable if present
    config = payload.get("config", {})
    if isinstance(config, dict):
        configurable = config.get("configurable", {})
        if isinstance(configurable, dict):
            assistant_id = assistant_id or configurable.get("assistant_id")
            graph_id = graph_id or configurable.get("graph_id")

    if assistant_id:
        metadata.setdefault("assistant_id", assistant_id)
    if graph_id:
        metadata.setdefault("graph_id", graph_id)

    # If only one identifier is present, mirror it so filtering works for both patterns.
    if assistant_id and "graph_id" not in metadata:
        metadata["graph_id"] = assistant_id
    if graph_id and "assistant_id" not in metadata:
        metadata["assistant_id"] = graph_id

    return metadata


async def parse_json_body(request: Request) -> Dict[str, Any]:
    raw_payload = await request.body()
    if not raw_payload:
        return {}
    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")


class ThreadCreate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None


class ThreadUpdate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None


class RunCreate(BaseModel):
    input: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    multitask_strategy: Optional[str] = None


@app.get("/")
def root():
    return {"message": "LangGraph Server is running"}


@app.get("/info")
def info():
    return {
        "name": "LangChain Chatbot",
        "description": "LangGraph + GPT-4o + RAG agent",
        "version": "2.0.2",
        "model": OPENAI_MODEL,
    }


@app.get("/assistants/{assistant_id}")
def get_assistant(assistant_id: str):
    return upsert_thread(assistant_id)


@app.get("/threads")
def list_threads(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    metadata: Optional[str] = Query(None),
):
    metadata_filter = json.loads(metadata) if metadata else None
    return list_threads_db(limit=limit, offset=offset, metadata_filter=metadata_filter)


@app.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    thread = get_thread_row(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


@app.patch("/threads/{thread_id}")
async def update_thread(thread_id: str, request: Request):
    existing = get_thread_row(thread_id, include_messages=False)
    if not existing:
        raise HTTPException(status_code=404, detail="Thread not found")
    payload = await parse_json_body(request)
    thread_update = ThreadUpdate.model_validate(payload)
    metadata = existing["metadata"]
    if thread_update.metadata:
        metadata.update(thread_update.metadata)
    merge_thread_metadata(thread_id, metadata)
    return get_thread_row(thread_id)


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    existing = get_thread_row(thread_id, include_messages=False)
    if not existing:
        raise HTTPException(status_code=404, detail="Thread not found")
    delete_thread_db(thread_id)
    return {"message": "Thread deleted successfully"}


@app.post("/threads")
async def create_thread(request: Request):
    data = await parse_json_body(request)
    base_metadata = extract_metadata_from_payload(data)
    thread_data = ThreadCreate.model_validate(data)
    merged = thread_data.metadata or {}
    merged.update(base_metadata)
    thread_id = str(uuid.uuid4())
    if DEBUG_LOGGING_ENABLED:
        logger.debug("create_thread generated %s with metadata=%s", thread_id, merged)
    return upsert_thread(thread_id, merged)


@app.get("/threads/search")
@app.post("/threads/search")
async def search_threads(
    request: Request,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    metadata: Optional[str] = None,
):
    body: Dict[str, Any] = {}
    if request.method == "POST":
        body = await parse_json_body(request)
        limit = limit if limit is not None else body.get("limit")
        offset = offset if offset is not None else body.get("offset")
        metadata_filter = body.get("metadata")
    else:
        metadata_filter = json.loads(metadata) if metadata else None

    limit = int(limit) if limit is not None else 10
    offset = int(offset) if offset is not None else 0
    logger.info("search_threads with metadata_filter: %s", metadata_filter)
    result = list_threads_db(limit=limit, offset=offset, metadata_filter=metadata_filter)
    logger.info("search_threads returned %s threads", len(result))
    return result


@app.get("/threads/{thread_id}/state")
def get_thread_state(thread_id: str):
    return checkpoint_payload(thread_id)


@app.get("/threads/{thread_id}/history")
@app.post("/threads/{thread_id}/history")
def get_thread_history(
    thread_id: str,
    limit: int = Query(100, ge=1, le=200),
    before: Optional[str] = Query(None),
):
    if not get_thread_row(thread_id, include_messages=False):
        if DEBUG_LOGGING_ENABLED:
            logger.debug("get_thread_history(%s) -> thread missing", thread_id)
        return []
    checkpoint = checkpoint_payload(thread_id, limit=limit)
    if DEBUG_LOGGING_ENABLED:
        logger.debug("get_thread_history(%s, limit=%s) -> %s messages", thread_id, limit, len(checkpoint.get("values", {}).get("messages", [])))
    return [checkpoint]


def checkpoint_payload(thread_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
    # Use MAX_CONTEXT_MESSAGES if no limit specified
    effective_limit = limit if limit is not None else MAX_CONTEXT_MESSAGES
    messages = fetch_messages(thread_id, limit=effective_limit)
    return {
        "checkpoint_id": f"{thread_id}-latest",
        "parent_checkpoint_id": None,
        "created_at": utc_now_iso(),
        "config": {"configurable": {"thread_id": thread_id}},
        "metadata": {},
        "values": {"messages": messages},
    }


@app.post("/threads/{thread_id}/runs")
async def create_run(
    thread_id: str, 
    request: Request,
    assistant_id: Optional[str] = None,
    graph_id: Optional[str] = None,
):
    if not get_thread_row(thread_id, include_messages=False):
        # Create thread with metadata from query params if available
        initial_metadata = {}
        if assistant_id:
            initial_metadata["assistant_id"] = assistant_id
            initial_metadata["graph_id"] = assistant_id
        elif graph_id:
            initial_metadata["graph_id"] = graph_id
            initial_metadata["assistant_id"] = graph_id
        upsert_thread(thread_id, initial_metadata if initial_metadata else None)

    run_payload = await parse_json_body(request)
    run_data = RunCreate.model_validate(run_payload)

    metadata_update = extract_metadata_from_payload(run_payload)
    # Also use query params if present
    if assistant_id and "assistant_id" not in metadata_update:
        metadata_update["assistant_id"] = assistant_id
        metadata_update["graph_id"] = assistant_id
    elif graph_id and "graph_id" not in metadata_update:
        metadata_update["graph_id"] = graph_id
        metadata_update["assistant_id"] = graph_id
    
    if run_data.metadata:
        metadata_update.update(run_data.metadata)
    merge_thread_metadata(thread_id, metadata_update)

    if DEBUG_LOGGING_ENABLED:
        logger.debug("create_run(%s) metadata_update=%s", thread_id, metadata_update)

    run_id = str(uuid.uuid4())
    messages = run_data.input.get("messages", [])

    if DEBUG_LOGGING_ENABLED:
        logger.debug("create_run(%s) received %s incoming messages", thread_id, len(messages))

    # Convert and persist incoming messages
    ui_messages = [to_ui_message(msg) for msg in messages]
    persist_messages(thread_id, ui_messages)
    
    # Load full conversation history from the thread
    all_messages = fetch_messages(thread_id, limit=MAX_CONTEXT_MESSAGES)
    if DEBUG_LOGGING_ENABLED:
        logger.debug("create_run(%s) loaded %s historical messages", thread_id, len(all_messages))
    
    # Convert all messages to LangChain format for the agent
    lc_messages = [to_lc_message(msg) for msg in all_messages]
    original_message_count = len(lc_messages)

    result_state = agent.invoke({"messages": lc_messages})
    result_msgs = result_state.get("messages", [])
    
    # Only process NEW messages (those added by the LLM, not the input history)
    new_messages = result_msgs[original_message_count:]
    result_ui = [to_ui_message(msg) for msg in new_messages]
    persist_messages(thread_id, result_ui)
    if DEBUG_LOGGING_ENABLED:
        logger.debug("create_run(%s) original=%s total=%s new=%s", 
                    thread_id, original_message_count, len(result_msgs), len(new_messages))
    touch_thread(thread_id)

    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "status": "success",
        "output": {"messages": result_ui},
        "created_at": utc_now_iso(),
    }


@app.get("/threads/{thread_id}/runs")
def list_runs(thread_id: str):
    if not get_thread_row(thread_id, include_messages=False):
        upsert_thread(thread_id)
    return {"runs": [], "total": 0}


@app.post("/threads/{thread_id}/runs/stream")
async def create_stream_run(
    thread_id: str, 
    request: Request,
    assistant_id: Optional[str] = None,
    graph_id: Optional[str] = None,
):
    if not get_thread_row(thread_id, include_messages=False):
        logger.info("Creating new thread %s", thread_id)
        # Create thread with metadata from query params if available
        initial_metadata = {}
        if assistant_id:
            initial_metadata["assistant_id"] = assistant_id
            initial_metadata["graph_id"] = assistant_id
        elif graph_id:
            initial_metadata["graph_id"] = graph_id
            initial_metadata["assistant_id"] = graph_id
        upsert_thread(thread_id, initial_metadata if initial_metadata else None)

    run_payload = await parse_json_body(request)
    run_data = RunCreate.model_validate(run_payload)

    metadata_update = extract_metadata_from_payload(run_payload)
    # Also use query params if present
    if assistant_id and "assistant_id" not in metadata_update:
        metadata_update["assistant_id"] = assistant_id
        metadata_update["graph_id"] = assistant_id
    elif graph_id and "graph_id" not in metadata_update:
        metadata_update["graph_id"] = graph_id
        metadata_update["assistant_id"] = graph_id
    
    if run_data.metadata:
        metadata_update.update(run_data.metadata)
    logger.info("Thread %s metadata_update: %s", thread_id, metadata_update)
    merge_thread_metadata(thread_id, metadata_update)

    run_id = str(uuid.uuid4())

    async def generate_stream():
        try:
            incoming = run_data.input.get("messages", [])
            if not incoming:
                error_event = {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "error": "No messages provided",
                }
                yield "event: error\n"
                yield f"data: {json.dumps(error_event)}\n\n"
                return

            # Convert and persist incoming messages
            ui_messages = [to_ui_message(msg) for msg in incoming]
            persist_messages(thread_id, ui_messages)
            
            # Load full conversation history from the thread
            all_messages = fetch_messages(thread_id, limit=MAX_CONTEXT_MESSAGES)
            if DEBUG_LOGGING_ENABLED:
                logger.debug("create_stream_run(%s) loaded %s historical messages", thread_id, len(all_messages))
            
            # Convert all messages to LangChain format for the agent
            lc_messages = [to_lc_message(msg) for msg in all_messages]
            original_message_count = len(lc_messages)

            result_state = agent.invoke({"messages": lc_messages})
            result_msgs = result_state.get("messages", [])
            
            # Only persist NEW messages (those added by the LLM, not the input history)
            new_messages = result_msgs[original_message_count:]
            if DEBUG_LOGGING_ENABLED:
                logger.debug("create_stream_run(%s) original=%s total=%s new=%s", 
                            thread_id, original_message_count, len(result_msgs), len(new_messages))
            
            for message in new_messages:
                ui_message = to_ui_message(message)
                persist_messages(thread_id, [ui_message])
            
            # Return the full updated conversation history to prevent flicker
            all_messages_updated = fetch_messages(thread_id, limit=MAX_CONTEXT_MESSAGES)
            event_payload = {
                "values": {"messages": all_messages_updated},
                "run_id": run_id,
                "thread_id": thread_id,
            }
            yield "event: values\n"
            yield f"data: {json.dumps(event_payload)}\n\n"
            await asyncio.sleep(0.05)

            touch_thread(thread_id)
            completion_event = {
                "run_id": run_id,
                "thread_id": thread_id,
                "status": "success",
            }
            yield "event: end\n"
            yield f"data: {json.dumps(completion_event)}\n\n"
        except Exception as exc:
            error_event = {
                "run_id": run_id,
                "thread_id": thread_id,
                "error": str(exc),
            }
            yield "event: error\n"
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2024)
