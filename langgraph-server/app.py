#!/usr/bin/env python3
# langgraph-server/app.py
import asyncio
import json
import os
import pathlib
import uuid
from typing import Any, Dict, List, Optional

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

# -----------------------------------------------------------------------------
# Environment / model setup
# -----------------------------------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY environment variable must be set before starting the server."
    )

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MODEL_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
DOCS_PATH = pathlib.Path(os.getenv("DOCS_PATH", "data/critical_role"))

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=MODEL_TEMPERATURE)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a meticulous Critical Role lore expert. Use the context when relevant, "
            "cite specific story beats or episodes when possible, and acknowledge uncertainty "
            "rather than guessing.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chat_chain = prompt | llm


def call_model(state: MessagesState) -> Dict[str, List[AIMessage]]:
    """LangGraph node: augment the latest user question with retrieved context, then answer."""
    messages = state["messages"]
    query_text = extract_last_user_question(messages)
    retrieved_docs = retriever.search(query_text) if query_text else []
    context = build_context_string(retrieved_docs)
    ai_msg = chat_chain.invoke({"messages": messages, "context": context})
    return {"messages": [ai_msg]}


# -----------------------------------------------------------------------------
# Retrieval helpers
# -----------------------------------------------------------------------------

class Retriever:
    """Simple wrapper around a FAISS vector store."""

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

# -----------------------------------------------------------------------------
# FastAPI + LangGraph wiring
# -----------------------------------------------------------------------------

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


def ensure_thread_storage(thread_id: str):
    if thread_id not in threads_storage:
        now = utc_now_iso()
        threads_storage[thread_id] = {
            "thread_id": thread_id,
            "created_at": now,
            "updated_at": now,
            "metadata": {},
            "status": "idle",
        }
    thread_messages.setdefault(thread_id, [])


def touch_thread(thread_id: str):
    if thread_id in threads_storage:
        threads_storage[thread_id]["updated_at"] = utc_now_iso()


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

    return {
        "id": raw.get("id", str(uuid.uuid4())),
        "type": msg_type,
        "role": role,
        "content": content_blocks,
    }


def checkpoint_payload(thread_id: str, limit: Optional[int] = None) -> Dict[str, Any]:
    messages = thread_messages.get(thread_id, [])
    if limit is not None and limit >= 0:
        messages = messages[-limit:]
    return {
        "checkpoint_id": f"{thread_id}-latest",
        "parent_checkpoint_id": None,
        "created_at": utc_now_iso(),
        "config": {"configurable": {"thread_id": thread_id}},
        "metadata": {},
        "values": {"messages": messages},
    }


async def parse_json_body(request: Request) -> Dict[str, Any]:
    raw_payload = await request.body()
    if not raw_payload:
        return {}
    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")


threads_storage: Dict[str, Dict[str, Any]] = {}
thread_messages: Dict[str, List[Dict[str, Any]]] = {}


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
        "version": "2.0.0",
        "model": OPENAI_MODEL,
    }


@app.get("/assistants/{assistant_id}")
def get_assistant(assistant_id: str):
    ensure_thread_storage(assistant_id)
    thread = threads_storage[assistant_id]
    return {
        "assistant_id": assistant_id,
        "graph_id": assistant_id,
        "created_at": thread["created_at"],
        "updated_at": thread["updated_at"],
        "config": {},
        "metadata": {},
    }


@app.get("/threads")
def list_threads(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    metadata: Optional[str] = Query(None),
):
    thread_list = list(threads_storage.values())
    return thread_list[offset : offset + limit]


@app.get("/threads/search")
@app.post("/threads/search")
def search_threads(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    metadata: Optional[str] = Query(None),
):
    return list_threads(limit=limit, offset=offset, metadata=metadata)


@app.post("/threads")
async def create_thread(request: Request):
    data = await parse_json_body(request)
    thread_data = ThreadCreate.model_validate(data)
    thread_id = str(uuid.uuid4())
    ensure_thread_storage(thread_id)
    if thread_data.metadata:
        threads_storage[thread_id]["metadata"] = thread_data.metadata
    return threads_storage[thread_id]


@app.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    return threads_storage[thread_id]


@app.patch("/threads/{thread_id}")
async def update_thread(thread_id: str, request: Request):
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    payload = await parse_json_body(request)
    thread_update = ThreadUpdate.model_validate(payload)
    if thread_update.metadata:
        threads_storage[thread_id]["metadata"].update(thread_update.metadata)
    touch_thread(thread_id)
    return threads_storage[thread_id]


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    del threads_storage[thread_id]
    thread_messages.pop(thread_id, None)
    return {"message": "Thread deleted successfully"}


@app.get("/threads/{thread_id}/state")
def get_thread_state(thread_id: str):
    return checkpoint_payload(thread_id)


@app.get("/threads/{thread_id}/history")
@app.post("/threads/{thread_id}/history")
def get_thread_history(
    thread_id: str,
    limit: int = Query(10, ge=1, le=100),
    before: Optional[str] = Query(None),
):
    ensure_thread_storage(thread_id)
    return [checkpoint_payload(thread_id, limit=limit)]


@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request):
    ensure_thread_storage(thread_id)
    run_payload = await parse_json_body(request)
    run_data = RunCreate.model_validate(run_payload)

    run_id = str(uuid.uuid4())
    messages = run_data.input.get("messages", [])

    ui_messages = [to_ui_message(msg) for msg in messages]
    thread_messages[thread_id].extend(ui_messages)
    lc_messages = [to_lc_message(msg) for msg in ui_messages]

    result_state = agent.invoke({"messages": lc_messages})
    result_ui = [to_ui_message(msg) for msg in result_state.get("messages", [])]
    thread_messages[thread_id].extend(result_ui)
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
    ensure_thread_storage(thread_id)
    return {"runs": [], "total": 0}


@app.post("/threads/{thread_id}/runs/stream")
async def create_stream_run(thread_id: str, request: Request):
    ensure_thread_storage(thread_id)
    run_payload = await parse_json_body(request)
    run_data = RunCreate.model_validate(run_payload)
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

            ui_messages = [to_ui_message(msg) for msg in incoming]
            thread_messages[thread_id].extend(ui_messages)
            lc_messages = [to_lc_message(msg) for msg in ui_messages]

            result_state = agent.invoke({"messages": lc_messages})
            result_msgs = result_state.get("messages", [])
            for message in result_msgs:
                ui_message = to_ui_message(message)
                thread_messages[thread_id].append(ui_message)

                if ui_message["type"] == "ai":
                    event_payload = {
                        "values": {"messages": [ui_message]},
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
