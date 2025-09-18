#!/usr/bin/env python3

from langgraph.graph import StateGraph, MessagesState, START, END
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import datetime
import os

# In-memory storage for threads (replace with database in production)
threads_storage = {}

class ThreadCreate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None

class ThreadUpdate(BaseModel):
    metadata: Optional[Dict[str, Any]] = None

class Thread(BaseModel):
    thread_id: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    status: str = "idle"

class Message(BaseModel):
    role: str
    content: str
    type: str = "human"

class RunCreate(BaseModel):
    input: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    multitask_strategy: Optional[str] = None

# Create a simple chatbot function that uses the new MessagesState
def mock_llm(state: MessagesState):
    # Simple echo bot for demo - replace with your actual LLM logic
    messages = state["messages"]
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content'):
            content = last_message.content
        elif isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = str(last_message)
        
        if content:
            return {"messages": [{"role": "assistant", "content": f"Echo: {content}"}]}
    
    return {"messages": [{"role": "assistant", "content": "Hello! I'm a simple echo bot."}]}

# Create the graph using the new v1-alpha API
def create_agent_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("mock_llm", mock_llm)
    graph.add_edge(START, "mock_llm")
    graph.add_edge("mock_llm", END)
    return graph.compile()

# Create FastAPI app
app = FastAPI(title="Simple LangGraph Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Allow the chat UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the agent
agent = create_agent_graph()

@app.get("/")
def root():
    return {"message": "LangGraph Server is running"}

@app.get("/info")
def info():
    return {
        "name": "Simple LangGraph Agent",
        "description": "A basic echo agent for testing",
        "version": "1.0.0"
    }

# LangGraph API endpoints
@app.get("/assistants/{assistant_id}")
def get_assistant(assistant_id: str):
    """Get assistant information"""
    return {
        "assistant_id": assistant_id,
        "graph_id": assistant_id,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "config": {},
        "metadata": {}
    }

@app.get("/threads")
def list_threads(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    metadata: Optional[str] = Query(None)
):
    """List threads"""
    thread_list = list(threads_storage.values())
    total = len(thread_list)
    
    # Apply pagination
    paginated_threads = thread_list[offset:offset + limit]
    
    # Return just the array of threads as expected by the frontend
    return paginated_threads

@app.get("/threads/search")
@app.post("/threads/search")
def search_threads(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    metadata: Optional[str] = Query(None)
):
    """Search threads - same as list for this simple implementation"""
    return list_threads(limit=limit, offset=offset, metadata=metadata)

@app.post("/threads")
def create_thread(thread_data: ThreadCreate):
    """Create a new thread"""
    thread_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow().isoformat() + "Z"
    
    thread = Thread(
        thread_id=thread_id,
        created_at=now,
        updated_at=now,
        metadata=thread_data.metadata or {}
    )
    
    threads_storage[thread_id] = thread.model_dump()
    return thread.model_dump()

@app.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    """Get a specific thread"""
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return threads_storage[thread_id]

@app.patch("/threads/{thread_id}")
def update_thread(thread_id: str, thread_update: ThreadUpdate):
    """Update a thread"""
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    thread = threads_storage[thread_id]
    if thread_update.metadata:
        thread["metadata"].update(thread_update.metadata)
    thread["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    
    return thread

@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    """Delete a thread"""
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    del threads_storage[thread_id]
    return {"message": "Thread deleted successfully"}

@app.get("/threads/{thread_id}/state")
def get_thread_state(thread_id: str):
    """Get thread state"""
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {
        "values": {"messages": []},
        "next": [],
        "config": {
            "configurable": {
                "thread_id": thread_id
            }
        },
        "metadata": {},
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "parent_config": None
    }

@app.get("/threads/{thread_id}/history")
@app.post("/threads/{thread_id}/history")
def get_thread_history(
    thread_id: str,
    limit: int = Query(10, ge=1, le=100),
    before: Optional[str] = Query(None)
):
    """Get thread history"""
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Return an array directly as expected by the frontend
    return []

@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, run_data: RunCreate):
    """Create a new run"""
    # Ensure thread exists or create it
    if thread_id not in threads_storage:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        thread = Thread(
            thread_id=thread_id,
            created_at=now,
            updated_at=now,
            metadata={}
        )
        threads_storage[thread_id] = thread.model_dump()
    
    run_id = str(uuid.uuid4())
    
    # For streaming, we'll return a simple response
    # In a real implementation, you'd use Server-Sent Events
    try:
        # Process the input through our agent
        messages = run_data.input.get("messages", [])
        if messages:
            # Convert to the format expected by our agent
            agent_input = {"messages": messages}
            result = agent.invoke(agent_input)
            
            return {
                "run_id": run_id,
                "thread_id": thread_id,
                "status": "success",
                "output": result,
                "created_at": datetime.datetime.utcnow().isoformat() + "Z"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads/{thread_id}/runs")
def list_runs(thread_id: str):
    """List runs for a thread"""
    if thread_id not in threads_storage:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return {
        "runs": [],
        "total": 0
    }

@app.post("/threads/{thread_id}/runs/stream")
async def create_stream_run(thread_id: str, run_data: RunCreate):
    """Create a streaming run - for now returns same as regular run"""
    # For this simple implementation, we'll just return the same as create_run
    # In a real implementation, you'd use Server-Sent Events for streaming
    return await create_run(thread_id, run_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2024)
