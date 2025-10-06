# LangGraph Server

A FastAPI-based backend server providing conversational AI capabilities with persistent conversation history, tool calling, and streaming responses.

## Architecture Overview

### Core Components

1. **FastAPI Application**: RESTful API server compatible with LangGraph SDK
2. **LangGraph Agent**: Conversational agent with tool-calling capabilities
3. **SQLite Database**: Persistent storage for conversation threads and messages
4. **Streaming Engine**: Server-Sent Events (SSE) for real-time response streaming

### Database Schema

The server uses SQLite with three main tables:

- **threads**: Stores conversation threads with metadata
- **messages**: Stores all messages with content, type, and timestamps
- **runs**: Tracks execution runs for each message

### Message Flow

1. **Incoming Request**:
   - Client sends message via `/threads/{thread_id}/runs/stream`
   - Server persists new message to database
   - Server loads conversation history (last N messages)

2. **Agent Processing**:
   - LangGraph agent receives full conversation context
   - Agent processes message with available tools
   - Agent generates response with potential tool calls

3. **Response Streaming**:
   - Server streams events back to client via SSE
   - Only new messages (not historical context) are sent
   - Client accumulates state for full conversation view

### Key Features

- **Persistent Conversations**: All messages stored in SQLite
- **Context Management**: Configurable message history limit
- **Tool Integration**: Extensible tool system via LangGraph
- **Metadata Support**: Assistant and graph ID tracking
- **Activity Logging**: Detailed operation logging
- **Error Handling**: Comprehensive error management

### API Endpoints

- `GET /info` - Server information and health check
- `GET /threads/search` - Search threads by metadata
- `GET /threads/{thread_id}/state` - Get thread state
- `GET /threads/{thread_id}/history` - Get message history
- `POST /threads/{thread_id}/runs/stream` - Create streaming run
- `DELETE /threads/{thread_id}` - Delete thread and messages

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python app.py
```

The server will run on http://localhost:2024

## Usage

This server provides a conversational AI agent with tool-calling capabilities. The agent uses LangGraph for orchestration and OpenAI for language understanding.

### Agent Capabilities

The agent includes the following built-in tools:
- **Search**: Web search capabilities for retrieving current information
- **Calculator**: Mathematical computation tool
- **Weather**: Weather information retrieval

### Extending the Agent

To add new tools or modify behaviour, edit the `agent` StateGraph in `app.py`:

```python
# Add new tools to the agent
from langchain_core.tools import tool

@tool
def custom_tool(query: str) -> str:
    """Your custom tool description."""
    return "Tool result"

# Add to model binding
model = model.bind_tools([search_tool, calculator_tool, custom_tool])
```

## Environment Variables

Required:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Optional configuration:
```bash
# Maximum number of historical messages to include in context (default: 50)
export MAX_CONTEXT_MESSAGES="50"

# OpenAI model to use (default: gpt-4o)
export OPENAI_MODEL="gpt-4o"

# Model temperature (default: 0.2)
export OPENAI_TEMPERATURE="0.2"

# Path to documents for RAG (default: data/critical_role)
export DOCS_PATH="data/critical_role"

# Enable debug logging (default: false)
export CHAT_DEBUG_LOG="true"
```

## Database Management

The server automatically creates and manages a SQLite database (`langgraph.db`) in the server directory.

### Database Schema

**threads table**:
- `thread_id` (TEXT PRIMARY KEY): Unique thread identifier
- `created_at` (TEXT): Creation timestamp
- `metadata` (TEXT): JSON metadata (assistant_id, graph_id)

**messages table**:
- `id` (TEXT PRIMARY KEY): Unique message identifier
- `thread_id` (TEXT): Associated thread
- `type` (TEXT): Message type (human, ai, tool)
- `content` (TEXT): JSON message content
- `created_at` (TEXT): Creation timestamp

**runs table**:
- `run_id` (TEXT PRIMARY KEY): Unique run identifier
- `thread_id` (TEXT): Associated thread
- `created_at` (TEXT): Creation timestamp
- `status` (TEXT): Run status

### Activity Logging

The server logs all operations with agent name prefixes for traceability:
```
[CREATE_STREAM_RUN] Created new thread: abc-123
[FETCH_MESSAGES] Loaded 15 messages from thread: abc-123
[PERSIST_MESSAGES] Persisted 2 messages to thread: abc-123
```

## Performance Considerations

- **Message History Limit**: Default 50 messages prevents excessive context size
- **Database Indexing**: Thread ID and timestamp indices for fast queries
- **Streaming**: Chunked responses reduce perceived latency
- **Connection Pooling**: SQLite connection management for concurrency

## Troubleshooting

**Database locked errors**: 
- Reduce concurrent requests or switch to PostgreSQL for production

**Context too long errors**:
- Reduce `MAX_CONTEXT_MESSAGES` environment variable
- Implement message summarisation for older messages

**Tool execution failures**:
- Check tool implementations return proper formats
- Verify external API keys and connectivity
