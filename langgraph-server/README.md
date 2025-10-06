# Simple LangGraph Server

This is a basic LangGraph server to work with the Agent Chat UI.

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

This server provides a simple echo agent for testing the Agent Chat UI. You can extend the `chatbot` function to integrate with actual LLM providers like OpenAI, Anthropic, etc.

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
