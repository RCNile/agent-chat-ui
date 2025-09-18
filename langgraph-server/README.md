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

If you want to use OpenAI or other LLM providers, set the appropriate API keys:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```
