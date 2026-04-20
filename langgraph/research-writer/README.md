# Research-Writer - LangGraph

A two-agent RAG pipeline that searches an in-memory vector store and synthesizes findings into polished responses, instrumented with [Lumoz](https://lumoz.ai) for optional observability.

## Architecture

This example uses a **linear two-node pipeline** — a Research Agent searches the knowledge base, then a Writer Agent polishes the findings:

```
documents/*.txt  ──(startup)──>  In-Memory Vector Store
                                        |
User Query (CLI)                        |
      |                                 v
+---------------------------------------------+
|           LangGraph StateGraph              |
|                                             |
|  START --> research --> write --> END        |
|               |            |                |
|               v            v                |
|         Research Agent   Writer Agent       |
|         (gpt-4o-mini)   (gpt-4o-mini)      |
|               |                             |
|               v                             |
|         vector_search tool                  |
|         (cosine similarity)                 |
+---------------------------------------------+
      |
      v
  Final Response (printed to terminal)
```

## Agents

| Agent | Description | Tools |
|-------|-------------|-------|
| `research` | Searches the knowledge base via vector similarity, synthesizes findings | `vector_search` |
| `write` | Transforms research into a well-structured, reader-friendly response | None |

## What Gets Traced

When running with Lumoz instrumentation, you'll see:

- **Research agent**: ReAct loop with vector search tool calls
- **Writer agent**: LLM synthesis of research into final response
- **Tool calls**: `vector_search` with query inputs and retrieved chunks
- **LLM calls**: OpenAI requests with token counts and latency
- **Root span**: Per-query span with `user.id` and `session.id` attributes

```
research_writer_query (root)
├── research (NODE)
│   └── Research Agent (AGENT)
│       ├── ChatOpenAI (LLM)
│       ├── vector_search (TOOL)
│       └── ChatOpenAI (LLM)
└── write (NODE)
    └── Writer Agent (AGENT)
        └── ChatOpenAI (LLM)
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key

# Optional — Lumoz observability
LUMOZ_API_KEY=your_client_id:your_client_secret
OTEL_ENDPOINT=https://api.lumoz.ai/proxy/v1/traces
```

**Get your Lumoz API key:**
1. Log in to [Lumoz Console](https://console.lumoz.ai)
2. Go to **Settings > API Keys**
3. Click **Create API Key**
4. Copy the key (format: `client_id:client_secret`)

Tracing is optional. The app works with only `OPENAI_API_KEY` set.

## Add Lumoz to Your Own LangGraph App

> **Just want the instrumentation code?** → [INSTRUMENTATION.md](INSTRUMENTATION.md)
>
> Covers: exporter setup, `LangChainInstrumentor`, user/session tracking with `using_attributes`, root spans, graceful shutdown.

## Run

```bash
python research_writer.py
```

### Usage

```
=== LangGraph Research-Writer ===

Enter your user ID (or press Enter for 'demo-user'):
Session: a1b2c3d4-...

[knowledge] Loading documents from documents/...
[knowledge] Loaded 5 files (42 chunks)

Knowledge base: 42 chunks ready

Commands:
  /ingest <text>  — Add text to the knowledge base
  /quit           — Exit

You: What's the most dangerous creature in the wizarding world?

[researching...]

Agent: Based on the knowledge base, several creatures stand out as
particularly dangerous...

You: /ingest The Elder Wand is the most powerful wand ever made.
[knowledge] Ingested 1 chunks (43 total)

You: /quit
Goodbye!
```

### Adding Documents

Drop `.txt` files into the `documents/` directory and restart the app, or use `/ingest <text>` at runtime to add content on the fly.

## View Traces in Lumoz

1. Run the app and send a query
2. Open [Lumoz Console](https://console.lumoz.ai)
3. Go to **Home** and click the **langgraph-research-writer** app card
4. Navigate to the **Telemetry → Traces** tab
5. Click any trace to explore the research and write agent spans, tool calls, and LLM interactions

## Requirements

- Python 3.11+
- OpenAI API key (for GPT-4o-mini and embeddings)
- Lumoz API key (optional, for observability)

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenInference](https://github.com/Arize-ai/openinference)
- Questions? [support@lumoz.ai](mailto:support@lumoz.ai)
