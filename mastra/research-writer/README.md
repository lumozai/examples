# Mastra Research-Writer

A multi-agent workflow example using the [Mastra](https://mastra.ai) framework with Lumoz observability. Two agents collaborate in a sequential pipeline: a **Research Agent** searches a knowledge base using vector search, then a **Writer Agent** synthesizes the findings into a polished response.

## Architecture

```
User Query
  |
  v
research-write workflow
  |-- research step
  |     |-- Research Agent (gpt-4o-mini)
  |     |     |-- vectorSearchTool (in-memory vector store)
  |     |     +-- synthesize results
  |     +-- returns research summary
  |
  +-- write step
        |-- Writer Agent (gpt-4o-mini)
        +-- returns final response
```

### Components

| File | Description |
|------|-------------|
| `src/index.ts` | Entry point — initializes tracing, seeds knowledge base, runs interactive CLI |
| `src/tracing.ts` | OpenTelemetry setup with Lumoz/Arize exporter |
| `src/agents/researcher.ts` | Research Agent with vector search tool |
| `src/agents/writer.ts` | Writer Agent for content synthesis |
| `src/workflows/research-write.ts` | Two-step workflow connecting both agents |
| `src/vectorStore.ts` | In-memory vector store using OpenAI embeddings |
| `src/env.ts` | Environment variable loading |

## Add Lumoz to Your Own Mastra App

> **Just want the instrumentation code?** → [INSTRUMENTATION.md](INSTRUMENTATION.md)
>
> Covers: exporter setup, init order, user/session tracking, trace context propagation, graceful shutdown.

## Prerequisites

- Node.js 18+
- An OpenAI API key
- A Lumoz API key (optional, for observability)

## Setup

1. Create a `.env` file in this directory (or in the parent):

```bash
OPENAI_API_KEY=sk-...
LUMOZ_API_KEY=client_id:client_secret
OTEL_ENDPOINT=https://api.lumoz.ai/proxy/v1/traces
```

2. Install dependencies:

```bash
npm install
```

## Usage

```bash
npm start
```

The CLI will prompt for a user ID (defaults to `demo-user`) and start an interactive session. A knowledge base is seeded with sample documents on startup.

### Commands

| Command | Description |
|---------|-------------|
| Any text | Sends a query through the research-write workflow |
| `/ingest <text>` | Adds text to the knowledge base |
| `/quit` | Exits the application |

## How Tracing Works

See [INSTRUMENTATION.md](INSTRUMENTATION.md) for full setup details, code snippets, and trace structure.

## Example Output

```
=== Mastra Research-Writer Agent ===

User: demo-user
Session: a1b2c3d4-...

[knowledge] Ingested 5 documents (5 chunks)
Knowledge base: 5 chunks ready

You: what's new in AI?

[researching...]

Agent: ### What's New in AI: Key Developments
...
```
