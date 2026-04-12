# LangGraph Research-Writer — Specification

Port of `mastra/research-writer` to LangGraph (Python). A two-agent RAG pipeline
where a Research Agent searches an in-memory vector store and a Writer Agent
synthesizes findings into a polished response.

---

## What This Is

A CLI chat application that:

1. On startup, loads all `.txt` files from a `documents/` directory into an
   in-memory vector store (chunk, embed, store)
2. Accepts user queries in an interactive loop
3. Runs a LangGraph two-node pipeline: **research** (vector search + synthesis)
   → **write** (polish into final response)
4. Prints the response and waits for the next query
5. Supports `/ingest <text>` to add content at runtime and `/quit` to exit

---

## Architecture

```
documents/*.txt  ──(startup)──▶  In-Memory Vector Store
                                        │
User Query (CLI)                        │
      │                                 │
      ▼                                 ▼
┌─────────────────────────────────────────────┐
│           LangGraph StateGraph              │
│                                             │
│  START ──▶ research ──▶ write ──▶ END       │
│               │            │                │
│               ▼            ▼                │
│         Research Agent   Writer Agent       │
│         (gpt-4o-mini)    (gpt-4o-mini)      │
│               │                             │
│               ▼                             │
│         vector_search tool                  │
│         (cosine similarity)                 │
└─────────────────────────────────────────────┘
      │
      ▼
  Final Response (printed to terminal)
```

---

## File Structure

```
langgraph/research-writer/
├── research_writer.py        # Single-file app (matches travel-video-analyzer convention)
├── requirements.txt
├── .env.example
├── README.md
└── documents/                # Drop .txt files here — loaded into vector store on startup
    ├── hogwarts.txt
    ├── magic-system.txt
    ├── magical-creatures.txt
    ├── quidditch.txt
    └── wizarding-world.txt
```

Single-file design matches the existing `langgraph/travel-video-analyzer` pattern.

---

## LangGraph Best Practices (Required)

This is a demo example meant to showcase LangGraph idioms. The implementation
must use framework-native features — not generic Python workarounds — for each
of the following:

- **Agent creation:** Use `create_react_agent` from `langgraph.prebuilt` with
  proper system prompts, tool binding, and model configuration. Look up the
  current API signature — parameter names may have changed.
- **State management:** Use `TypedDict` state with `Annotated` reducers
  (e.g. `operator.add` for message lists). Nodes return partial state dicts.
- **Session / conversation persistence:** Use LangGraph's built-in checkpointer
  (`MemorySaver` or equivalent) so the graph retains conversation history across
  queries in the interactive loop. Pass `thread_id` via config on each invoke.
- **Config propagation:** Pass `user_id` and `session_id` through the
  `configurable` dict on `graph.invoke()` so metadata flows through the graph
  and is available to tracing.
- **User info in traces:** Attach `user.id` and `session.id` as OpenTelemetry
  span attributes on a root span per query. All child spans from
  `LangChainInstrumentor` nest under it automatically.
- **Tool definition:** Use the `@tool` decorator from `langchain_core.tools`
  with clear docstrings and `Args:` sections.

Consult the latest LangGraph documentation for current API signatures and
import paths before implementing.

---

## Components

### 1. State

```python
class ResearchWriterState(TypedDict):
    query: str          # Original user query
    research: str       # Research agent output
    response: str       # Writer agent output (final answer)
    messages: Annotated[list, operator.add]
```

### 2. Vector Store (in-memory)

| Function | Description |
|----------|-------------|
| `embed(text) -> list[float]` | OpenAI `text-embedding-3-small` (1536 dims) |
| `chunk(text, size=512, overlap=50) -> list[str]` | Character-based overlapping chunks |
| `ingest(text) -> int` | Chunk + embed + store. Returns chunk count |
| `search(query, top_k=5) -> list[dict]` | Cosine similarity. Returns `[{text, score}]` |
| `load_documents(directory) -> int` | Load all `.txt` files from directory, ingest each. Returns total chunks |

Cosine similarity: numpy dot product / norms. No vector DB library.

### 3. Document Loading

On startup, call `load_documents("documents/")`:
- Glob for `documents/*.txt`
- Read each file, call `ingest(content)`
- Print count of files and chunks loaded
- If directory is empty or missing, warn and continue (store will be empty)

Users add their own content by dropping `.txt` files into `documents/` and
restarting, or by using `/ingest <text>` at runtime.

### 4. Research Agent

- **Built with:** `create_react_agent(llm, tools=[vector_search], prompt=...)`
- **LLM:** `ChatOpenAI(model="gpt-4o-mini")`
- **Tool:** `vector_search` — calls `search(query, top_k=5)`, returns JSON results
- **Prompt:** Research assistant that searches the knowledge base, synthesizes
  findings, cites retrieved documents, states clearly when nothing relevant is found

### 5. Writer Agent

- **Built with:** `create_react_agent(llm, tools=[], prompt=...)`
- **LLM:** `ChatOpenAI(model="gpt-4o-mini")`
- **No tools** — purely synthesis
- **Prompt:** Skilled writer that transforms research into a well-structured,
  reader-friendly response. Conversational but informative tone.

### 6. Graph

```
START → research_node → write_node → END
```

Linear, no conditionals. `research_node` invokes the research agent and writes
to `state["research"]`. `write_node` invokes the writer agent and writes to
`state["response"]`.

### 7. Tracing (OpenTelemetry + Lumoz)

Follow `travel-video-analyzer` patterns:

- `ImageStrippingSpanProcessor` extending `BatchSpanProcessor` — strip base64 data
- OTLP HTTP exporter with `LUMOZ_API_KEY` (Basic auth) and `OTEL_ENDPOINT`
- `LangChainInstrumentor().instrument()` for automatic LLM/tool span capture
- Custom root span per query with `user.id` and `session.id` attributes
- Service name: `langgraph-research-writer`
- Tracing is optional — app works without `LUMOZ_API_KEY` set

### 8. CLI Interface

```
=== LangGraph Research-Writer ===

Enter your user ID (or press Enter for 'demo-user'): _
Session: <uuid>

[knowledge] Loading documents from documents/...
[knowledge] Loaded 5 files (N chunks)

Knowledge base: N chunks ready

Commands:
  /ingest <text>  — Add text to the knowledge base
  /quit           — Exit

You: What's the most dangerous creature in the wizarding world?

[researching...]

Agent: <response>

You: _
```

---

## Seed Documents

5 `.txt` files already exist in `documents/`, themed around the Harry Potter
wizarding world. Each file is a rich, detailed reference document (~500-800
words) covering a different aspect:

| File | Content |
|------|---------|
| `hogwarts.txt` | The school, four houses, sorting, curriculum, professors, castle layout, grounds |
| `magic-system.txt` | Wands, spells, Unforgivable Curses, potions, Apparition, Animagi, Horcruxes |
| `magical-creatures.txt` | Dragons, phoenixes, hippogriffs, thestrals, house-elves, basilisks, Dementors |
| `quidditch.txt` | Rules, positions, Snitch, broomsticks, famous matches, World Cup, fouls |
| `wizarding-world.txt` | Ministry of Magic, Diagon Alley, Gringotts, Hogsmeade, Azkaban, economy, communication |

These documents have plenty of cross-references and overlapping themes (e.g.
Fawkes the phoenix appears in both `hogwarts.txt` and `magical-creatures.txt`),
making vector search results interesting and multi-source.

---

## Configuration

### `.env.example`

```bash
# Required — LLM and embeddings
OPENAI_API_KEY=sk-...

# Optional — Lumoz observability
LUMOZ_API_KEY=client_id:client_secret
OTEL_ENDPOINT=https://api.lumoz.ai/proxy/v1/traces
```

### `requirements.txt`

```
langgraph>=0.2.0
langchain-openai>=0.3.0
langchain-core>=0.3.0
openai>=1.0.0
numpy>=1.24.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp-proto-http>=1.20.0
openinference-instrumentation-langchain>=0.1.0
python-dotenv>=1.0.0
```

---

## Acceptance Criteria

1. `python research_writer.py` starts, loads `documents/*.txt`, enters interactive chat
2. Queries go through `research → write` LangGraph pipeline and produce grounded responses
3. `/ingest <text>` adds content to the vector store at runtime
4. `/quit` exits cleanly (flushes traces if tracing is active)
5. Dropping new `.txt` files into `documents/` and restarting picks them up
6. Works with only `OPENAI_API_KEY` set (tracing is optional)
7. When `LUMOZ_API_KEY` + `OTEL_ENDPOINT` are set, traces appear in Lumoz with
   correct nesting (workflow → agent → LLM/tool spans)
8. Single-file `research_writer.py`, Python 3.11+, `requirements.txt`
9. `README.md` with description, setup instructions, architecture diagram
   (text-based), and usage examples

---

## Out of Scope

- No persistent vector store — in-memory only
- No streaming responses
- No web UI — CLI only
- No tests
- No Docker
- No lumoz.yaml / programmatic entry point — this is a standalone CLI app
