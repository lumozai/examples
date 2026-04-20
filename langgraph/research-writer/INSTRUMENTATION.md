# Instrumenting LangGraph with Lumoz

This guide covers how this example integrates LangGraph with Lumoz observability to capture LLM calls, tool usage, agent execution, and multi-step reasoning traces.

LangGraph instrumentation uses **auto-instrumentation via OpenInference** — the `LangChainInstrumentor` patches LangChain/LangGraph internals and automatically captures spans for every node, agent, tool call, and LLM request. You only write a few lines of setup code and a root span per query.

## Dependencies

```bash
pip install \
  opentelemetry-sdk \
  opentelemetry-exporter-otlp-proto-http \
  openinference-instrumentation-langchain
```

These are already in `requirements.txt` and are optional — the app works without them if `LUMOZ_API_KEY` / `OTEL_ENDPOINT` are not set.

## 1. Configure the Exporter

See [`research_writer.py`](research_writer.py) → `configure_lumoz_tracing()`.

```python
import base64
from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

def configure_lumoz_tracing():
    otel_endpoint = os.environ.get("OTEL_ENDPOINT")
    api_key = os.environ.get("LUMOZ_API_KEY")

    if not otel_endpoint or not api_key:
        return None  # tracing is optional

    encoded = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
    headers = {"authorization": f"Basic {encoded}"}

    resource = Resource.create({
        "service.name": "langgraph-research-writer",
        "deployment.environment": "development",
    })

    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    otlp_exporter = OTLPSpanExporter(endpoint=otel_endpoint, headers=headers)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Auto-instruments all LangChain/LangGraph spans
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider
```

`LangChainInstrumentor` is the key difference from frameworks like Mastra — it monkey-patches LangChain internals so every agent, node, tool call, and LLM request is captured automatically. No manual span creation needed for the graph itself.

## 2. Initialize at Module Level

Call `configure_lumoz_tracing()` at module load time, **before** building the graph or creating any LangChain objects:

```python
# Module-level — runs before graph construction
tracer_provider = configure_lumoz_tracing()
tracing_enabled = tracer_provider is not None
```

This ensures the instrumentor is active before `StateGraph`, agents, or tools are instantiated.

## 3. Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...
LUMOZ_API_KEY=client_id:client_secret
OTEL_ENDPOINT=https://api.lumoz.ai/proxy/v1/traces
```

Tracing is optional. The app runs normally with only `OPENAI_API_KEY` set.

## 4. User and Session Tracking

LangGraph does not have a native session/user concept like Mastra's `tracingOptions.metadata`. Instead, use two mechanisms together:

### Root span with explicit attributes

Wrap each query in a manual root span and set `user.id` and `session.id` directly:

```python
tracer = trace.get_tracer("research_writer")

with tracer.start_as_current_span(
    "research_writer_query",
    attributes={
        "user.id": user_id,
        "session.id": session_id,
        "query": user_input,
    },
):
    result = app.invoke(...)
```

### `using_attributes` context manager

Wrap the `app.invoke()` call with `using_attributes` from `openinference.instrumentation`. This propagates `user.id` and `session.id` to **all auto-instrumented child spans** (nodes, agents, tool calls, LLM requests) within that context:

```python
from openinference.instrumentation import using_attributes

with tracer.start_as_current_span("research_writer_query", attributes={...}):
    with using_attributes(user_id=user_id, session_id=session_id):
        result = app.invoke(
            {"query": user_input, ...},
            config=config,
        )
```

Without `using_attributes`, agent and LLM spans won't carry the user/session attributes — they'll appear in Lumoz without session grouping.

### Optional tracing — use `nullcontext` as a no-op

When tracing is disabled, use `contextlib.nullcontext` so the rest of the code doesn't need conditionals:

```python
from contextlib import nullcontext

span_ctx = (
    tracer.start_as_current_span("research_writer_query", attributes={...})
    if tracing_enabled else nullcontext()
)
attr_ctx = (
    using_attributes(user_id=user_id, session_id=session_id)
    if tracing_enabled else nullcontext()
)

with span_ctx, attr_ctx:
    result = app.invoke(...)
```

## 5. LangGraph Config — Thread and Session IDs

Pass session context through the LangGraph `config` dict. This drives `MemorySaver` (conversation persistence) and also makes the IDs available to any graph node that reads config:

```python
config = {
    "configurable": {
        "thread_id": session_id,   # MemorySaver key — persists conversation per session
        "user_id": user_id,
        "session_id": session_id,
    },
    "metadata": {
        "session_id": session_id,
        "user_id": user_id,
    },
}

result = app.invoke(input_state, config=config)
```

`thread_id` is the `MemorySaver` key — using `session_id` as the thread ID means each user session gets its own persistent conversation history.

## 6. Graceful Shutdown

Flush buffered spans before the process exits to avoid losing trace data:

```python
if tracing_enabled and tracer_provider:
    tracer_provider.force_flush()
```

## What Gets Captured

`LangChainInstrumentor` automatically captures:

| Span Type | Attributes |
|-----------|-----------|
| **Graph node** | Node name, input/output state |
| **Agent** | Agent name, instructions, messages |
| **LLM** | Model name, provider, temperature, token counts, input/output messages |
| **Tool** | Tool name, description, input arguments, retrieved documents |

## Trace Structure

Running this example produces a trace like:

```
research_writer_query               (root — manual span)
├── prepare_research                (NODE)
├── researcher                      (NODE — agent subgraph)
│   └── Research Agent              (AGENT)
│       ├── ChatOpenAI              (LLM)
│       ├── vector_search           (TOOL)
│       └── ChatOpenAI              (LLM)
├── save_research                   (NODE)
├── prepare_writing                 (NODE)
├── writer                          (NODE — agent subgraph)
│   └── Writer Agent                (AGENT)
│       └── ChatOpenAI              (LLM)
└── save_response                   (NODE)
```

The root span carries `user.id`, `session.id`, and `query`. All child spans inherit `user.id` and `session.id` via `using_attributes`, which is what Lumoz uses to group traces into sessions.
