# Agent-to-Agent OTel Transparency

## Overview / Goal

Each agent process normally produces an **independent trace** with its own `trace_id`. When the host agent calls a friend agent over A2A, the resulting spans appear as two unrelated traces in the observability platform — you cannot see the full call chain.

The goal is a **single connected trace tree** across all agent processes:

```
Host Agent (trace_id: abc123)
  └─ agent run
       └─ send_message → Karley Agent (trace_id: abc123)
                              └─ ADK/crew/graph execution
                                   └─ LLM call
```

This is achieved by propagating the W3C `traceparent` header on every outbound A2A HTTP call and extracting it on every inbound one. No changes to the a2a-sdk are needed — the SDK already copies `service_parameters` verbatim as HTTP headers on the client side, and the ASGI scope already contains incoming headers on the server side.

---

## Changes on the Sender Side

**File: `host_agent_adk/host/remote_agent_connection.py`**

One addition per outbound call: inject the current active span into a dict and pass that dict as `service_parameters` on `ClientCallContext`. The a2a-sdk transport layer copies this dict as the outgoing HTTP request headers.

```python
# Added import
from instrumentation import make_trace_service_parameters

# send_message() now injects traceparent on every outbound A2A call
async def send_message(self, message_request: SendMessageRequest) -> Task | None:
    context = ClientCallContext(service_parameters=make_trace_service_parameters())
    task: Task | None = None
    async for response in self.agent_client.send_message(message_request, context=context):
        if response.HasField("task"):
            task = response.task
    return task
```

**In `instrumentation.py` (shared module):**

```python
def make_trace_service_parameters() -> dict:
    """Returns a service_parameters dict with W3C traceparent injected."""
    headers: dict = {}
    inject(headers)      # writes traceparent + tracestate from the current active span
    return headers
```

`inject(headers)` reads the currently active OTel span and writes
`{"traceparent": "00-<trace_id>-<span_id>-01"}` into the dict. That becomes
the HTTP headers for the outbound A2A call, carrying the trace context to the
remote agent.

---

## Changes on the Receiver Side

**All friend agent `__main__.py` files (karley, nate, kaitlynn):**

One change: wrap the Starlette app with `OTelPropagationMiddleware` before handing it to uvicorn.

```python
# Before:
uvicorn.run(app, host=host, port=port)

# After:
uvicorn.run(OTelPropagationMiddleware(app), host=host, port=port)
```

**In `instrumentation.py` — the middleware:**

```python
class OTelPropagationMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        # Decode ASGI headers (List[Tuple[bytes, bytes]]) into a plain dict
        headers = {
            k.decode("latin-1"): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        parent_ctx = extract(headers)           # reads traceparent → OTel Context
        token = otel_context.attach(parent_ctx) # makes it the active context for this request
        try:
            await self.app(scope, receive, send)
        finally:
            otel_context.detach(token)
```

This runs before any handler or executor. Any span created downstream (by ADK,
CrewAI, LangGraph, LiteLLM instrumentors) is automatically a child of the remote
parent span that the host injected.

**Why raw ASGI instead of `BaseHTTPMiddleware`:**
Starlette's `BaseHTTPMiddleware` switches asyncio tasks for the response body. The
`contextvars` token created in the request task is invalid in the response task,
causing `ValueError: Failed to detach context` on streaming responses. Raw ASGI
keeps the same task for the full request/response cycle.

**CrewAI-specific — import ordering in `nate_agent_crewai/__main__.py`:**

CrewAI's `EventListener` calls `trace.set_tracer_provider(crewai_provider)` at
module-level the moment `crewai` is imported. If that import runs before
`configure_openinference()`, CrewAI owns the global OTel provider and remote
parents don't propagate correctly through CrewAI spans.

```python
# WRONG — crewai import hijacks global provider before we set ours
from agent import SchedulingAgent            # triggers crewai import → set_tracer_provider(crewai)
tracer_provider = configure_openinference(service_name="nate-agent-crewai")  # silently ignored

# CORRECT — our provider wins the race
load_dotenv(...)
from instrumentation import configure_openinference, OTelPropagationMiddleware
tracer_provider = configure_openinference(service_name="nate-agent-crewai")  # sets our provider first
...
from agent import SchedulingAgent            # crewai import now — its set_tracer_provider() is ignored
```
