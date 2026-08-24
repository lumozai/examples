# Claude Agent SDK — Get Started

A minimal Claude Agent SDK app instrumented with OpenInference and traced to Lumoz.

One query delegates to a sub-agent, which runs a shell command. That produces nested agent spans with a tool span underneath, so the first run shows you the shape of an agent trace in the Console.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
```

The Claude Agent SDK runs the Claude Code CLI, which requires Node.js 18+.

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | used by the Claude Agent SDK |
| `LUMOZ_API_KEY` | no | enables Lumoz tracing |
| `OTEL_ENDPOINT` | no | Lumoz ingest endpoint |

Without the Lumoz variables the agent still runs; tracing is skipped.

## Run

```bash
python agent.py
```

```
[tracing] Lumoz tracing configured — sending traces to https://api.lumoz.ai/proxy/v1/traces

Result: '5'
```

## What gets traced

```
AGENT  ClaudeAgentSDK.query      the top-level agent
TOOL   Agent                     the Task tool call
AGENT  ClaudeAgentSDK.Agent      the sub-agent
TOOL   Bash                      the shell command
```

Agent spans carry the model name, token counts (including cache reads and writes), cost, and a session id. Tool spans carry the tool name, its parameters, and its input and output.

## Instrumentation, in three steps

```python
tracer_provider = TracerProvider(resource=Resource.create({
    "service.name": "claude-agent-sdk-get-started",
    "deployment.environment": "development",
}))
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=..., headers=...)))
ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)
```

Two ordering rules:

- **Instrument before importing `claude_agent_sdk`.** Instrumentors work by patching, so anything imported first emits nothing.
- **Call `tracer_provider.shutdown()` before exiting.** A batch processor holds spans briefly, and a short-lived script can otherwise drop the last batch.

## No LLM spans

You get agent and tool spans, not LLM spans. The Claude Agent SDK runs the model loop inside the Claude Code CLI as a separate process, so individual model calls are not visible to Python instrumentation. The agent span still carries the model name, token counts, and cost.

---

Adapted from the [OpenInference instrumentor example](https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-claude-agent-sdk/examples) (Apache-2.0).
