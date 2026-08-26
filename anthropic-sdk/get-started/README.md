# Anthropic Python SDK — Get Started

A minimal tool-calling agent built on the Anthropic Python SDK, instrumented with OpenInference and traced to Lumoz.

The model asks for a tool, the code runs it, the result goes back, the model answers. That loop produces all three span kinds a real agent trace has.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
```

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Anthropic API access |
| `LUMOZ_API_KEY` | no | enables Lumoz tracing |
| `OTEL_ENDPOINT` | no | Lumoz ingest endpoint |

Without the Lumoz variables the agent still runs; tracing is skipped.

## Run

```bash
python agent.py
```

```
[tracing] Lumoz tracing configured — sending traces to https://api.lumoz.ai/proxy/v1/traces
  → get_weather({"city": "Paris"})

Answer: The weather in Paris is currently 18°C and partly cloudy.
```

## What gets traced

```
AGENT  weather_agent        your agent loop
  LLM  messages.create      first call — the model asks for the tool
  TOOL get_weather          your tool, executed locally
  LLM  messages.create      second call — the model answers
```

LLM spans carry the full request: `llm.input_messages` including the **system prompt**, `llm.tools` with each tool's JSON schema, `llm.output_messages` with the tool calls the model returned, token counts, and the finish reason.

Two mechanisms are at work:

- **`AnthropicInstrumentor`** patches the client, so every `messages.create` becomes an LLM span automatically.
- **`@tracer.agent` / `@tracer.tool`** cover your own code, which the instrumentor cannot see.

## Instrumentation, in three steps

```python
tracer_provider = TracerProvider(resource=Resource.create({
    "service.name": "anthropic-sdk-get-started",
    "deployment.environment": "development",
}))
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=..., headers=...)))
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

tracer = OITracer(trace.get_tracer(__name__), TraceConfig())
```

Two ordering rules:

- **Instrument before constructing `anthropic.Anthropic()`.** Instrumentors work by patching; a client built first emits nothing.
- **Call `tracer_provider.shutdown()` before exiting.** A batch processor holds spans briefly, and a short script can otherwise drop the last batch.

## Version pin

`requirements.txt` pins `anthropic>=1.0.0` and `openinference-instrumentation-anthropic>=2.0.0` — keep these in lockstep.

`openinference-instrumentation-anthropic` 2.x declares `anthropic>=1.0.0` as its instrumentation target. If the two packages fall out of sync (e.g. `anthropic<1.0` installed alongside instrumentor 2.x, or vice versa with 1.x), `AnthropicInstrumentor().instrument()` does **not** raise — `BaseInstrumentor.instrument()` checks the declared dependency range up front, and on a mismatch it logs one `ERROR`-level line (`DependencyConflict: requested "anthropic >= 1.0.0" but found "anthropic 0.125.0"`) and returns `None` without ever patching `messages.create`. The script keeps running — real API calls, correct answers, correct `@tracer.agent`/`@tracer.tool` spans — just with zero LLM spans, and nothing crashes to tell you why.

(Older versions of this file recommended `anthropic<1.0` paired with `openinference-instrumentation-anthropic` 1.x, because instrumentor 1.1.2 imported `anthropic.resources.completions`, which the 1.x SDK removed. That combination still works, but the 1.x line of the instrumentor is no longer the actively developed one — this project now tracks the 2.x/anthropic-1.x pairing instead.)
