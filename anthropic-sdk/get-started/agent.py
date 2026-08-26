"""Anthropic Python SDK instrumented for Lumoz.

A minimal tool-calling agent: the model asks for a tool, the code runs it,
the result goes back, the model answers. That loop produces the three span
kinds a real agent trace has — AGENT, TOOL, and LLM.

AnthropicInstrumentor captures the LLM calls automatically, including the
system prompt, the tool schemas sent to the model, and the tool calls it
returns. The @tracer.agent and @tracer.tool decorators cover your own code,
which the instrumentor cannot see.

Tracing setup follows vanilla/travel-video-analyzer.
"""

import base64
import json
import os
import sys

from dotenv import load_dotenv

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation import OITracer
    from openinference.instrumentation.config import TraceConfig
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable is required.")
    sys.exit(1)


# ============================================================================
# Lumoz Tracing
# ============================================================================

def configure_lumoz_tracing():
    """Configure OpenInference instrumentation for Lumoz observability.

    Returns an OITracer if both LUMOZ_API_KEY and OTEL_ENDPOINT are set,
    otherwise None. The agent works fine without tracing.
    """
    if not _OTEL_AVAILABLE:
        print("[tracing] OpenTelemetry packages not installed — tracing disabled")
        return None, None

    otel_endpoint = os.environ.get("OTEL_ENDPOINT")
    api_key = os.environ.get("LUMOZ_API_KEY")

    if not otel_endpoint or not api_key:
        print("[tracing] LUMOZ_API_KEY or OTEL_ENDPOINT not set — tracing disabled")
        return None, None

    encoded = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
    headers = {"authorization": f"Basic {encoded}"}

    resource = Resource.create({
        "service.name": "anthropic-sdk-get-started",
        "deployment.environment": "development",
    })

    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otel_endpoint, headers=headers))
    )

    # Must run before the Anthropic client is constructed.
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"[tracing] Lumoz tracing configured — sending traces to {otel_endpoint}")
    return OITracer(trace.get_tracer(__name__), TraceConfig()), tracer_provider


tracer, tracer_provider = configure_lumoz_tracing()

import anthropic  # noqa: E402  — imported after instrument()

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = "You are a concise weather assistant. Use the tool, then answer in one sentence."

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    }
]


# ============================================================================
# Tool
# ============================================================================

def _get_weather(city: str) -> dict:
    """Stand-in for a real weather API."""
    return {"city": city, "temp_c": 18, "condition": "partly cloudy"}


get_weather = tracer.tool(name="get_weather", description="Get the current weather for a city")(
    _get_weather
) if tracer else _get_weather


# ============================================================================
# Agent loop
# ============================================================================

def _run(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → {block.name}({json.dumps(block.input)})")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(get_weather(**block.input)),
                })
        messages.append({"role": "user", "content": results})


run = tracer.agent(name="weather_agent")(_run) if tracer else _run


if __name__ == "__main__":
    try:
        print(f"\nAnswer: {run('What is the weather in San Ramon?')}")
    finally:
        if tracer_provider is not None:
            tracer_provider.shutdown()
