"""Claude Agent SDK with your own prompt, tools, and sub-agent captured in traces.

The get-started example shows what ClaudeAgentSDKInstrumentor gives you on its
own: AGENT and TOOL spans with the model, token counts, and cost. What it does
not give you is anything you wrote — your system prompt, the tool schemas you
defined, or your sub-agent instructions. The instrumentor never reads
ClaudeAgentOptions, and the SDK runs the model loop in a separate process, so
none of it reaches a span.

All of it is in your process, though. This example opens one span around the
run and attaches it with the standard OpenInference keys, so a trace carries
both halves: what you configured, and what the agent did.

Adapted from the OpenInference instrumentor example (Apache-2.0):
https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-claude-agent-sdk/examples
"""

import asyncio
import base64
import os
import sys

from dotenv import load_dotenv

try:
    from opentelemetry import trace
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation.claude_agent_sdk import ClaudeAgentSDKInstrumentor
    from openinference.semconv.trace import SpanAttributes

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

from config_attributes import config_attributes

load_dotenv()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable is required.")
    sys.exit(1)


# ============================================================================
# Lumoz Tracing
# ============================================================================

def configure_lumoz_tracing():
    """Configure OpenInference instrumentation for Lumoz observability.

    Returns a TracerProvider if both LUMOZ_API_KEY and OTEL_ENDPOINT are set,
    otherwise returns None. The agent works fine without tracing.
    """
    if not _OTEL_AVAILABLE:
        print("[tracing] OpenTelemetry packages not installed — tracing disabled")
        return None

    otel_endpoint = os.environ.get("OTEL_ENDPOINT")
    api_key = os.environ.get("LUMOZ_API_KEY")

    if not otel_endpoint or not api_key:
        print("[tracing] LUMOZ_API_KEY or OTEL_ENDPOINT not set — tracing disabled")
        return None

    encoded = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
    headers = {"authorization": f"Basic {encoded}"}

    resource = Resource.create({
        "service.name": "claude-agent-sdk-custom-tools",
        "deployment.environment": "development",
    })

    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otel_endpoint, headers=headers))
    )

    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"[tracing] Lumoz tracing configured — sending traces to {otel_endpoint}")
    return tracer_provider


# Module-level tracing setup — must run before claude_agent_sdk is imported.
tracer_provider = configure_lumoz_tracing()

PROMPT = "Look up the price of SKU ABC-123, then have the auditor check it."


async def run() -> None:
    from claude_agent_sdk import (
        AgentDefinition,
        ClaudeAgentOptions,
        create_sdk_mcp_server,
        query,
        tool,
    )

    # --- Everything in this block is yours. None of it reaches a span unless
    # --- you put it there, which is what config_attributes() does below.

    @tool("lookup_price", "Look up the list price of a SKU", {"sku": str})
    async def lookup_price(args):
        return {"content": [{"type": "text", "text": "19.99"}]}

    tools = [lookup_price]
    pricing = create_sdk_mcp_server(name="pricing", version="1.0.0", tools=tools)

    options = ClaudeAgentOptions(
        system_prompt="You are a pricing assistant. Never quote a price you did not look up.",
        agents={
            "auditor": AgentDefinition(
                description="Checks a quote against policy",
                prompt="You audit prices. Flag anything above 100.",
                tools=["mcp__pricing__lookup_price"],
            )
        },
        mcp_servers={"pricing": pricing},
        allowed_tools=["mcp__pricing__lookup_price", "Task", "TaskOutput"],
        permission_mode="bypassPermissions",
    )

    if tracer_provider is None:
        await _query(options)
        return

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("agent.request") as span:
        # LLM, not AGENT: Lumoz only records tool schemas on LLM spans, so an
        # AGENT span would drop them silently at ingest.
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")

        for key, value in config_attributes(options, PROMPT, tools=tools).items():
            span.set_attribute(key, value)

        result = await _query(options)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))


async def _query(options) -> str | None:
    from claude_agent_sdk import query

    result = None
    async for message in query(prompt=PROMPT, options=options):
        if getattr(message, "result", None) is not None:
            result = message.result

    print(f"\nResult: {result!r}")
    return result


if __name__ == "__main__":
    try:
        asyncio.run(run())
    finally:
        # Batch processors hold spans briefly — flush before the process exits.
        if tracer_provider is not None:
            tracer_provider.shutdown()
