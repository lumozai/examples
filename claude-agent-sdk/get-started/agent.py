"""Claude Agent SDK instrumented for Lumoz.

Runs one query that delegates to a sub-agent, which runs a shell command. That
produces nested AGENT spans with a TOOL span underneath, so you can see the
shape of an agent trace in the Lumoz Console on the first run.

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
        "service.name": "claude-agent-sdk-get-started",
        "deployment.environment": "development",
    })

    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    otlp_exporter = OTLPSpanExporter(endpoint=otel_endpoint, headers=headers)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Instrument before importing claude_agent_sdk, so the traced query is used.
    ClaudeAgentSDKInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"[tracing] Lumoz tracing configured — sending traces to {otel_endpoint}")
    return tracer_provider


# Module-level tracing setup — must run before claude_agent_sdk is imported.
tracer_provider = configure_lumoz_tracing()


# ============================================================================
# Agent
# ============================================================================

PROMPT = (
    "Use the Task tool to delegate a sub-agent. The sub-agent must use the Bash "
    "tool to run: `printf 'hello' | wc -c`. Return exactly the numeric output "
    "and nothing else."
)


async def run() -> None:
    from claude_agent_sdk import ClaudeAgentOptions, query

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Task", "TaskOutput"],
        permission_mode="bypassPermissions",
    )

    result = None
    async for message in query(prompt=PROMPT, options=options):
        if getattr(message, "result", None) is not None:
            result = message.result

    print(f"\nResult: {result!r}")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    finally:
        # Batch processors hold spans briefly — flush before the process exits.
        if tracer_provider is not None:
            tracer_provider.shutdown()
