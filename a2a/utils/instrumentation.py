# Copyright 2025 Lumoz AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenInference instrumentation for A2A Friend Scheduling Agents
Sends telemetry data to Lumoz platform
"""

import os
import base64
from typing import Optional
from opentelemetry import trace, context as otel_context
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource


class OTelPropagationMiddleware:
    """Pure ASGI middleware that extracts W3C traceparent from incoming HTTP
    headers and attaches it as the current OTel context for the request lifetime.

    Uses raw ASGI (not BaseHTTPMiddleware) to avoid the Starlette task-switching
    issue where ContextVar tokens created in one asyncio Task cannot be reset in
    the streaming response task, causing 'Failed to detach context' errors.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        # ASGI headers are List[Tuple[bytes, bytes]] — decode with latin-1 per HTTP spec
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])}
        parent_ctx = extract(headers)
        token = otel_context.attach(parent_ctx)
        try:
            await self.app(scope, receive, send)
        finally:
            otel_context.detach(token)


def make_trace_http_kwargs() -> dict:
    """Returns http_kwargs dict with W3C traceparent injected, for use as
    the http_kwargs= argument to A2AClient.send_message() (a2a-sdk 0.2.x)."""
    headers = {}
    inject(headers)
    return {"headers": headers} if headers else {}


def make_trace_service_parameters() -> dict:
    """Returns a service_parameters dict with W3C traceparent injected, for use as
    ClientCallContext(service_parameters=...) in a2a-sdk 1.0.x."""
    headers: dict = {}
    inject(headers)
    return headers


def configure_openinference(service_name: str = "a2a-friend-scheduling") -> Optional[trace_sdk.TracerProvider]:
    """
    Configure OpenInference with Lumoz SaaS endpoint

    Args:
        service_name: Name of the service for telemetry identification

    Returns:
        TracerProvider instance if configuration successful, None otherwise
    """
    # Get configuration from environment
    otel_endpoint = os.environ.get(
        "OTEL_ENDPOINT",
        "https://tp9jv7tcq3.execute-api.us-east-1.amazonaws.com/dev/proxy/v1/traces"
    )
    api_key = os.environ.get("LUMOZ_API_KEY", "")

    if not api_key:
        print(f"⚠️  LUMOZ_API_KEY not found for {service_name} - telemetry will not be sent")
        return None

    # Setup OTLP exporter with Basic Auth headers
    # API key contains client_id:client_secret format
    encoded_credentials = base64.b64encode(api_key.encode('utf-8')).decode('utf-8')
    headers = {
        "authorization": f"Basic {encoded_credentials}"
    }

    print(f"✅ LUMOZ_API_KEY found: {api_key[:20]}... (length: {len(api_key)})")
    print(f"✅ Sending traces from {service_name} to: {otel_endpoint}")

    # Create OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=otel_endpoint,
        headers=headers
    )

    # Configure tracer with service name
    resource = Resource.create({
        "service.name": service_name,
        "service.namespace": "demo",
        "deployment.environment": "dev"
    })
    tracer_provider = trace_sdk.TracerProvider(resource=resource)

    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Instrument Google ADK if available
    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor
        GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
        print(f"✅ Google ADK instrumentation enabled for {service_name}")
    except ImportError:
        print(f"⚠️  Google ADK instrumentor not available for {service_name}")

    # Instrument LangChain if available
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print(f"✅ LangChain instrumentation enabled for {service_name}")
    except ImportError:
        pass

    # Instrument CrewAI if available
    try:
        from openinference.instrumentation.crewai import CrewAIInstrumentor
        CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
        print(f"✅ CrewAI instrumentation enabled for {service_name}")
    except ImportError:
        pass

    # Instrument LiteLLM if available (used by CrewAI for LLM calls)
    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
        print(f"✅ LiteLLM instrumentation enabled for {service_name}")
    except ImportError:
        pass

    print(f"✅ OpenTelemetry instrumentation enabled for {service_name}")

    # Set up OTLP exporter with SimpleSpanProcessor for immediate export
    span_processor = SimpleSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Add console exporter for local debugging (optional - comment out if too verbose)
    enable_console_debug = os.environ.get("OTEL_DEBUG_CONSOLE", "false").lower() == "true"
    if enable_console_debug:
        console_processor = SimpleSpanProcessor(ConsoleSpanExporter())
        tracer_provider.add_span_processor(console_processor)
        print(f"✅ Console span exporter enabled for debugging ({service_name})")

    return tracer_provider


def flush_telemetry(tracer_provider: Optional[trace_sdk.TracerProvider]) -> None:
    """
    Flush remaining telemetry data to Lumoz platform

    Args:
        tracer_provider: TracerProvider instance to flush
    """
    if tracer_provider:
        print("\n✓ Flushing traces to Lumoz platform...")
        tracer_provider.force_flush()
        print("✓ Telemetry flushed successfully")
