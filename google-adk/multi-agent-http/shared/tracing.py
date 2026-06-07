"""OpenTelemetry and OpenInference setup shared by both services."""

from __future__ import annotations

import base64
import os
from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


_TRACING_CONFIGURED = False
_REQUESTS_INSTRUMENTED = False


def _instrument_openinference(tracer_provider: TracerProvider) -> None:
    """Install optional OpenInference instrumentors if the packages are present."""
    try:
        from openinference.instrumentation.google_adk import GoogleADKInstrumentor

        GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # pragma: no cover - defensive for version drift.
        print(f"OpenInference Google ADK instrumentation skipped: {exc}")

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # pragma: no cover - defensive for version drift.
        print(f"OpenInference OpenAI instrumentation skipped: {exc}")

    try:
        from openinference.instrumentation.litellm import LiteLLMInstrumentor

        LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception as exc:  # pragma: no cover - defensive for version drift.
        print(f"OpenInference LiteLLM instrumentation skipped: {exc}")


def configure_tracing(service_name: str) -> TracerProvider:
    """Configure tracing and Lumoz export for a single service process."""
    global _TRACING_CONFIGURED

    if _TRACING_CONFIGURED:
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            return provider
        raise RuntimeError("Tracing was configured by a non-SDK provider")

    resource = Resource.create(
        {
            "service.name": service_name,
            "deployment.environment": os.environ.get("DEPLOYMENT_ENVIRONMENT", "development"),
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    otel_endpoint = os.environ.get("OTEL_ENDPOINT")
    lumoz_api_key = os.environ.get("LUMOZ_API_KEY")
    if otel_endpoint and lumoz_api_key:
        encoded_key = base64.b64encode(lumoz_api_key.encode("utf-8")).decode("utf-8")
        exporter = OTLPSpanExporter(
            endpoint=otel_endpoint,
            headers={"authorization": f"Basic {encoded_key}"},
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        print(f"Lumoz tracing enabled for {service_name}: {otel_endpoint}")
    else:
        print(
            f"Lumoz tracing not enabled for {service_name}; "
            "set OTEL_ENDPOINT and LUMOZ_API_KEY to export traces."
        )

    _instrument_openinference(tracer_provider)
    _TRACING_CONFIGURED = True
    return tracer_provider


def instrument_fastapi_app(app: FastAPI, service_name: str) -> None:
    """Configure tracing and instrument the given FastAPI app."""
    tracer_provider = configure_tracing(service_name)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)


def instrument_requests() -> None:
    """Instrument requests once so outbound HTTP spans carry trace context."""
    global _REQUESTS_INSTRUMENTED
    if not _REQUESTS_INSTRUMENTED:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor

        RequestsInstrumentor().instrument()
        _REQUESTS_INSTRUMENTED = True


def set_span_attributes(attributes: dict[str, Any]) -> None:
    """Attach safe attributes to the current span when one exists."""
    span = trace.get_current_span()
    if not span or not span.is_recording():
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)
