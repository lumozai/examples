"""
Travel Video Analyzer - Framework-less Multi-Agent

Uses OpenInference decorators (@tracer.chain, @tracer.tool) with direct Anthropic API.
No LangGraph, LangChain, or other agent frameworks.

Pipeline: extract_frames → analyze_frames → generate_tags
"""

import base64
import json
import os
import re
import sys
import tempfile

import anthropic
import cv2
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation import OITracer
from openinference.instrumentation.config import TraceConfig
from openinference.instrumentation.anthropic import AnthropicInstrumentor

load_dotenv()

# ============================================================================
# Image Stripping Span Processor
# ============================================================================

# Patterns to match base64 image data in span attributes
BASE64_IMAGE_PATTERNS = [
    re.compile(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}'),  # data URI
    re.compile(r'/9j/[A-Za-z0-9+/=]{100,}'),  # JPEG magic bytes
    re.compile(r'iVBOR[A-Za-z0-9+/=]{100,}'),  # PNG magic bytes
]


class ImageStrippingSpanProcessor(BatchSpanProcessor):
    """Strips base64 image data from spans before export to prevent large payloads."""

    def __init__(self, span_exporter: SpanExporter, max_image_chars: int = 100):
        super().__init__(span_exporter)
        self.max_image_chars = max_image_chars

    def _strip_images_from_value(self, value):
        """Recursively strip base64 image data from a value."""
        if isinstance(value, str):
            result = value
            for pattern in BASE64_IMAGE_PATTERNS:
                result = pattern.sub(
                    lambda m: m.group(0)[:self.max_image_chars] + '...[IMAGE_TRUNCATED]',
                    result
                )
            return result
        elif isinstance(value, (list, tuple)):
            return type(value)(self._strip_images_from_value(v) for v in value)
        elif isinstance(value, dict):
            return {k: self._strip_images_from_value(v) for k, v in value.items()}
        return value

    def on_end(self, span: ReadableSpan) -> None:
        """Process span on end, stripping image data from attributes."""
        if span.attributes:
            modified_attrs = {}
            for key, value in span.attributes.items():
                modified_attrs[key] = self._strip_images_from_value(value)
            # Modify the internal attributes (this is a private API but necessary)
            if hasattr(span, '_attributes'):
                span._attributes = modified_attrs
        super().on_end(span)


# ============================================================================
# Tracing Setup
# ============================================================================

def _init_tracing():
    endpoint = os.environ.get("OTEL_ENDPOINT")
    api_key = os.environ.get("LUMOZ_API_KEY")
    if not endpoint or not api_key:
        raise ValueError("OTEL_ENDPOINT and LUMOZ_API_KEY required")

    resource = Resource.create({"service.name": "travel-video-analyzer-vanilla"})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    headers = {"authorization": f"Basic {base64.b64encode(api_key.encode()).decode()}"}
    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    # Use ImageStrippingSpanProcessor to truncate base64 images before export
    # This prevents large payloads that cause timeouts/retries and duplicate spans
    provider.add_span_processor(ImageStrippingSpanProcessor(exporter))

    # Auto-instrument Anthropic client (must be before client creation)
    AnthropicInstrumentor().instrument(tracer_provider=provider)

    print(f"Tracing → {endpoint}")
    return OITracer(trace.get_tracer(__name__), TraceConfig())


tracer = _init_tracing()
client = anthropic.Anthropic()  # Created AFTER instrumentation
MODEL = "claude-sonnet-4-20250514"


# ============================================================================
# Tools
# ============================================================================

@tracer.tool(name="extract_frames", description="Extract key frames from video using OpenCV")
def extract_frames(video_path: str, num_frames: int = 2) -> dict:
    """Extract key frames from a video."""
    print(f"\n[extract_frames] {video_path}")

    if not os.path.exists(video_path):
        return {"error": f"Video not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open: {video_path}"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_dir = tempfile.mkdtemp(prefix="frames_")
    frames = []

    for i, idx in enumerate([int(i * total / num_frames) for i in range(num_frames)]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(path, frame)
            frames.append(path)

    cap.release()
    print(f"[extract_frames] → {len(frames)} frames")

    return {
        "success": True,
        "frame_paths": frames,
        "metadata": {
            "filename": os.path.basename(video_path),
            "duration_seconds": round(total / fps, 2) if fps > 0 else 0,
        }
    }


@tracer.tool(name="analyze_frames", description="Analyze frames with Claude Vision")
def analyze_frames(frame_paths: list[str]) -> dict:
    """Analyze frames with Claude Vision to identify destination."""
    print(f"\n[analyze_frames] {len(frame_paths)} frames")

    if not frame_paths:
        return {"error": "No frames provided"}

    # Build image content for Claude Vision
    images = []
    for path in frame_paths[:8]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            images.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": data}
            })

    if not images:
        return {"error": "Could not load images"}

    # Call Claude Vision
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": images + [{
                "type": "text",
                "text": """Analyze these travel video frames and identify the destination.

Return JSON:
{
    "location": {"country": "", "city": "", "region": "", "specific_places": []},
    "landmarks": [],
    "activities": [],
    "landscape_type": [],
    "description": "2-3 sentences describing the destination"
}

Return ONLY the JSON object."""
            }]
        }]
    )

    text = response.content[0].text
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        destination = json.loads(text[start:end]) if start >= 0 and end > start else {"raw": text}
    except json.JSONDecodeError:
        destination = {"raw": text}

    loc = destination.get("location", {})
    print(f"[analyze_frames] → {loc.get('city', '?')}, {loc.get('country', '?')}")

    return {"success": True, "destination_info": destination}


@tracer.tool(name="generate_tags", description="Generate searchable tags from destination info")
def generate_tags(destination_info: dict) -> dict:
    """Generate searchable tags from destination info."""
    print(f"\n[generate_tags]")

    if not destination_info:
        return {"error": "No destination info"}

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Generate tags for this travel destination:

{json.dumps(destination_info, indent=2)}

Return JSON with lowercase, hyphenated tags:
{{
    "location_tags": [],
    "activity_tags": [],
    "landscape_tags": [],
    "all_tags": ["all unique tags combined"]
}}

Return ONLY the JSON object."""
        }]
    )

    text = response.content[0].text
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        tags = json.loads(text[start:end]) if start >= 0 and end > start else {"all_tags": []}
    except json.JSONDecodeError:
        tags = {"all_tags": []}

    print(f"[generate_tags] → {len(tags.get('all_tags', []))} tags")
    return {"success": True, "tags": tags}


# ============================================================================
# Orchestrator
# ============================================================================

@tracer.chain(name="analyze_video")
def analyze_video(video_path: str) -> dict:
    """Run the video analysis pipeline: extract → analyze → tag."""
    print(f"\n{'='*50}")
    print("TRAVEL VIDEO ANALYZER (Framework-less)")
    print(f"{'='*50}")
    print(f"Video: {video_path}")

    # Step 1: Extract frames
    extraction = extract_frames(video_path)
    if not extraction.get("success"):
        return {"error": extraction.get("error")}

    # Step 2: Analyze with Vision
    analysis = analyze_frames(extraction["frame_paths"])
    if not analysis.get("success"):
        return {"error": analysis.get("error")}

    # Step 3: Generate tags
    tagging = generate_tags(analysis["destination_info"])

    return {
        "destination_info": analysis.get("destination_info", {}),
        "tags": tagging.get("tags", {}),
        "metadata": extraction.get("metadata", {}),
    }


def run_analysis(input: str) -> dict:
    """Entry point for experiment harness."""
    path = input.rsplit(":", 1)[-1].strip() if ":" in input else input
    return analyze_video(path)


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "./videos/travel-video-0.mp4"
    result = analyze_video(video)

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)

    if dest := result.get("destination_info"):
        loc = dest.get("location", {})
        print(f"Location: {loc.get('city', '?')}, {loc.get('country', '?')}")
        if desc := dest.get("description"):
            print(f"Description: {desc}")

    if tags := result.get("tags", {}).get("all_tags"):
        print(f"Tags: {', '.join(tags)}")
