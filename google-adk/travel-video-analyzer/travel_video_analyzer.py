"""
Travel Video Analyzer - Google ADK with Anthropic Claude

A multi-agent app using Google Agent Development Kit (ADK) with
Anthropic Claude models via LiteLLM integration.

Architecture:
    Orchestrator Agent (coordinates sub-agents)
        │
        ├── frame_extraction_agent - Extracts frames from videos
        ├── content_analysis_agent - Analyzes frames with Claude Vision
        └── tagging_agent - Generates tags from analysis

The Orchestrator LLM decides which sub-agents to call based on the task.

Instrumented with OpenInference for Lumoz observability.
"""

# Standard library
import asyncio
import base64
import json
import os
import sys
import tempfile
import traceback

# Third-party - Core
import anthropic
import cv2
from dotenv import load_dotenv

# Third-party - Google ADK
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Third-party - OpenTelemetry / OpenInference
import re
from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from openinference.instrumentation.anthropic import AnthropicInstrumentor

# Load environment variables
demo_env_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"))
load_dotenv(dotenv_path=demo_env_path)


# ============================================================================
# Image Stripping Span Processor
# ============================================================================
# When using Claude Vision (multimodal), base64 image data is captured in span
# attributes by OpenInference instrumentation. These large payloads can cause
# export failures (502 errors) due to size limits. This processor strips the
# base64 data before export while preserving the rest of the trace.

# Regex patterns to match base64 image data in various formats:
# 1. Data URI format: "data:image/jpeg;base64,/9j/4AAQ..." (used by LangChain)
# 2. Raw base64 JPEG: starts with "/9j/" (JPEG magic bytes in base64, used by Anthropic SDK)
# 3. Raw base64 PNG: starts with "iVBOR" (PNG magic bytes in base64)
BASE64_IMAGE_PATTERNS = [
    re.compile(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}'),  # Data URI format
    re.compile(r'/9j/[A-Za-z0-9+/=]{100,}'),  # Raw JPEG base64
    re.compile(r'iVBOR[A-Za-z0-9+/=]{100,}'),  # Raw PNG base64
]


class ImageStrippingSpanProcessor(BatchSpanProcessor):
    """
    Custom span processor that strips base64 image data from span attributes
    before exporting to prevent large payloads from causing export failures.

    Why this is needed:
    - OpenInference captures LLM inputs/outputs in span attributes
    - Vision calls include base64-encoded images which can be very large
    - Large spans cause 502 errors when exported to observability backends
    - OpenInference's TraceConfig masking doesn't work for nested JSON attributes

    Solution:
    - Intercept spans before export
    - Strip base64 image data, replacing with truncated placeholder
    - Preserve all other span data for debugging and analysis
    """

    def __init__(self, span_exporter: SpanExporter, max_image_chars: int = 100):
        """
        Args:
            span_exporter: The underlying exporter (e.g., OTLPSpanExporter)
            max_image_chars: Number of chars to keep from image data (for debugging)
        """
        super().__init__(span_exporter)
        self.max_image_chars = max_image_chars

    def _strip_images_from_value(self, value):
        """Recursively strip base64 image data from a value."""
        if isinstance(value, str):
            # Apply all patterns to strip different base64 image formats
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
        """Process span before export - strip image data from attributes."""
        if span.attributes:
            # Create modified attributes with images stripped
            modified_attrs = {}
            for key, value in span.attributes.items():
                modified_attrs[key] = self._strip_images_from_value(value)

            # Replace attributes on the span
            # Note: Accessing internal _attributes is necessary for modification
            if hasattr(span, '_attributes'):
                span._attributes = modified_attrs

        # Call parent to queue for batch export
        super().on_end(span)


# ============================================================================
# Lumoz Instrumentation Setup
# ============================================================================
# This section configures OpenInference tracing to send telemetry to Lumoz.
# Both ADK and Anthropic calls are instrumented under a single trace.

def configure_lumoz_tracing():
    """
    Configure OpenInference instrumentation for Lumoz observability.

    Required environment variables:
        OTEL_ENDPOINT: Lumoz trace endpoint
        LUMOZ_API_KEY: API key in format client_id:client_secret

    Returns:
        TracerProvider configured for Lumoz
    """
    # Get Lumoz endpoint and API key from environment
    otel_endpoint = os.environ.get("OTEL_ENDPOINT")
    api_key = os.environ.get("LUMOZ_API_KEY")

    if not otel_endpoint:
        raise ValueError("OTEL_ENDPOINT environment variable is required")
    if not api_key:
        raise ValueError("LUMOZ_API_KEY environment variable is required")

    # Configure authentication headers
    encoded = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
    headers = {"authorization": f"Basic {encoded}"}

    print(f"✅ Lumoz tracing configured")
    print(f"📡 Sending traces to: {otel_endpoint}")

    # Create resource with app metadata
    resource = Resource.create({
        "service.name": "travel-video-analyzer",
        "deployment.environment": "development",
    })

    # Create tracer provider with resource
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Configure OTLP exporter to send traces to Lumoz
    otlp_exporter = OTLPSpanExporter(
        endpoint=otel_endpoint,
        headers=headers
    )

    # Use ImageStrippingSpanProcessor to handle multimodal (vision) spans
    # This prevents 502 errors from large base64 image payloads
    span_processor = ImageStrippingSpanProcessor(otlp_exporter, max_image_chars=100)
    tracer_provider.add_span_processor(span_processor)

    # Instrument Google ADK - captures agent orchestration and tool calls
    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

    # Instrument Anthropic - captures LLM calls within ADK trace context
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider


# Initialize tracing before any ADK or Anthropic calls
tracer_provider = configure_lumoz_tracing()


# ============================================================================
# Configuration
# ============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

# Set for LiteLLM
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# Model configuration - using Claude via LiteLLM
MODEL = LiteLlm(model="anthropic/claude-sonnet-4-20250514")

# Direct Anthropic client for vision tool
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ============================================================================
# Tool Functions
# ============================================================================

def extract_video_frames(video_path: str, num_frames: int = 8) -> dict:
    """
    Extract key frames from a travel video for destination analysis.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract (default: 8).

    Returns:
        dict with video metadata and list of extracted frame paths.
    """
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    metadata = {
        "filename": os.path.basename(video_path),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}"
    }

    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    temp_dir = tempfile.mkdtemp(prefix="travel_frames_")
    extracted_frames = []

    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_frames.append({
                "path": frame_path,
                "frame_number": frame_num,
                "timestamp_seconds": round(frame_num / fps, 2) if fps > 0 else 0
            })

    cap.release()

    return {
        "success": True,
        "metadata": metadata,
        "frames": extracted_frames,
        "frame_paths": [f["path"] for f in extracted_frames],
        "frames_directory": temp_dir
    }


def analyze_frames_with_vision(frame_paths_json: str) -> dict:
    """
    Analyze video frames using Claude Vision to identify travel destination.

    Args:
        frame_paths_json: JSON string containing list of frame paths.

    Returns:
        dict with destination information including location, landmarks, activities.
    """
    # Parse frame paths
    try:
        frame_paths = json.loads(frame_paths_json) if isinstance(frame_paths_json, str) else frame_paths_json
    except:
        return {"error": "Invalid frame_paths_json format"}

    if not frame_paths:
        return {"error": "No frame paths provided"}

    # Build image content for Claude Vision
    image_content = []
    for frame_path in frame_paths[:8]:
        if os.path.exists(frame_path):
            with open(frame_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            image_content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}
            })

    if not image_content:
        return {"error": "Could not load any frame images"}

    image_content.append({
        "type": "text",
        "text": """Analyze these travel video frames and identify the destination.

Return JSON with this structure:
{
    "location": {"country": "", "city": "", "region": "", "specific_places": []},
    "landmarks": [],
    "activities": [],
    "landscape_type": [],
    "climate_indicators": "",
    "travel_style": "",
    "best_for": [],
    "estimated_budget": "",
    "description": "2-3 sentences describing the destination"
}

Return ONLY the JSON object, no other text."""
    })

    # Call Claude Vision
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": image_content}]
    )

    text = response.content[0].text if response.content else ""

    try:
        start, end = text.find("{"), text.rfind("}") + 1
        destination_info = json.loads(text[start:end]) if start >= 0 and end > start else {"raw": text}
    except json.JSONDecodeError:
        destination_info = {"raw": text}

    return {
        "success": True,
        "destination_info": destination_info,
        "frames_analyzed": len(image_content) - 1
    }


def generate_destination_tags(destination_info_json: str) -> dict:
    """
    Generate searchable tags for a travel destination.

    Args:
        destination_info_json: JSON string of destination information.

    Returns:
        dict with categorized tags for the destination.
    """
    try:
        destination_info = json.loads(destination_info_json) if isinstance(destination_info_json, str) else destination_info_json
    except:
        return {"error": "Invalid destination_info_json format"}

    if not destination_info:
        return {"error": "No destination info provided"}

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""Generate tags for this travel destination:

{json.dumps(destination_info, indent=2)}

Return JSON with lowercase, hyphenated tags:
{{
    "location_tags": ["country", "region", "city tags"],
    "activity_tags": ["activity-related tags"],
    "landscape_tags": ["landscape/scenery tags"],
    "travel_style_tags": ["travel style tags"],
    "all_tags": ["all unique tags combined"]
}}

Return ONLY the JSON object."""}]
    )

    text = response.content[0].text if response.content else ""

    try:
        start, end = text.find("{"), text.rfind("}") + 1
        tags = json.loads(text[start:end]) if start >= 0 and end > start else {"all_tags": []}
    except json.JSONDecodeError:
        tags = {"all_tags": []}

    return {"success": True, "tags": tags}


# ============================================================================
# Sub-Agent Definitions
# ============================================================================

frame_extraction_agent = LlmAgent(
    model=MODEL,
    name="frame_extraction_agent",
    description="Extracts key frames from video files for visual analysis.",
    instruction="""You are a frame extraction specialist.
When asked to extract frames from a video, use the extract_video_frames tool.
Return the frame paths so they can be analyzed by other agents.""",
    tools=[extract_video_frames],
)

content_analysis_agent = LlmAgent(
    model=MODEL,
    name="content_analysis_agent",
    description="Analyzes video frames using vision AI to identify travel destinations, landmarks, and activities.",
    instruction="""You are a travel content analyst with vision capabilities.
When given frame paths, use the analyze_frames_with_vision tool to identify:
- Location (country, city, specific places)
- Landmarks and points of interest
- Activities visible in the content
- Travel style and budget indicators
Return detailed destination information.""",
    tools=[analyze_frames_with_vision],
)

tagging_agent = LlmAgent(
    model=MODEL,
    name="tagging_agent",
    description="Generates searchable tags and categories for travel destinations.",
    instruction="""You are a travel content tagger.
When given destination information, use the generate_destination_tags tool to create:
- Location tags (country, region, city)
- Activity tags
- Landscape/scenery tags
- Travel style tags
Return comprehensive, searchable tags.""",
    tools=[generate_destination_tags],
)


# ============================================================================
# Orchestrator Agent
# ============================================================================

orchestrator_agent = LlmAgent(
    model=MODEL,
    name="travel_video_orchestrator",
    description="Coordinates the analysis of travel videos by delegating to specialized sub-agents.",
    instruction="""You are the orchestrator for travel video analysis.

Your job is to coordinate a team of specialized agents to analyze travel videos:

1. **frame_extraction_agent**: Call this first to extract frames from the video file
2. **content_analysis_agent**: Call this with the extracted frame paths to identify the destination
3. **tagging_agent**: Call this with the destination info to generate searchable tags

For each user request:
1. Understand what video needs to be analyzed
2. Delegate to frame_extraction_agent to get frames
3. Pass frame paths to content_analysis_agent for vision analysis
4. Pass destination info to tagging_agent for tag generation
5. Compile and present the final results

Always explain what you're doing and provide clear status updates.
When complete, summarize the destination info and tags.""",
    sub_agents=[frame_extraction_agent, content_analysis_agent, tagging_agent],
)


# ============================================================================
# Runner
# ============================================================================

async def analyze_video_async(
    video_path: str,
    user_id: str = "demo_user",
    session_id: str = "demo_session"
) -> dict:
    """
    Analyze a travel video using the ADK multi-agent system.

    Args:
        video_path: Path to the video file
        user_id: User identifier for tracing
        session_id: Session identifier for grouping runs

    Returns:
        dict with analysis results
    """
    print(f"\n{'='*60}")
    print("TRAVEL VIDEO ANALYZER - Google ADK + Anthropic")
    print(f"{'='*60}")
    print(f"\nVideo: {video_path}")
    print(f"Session: {session_id}")
    print(f"User: {user_id}\n")

    # Create session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator_agent,
        app_name="travel_video_analyzer",
        session_service=session_service,
    )

    # Create a session with the specified ID
    session = await session_service.create_session(
        app_name="travel_video_analyzer",
        user_id=user_id,
        session_id=session_id,
    )

    # Get tracer for creating root span with session/user attributes
    tracer = trace.get_tracer("travel_video_analyzer")

    # Run the analysis
    user_message = f"Please analyze this travel video and generate destination tags: {video_path}"

    print("Starting analysis...\n")

    # Execute and collect results within a traced span
    results = {"messages": []}

    with tracer.start_as_current_span(
        "analyze_video",
        attributes={
            "session.id": session_id,
            "user.id": user_id,
        }
    ) as span:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=user_message)]
            ),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"Agent: {part.text[:300]}{'...' if len(part.text) > 300 else ''}")
                        results["messages"].append(part.text)

    return results


def analyze_video(
    video_path: str,
    user_id: str = "demo_user",
    session_id: str = "demo_session"
) -> dict:
    """Sync wrapper for analyze_video_async."""
    return asyncio.run(analyze_video_async(video_path, user_id, session_id))


def print_results(results: dict):
    """Pretty print results."""
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    if results.get("messages"):
        print("\nFinal output from agents:")
        for msg in results["messages"][-3:]:  # Show last 3 messages
            print(f"\n{msg}")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "./videos/travel_sample.mp4"

    print(f"🎬 Travel Video Analyzer (Google ADK + Anthropic)")

    try:
        results = analyze_video(video_path)
        print_results(results)
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
