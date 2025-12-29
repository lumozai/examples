"""
Travel Video Analyzer - LangGraph Multi-Agent with Supervisor

A multi-agent app using LangGraph with a supervisor pattern, similar to ADK's orchestrator.
The supervisor LLM decides which specialized agent to call next.

Architecture:
    Supervisor Agent (coordinates worker agents)
        │
        ├── frame_extraction_agent - Extracts frames from videos
        ├── content_analysis_agent - Analyzes frames with Claude Vision
        └── tagging_agent - Generates tags from analysis

The Supervisor LLM decides which worker agent to call based on the task.

Instrumented with OpenInference for Lumoz observability.
"""

# Standard library
import base64
import json
import operator
import os
import re
import sys
import tempfile
import traceback
from typing import Annotated, Literal, TypedDict

# Third-party - Core
import anthropic
import cv2
from dotenv import load_dotenv

# Third-party - LangGraph/LangChain
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool

# Third-party - OpenTelemetry / OpenInference
from opentelemetry import trace
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

# Load environment variables
env_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))
load_dotenv(dotenv_path=env_path)


# ============================================================================
# Image Stripping Span Processor
# ============================================================================

BASE64_IMAGE_PATTERNS = [
    re.compile(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{100,}'),
    re.compile(r'/9j/[A-Za-z0-9+/=]{100,}'),
    re.compile(r'iVBOR[A-Za-z0-9+/=]{100,}'),
]


class ImageStrippingSpanProcessor(BatchSpanProcessor):
    """Strips base64 image data from spans before export."""

    def __init__(self, span_exporter: SpanExporter, max_image_chars: int = 100):
        super().__init__(span_exporter)
        self.max_image_chars = max_image_chars

    def _strip_images_from_value(self, value):
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
        if span.attributes:
            modified_attrs = {}
            for key, value in span.attributes.items():
                modified_attrs[key] = self._strip_images_from_value(value)
            if hasattr(span, '_attributes'):
                span._attributes = modified_attrs
        super().on_end(span)


# ============================================================================
# Lumoz Instrumentation Setup
# ============================================================================

def configure_lumoz_tracing():
    """Configure OpenInference instrumentation for Lumoz observability."""
    otel_endpoint = os.environ.get("OTEL_ENDPOINT")
    api_key = os.environ.get("LUMOZ_API_KEY")

    if not otel_endpoint:
        raise ValueError("OTEL_ENDPOINT environment variable is required")
    if not api_key:
        raise ValueError("LUMOZ_API_KEY environment variable is required")

    encoded = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
    headers = {"authorization": f"Basic {encoded}"}

    print(f"Lumoz tracing configured")
    print(f"Sending traces to: {otel_endpoint}")

    resource = Resource.create({
        "service.name": "travel-video-analyzer-langgraph",
        "deployment.environment": "development",
    })

    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    otlp_exporter = OTLPSpanExporter(endpoint=otel_endpoint, headers=headers)
    span_processor = ImageStrippingSpanProcessor(otlp_exporter, max_image_chars=100)
    tracer_provider.add_span_processor(span_processor)

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider


# Initialize tracing first
tracer_provider = configure_lumoz_tracing()


# ============================================================================
# Configuration
# ============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

# LangChain model
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Direct Anthropic client for vision tool
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ============================================================================
# Shared State
# ============================================================================

class AgentState(TypedDict):
    """State shared across all agents."""
    messages: Annotated[list[BaseMessage], operator.add]
    video_path: str
    frame_paths: list[str]
    destination_info: dict
    tags: dict
    next_agent: str


# ============================================================================
# Tool Definitions
# ============================================================================

@tool
def extract_video_frames(video_path: str, num_frames: int = 8) -> str:
    """
    Extract key frames from a travel video for destination analysis.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract (default: 8).

    Returns:
        JSON string with video metadata and list of extracted frame paths.
    """
    if not os.path.exists(video_path):
        return json.dumps({"error": f"Video file not found: {video_path}"})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return json.dumps({"error": f"Could not open video: {video_path}"})

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
            extracted_frames.append(frame_path)

    cap.release()

    return json.dumps({
        "success": True,
        "metadata": metadata,
        "frame_paths": extracted_frames,
        "message": f"Extracted {len(extracted_frames)} frames from {metadata['filename']}"
    })


@tool
def analyze_frames_with_vision(frame_paths_json: str) -> str:
    """
    Analyze video frames using Claude Vision to identify travel destination.

    Args:
        frame_paths_json: JSON string containing list of frame paths.

    Returns:
        JSON string with destination information.
    """
    try:
        frame_paths = json.loads(frame_paths_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format"})

    if not frame_paths:
        return json.dumps({"error": "No frame paths provided"})

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
        return json.dumps({"error": "Could not load any frame images"})

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

    return json.dumps({
        "success": True,
        "destination_info": destination_info,
        "frames_analyzed": len(image_content) - 1
    })


@tool
def generate_destination_tags(destination_info_json: str) -> str:
    """
    Generate searchable tags for a travel destination.

    Args:
        destination_info_json: JSON string of destination information.

    Returns:
        JSON string with categorized tags.
    """
    try:
        destination_info = json.loads(destination_info_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format"})

    if not destination_info:
        return json.dumps({"error": "No destination info provided"})

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

    return json.dumps({"success": True, "tags": tags})


# ============================================================================
# Worker Agents (ReAct agents with specialized tools)
# ============================================================================

# Frame Extraction Agent
frame_extraction_agent = create_react_agent(
    llm,
    tools=[extract_video_frames],
    prompt="""You are a frame extraction specialist.
Your job is to extract key frames from video files for visual analysis.
When given a video path, use the extract_video_frames tool to extract frames.
Report what you extracted and the frame paths."""
)

# Content Analysis Agent
content_analysis_agent = create_react_agent(
    llm,
    tools=[analyze_frames_with_vision],
    prompt="""You are a travel content analyst with vision capabilities.
Your job is to analyze video frames and identify travel destinations.
When given frame paths, use the analyze_frames_with_vision tool to identify:
- Location (country, city, specific places)
- Landmarks and points of interest
- Activities visible in the content
Report the destination information you discovered."""
)

# Tagging Agent
tagging_agent = create_react_agent(
    llm,
    tools=[generate_destination_tags],
    prompt="""You are a travel content tagger.
Your job is to generate searchable tags for travel destinations.
When given destination information, use the generate_destination_tags tool to create:
- Location tags (country, region, city)
- Activity tags
- Landscape/scenery tags
- Travel style tags
Report all the tags you generated."""
)


# ============================================================================
# Worker Agent Nodes
# ============================================================================

def frame_extraction_node(state: AgentState) -> AgentState:
    """Run the frame extraction agent."""
    print("\n[Frame Extraction Agent] Starting...")

    video_path = state["video_path"]
    result = frame_extraction_agent.invoke({
        "messages": [HumanMessage(content=f"Extract frames from this video: {video_path}")]
    })

    # Extract frame paths from the result
    last_message = result["messages"][-1]
    frame_paths = []

    # Parse frame paths from tool results
    for msg in result["messages"]:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            if "frame_paths" in msg.content:
                try:
                    data = json.loads(msg.content)
                    if "frame_paths" in data:
                        frame_paths = data["frame_paths"]
                except:
                    pass

    print(f"[Frame Extraction Agent] Extracted {len(frame_paths)} frames")

    return {
        "messages": [AIMessage(content=f"[Frame Extraction Agent] {last_message.content}")],
        "frame_paths": frame_paths
    }


def content_analysis_node(state: AgentState) -> AgentState:
    """Run the content analysis agent."""
    print("\n[Content Analysis Agent] Starting...")

    frame_paths = state.get("frame_paths", [])
    if not frame_paths:
        return {
            "messages": [AIMessage(content="[Content Analysis Agent] No frames to analyze")],
            "destination_info": {}
        }

    frame_paths_json = json.dumps(frame_paths)
    result = content_analysis_agent.invoke({
        "messages": [HumanMessage(content=f"Analyze these frames to identify the travel destination: {frame_paths_json}")]
    })

    last_message = result["messages"][-1]
    destination_info = {}

    # Parse destination info from tool results
    for msg in result["messages"]:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            if "destination_info" in msg.content:
                try:
                    data = json.loads(msg.content)
                    if "destination_info" in data:
                        destination_info = data["destination_info"]
                except:
                    pass

    location = destination_info.get("location", {})
    print(f"[Content Analysis Agent] Identified: {location.get('city', 'Unknown')}, {location.get('country', 'Unknown')}")

    return {
        "messages": [AIMessage(content=f"[Content Analysis Agent] {last_message.content}")],
        "destination_info": destination_info
    }


def tagging_node(state: AgentState) -> AgentState:
    """Run the tagging agent."""
    print("\n[Tagging Agent] Starting...")

    destination_info = state.get("destination_info", {})
    if not destination_info:
        return {
            "messages": [AIMessage(content="[Tagging Agent] No destination info for tagging")],
            "tags": {}
        }

    destination_json = json.dumps(destination_info)
    result = tagging_agent.invoke({
        "messages": [HumanMessage(content=f"Generate tags for this destination: {destination_json}")]
    })

    last_message = result["messages"][-1]
    tags = {}

    # Parse tags from tool results
    for msg in result["messages"]:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            if "tags" in msg.content:
                try:
                    data = json.loads(msg.content)
                    if "tags" in data:
                        tags = data["tags"]
                except:
                    pass

    all_tags = tags.get("all_tags", [])
    print(f"[Tagging Agent] Generated {len(all_tags)} tags")

    return {
        "messages": [AIMessage(content=f"[Tagging Agent] {last_message.content}")],
        "tags": tags
    }


# ============================================================================
# Supervisor Agent
# ============================================================================

WORKERS = ["frame_extraction_agent", "content_analysis_agent", "tagging_agent", "FINISH"]

SUPERVISOR_PROMPT = """You are the supervisor coordinating travel video analysis.

You manage a team of specialized agents:
1. frame_extraction_agent - Extracts frames from video files
2. content_analysis_agent - Analyzes frames with vision AI to identify destinations
3. tagging_agent - Generates searchable tags from destination info

For each video analysis request, you MUST call agents in this order:
1. First: frame_extraction_agent (to get frames)
2. Then: content_analysis_agent (to analyze the frames)
3. Then: tagging_agent (to generate tags)
4. Finally: FINISH (when all agents have completed)

Current state:
- Video path: {video_path}
- Frames extracted: {has_frames}
- Destination analyzed: {has_destination}
- Tags generated: {has_tags}

Based on the current state, which agent should act next?
Respond with ONLY the agent name: frame_extraction_agent, content_analysis_agent, tagging_agent, or FINISH"""


def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor decides which agent to call next."""
    has_frames = len(state.get("frame_paths", [])) > 0
    has_destination = bool(state.get("destination_info"))
    has_tags = bool(state.get("tags"))

    prompt = SUPERVISOR_PROMPT.format(
        video_path=state.get("video_path", ""),
        has_frames=has_frames,
        has_destination=has_destination,
        has_tags=has_tags
    )

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Which agent should act next?")
    ])

    # Parse the response to get the next agent
    next_agent = response.content.strip()

    # Validate and normalize the response
    if "frame_extraction" in next_agent.lower():
        next_agent = "frame_extraction_agent"
    elif "content_analysis" in next_agent.lower():
        next_agent = "content_analysis_agent"
    elif "tagging" in next_agent.lower():
        next_agent = "tagging_agent"
    elif "finish" in next_agent.lower():
        next_agent = "FINISH"
    else:
        # Default logic based on state
        if not has_frames:
            next_agent = "frame_extraction_agent"
        elif not has_destination:
            next_agent = "content_analysis_agent"
        elif not has_tags:
            next_agent = "tagging_agent"
        else:
            next_agent = "FINISH"

    print(f"\n[Supervisor] Next agent: {next_agent}")

    return {"next_agent": next_agent}


def route_to_agent(state: AgentState) -> Literal["frame_extraction_agent", "content_analysis_agent", "tagging_agent", "FINISH"]:
    """Route to the next agent based on supervisor decision."""
    return state.get("next_agent", "FINISH")


# ============================================================================
# Build the Multi-Agent Graph
# ============================================================================

def build_multi_agent_graph():
    """Build the supervisor-worker graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("frame_extraction_agent", frame_extraction_node)
    graph.add_node("content_analysis_agent", content_analysis_node)
    graph.add_node("tagging_agent", tagging_node)

    # Start with supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor routes to workers or finish
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "frame_extraction_agent": "frame_extraction_agent",
            "content_analysis_agent": "content_analysis_agent",
            "tagging_agent": "tagging_agent",
            "FINISH": END
        }
    )

    # Workers report back to supervisor
    graph.add_edge("frame_extraction_agent", "supervisor")
    graph.add_edge("content_analysis_agent", "supervisor")
    graph.add_edge("tagging_agent", "supervisor")

    return graph.compile()


# ============================================================================
# Runner
# ============================================================================

def analyze_video(
    video_path: str,
    user_id: str = "demo_user",
    session_id: str = "demo_session"
) -> dict:
    """
    Analyze a travel video using the LangGraph multi-agent system.

    Args:
        video_path: Path to the video file
        user_id: User identifier for tracing
        session_id: Session identifier for grouping runs

    Returns:
        dict with analysis results
    """
    print(f"\n{'='*60}")
    print("TRAVEL VIDEO ANALYZER - LangGraph Multi-Agent")
    print(f"{'='*60}")
    print(f"\nVideo: {video_path}")
    print(f"Session: {session_id}")
    print(f"User: {user_id}")

    # Build the graph
    app = build_multi_agent_graph()

    # Get tracer for creating root span
    tracer = trace.get_tracer("travel_video_analyzer")

    # Execute within a traced span
    with tracer.start_as_current_span(
        "analyze_video",
        attributes={
            "session.id": session_id,
            "user.id": user_id,
            "video.path": video_path,
        }
    ):
        result = app.invoke({
            "messages": [HumanMessage(content=f"Analyze this travel video: {video_path}")],
            "video_path": video_path,
            "frame_paths": [],
            "destination_info": {},
            "tags": {},
            "next_agent": ""
        })

    return result


def print_results(results: dict):
    """Pretty print results."""
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    if results.get("destination_info"):
        print("\nDestination Info:")
        dest = results["destination_info"]
        location = dest.get("location", {})
        print(f"  Location: {location.get('city', 'Unknown')}, {location.get('country', 'Unknown')}")
        if dest.get("landmarks"):
            print(f"  Landmarks: {', '.join(dest.get('landmarks', []))}")
        if dest.get("activities"):
            print(f"  Activities: {', '.join(dest.get('activities', []))}")
        if dest.get("description"):
            print(f"  Description: {dest.get('description')}")

    if results.get("tags"):
        tags = results["tags"]
        all_tags = tags.get("all_tags", [])
        if all_tags:
            print(f"\nTags: {', '.join(all_tags)}")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "./videos/travel_sample.mp4"

    print(f"Travel Video Analyzer (LangGraph Multi-Agent)")

    try:
        results = analyze_video(video_path)
        print_results(results)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
