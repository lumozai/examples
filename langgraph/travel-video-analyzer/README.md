# Travel Video Analyzer - LangGraph

A multi-agent app that analyzes travel videos using LangGraph with Anthropic Claude, instrumented with [Lumoz](https://lumoz.ai) for observability.

## Architecture

This example uses the **Supervisor pattern** - similar to Google ADK's orchestrator:

```
supervisor (coordinates agents)
    │
    ├── frame_extraction_agent
    │   └── Extracts key frames from video files using OpenCV
    │
    ├── content_analysis_agent
    │   └── Analyzes frames with Claude Vision to identify destinations
    │
    └── tagging_agent
        └── Generates searchable tags from destination info
```

## Agents

| Agent | Description | Tools |
|-------|-------------|-------|
| `supervisor` | Decides which agent to call next based on current state | Routes to workers |
| `frame_extraction_agent` | Extracts evenly-spaced frames from video | `extract_video_frames` |
| `content_analysis_agent` | Uses Claude Vision to identify location, landmarks, activities | `analyze_frames_with_vision` |
| `tagging_agent` | Generates categorized tags for search/discovery | `generate_destination_tags` |

## What Gets Traced

When running with Lumoz instrumentation, you'll see:

- **Supervisor decisions**: How the supervisor routes to each agent
- **Agent execution**: Each worker agent's ReAct loop
- **Tool calls**: Frame extraction, vision analysis, and tagging with inputs/outputs
- **LLM calls**: Claude requests with token counts and latency
- **Full trace hierarchy**: Parent-child relationships between all operations

```
travel_video_analyzer (root)
└── supervisor (decides: frame_extraction_agent)
    └── frame_extraction_agent (AGENT)
        └── extract_video_frames (TOOL)
└── supervisor (decides: content_analysis_agent)
    └── content_analysis_agent (AGENT)
        └── analyze_frames_with_vision (TOOL)
            └── Claude Vision API call (LLM)
└── supervisor (decides: tagging_agent)
    └── tagging_agent (AGENT)
        └── generate_destination_tags (TOOL)
└── supervisor (decides: FINISH)
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Lumoz Configuration
LUMOZ_API_KEY=your_client_id:your_client_secret
OTEL_ENDPOINT=https://api.lumoz.ai/v1/traces

# Anthropic API Key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

**Get your Lumoz API key:**
1. Log in to [Lumoz Console](https://console.lumoz.ai)
2. Go to **Settings > API Keys**
3. Click **Create API Key**
4. Copy the key (format: `client_id:client_secret`)

## Run

```bash
python travel_video_analyzer.py ./videos/travel-video.mp4
```

Or use one of the included sample videos:
```bash
python travel_video_analyzer.py ./videos/travel-video-0.mp4
```

## View Traces in Lumoz

1. Run the analyzer on a video
2. Open [Lumoz Console](https://console.lumoz.ai)
3. Navigate to **Traces**
4. See your supervisor routing, agent execution, and LLM interactions

## Requirements

- Python 3.11+
- Anthropic API key (for Claude Vision)
- Lumoz API key (for observability)
- Video file (mp4, mov, avi)

## Span Processing

Use a span processor to batch and export spans:

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider.add_span_processor(BatchSpanProcessor(exporter))
```

### Image Payload Handling (Vision API)

If your app sends images to Claude Vision, use `ImageStrippingSpanProcessor` instead. Large base64-encoded images in span attributes cause export timeouts and duplicate spans.

```python
# For apps WITHOUT Vision API - use standard BatchSpanProcessor
provider.add_span_processor(BatchSpanProcessor(exporter))

# For apps WITH Vision API - use ImageStrippingSpanProcessor
provider.add_span_processor(ImageStrippingSpanProcessor(exporter))
```

This example includes an `ImageStrippingSpanProcessor` that extends `BatchSpanProcessor` and automatically truncates base64 image data before export.

## Comparison with ADK Version

| Aspect | ADK Version | LangGraph Version |
|--------|-------------|-------------------|
| Orchestration | `LlmAgent` with `sub_agents` | Supervisor StateGraph |
| Worker Agents | `LlmAgent` with tools | `create_react_agent` with tools |
| Routing | ADK's built-in orchestration | Conditional edges in graph |
| State | Session-based | TypedDict state |

Both versions achieve the same result - the LLM decides which agent to call. LangGraph makes the routing explicit in the graph structure.

## Learn More

- [Lumoz Documentation](https://docs.lumoz.ai)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenInference](https://github.com/Arize-ai/openinference)
