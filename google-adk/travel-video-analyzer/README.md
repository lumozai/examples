# Travel Video Analyzer - Google ADK

A multi-agent app that analyzes travel videos using Google Agent Development Kit (ADK) with Anthropic Claude, instrumented with [Lumoz](https://lumoz.ai) for observability.

## Architecture

```
orchestrator_agent
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
| `orchestrator_agent` | Coordinates the workflow by deciding which sub-agents to call | Sub-agents as tools |
| `frame_extraction_agent` | Extracts evenly-spaced frames from video | `extract_video_frames` |
| `content_analysis_agent` | Uses Claude Vision to identify location, landmarks, activities | `analyze_frames_with_vision` |
| `tagging_agent` | Generates categorized tags for search/discovery | `generate_destination_tags` |

## What Gets Traced

When running with Lumoz instrumentation, you'll see:

- **Agent orchestration**: How the orchestrator delegates to sub-agents
- **Tool calls**: Frame extraction, vision analysis, and tagging with inputs/outputs
- **LLM calls**: Claude Vision requests with token counts and latency
- **Full trace hierarchy**: Parent-child relationships between all operations

```
travel_video_analyzer (root)
└── orchestrator_agent (AGENT)
    ├── frame_extraction_agent (AGENT)
    │   └── extract_video_frames (TOOL)
    ├── content_analysis_agent (AGENT)
    │   └── analyze_frames_with_vision (TOOL)
    │       └── Claude Vision API call (LLM)
    └── tagging_agent (AGENT)
        └── generate_destination_tags (TOOL)
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
python travel_video_analyzer.py ./videos/sample.mp4
```

## View Traces in Lumoz

1. Run the analyzer on a video
2. Open [Lumoz Console](https://console.lumoz.ai)
3. Navigate to **Traces**
4. See your agent orchestration, tool calls, and LLM interactions

## Requirements

- Python 3.11+
- Anthropic API key (for Claude Vision)
- Lumoz API key (for observability)
- Video file (mp4, mov, avi)

## Handling Large Vision Payloads

This example includes an `ImageStrippingSpanProcessor` that automatically truncates base64 image data in traces. This prevents large payloads from causing export failures while preserving all other trace information.

## Learn More

- [Lumoz Documentation](https://docs.lumoz.ai)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [OpenInference](https://github.com/Arize-ai/openinference)
