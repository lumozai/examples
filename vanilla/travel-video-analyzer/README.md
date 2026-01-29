# Travel Video Analyzer - Framework-less

A multi-agent app using **plain Python** with direct Anthropic API calls. No LangGraph, LangChain, or other agent frameworks - just OpenInference decorators.

## Architecture

```
analyze_video (chain)
    │
    ├── extract_frames (tool) - OpenCV frame extraction
    ├── analyze_frames (tool) - Claude Vision analysis
    └── generate_tags (tool)  - Tag generation
```

## Comparison with LangGraph Version

| Aspect | LangGraph Version | Framework-less Version |
|--------|-------------------|------------------------|
| Orchestration | StateGraph + Supervisor | Simple function calls |
| Agents | `create_react_agent` | Direct API calls |
| State | TypedDict + message passing | Return values |
| Tracing | LangChain auto-instrumentation | OpenInference decorators |
| Lines of code | ~740 | ~300 |
| Dependencies | 15+ | 6 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

## Run

```bash
# Copy sample videos from LangGraph version
cp -r ../langgraph/travel-video-analyzer/videos .

# Run analysis
python travel_video_analyzer.py ./videos/travel-video-0.mp4
```

## What Gets Traced

```
analyze_video (CHAIN)
├── extract_frames (TOOL)
├── analyze_frames (TOOL)
│   └── Claude Vision API call
└── generate_tags (TOOL)
    └── Claude API call
```

## Key Decorators

```python
from openinference.instrumentation import OITracer

tracer = OITracer(trace.get_tracer(__name__), TraceConfig())

@tracer.chain(name="workflow")      # For orchestration
@tracer.tool(name="...", desc="...") # For tools
@tracer.llm                          # For LLM calls
```

## Custom Span Attributes

Add custom attributes from within decorated functions:

```python
from opentelemetry import trace

@tracer.tool(name="analyze_frames", description="Analyze frames with Vision")
def analyze_frames(frame_paths: list[str]) -> dict:
    span = trace.get_current_span()
    span.set_attribute("vision.frame_count", len(frame_paths))
    span.set_attribute("vision.model", MODEL)

    # ... processing ...

    # Add attributes after computation
    span.set_attribute("vision.detected_city", city)
    return result
```

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

The `ImageStrippingSpanProcessor` (included in this example) extends `BatchSpanProcessor` and truncates base64 image data before export:

```python
class ImageStrippingSpanProcessor(BatchSpanProcessor):
    """Strips base64 image data from spans before export."""

    def on_end(self, span):
        # Truncate base64 patterns: data:image/..., /9j/..., iVBOR...
        # Keeps first 100 chars + "...[IMAGE_TRUNCATED]"
        super().on_end(span)
```
