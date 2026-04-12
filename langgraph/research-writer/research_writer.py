"""
LangGraph Research-Writer — Two-Agent RAG Pipeline

A CLI chat application with a two-node LangGraph pipeline:
  1. Research Agent — searches an in-memory vector store for relevant context
  2. Writer Agent  — synthesizes research into a polished response

Documents are loaded from documents/*.txt on startup. Users can add content
at runtime with /ingest or drop .txt files and restart.

Instrumented with OpenInference for optional Lumoz observability.
"""

# Standard library
import base64
import glob
import json
import operator
import os
import re
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, TypedDict

# Third-party - Core
import numpy as np
from dotenv import load_dotenv

# Third-party - LangGraph / LangChain
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Third-party - OpenTelemetry / OpenInference (optional)
try:
    from opentelemetry import trace
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation.langchain import LangChainInstrumentor
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

# Third-party - OpenAI (for embeddings)
import openai as openai_lib

# Load environment variables
load_dotenv()

# Validate required config
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is required.")
    sys.exit(1)


# ============================================================================
# Image Stripping Span Processor
# ============================================================================

if _OTEL_AVAILABLE:
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
# Lumoz Tracing (Optional)
# ============================================================================

def configure_lumoz_tracing():
    """Configure OpenInference instrumentation for Lumoz observability.

    Returns a TracerProvider if both LUMOZ_API_KEY and OTEL_ENDPOINT are set,
    otherwise returns None. The app works fine without tracing.
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
        "service.name": "langgraph-research-writer",
        "deployment.environment": "development",
    })

    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    otlp_exporter = OTLPSpanExporter(endpoint=otel_endpoint, headers=headers)
    span_processor = ImageStrippingSpanProcessor(otlp_exporter, max_image_chars=100)
    tracer_provider.add_span_processor(span_processor)

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"[tracing] Lumoz tracing configured — sending traces to {otel_endpoint}")
    return tracer_provider


# Module-level tracing setup
tracer_provider = configure_lumoz_tracing()
tracing_enabled = tracer_provider is not None


# ============================================================================
# In-Memory Vector Store
# ============================================================================

_store: list[dict] = []  # Each entry: {"text": str, "embedding": list[float]}
_openai_client = openai_lib.OpenAI()


def embed(text: str) -> list[float]:
    """Embed text using OpenAI text-embedding-3-small."""
    response = _openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def chunk(text: str, size: int = 512, overlap: int = 50) -> list[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def ingest(text: str) -> int:
    """Chunk, embed, and store text. Returns the number of chunks added."""
    chunks = chunk(text)
    for c in chunks:
        embedding = embed(c)
        _store.append({"text": c, "embedding": embedding})
    return len(chunks)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search the vector store by cosine similarity. Returns [{text, score}]."""
    if not _store:
        return []
    query_embedding = embed(query)
    scored = [
        {"text": entry["text"], "score": cosine_similarity(query_embedding, entry["embedding"])}
        for entry in _store
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def load_documents(directory: str) -> tuple[int, int]:
    """Load all .txt files from a directory into the vector store.

    Returns (file_count, chunk_count).
    """
    pattern = os.path.join(directory, "*.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[knowledge] No .txt files found in {directory}")
        return 0, 0
    total_chunks = 0
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        n = ingest(content)
        total_chunks += n
    return len(files), total_chunks


# ============================================================================
# Tool Definition
# ============================================================================

@tool
def vector_search(query: str) -> str:
    """Search the knowledge base for information relevant to a query.

    Args:
        query: The search query to find relevant documents.

    Returns:
        JSON string with search results and whether any were found.
    """
    results = search(query, top_k=5)
    return json.dumps({"results": results, "found": len(results) > 0})


# ============================================================================
# State
# ============================================================================

class ResearchWriterState(TypedDict):
    query: str
    research: str
    response: str
    messages: Annotated[list, operator.add]


# ============================================================================
# Agents
# ============================================================================

llm = ChatOpenAI(model="gpt-4o-mini")

RESEARCH_PROMPT = """You are a research assistant that finds and synthesizes information from a knowledge base.

Your workflow:
1. Use the vector_search tool to find relevant information for the user's query
2. Synthesize the search results into a coherent research summary
3. Cite specific findings from the retrieved documents
4. If the knowledge base has no relevant information, clearly state that and provide what general knowledge you can

Be thorough but concise. Focus on facts and evidence from the knowledge base."""

WRITER_PROMPT = """You are a skilled writer that creates clear, engaging responses based on research provided to you.

Your workflow:
1. Read the research summary provided in the user's message
2. Transform it into a well-structured, reader-friendly response
3. Maintain accuracy — only include information from the research
4. Use clear headings, bullet points, or paragraphs as appropriate
5. If the research indicates no relevant information was found, acknowledge this honestly and offer what insight you can

Write in a conversational but informative tone. Be concise and direct."""

research_agent = create_agent(
    llm,
    tools=[vector_search],
    system_prompt=RESEARCH_PROMPT,
)

writer_agent = create_agent(
    llm,
    system_prompt=WRITER_PROMPT,
)


# ============================================================================
# Graph Nodes
# ============================================================================

def research_node(state: ResearchWriterState) -> dict:
    """Invoke the research agent and return its findings."""
    query = state["query"]
    result = research_agent.invoke({
        "messages": [HumanMessage(content=query)],
    })
    last_msg = result["messages"][-1]
    research_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return {"research": research_text}


def write_node(state: ResearchWriterState) -> dict:
    """Invoke the writer agent to polish the research into a final response."""
    query = state["query"]
    research = state["research"]
    prompt = (
        f"Original question: {query}\n\n"
        f"Research findings:\n{research}\n\n"
        "Please write a polished, well-structured response based on the research above."
    )
    result = writer_agent.invoke({
        "messages": [HumanMessage(content=prompt)],
    })
    last_msg = result["messages"][-1]
    response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return {"response": response_text}


# ============================================================================
# Graph Construction
# ============================================================================

def build_graph():
    """Build the research-writer LangGraph pipeline."""
    graph = StateGraph(ResearchWriterState)
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_edge(START, "research")
    graph.add_edge("research", "write")
    graph.add_edge("write", END)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ============================================================================
# CLI Main Loop
# ============================================================================

def main():
    print("\n=== LangGraph Research-Writer ===\n")

    # User and session setup
    user_id = input("Enter your user ID (or press Enter for 'demo-user'): ").strip()
    if not user_id:
        user_id = "demo-user"
    session_id = str(uuid.uuid4())
    print(f"Session: {session_id}")

    # Resolve documents directory relative to this script
    script_dir = Path(__file__).resolve().parent
    docs_dir = script_dir / "documents"

    # Load documents
    print(f"\n[knowledge] Loading documents from {docs_dir}...")
    file_count, chunk_count = load_documents(str(docs_dir))
    if file_count > 0:
        print(f"[knowledge] Loaded {file_count} files ({chunk_count} chunks)")
    print(f"\nKnowledge base: {len(_store)} chunks ready")

    print("\nCommands:")
    print("  /ingest <text>  — Add text to the knowledge base")
    print("  /quit           — Exit\n")

    # Build graph
    app = build_graph()

    # Get tracer if tracing enabled
    tracer = trace.get_tracer("research_writer") if tracing_enabled else None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            if tracing_enabled and tracer_provider:
                print("[tracing] Flushing traces...")
                tracer_provider.force_flush()
            print("Goodbye!")
            break

        if user_input.lower().startswith("/ingest "):
            text = user_input[8:].strip()
            if text:
                n = ingest(text)
                print(f"[knowledge] Ingested {n} chunks ({len(_store)} total)\n")
            else:
                print("[knowledge] No text provided.\n")
            continue

        # Run the research-writer pipeline
        print("\n[researching...]\n")

        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "session_id": session_id,
            }
        }

        # Wrap in root span if tracing is enabled
        if tracing_enabled and tracer:
            ctx = tracer.start_as_current_span(
                "research_writer_query",
                attributes={
                    "user.id": user_id,
                    "session.id": session_id,
                    "query": user_input,
                },
            )
        else:
            ctx = nullcontext()

        with ctx:
            result = app.invoke(
                {
                    "query": user_input,
                    "research": "",
                    "response": "",
                    "messages": [],
                },
                config=config,
            )

        print(f"Agent: {result['response']}\n")


if __name__ == "__main__":
    main()
