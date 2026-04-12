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
from dotenv import load_dotenv

# Third-party - LangGraph / LangChain
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Third-party - OpenTelemetry / OpenInference (optional)
try:
    from opentelemetry import trace
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation import using_attributes
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

if not _OTEL_AVAILABLE:
    from contextlib import contextmanager

    @contextmanager
    def using_attributes(**kwargs):
        yield

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
# In-Memory Vector Store (LangChain-native)
# ============================================================================

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embedding=embeddings)
_chunk_count = 0  # Track total chunks for CLI display

_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)


def load_documents(directory: str) -> tuple[int, int]:
    """Load all .txt files from a directory into the vector store.

    Returns (file_count, chunk_count).
    """
    global _chunk_count
    loader = DirectoryLoader(directory, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        print(f"[knowledge] No .txt files found in {directory}")
        return 0, 0
    chunks = _splitter.split_documents(docs)
    vector_store.add_documents(chunks)
    _chunk_count += len(chunks)
    # Count unique source files
    files = set(doc.metadata.get("source", "") for doc in docs)
    return len(files), len(chunks)


def ingest(text: str) -> int:
    """Add text to the vector store at runtime. Returns chunk count."""
    global _chunk_count
    chunks = _splitter.split_documents([Document(page_content=text)])
    vector_store.add_documents(chunks)
    _chunk_count += len(chunks)
    return len(chunks)


# ============================================================================
# Tool Definition
# ============================================================================

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
vector_search = create_retriever_tool(
    retriever,
    "vector_search",
    "Search the knowledge base for information relevant to a query.",
)


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
    name="researcher",
)

writer_agent = create_agent(
    llm,
    system_prompt=WRITER_PROMPT,
    name="writer",
)


# ============================================================================
# State Adapter Nodes
# ============================================================================

def prepare_research(state: ResearchWriterState) -> dict:
    """Set up messages for the research agent from the user's query."""
    return {"messages": [HumanMessage(content=state["query"])]}


def save_research(state: ResearchWriterState) -> dict:
    """Extract research findings from the agent's last message."""
    last_msg = state["messages"][-1]
    research_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return {"research": research_text}


def prepare_writing(state: ResearchWriterState) -> dict:
    """Set up messages for the writer agent with query and research context."""
    prompt = (
        f"Original question: {state['query']}\n\n"
        f"Research findings:\n{state['research']}\n\n"
        "Please write a polished, well-structured response based on the research above."
    )
    return {"messages": [HumanMessage(content=prompt)]}


def save_response(state: ResearchWriterState) -> dict:
    """Extract the final response from the writer agent's last message."""
    last_msg = state["messages"][-1]
    response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    return {"response": response_text}


# ============================================================================
# Graph Construction
# ============================================================================

def build_graph():
    """Build the research-writer LangGraph pipeline with agent subgraph nodes."""
    graph = StateGraph(ResearchWriterState)

    # State adapters and agent subgraph nodes
    graph.add_node("prepare_research", prepare_research)
    graph.add_node("researcher", research_agent)      # Agent as subgraph node
    graph.add_node("save_research", save_research)
    graph.add_node("prepare_writing", prepare_writing)
    graph.add_node("writer", writer_agent)             # Agent as subgraph node
    graph.add_node("save_response", save_response)

    # Linear flow
    graph.add_edge(START, "prepare_research")
    graph.add_edge("prepare_research", "researcher")
    graph.add_edge("researcher", "save_research")
    graph.add_edge("save_research", "prepare_writing")
    graph.add_edge("prepare_writing", "writer")
    graph.add_edge("writer", "save_response")
    graph.add_edge("save_response", END)

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
    print(f"\nKnowledge base: {_chunk_count} chunks ready")

    print("\nCommands:")
    print("  /ingest <text>  — Add text to the knowledge base")
    print("  /quit           — Exit\n")

    # Build graph
    app = build_graph()

    # Get tracer if tracing enabled
    tracer = trace.get_tracer("research_writer") if tracing_enabled else None

    # Config uses session_id as thread_id so MemorySaver persists conversation
    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
            "session_id": session_id,
        },
        "metadata": {
            "session_id": session_id,
            "user_id": user_id,
        },
    }

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
                print(f"[knowledge] Ingested {n} chunks ({_chunk_count} total)\n")
            else:
                print("[knowledge] No text provided.\n")
            continue

        # Run the research-writer pipeline
        print("\n[researching...]\n")

        # Wrap in root span with OpenInference context attributes
        if tracing_enabled and tracer:
            span_ctx = tracer.start_as_current_span(
                "research_writer_query",
                attributes={
                    "user.id": user_id,
                    "session.id": session_id,
                    "query": user_input,
                },
            )
        else:
            span_ctx = nullcontext()

        # using_attributes propagates user.id and session.id to all
        # auto-instrumented LangChain/LangGraph spans
        if tracing_enabled:
            attr_ctx = using_attributes(
                user_id=user_id,
                session_id=session_id,
            )
        else:
            attr_ctx = nullcontext()

        with span_ctx, attr_ctx:
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
