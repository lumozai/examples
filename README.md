# Lumoz Integrations

Example applications demonstrating how to instrument AI agents with [Lumoz](https://lumoz.ai) for observability.

## What is Lumoz?

Lumoz is a reliability and security platform for AI applications. It provides deep observability into your agentic workflows, including:

- **Agent orchestration** - Track how orchestrators delegate to sub-agents
- **Tool execution** - Monitor tool calls, inputs, outputs, and latency
- **LLM calls** - Capture model, tokens, prompts, and responses
- **Multi-modal support** - Handle vision/image analysis with automatic payload optimization

Lumoz uses **OpenInference**, the open source SDK for LLM application tracing, ensuring your instrumentation is portable and not locked into proprietary formats.

## Getting Started

1. **Get your Lumoz API key**
   - Log in to [Lumoz Console](https://console.lumoz.ai)
   - Enter your email and you will be sent login link
   - Enter the required information. 
   - You will be presented with the dialog to create and copy your Lumoz API Key 

2. **Test connectivity** — verify your environment can reach Lumoz before running an example:
   ```bash
   python connectivity/test_connectivity.py "your Lumoz API key"
   ```
   A `200 OK` response confirms your API key and network are working. See [connectivity/README.md](connectivity/README.md) for expected output and troubleshooting.

3. **Choose an example** from the table below

4. **Follow the example's README** for setup instructions and run the example

5. **View traces** in the Lumoz Console

## Examples

### Google ADK

| Example | Description |
|---------|-------------|
| [travel-video-analyzer](google-adk/travel-video-analyzer) | Multi-agent app that analyzes travel videos using Claude Vision |

### LangGraph

| Example | Description |
|---------|-------------|
| [travel-video-analyzer](langgraph/travel-video-analyzer) | Multi-agent supervisor pattern that analyzes travel videos using Claude Vision |
| [research-writer](langgraph/research-writer) | Two-agent RAG pipeline — researcher searches a vector store, writer synthesizes findings |

### Mastra

| Example | Description |
|---------|-------------|
| [simple/research-writer](mastra/simple/research-writer) | Sequential multi-agent workflow — single shared OTLP exporter, simplest setup to get started |
| [subtenant/research-writer](mastra/subtenant/research-writer) | Same workflow with per-subtenant OTLP exporter isolation — each tenant gets its own export queue |

### Vanilla (plain Python)

| Example | Description |
|---------|-------------|
| [travel-video-analyzer](vanilla/travel-video-analyzer) | Multi-agent app using direct Anthropic API calls with OpenInference decorators — no framework required |

## Documentation & Support

- [OpenInference](https://github.com/Arize-ai/openinference) - Open source instrumentation SDK
- Email: support@lumoz.ai

## License

These examples are provided under the Apache License 2.0. See [LICENSE](LICENSE) for details.
