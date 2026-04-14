# Lumoz Examples

Example applications demonstrating how to instrument AI agents with [Lumoz](https://lumoz.ai) for observability.

## What is Lumoz?

Lumoz is a reliability and security platform for AI applications. It provides deep observability into your agentic workflows, including:

- **Agent orchestration** - Track how orchestrators delegate to sub-agents
- **Tool execution** - Monitor tool calls, inputs, outputs, and latency
- **LLM calls** - Capture model, tokens, prompts, and responses
- **Multi-modal support** - Handle vision/image analysis with automatic payload optimization

Lumoz uses **OpenInference**, the open source SDK for LLM application tracing, ensuring your instrumentation is portable and not locked into proprietary formats.

## Connectivity Test

Before running any example, verify your environment can reach Lumoz and send traces:

```bash
python connectivity/test_connectivity.py "your-client-id:your-client-secret"
```

A `200 OK` response confirms your API key and network connectivity are working. See [connectivity/README.md](connectivity/README.md) for expected output and troubleshooting.

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
| [research-writer](mastra/research-writer) | Sequential multi-agent workflow — research agent queries a knowledge base, writer agent produces a polished response |

### Vanilla (plain Python)

| Example | Description |
|---------|-------------|
| [travel-video-analyzer](vanilla/travel-video-analyzer) | Multi-agent app using direct Anthropic API calls with OpenInference decorators — no framework required |

## Getting Started

1. **Get your Lumoz API key**
   - Log in to [Lumoz Console](https://console.lumoz.ai)
   - Go to **Settings > API Keys**
   - Create and copy your API key

2. **Choose an example** from the table above

3. **Follow the example's README** for setup instructions

4. **View traces** in the Lumoz Console

## Documentation

- [Lumoz Setup Guide - Google ADK](https://docs.lumoz.ai/setup/google-adk) - Detailed setup instructions
- [OpenInference](https://github.com/Arize-ai/openinference) - Open source instrumentation SDK

## Support

- Email: support@lumoz.ai
- Discord: [Join our community](https://discord.gg/lumoz)

## License

These examples are provided under the Apache License 2.0. See [LICENSE](LICENSE) for details.
