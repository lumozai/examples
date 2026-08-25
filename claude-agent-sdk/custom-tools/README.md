# Claude Agent SDK — Custom Tools and Prompts

Your system prompt, your tool definitions, and your sub-agent instructions, captured in the trace.

The [get-started](../get-started) example shows what `ClaudeAgentSDKInstrumentor` gives you on its own: agent and tool spans with the model, token counts, and cost. It does not include anything you wrote. The instrumentor never reads `ClaudeAgentOptions`, and the SDK runs the model loop in a separate process, so your configuration never reaches a span.

This example adds it. `config_attributes.py` maps your options onto the standard OpenInference keys, and `agent.py` attaches them to one span around the run.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys
```

The Claude Agent SDK runs the Claude Code CLI, which requires Node.js 18+.

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | used by the Claude Agent SDK |
| `LUMOZ_API_KEY` | no | enables Lumoz tracing |
| `OTEL_ENDPOINT` | no | Lumoz ingest endpoint |

## Run

```bash
python agent.py
```

A pricing agent looks up a SKU with a tool you defined, then delegates to a sub-agent you defined to audit the quote.

## What gets captured

```
LLM    agent.request                your span, carrying your configuration
  AGENT  ClaudeAgentSDK.query       the instrumentor's spans, nested underneath
    TOOL   mcp__pricing__lookup_price
    TOOL   Agent
      AGENT  ClaudeAgentSDK.Agent
```

| What you supply | Attribute |
|---|---|
| `system_prompt` (or a preset's `append`) | `llm.input_messages.0.message.{role,content}` |
| the prompt | `llm.input_messages.1.message.{role,content}` |
| `@tool` definitions via SDK MCP servers | `llm.tools.N.tool.json_schema` |
| `agents` (`AgentDefinition`) | `lumoz.agent_definitions` |
| `allowed_tools` | `lumoz.allowed_tools` |

## Adding this to your own agent

Copy `config_attributes.py` into your project, then wrap your `query()` call:

```python
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from config_attributes import config_attributes

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("agent.request") as span:
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")

    for key, value in config_attributes(options, prompt, tools=tools).items():
        span.set_attribute(key, value)

    result = None
    async for message in query(prompt=prompt, options=options):
        if getattr(message, "result", None) is not None:
            result = message.result

    span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result))
```

Two details that matter:

**Set the span kind to `LLM`, not `AGENT`.** Lumoz only records tool schemas on LLM spans. With `AGENT` the schemas are dropped at ingest, with no error anywhere.

**Pass `tools=` explicitly.** `create_sdk_mcp_server` keeps its tool list in a closure, so the schemas cannot be recovered from `options`.

## What this does not cover

Built-in tool schemas (`Bash`, `Read`, `Glob`, `Task`, …) and Claude Code's own framing are assembled inside the CLI. They are roughly 37,000 tokens of every request and are not reachable from your process. `allowed_tools` controls which tools the agent may call, not which definitions are sent — the built-in schemas go to the model regardless.

So the span carries everything you authored, and none of what the CLI adds.

---

Adapted from the [OpenInference instrumentor example](https://github.com/Arize-ai/openinference/tree/main/python/instrumentation/openinference-instrumentation-claude-agent-sdk/examples) (Apache-2.0).
