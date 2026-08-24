# Claude Context - Lumoz Integrations

Public repository of runnable examples showing how to instrument AI agents with Lumoz. Apache 2.0 licensed.

**This repo is public and customer-facing by default.** Customers clone it, run it, and copy code out of it. There is no internal side to this repository. Treat every file as documentation.

## Repository Map

| Directory | What it demonstrates |
|---|---|
| `connectivity/` | Pre-flight API key and network check. No dependencies. First thing a customer runs. |
| `langgraph/` | LangGraph via `LangChainInstrumentor` auto-instrumentation |
| `google-adk/` | Google ADK, single-process and multi-agent over HTTP |
| `mastra/` | Mastra in TypeScript, including per-subtenant exporter isolation |
| `vanilla/` | Plain Python with OpenInference decorators, no framework |
| `a2a/` | Agent-to-agent across ADK, LangGraph, and CrewAI |
| `cli/subtenant/` | Sub-tenant routing from the CLI |

## Customer-Facing Docs Check

The published documentation built from these examples does NOT live here. It lives in the `mvp` repo:

```
lumozai/mvp → customer-docs/instrument/*.mdx
```

Each example's own `README.md` and `INSTRUMENTATION.md` are the source of truth. The pages in `customer-docs/instrument/` are the customer-facing rendering of them. The two drift easily. They must not.

**After completing any change, decide whether it alters what a customer sees, types, or depends on.** Because this repo is entirely customer-facing, the default answer is yes.

| Change | Update |
|---|---|
| Ingest endpoint, auth header, or env var name | Every example that uses it, AND `mvp/customer-docs/instrument/overview.mdx` and `quickstart.mdx` |
| Instrumentation setup code that the docs quote | The example's `INSTRUMENTATION.md` AND the matching `customer-docs/instrument/` page |
| Example added, renamed, moved, or deleted | Root `README.md` table AND `customer-docs/instrument/overview.mdx` framework cards |
| Required resource attribute (`service.name`, `deployment.environment`, `session.id`, `user.id`) | Every example plus the instrumentation pages |
| Minimum framework or SDK version | The example's README and its docs page |
| Expected console navigation path in a README | Everywhere that path appears (console tab labels change) |

**Root README currency.** The root `README.md` example table has drifted before: `google-adk/multi-agent-http`, the whole `a2a/` suite, and `cli/subtenant` were all missing from it. Any directory added under an example root must be added to that table in the same change.

**How to surface it.** End the work with one line:

`Customer-facing: <what changed>. Update <example README> and mvp/customer-docs/instrument/<page>? (y/n)`

**Note the cross-repo edit.** Updating `customer-docs/` means a change in a different repository and a separate commit. Say so explicitly.

### What, Not How

**Show HOW to instrument. Never show HOW Lumoz processes what you send.**

This repo is unusual: teaching the customer's own How is its entire purpose. Exporter setup, auth headers, resource attributes, initialization ordering, and shutdown handling are all in scope and should be as concrete as possible.

The line falls at the network boundary. Once a span leaves the customer's process, what happens to it is not documented here.

Publish: instrumentation code, dependencies and versions, env vars, the ingest endpoint and auth format, required and optional attributes, what a customer sees in the console afterward.

Never publish: how spans are stored, processed, or scored. Pipeline stages, model or algorithm names, internal service, table, or column names, or the schedule anything runs on.

Signal **definitions**, including their detection criteria, are publishable; the machinery that evaluates them is not. In practice this rarely comes up here, since examples should point at the docs site rather than restate the signal catalog.

| Do not write | Write instead |
|---|---|
| "spans land in ClickHouse and a Kafka consumer scores them" | "traces appear in the console within a few seconds" |
| "the hallucination classifier runs a verifier model over the LLM input" | "Lumoz evaluates response groundedness automatically" |
| "workflows are inferred by the naming worker from root span names" | "Lumoz identifies workflows automatically; you can rename them in the console" |
| "signals fire when the classifier score exceeds its threshold" | "signals fire when Lumoz detects the pattern" |

**The test:** if a sentence describes something happening on Lumoz's side rather than the customer's, it is probably How. Say what the customer observes instead.

**Comments in example code count.** A `# the backend re-embeds this field` comment is as public as the README.

## Working Rules

**Examples must actually run.** This is not pseudo-code. A customer will clone and execute it. Do not commit an example that has not been run end to end against real Lumoz ingest.

**No secrets, ever.** Public repo. API keys, tokens, and endpoints beyond the public `https://api.lumoz.ai/proxy/v1/traces` belong in `.env.example` as placeholders only.

**Tracing stays optional.** Existing examples run normally when `LUMOZ_API_KEY` and `OTEL_ENDPOINT` are unset. Preserve that. A customer evaluating the framework should not be blocked on having a Lumoz account.

**Initialize tracing first.** OpenInference instrumentors patch library internals, so tracing setup must run before any agent, graph, or model object is constructed. This is the single most common instrumentation bug. Every example must demonstrate the correct ordering.
