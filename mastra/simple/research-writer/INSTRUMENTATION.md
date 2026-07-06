# Instrumenting Mastra with Lumoz

This guide covers how this example integrates [Mastra](https://mastra.ai) with Lumoz observability to capture LLM calls, tool usage, agent workflows, and multi-step reasoning traces.

## Dependencies

```bash
npm install @mastra/observability @mastra/arize \
  @opentelemetry/api @opentelemetry/sdk-trace-node \
  @opentelemetry/exporter-trace-otlp-http
```

## 1. Configure the Exporter

Create a tracing module that initializes the OpenTelemetry exporter with your Lumoz API key.

See [`src/tracing.ts`](src/tracing.ts) for the full implementation.

```typescript
import { Observability } from "@mastra/observability";
import { ArizeExporter } from "@mastra/arize";

let observability: Observability;

export function initTracing() {
  const otelEndpoint = process.env.OTEL_ENDPOINT || "https://api.lumoz.ai/proxy/v1/traces";
  const apiKey = process.env.LUMOZ_API_KEY || "";

  const headers: Record<string, string> = {
    accept: "application/x-protobuf",
  };

  if (apiKey) {
    const encoded = Buffer.from(apiKey, "utf-8").toString("base64");
    headers["authorization"] = `Basic ${encoded}`;
  }

  const arizeExporter = new ArizeExporter({
    endpoint: otelEndpoint,
    headers,
  });

  observability = new Observability({
    configs: {
      default: {
        serviceName: "mastra-research-writer",
        exporters: [arizeExporter],
      },
    },
  });
}

export function getObservability() {
  return observability;
}

export async function shutdown() {
  if (observability) {
    await observability.shutdown();
  }
}
```

## 2. Initialize Before App Startup

Tracing **must** be initialized before importing agents, workflows, or any code that makes LLM calls. This ensures the OpenTelemetry SDK is configured before spans are created.

See [`src/index.ts`](src/index.ts) for the full implementation.

```typescript
// Load env vars first
import "./env.js";

// Initialize tracing before anything else
import { initTracing, getObservability, shutdown } from "./tracing.js";
initTracing();

// Now import application code
import { Mastra } from "@mastra/core";
import { researchAgent } from "./agents/researcher.js";
import { writerAgent } from "./agents/writer.js";
import { researchWriteWorkflow } from "./workflows/research-write.js";

const mastra = new Mastra({
  agents: { researchAgent, writerAgent },
  workflows: { "research-write": researchWriteWorkflow },
  observability: getObservability(),
});
```

## 3. User and Session Tracking

Pass `userId` and `threadId` via `tracingOptions.metadata` when starting a workflow run. Lumoz uses these to group traces by user and session in the console.

```typescript
import { randomUUID } from "node:crypto";

const userId = "user-123";
const sessionId = randomUUID();

const workflow = mastra.getWorkflow("research-write");
const run = await workflow.createRun();
const result = await run.start({
  inputData: { query: "Tell me about AI agents" },
  tracingOptions: {
    metadata: {
      userId: userId,
      threadId: sessionId,
    },
  },
});
```

These appear as `user.id` and `session.id` attributes on the root span and propagate to all child spans.

## 4. Passing Trace Context to Agents

When using workflows with multiple steps, pass `tracing` and `tracingContext` from the step execution context to each agent's `generate()` call. This connects all agent calls, tool executions, and LLM spans under a single trace.

See [`src/workflows/research-write.ts`](src/workflows/research-write.ts) for the full implementation.

```typescript
const researchStep = createStep({
  id: "research",
  inputSchema: z.object({ query: z.string() }),
  outputSchema: z.object({ research: z.string(), query: z.string() }),
  execute: async ({ inputData, tracing, tracingContext }) => {
    // Pass tracing + tracingContext so the agent's spans are children of this step
    const result = await researchAgent.generate(
      `Research: ${inputData.query}`,
      { tracing, tracingContext },
    );
    return { research: result.text, query: inputData.query };
  },
});
```

**Without `{ tracing, tracingContext }`**, agent spans appear as separate disconnected traces instead of being nested under the workflow.

## 5. Graceful Shutdown

Always flush pending spans before the process exits to avoid losing trace data:

```typescript
process.on("SIGINT", async () => {
  await shutdown();
  process.exit(0);
});
```

## What Gets Captured

Once instrumented, Lumoz automatically captures:

| Span Type | Attributes |
|-----------|-----------|
| Workflow | Workflow name, step inputs/outputs, execution status |
| Agent | Agent name, instructions, input/output text |
| LLM | Model name, provider, temperature, token counts, input/output messages |
| Tool | Tool name, description, input arguments, output results |

## Trace Structure

Running this example produces a trace like:

```
invoke_workflow research-write          (WORKFLOW)
  workflow_step research-write          (WORKFLOW)
    invoke_agent Research Agent         (AGENT)
      chat gpt-4o-mini                  (LLM)
      model_step Research Agent         (LLM)
        execute_tool vectorSearchTool   (TOOL)
      model_step Research Agent         (LLM)
    invoke_agent Writer Agent           (AGENT)
      chat gpt-4o-mini                  (LLM)
      model_step Writer Agent           (LLM)
  workflow_step research-write          (WORKFLOW)
```

## Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...
LUMOZ_API_KEY=client_id:client_secret
OTEL_ENDPOINT=https://api.lumoz.ai/proxy/v1/traces
```
