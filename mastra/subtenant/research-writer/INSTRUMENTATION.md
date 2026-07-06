# Instrumenting Mastra with Lumoz

This guide covers how this example integrates [Mastra](https://mastra.ai) with Lumoz observability to capture LLM calls, tool usage, agent workflows, and multi-step reasoning traces.

## Dependencies

```bash
npm install @mastra/observability @mastra/arize \
  @opentelemetry/api @opentelemetry/sdk-trace-node \
  @opentelemetry/exporter-trace-otlp-http
```

## 1. Configure the Exporter

Create a tracing module that initializes tenant exporters with your Lumoz API key. The utility class in [`src/tenantExporters.ts`](src/tenantExporters.ts) owns the per-tenant exporter registry.

See [`src/tracing.ts`](src/tracing.ts) for the full implementation.

```typescript
import { TenantExporters } from "./tenantExporters.js";

const SERVICE_NAME = "mastra-research-writer";
let tenantExporters: TenantExporters;

export function initTracing() {
  tenantExporters = new TenantExporters({
    endpoint: process.env.OTEL_ENDPOINT || "https://api.lumoz.ai/proxy/v1/traces",
    apiKey: process.env.LUMOZ_API_KEY || "",
    serviceName: SERVICE_NAME,
    logLevel: "error",
  });
}

export function getObservability() {
  return tenantExporters.getObservability();
}

export function ensureTenantExporter(tenant_id: string) {
  return tenantExporters.ensureTenantExporter(tenant_id);
}

export function createTenantRequestContext(tenant_id: string) {
  return tenantExporters.createRequestContext(tenant_id);
}

export async function shutdown() {
  if (tenantExporters) {
    await tenantExporters.shutdown();
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

## 3. Tenant Exporters

For a true multi-tenant agent service, do not use one global exporter with one tenant header. OTLP exporters batch spans, so one export request can contain spans from multiple app requests.

Instead, create one exporter and batch queue per `tenant_id`. The Lumoz API key identifies the Lumoz tenant, and `x-lumoz-subtenant-id` carries the customer's `tenant_id` value for Lumoz to map to its internal subtenant ID before storing traces. Mastra's `configSelector` chooses the correct observability instance from validated `requestContext` before the root span is created.

## 4. User and Session Tracking

After validating `tenant_id` at request ingress, ensure the tenant exporter exists and pass `tenant_id` through `requestContext`. Pass `userId` and `threadId` via `tracingOptions.metadata` for trace grouping in the console.

```typescript
import { randomUUID } from "node:crypto";
import {
  createTenantRequestContext,
  ensureTenantExporter,
} from "./tracing.js";

const userId = "user-123";
const tenant_id = "demo-tenant";
const sessionId = randomUUID();

ensureTenantExporter(tenant_id);
const requestContext = createTenantRequestContext(tenant_id);

const workflow = mastra.getWorkflow("research-write");
const run = await workflow.createRun();
const result = await run.start({
  inputData: { query: "Tell me about AI agents" },
  requestContext,
  tracingOptions: {
    metadata: {
      userId,
      threadId: sessionId,
    },
  },
});
```

These appear as `user.id` and `session.id` attributes on the root span and propagate to all child spans.

## 5. Server Request Handling

In a server, initialize Mastra and tracing once at process startup. For each incoming request, resolve and validate `tenant_id`, ensure its exporter exists, create a request context, then start a new workflow run.

```typescript
app.post("/agent", async (req, res) => {
  const tenant_id = validateTenantId(req.headers["x-tenant-id"]);
  const userId = req.user.id;
  const threadId = req.body.sessionId;

  ensureTenantExporter(tenant_id);
  const requestContext = createTenantRequestContext(tenant_id);

  const workflow = mastra.getWorkflow("research-write");
  const run = await workflow.createRun();
  const result = await run.start({
    inputData: { query: req.body.query },
    requestContext,
    tracingOptions: {
      metadata: {
        userId,
        threadId,
      },
    },
  });

  res.json(result);
});
```

`run.start()` is per request/workflow execution. The Mastra instance and tenant exporters are process-scoped.

## 6. Inspecting OTLP Headers

To verify that the custom subtenant header is sent, run the sample with:

```bash
LUMOZ_PRINT_OTEL_HEADERS=true npm start
```

When a tenant exporter is created, the sample prints the OTLP headers with `authorization` redacted:

```text
[tracing] OTLP headers {
  accept: 'application/x-protobuf',
  authorization: 'REDACTED',
  'x-lumoz-subtenant-id': 'demo-tenant'
}
```

## 7. Passing Trace Context to Agents

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

## 8. Graceful Shutdown

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
LUMOZ_PRINT_OTEL_HEADERS=false
```
