import { Observability } from "@mastra/observability";
import { LumozExporter } from "@mastra/lumoz";

let observability: Observability;

export function initTracing() {
  const otelEndpoint =
    process.env.LUMOZ_ENDPOINT ||
    process.env.OTEL_ENDPOINT ||
    "https://api.lumoz.ai/proxy/v1/traces";

  const lumozExporter = new LumozExporter({
    apiKey: process.env.LUMOZ_API_KEY,
    endpoint: otelEndpoint,
  });

  observability = new Observability({
    configs: {
      default: {
        serviceName: "mastra-research-writer",
        exporters: [lumozExporter],
      },
    },
  });

  console.log(`[tracing] Mastra observability with Lumoz enabled`);
  console.log(`[tracing] Exporting to ${otelEndpoint}`);
}

export function getObservability() {
  return observability;
}

export async function shutdown() {
  if (observability) {
    await observability.shutdown();
  }
}
