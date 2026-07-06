import { Observability } from "@mastra/observability";
import { ArizeExporter } from "@mastra/arize";

let observability: Observability;

export function initTracing() {
  const otelEndpoint =
    process.env.OTEL_ENDPOINT ||
    "https://api.lumoz.ai/proxy/v1/traces";
  const apiKey = process.env.LUMOZ_API_KEY || "";

  const headers: Record<string, string> = {
    // Required so API Gateway decodes the binary protobuf response.
    // REST API only base64-decodes isBase64Encoded Lambda responses when
    // the request includes a matching Accept header.
    "accept": "application/x-protobuf",
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

  console.log(`[tracing] Mastra observability with OpenInference enabled`);
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
