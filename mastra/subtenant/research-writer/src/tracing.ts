import { TenantExporters } from "./tenantExporters.js";

const SERVICE_NAME = "mastra-research-writer";
const LOG_LEVELS = ["debug", "info", "warn", "error"] as const;
type LogLevel = (typeof LOG_LEVELS)[number];

let tenantExporters: TenantExporters;

export function initTracing() {
  const otelEndpoint =
    process.env.OTEL_ENDPOINT ||
    "https://api.lumoz.ai/proxy/v1/traces";
  const apiKey = process.env.LUMOZ_API_KEY || "";

  tenantExporters = new TenantExporters({
    endpoint: otelEndpoint,
    apiKey,
    serviceName: SERVICE_NAME,
    logLevel: getExporterLogLevel(),
  });

  console.log(`[tracing] Mastra observability with OpenInference enabled`);
  console.log(`[tracing] Exporting to ${otelEndpoint}`);
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

function getExporterLogLevel(): LogLevel {
  const logLevel = process.env.LUMOZ_LOG_LEVEL?.toLowerCase();
  if (LOG_LEVELS.includes(logLevel as LogLevel)) {
    return logLevel as LogLevel;
  }
  return "error";
}
