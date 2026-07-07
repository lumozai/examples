import { LumozExporter } from "@mastra/lumoz";
import {
  DefaultObservabilityInstance,
  Observability,
} from "@mastra/observability";
import type { ObservabilityInstanceConfig } from "@mastra/observability";
import { BaseExporter } from "@mastra/observability";
import { RequestContext } from "@mastra/core/request-context";
import type { TracingEvent } from "@mastra/core/observability";

const TENANT_ID_CONTEXT_KEY = "tenant_id";
const DEFAULT_CONFIG_NAME = "__lumoz_default__";

type TenantExportersOptions = {
  endpoint: string;
  apiKey: string;
  serviceName: string;
  logLevel?: "debug" | "info" | "warn" | "error";
};

export class TenantExporters {
  private readonly observability: Observability;
  private readonly tenantIds = new Set<string>();

  constructor(private readonly options: TenantExportersOptions) {
    this.observability = new Observability({
      configs: {
        [DEFAULT_CONFIG_NAME]: {
          serviceName: options.serviceName,
          exporters: [new NoopExporter()],
        },
      },
      configSelector: ({ requestContext }) => {
        const tenant_id = requestContext?.get(TENANT_ID_CONTEXT_KEY);
        if (typeof tenant_id !== "string") {
          return DEFAULT_CONFIG_NAME;
        }

        const normalizedTenantId = normalizeTenantId(tenant_id);
        if (!this.tenantIds.has(normalizedTenantId)) {
          throw new Error(
            `No Lumoz exporter registered for tenant_id ${normalizedTenantId}`,
          );
        }
        return normalizedTenantId;
      },
    });
  }

  getObservability(): Observability {
    return this.observability;
  }

  ensureTenantExporter(tenant_id: string): string {
    const normalizedTenantId = normalizeTenantId(tenant_id);

    if (this.tenantIds.has(normalizedTenantId)) {
      return normalizedTenantId;
    }

    const exporter = this.createExporter(normalizedTenantId);
    const config: ObservabilityInstanceConfig = {
      name: normalizedTenantId,
      serviceName: this.options.serviceName,
      exporters: [exporter],
      requestContextKeys: [TENANT_ID_CONTEXT_KEY],
    };

    // Dynamically registered instances are added after Mastra's startup-time
    // exporter initialization, so initialize this exporter before first use.
    exporter.init({ config });
    this.observability.registerInstance(
      normalizedTenantId,
      new DefaultObservabilityInstance(config),
    );
    this.tenantIds.add(normalizedTenantId);

    return normalizedTenantId;
  }

  createRequestContext(tenant_id: string): RequestContext {
    const requestContext = new RequestContext();
    requestContext.set(TENANT_ID_CONTEXT_KEY, normalizeTenantId(tenant_id));
    return requestContext;
  }

  async shutdown(): Promise<void> {
    await this.observability.shutdown();
  }

  private createExporter(tenant_id: string): LumozExporter {
    if (!this.options.apiKey) {
      throw new Error("LUMOZ_API_KEY is required to create a tenant exporter");
    }

    if (process.env.LUMOZ_PRINT_OTEL_HEADERS === "true") {
      console.log("[tracing] OTLP headers", {
        accept: "application/x-protobuf",
        authorization: "REDACTED",
        "x-lumoz-subtenant-id": tenant_id,
      });
    }

    return new LumozExporter({
      apiKey: this.options.apiKey,
      endpoint: this.options.endpoint,
      subtenantId: tenant_id,
      logLevel: this.options.logLevel,
    });
  }
}

class NoopExporter extends BaseExporter {
  name = "noop";

  protected async _exportTracingEvent(_event: TracingEvent): Promise<void> {}
}

function normalizeTenantId(tenant_id: string): string {
  const trimmed = tenant_id.trim();
  if (!/^[a-zA-Z0-9._:-]{1,128}$/.test(trimmed)) {
    throw new Error(
      "tenant_id must be 1-128 characters and contain only letters, numbers, dot, underscore, colon, or dash",
    );
  }
  return trimmed;
}
