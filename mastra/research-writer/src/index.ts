// env.ts must be the first import — it calls dotenv.config() before
// any other module reads process.env (ESM hoists all imports).
import "./env.js";

import { initTracing, getObservability, shutdown } from "./tracing.js";

// Initialize tracing before anything else
initTracing();

import readline from "node:readline";
import { randomUUID } from "node:crypto";
import { Mastra } from "@mastra/core";
import { researchAgent } from "./agents/researcher.js";
import { writerAgent } from "./agents/writer.js";
import { researchWriteWorkflow } from "./workflows/research-write.js";
import { ingest, getStoreSize } from "./vectorStore.js";

const mastra = new Mastra({
  agents: { researchAgent, writerAgent },
  workflows: { "research-write": researchWriteWorkflow },
  observability: getObservability(),
});

const SAMPLE_DOCS = [
  `Large Language Models (LLMs) are neural networks trained on massive text datasets that can generate, understand, and reason about natural language. Key architectures include the Transformer, introduced in the 2017 paper "Attention Is All You Need." Modern LLMs like GPT-4, Claude, and Gemini use decoder-only transformer architectures with billions of parameters. Training involves pre-training on internet-scale text followed by fine-tuning with human feedback (RLHF). Challenges include hallucination, where models generate plausible but incorrect information, and the high computational cost of training and inference.`,

  `Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving relevant documents from an external knowledge base before generating an answer. The process involves: 1) Converting documents into vector embeddings and storing them in a vector database. 2) When a query arrives, converting it to an embedding and finding similar documents via cosine similarity. 3) Feeding the retrieved context to the LLM along with the query. RAG reduces hallucination by grounding responses in actual data and allows knowledge updates without retraining the model. Chunking strategies (size, overlap) significantly affect retrieval quality.`,

  `Multi-agent systems in AI involve multiple autonomous agents working together to solve complex tasks. Each agent can have specialized roles, tools, and knowledge. Orchestration frameworks like LangGraph, CrewAI, AutoGen, and Mastra provide patterns for agent coordination including sequential workflows (agent A then agent B), parallel execution, and hierarchical delegation. Key challenges include managing agent communication, handling failures gracefully, and maintaining coherent context across agents. The research-writer pattern is a common multi-agent workflow where one agent gathers information and another synthesizes it into a final output.`,

  `Vector databases store high-dimensional embeddings and enable fast similarity search. Popular options include Pinecone, Weaviate, Chroma, Milvus, and pgvector. Key concepts: embeddings are dense vector representations of text/images created by models like text-embedding-3-small. Cosine similarity and dot product are common distance metrics. Approximate Nearest Neighbor (ANN) algorithms like HNSW enable sub-linear search time. In-memory vector stores are useful for development and small datasets but production systems need persistent storage with indexing for scalability.`,

  `OpenTelemetry (OTEL) is an observability framework for generating, collecting, and exporting telemetry data (traces, metrics, logs). For AI applications, OpenInference extends OTEL with semantic conventions specific to LLM operations — tracking token usage, model parameters, prompt/completion content, and retrieval quality. Instrumenting AI agents with OpenInference enables monitoring of multi-step reasoning, tool usage patterns, latency per agent, and end-to-end trace visualization. This observability is critical for debugging, optimization, and ensuring quality in production AI systems.`,
];

async function seedKnowledgeBase() {
  console.log("[knowledge] Seeding vector store with sample documents...");
  let totalChunks = 0;
  for (const doc of SAMPLE_DOCS) {
    totalChunks += await ingest(doc);
  }
  console.log(
    `[knowledge] Ingested ${SAMPLE_DOCS.length} documents (${totalChunks} chunks)\n`,
  );
}

async function runQuery(
  query: string,
  userId: string,
  sessionId: string,
): Promise<string> {
  const workflow = mastra.getWorkflow("research-write");
  const run = await workflow.createRun();
  const result = await run.start({
    inputData: { query },
    tracingOptions: {
      metadata: {
        "userId": userId,
        "threadId": sessionId,
      },
    },
  });

  const writeResult = result.steps?.write;
  if (
    writeResult &&
    typeof writeResult === "object" &&
    "status" in writeResult &&
    writeResult.status === "success" &&
    "output" in writeResult
  ) {
    return (writeResult.output as { response: string }).response;
  }
  return `Error: workflow ended with status "${result.status}"`;
}

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = (q: string): Promise<string> =>
    new Promise((resolve) => rl.question(q, resolve));

  console.log("\n=== Mastra Research-Writer Agent ===\n");

  const userId =
    (
      await prompt("Enter your user ID (or press Enter for 'demo-user'): ")
    ).trim() || "demo-user";
  const sessionId = randomUUID();

  console.log(`\nUser: ${userId}`);
  console.log(`Session: ${sessionId}\n`);

  await seedKnowledgeBase();

  console.log(`Knowledge base: ${getStoreSize()} chunks ready`);
  console.log("\nCommands:");
  console.log("  /ingest <text>  — Add text to the knowledge base");
  console.log("  /quit           — Exit\n");

  const ask = async () => {
    const input = await prompt("You: ");
    const trimmed = input.trim();

    if (!trimmed) {
      return ask();
    }

    if (trimmed === "/quit") {
      console.log("\nShutting down...");
      await shutdown();
      rl.close();
      process.exit(0);
    }

    if (trimmed.startsWith("/ingest ")) {
      const text = trimmed.slice(8);
      const chunks = await ingest(text);
      console.log(
        `\n[knowledge] Ingested ${chunks} chunks (total: ${getStoreSize()})\n`,
      );
      return ask();
    }

    try {
      console.log("\n[researching...]");
      const response = await runQuery(trimmed, userId, sessionId);
      console.log(`\nAgent: ${response}\n`);
    } catch (err) {
      console.error(
        "\nError:",
        err instanceof Error ? err.message : String(err),
        "\n",
      );
    }

    return ask();
  };

  await ask();
}

main().catch(async (err) => {
  console.error("Fatal:", err);
  await shutdown();
  process.exit(1);
});
