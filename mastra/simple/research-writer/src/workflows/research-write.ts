import { createStep, createWorkflow } from "@mastra/core/workflows";
import { z } from "zod";
import { researchAgent } from "../agents/researcher.js";
import { writerAgent } from "../agents/writer.js";

const researchStep = createStep({
  id: "research",
  inputSchema: z.object({ query: z.string() }),
  outputSchema: z.object({ research: z.string(), query: z.string() }),
  execute: async ({ inputData, tracing, tracingContext }) => {
    const result = await researchAgent.generate(
      `Research the following topic thoroughly using the knowledge base: ${inputData.query}`,
      { tracing, tracingContext },
    );
    return { research: result.text, query: inputData.query };
  },
});

const writeStep = createStep({
  id: "write",
  inputSchema: z.object({ research: z.string(), query: z.string() }),
  outputSchema: z.object({ response: z.string() }),
  execute: async ({ inputData, tracing, tracingContext }) => {
    const result = await writerAgent.generate(
      `Based on the following research, write a response to the question: "${inputData.query}"\n\n--- Research ---\n${inputData.research}`,
      { tracing, tracingContext },
    );
    return { response: result.text };
  },
});

export const researchWriteWorkflow = createWorkflow({
  id: "research-write",
  inputSchema: z.object({ query: z.string() }),
  outputSchema: z.object({ response: z.string() }),
  steps: [researchStep, writeStep],
});

researchWriteWorkflow.then(researchStep).then(writeStep).commit();
