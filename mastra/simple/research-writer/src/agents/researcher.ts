import { createOpenAI } from "@ai-sdk/openai";
import { Agent } from "@mastra/core/agent";
import { createTool } from "@mastra/core/tools";
import { z } from "zod";
import { search } from "../vectorStore.js";

const openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });

const vectorSearchTool = createTool({
  id: "vector-search",
  description:
    "Search the knowledge base for information relevant to a query. Returns the most relevant text chunks.",
  inputSchema: z.object({
    query: z.string().describe("The search query to find relevant documents"),
  }),
  outputSchema: z.object({
    results: z.array(
      z.object({
        text: z.string(),
        score: z.number(),
      }),
    ),
    found: z.boolean(),
  }),
  execute: async ({ query }) => {
    const results = await search(query, 5);
    return { results, found: results.length > 0 };
  },
});

export const researchAgent = new Agent({
  id: "research-agent",
  name: "Research Agent",
  instructions: `You are a research assistant that finds and synthesizes information from a knowledge base.

Your workflow:
1. Use the vector-search tool to find relevant information for the user's query
2. Synthesize the search results into a coherent research summary
3. Cite specific findings from the retrieved documents
4. If the knowledge base has no relevant information, clearly state that and provide what general knowledge you can

Be thorough but concise. Focus on facts and evidence from the knowledge base.`,
  model: openai("gpt-4o-mini"),
  tools: { vectorSearchTool },
});
