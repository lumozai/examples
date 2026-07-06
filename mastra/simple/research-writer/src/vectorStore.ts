import OpenAI from "openai";

let _openai: OpenAI;
function getOpenAI() {
  if (!_openai) _openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  return _openai;
}

interface Document {
  id: string;
  text: string;
  embedding: number[];
}

const store: Document[] = [];
let docCounter = 0;

async function embed(text: string): Promise<number[]> {
  const response = await getOpenAI().embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return response.data[0].embedding;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Split text into overlapping chunks */
function chunk(text: string, size = 512, overlap = 50): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < text.length; i += size - overlap) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

/** Ingest a document into the in-memory vector store */
export async function ingest(text: string): Promise<number> {
  const chunks = chunk(text);
  const embeddings = await Promise.all(chunks.map((c) => embed(c)));
  for (let i = 0; i < chunks.length; i++) {
    store.push({
      id: `doc-${++docCounter}`,
      text: chunks[i],
      embedding: embeddings[i],
    });
  }
  return chunks.length;
}

/** Search the vector store for the top-k most relevant chunks */
export async function search(
  query: string,
  topK = 3,
): Promise<{ text: string; score: number }[]> {
  if (store.length === 0) {
    return [];
  }

  const queryEmbedding = await embed(query);
  return store
    .map((doc) => ({
      text: doc.text,
      score: cosineSimilarity(queryEmbedding, doc.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

export function getStoreSize(): number {
  return store.length;
}
