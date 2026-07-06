import "./env.js";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

try {
  const models = await openai.models.list();
  console.log("[ok] Connected to OpenAI. Models available:", models.data.length);

  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: "hello world",
  });
  console.log("[ok] Embeddings work. Dimensions:", embedding.data[0].embedding.length);

  const chat = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "Say hi in 5 words." }],
    max_tokens: 20,
  });
  console.log("[ok] Chat works. Response:", chat.choices[0].message.content);
} catch (err: any) {
  console.error("[fail]", err.message);
  process.exit(1);
}
