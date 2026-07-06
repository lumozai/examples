import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const demoEnvPath = path.resolve(__dirname, "..", ".env");
dotenv.config({ path: demoEnvPath });

// Suppress OpenAI SDK debug logging (response 200, headers, etc.)
if (!process.env.OPENAI_LOG) {
  process.env.OPENAI_LOG = "error";
}

const openaiKey = process.env.OPENAI_API_KEY || "";
const lumozKey = process.env.LUMOZ_API_KEY || "";
console.log(
  `[env] OPENAI_API_KEY: ${openaiKey ? "ok" : "(missing)"}` +
  ` | LUMOZ_API_KEY: ${lumozKey ? "ok" : "(missing)"}`,
);
