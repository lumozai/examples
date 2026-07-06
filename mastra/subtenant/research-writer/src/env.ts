import dotenv from "dotenv";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const demoEnvPath = path.resolve(__dirname, "..", ".env");
dotenv.config({ path: demoEnvPath });

// Suppress OpenAI SDK debug logging (response 200, headers, etc.) by default.
// The installed OpenAI SDK emits those logs when the generic DEBUG flag is
// exactly "true", which may be present in the shared demo .env.
if (process.env.OPENAI_DEBUG !== "true") {
  process.env.OPENAI_LOG = "error";
  if (process.env.DEBUG === "true") {
    process.env.DEBUG = "false";
  }
}

const openaiKey = process.env.OPENAI_API_KEY || "";
const lumozKey = process.env.LUMOZ_API_KEY || "";
console.log(
  `[env] OPENAI_API_KEY: ${openaiKey ? "ok" : "(missing)"}` +
  ` | LUMOZ_API_KEY: ${lumozKey ? "ok" : "(missing)"}`,
);
