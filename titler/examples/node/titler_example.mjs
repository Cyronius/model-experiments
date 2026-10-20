// Minimal Node inference example (quantized ONNX, CPU).
// Run from repo root:  node examples/node/titler_example.mjs [model-name]
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { env, pipeline } from "@huggingface/transformers";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
env.localModelPath = join(root, "model");
env.allowRemoteModels = false;

const titler = await pipeline("text2text-generation", process.argv[2] ?? "titler-v1-web", { dtype: "q8" });

// product-boundary clamp: strip stray quotes/punctuation, cap at 8 words
const clampTitle = (t) =>
  t.replace(/["'“”‘’`]/g, "").replace(/[.!?:;,\s]+$/, "").split(/\s+/).slice(0, 8).join(" ");

const prompts = [
  "how do I center a div horizontally with flexbox",
  "write a poem about the ocean at sunset",
  "my docker compose file keeps failing with port is already allocated",
  "hi",
  "explain quantum entanglement like I'm five",
];
for (const prompt of prompts) {
  const start = performance.now();
  const out = await titler(prompt, { max_new_tokens: 32, num_beams: 2, no_repeat_ngram_size: 2 });
  const ms = Math.round(performance.now() - start);
  console.log(`(${String(ms).padStart(4)}ms) ${JSON.stringify(prompt.slice(0, 50))} -> ${JSON.stringify(clampTitle(out[0].generated_text))}`);
}
