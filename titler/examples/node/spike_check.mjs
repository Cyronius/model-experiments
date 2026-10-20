// Phase-0 spike: load the quantized model with transformers.js in Node,
// verify tokenization matches Python exactly, and run a generation.
// Run from repo root:  node examples/node/spike_check.mjs <model-name>
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AutoTokenizer, env, pipeline } from "@huggingface/transformers";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const modelName = process.argv[2] ?? "titler-spike-web";

env.localModelPath = join(root, "model");
env.allowRemoteModels = false;

const strings = JSON.parse(readFileSync(join(root, "scripts", "spike_test_strings.json"), "utf-8"));
const pythonDump = JSON.parse(readFileSync(join(root, "data", "spike_tokenization_python.json"), "utf-8"));

const tokenizer = await AutoTokenizer.from_pretrained(modelName);
let mismatches = 0;
for (const s of strings) {
  const jsIds = tokenizer.encode(s);
  const pyIds = pythonDump[s];
  const match = JSON.stringify(jsIds) === JSON.stringify(pyIds);
  if (!match) {
    mismatches++;
    console.log(`TOKENIZATION MISMATCH for ${JSON.stringify(s.slice(0, 40))}`);
    console.log(`  python: ${JSON.stringify(pyIds)}`);
    console.log(`  js:     ${JSON.stringify(jsIds)}`);
  }
}
console.log(`tokenization: ${strings.length - mismatches}/${strings.length} match python`);

const titler = await pipeline("text2text-generation", modelName, { dtype: "q8" });
const start = performance.now();
const out = await titler(strings[0], { max_new_tokens: 16, num_beams: 2 });
const ms = Math.round(performance.now() - start);
console.log(`js generation (${ms}ms): ${JSON.stringify(out[0].generated_text)}`);

if (mismatches > 0) process.exit(1);
