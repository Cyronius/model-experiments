# titler

A tiny model that turns a prompt into a short title.

> **New: titler v2** is a standard Llama-architecture GGUF (5.8M parameters, **4 MB**) that runs on the CPU in 15-100 ms. It wins about
> 65% of the head-to-head pairs against the v1.5 model described below.
> Try it in your browser: **https://cyronius.github.io/model-experiments/** (runs locally with WebAssembly, nothing is sent anywhere).
> Use it in your own app (Rust, C, Python, JavaScript, command line): [app/EMBEDDING.md](app/EMBEDDING.md).
> Code and model: [app/](app/). Training and evaluation scripts are in [scripts/](scripts/). The sections below describe v1.5.


**7.6M parameters · 10.3 MB int8 ONNX · CPU-only · browser / Node / Python**

Example:

```text
how do I center a div horizontally with flexbox
→ Centering a Div Humanly with Flexbox
```

The released model is on Hugging Face:

https://huggingface.co/Cyronius/titler

Live demo (server-side ONNX inference, no browser WASM): https://titler-production.up.railway.app

## Why?

Generating a conversation title is a tiny job, but it's usually handed to a general-purpose LLM.

titler is an experiment in doing the opposite: train the smallest practical model specifically for that task.

It uses a from-scratch T5 architecture with:

- 3 encoder layers
- 3 decoder layers
- 256 hidden dimensions
- custom 8,192-token vocabulary
- ~7.61M parameters

The current model was trained on about **481k prompt → title pairs** distilled from larger language models.

Getting here took five iterations, not one — full history, including what was tried and rejected
and why: **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Performance

| Metric | v1.2 | v1.4 | **v1.5 (current)** |
| --- | ---: | ---: | ---: |
| Parameters | 7.61M | 7.61M | 7.61M |
| int8 ONNX size | 10.3 MB | 10.3 MB | 10.3 MB |
| CPU latency | ~160–220 ms | ~160–220 ms | ~160–220 ms |
| ROUGE-L vs teacher | 0.484 | 0.473 | **0.513** |
| Format compliance | 98.0% | 97.8% | **99.8%** |

The target output is a useful **3–8 word title**. See **[EXPERIMENTS.md](EXPERIMENTS.md)** (v1.6)
for the format-compliance fix behind the 95.0% → 99.8% jump.

## Runtimes

Examples are included for:

- `examples/python/` — ONNX Runtime
- `examples/node/` — Transformers.js
- `examples/web/` — Transformers.js + WASM
- `llama.cpp` — GGUF builds

WebGPU currently has upstream issues with quantized seq2seq models, so the browser example uses WASM.

## Browser / Node

```javascript
import { pipeline } from "@huggingface/transformers";

const titler = await pipeline(
  "text2text-generation",
  "Cyronius/titler",
  { dtype: "q8" }
);

const result = await titler(
  "how do I center a div horizontally with flexbox",
  {
    max_new_tokens: 32,
    num_beams: 2,
    no_repeat_ngram_size: 2
  }
);

console.log(result[0].generated_text);
```

## Training pipeline

The repository contains the complete pipeline used to build the model.

Commands are run from the repo root with `uv run`.

### 1. Download and prepare prompts

```bash
uv run python scripts/01_download.py \
  --output data/01_raw_prompts.jsonl \
  --source wildchat oasst2 arena lmsys

uv run python scripts/02_clean_dedup.py \
  --input data/01_raw_prompts.jsonl \
  --output data/02_deduped.jsonl

uv run python scripts/03_stratify.py \
  --input data/02_deduped.jsonl \
  --output data/03_sampled.jsonl \
  --total 200000
```

Some datasets are gated on Hugging Face and require accepting their terms and running:

```bash
hf auth login
```

### 2. Generate teacher titles

The generation scripts use an OpenAI-compatible endpoint such as LM Studio.

```bash
uv run python scripts/bakeoff.py select \
  --input data/03_sampled.jsonl

uv run python scripts/bakeoff.py run \
  --model <model-id>

uv run python scripts/bakeoff.py report
```

Then generate the full training set:

```bash
uv run python scripts/04_generate_titles.py \
  --input data/03_sampled.jsonl \
  --output data/04_titled.jsonl \
  --model <model-id>

uv run python scripts/05_validate_split.py \
  --input data/04_titled.jsonl \
  --outdir data
```

### 3. Train tokenizer

```bash
uv run python scripts/train_tokenizer.py \
  --input data/train.jsonl \
  --output model/tokenizer \
  --vocab-size 8192
```

The model itself was trained using the notebook in:

```text
notebooks/train_titler.ipynb
```

### 4. Export to ONNX

```bash
uv run optimum-cli export onnx \
  --model model/titler-v1 \
  --task text2text-generation-with-past \
  model/titler-v1-onnx

uv run python scripts/quantize.py \
  --input model/titler-v1-onnx \
  --output model/titler-v1-web
```

### 5. Evaluate

```bash
uv run python scripts/eval.py \
  --model model/titler-v1-web \
  --test data/test.jsonl \
  --judge <model-id> \
  --hardcases data/hardcases.jsonl
```

## GGUF

GGUF exports are supported for `llama.cpp`.

```bash
uv run --extra gguf python scripts/export_gguf.py \
  --model model/titler-v15-scratch \
  --out model/titler-v15-gguf \
  --name titler-v1.5
```

Current builds, benchmarked against the f16 baseline on the same 100-prompt held-out set (greedy, temp 0):

| Quant | Size | ROUGE-L vs teacher | Outputs differing from f16 |
| --- | ---: | ---: | --- |
| F16 | 15.4 MB | 0.484 | baseline |
| Q8_0 | 8.3 MB | 0.485 | 6/100, paraphrases only |
| Q6_K | 6.4 MB | 0.490 | 11/100, paraphrases only — recommended if size matters |
| Q4_0 | 5.0 MB | 0.465 | 29/100, mostly paraphrases, a few real topic drifts |

Run GGUF models with `llama-completion`:

```bash
llama-completion \
  -m titler-v1.5-q8_0.gguf \
  -p "how do I center a div horizontally with flexbox" \
  -n 32 \
  --temp 0
```

LM Studio and Ollama currently don't work correctly with this T5 model because their inference paths do not execute the encoder pass required by encoder-decoder architectures.

## Dataset

Training prompts come from:

- WildChat
- OpenAssistant / OASST2
- Chatbot Arena
- LMSYS-Chat-1M
- synthetic task-specific examples

Large teacher models generate the target titles, and titler learns from those prompt/title pairs.

See the Hugging Face model card for dataset and licensing details.

## Known limitations

The model is intentionally tiny.

It works best on English prompts that contain a reasonably clear subject. Weak spots include:

- non-English text
- long or rambling prompts
- ambiguous one-word inputs
- uncommon jargon fragmented poorly by the tokenizer

Failures are usually awkward titles rather than total topic misses.

## Project structure

```text
data/          dataset build outputs
examples/      Python, Node and browser inference examples
hf-release/    Hugging Face release files
notebooks/     training notebooks
scripts/       data, training, export and evaluation tools
EXPERIMENTS.md full methodology and results, every version, what was tried and rejected
```

## License

Apache 2.0.

Model weights:

https://huggingface.co/Cyronius/titler