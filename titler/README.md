# titler

A tiny model that turns a prompt into a short title.

```text
how do I sort a list of dicts by a key in python
→ Sorting Dicts By Key in Python
```

| | v2 (current) | v1.5 (legacy) |
|---|---|---|
| Architecture | Llama, decoder-only, 8 layers | T5 encoder-decoder, 3 + 3 layers |
| Parameters | 5.84M | 7.61M |
| File | **4.1 MB** GGUF (Q4_K_M) | 10.3 MB int8 ONNX, or GGUF 5–15 MB |
| CPU speed | about 15–100 ms | 160–220 ms |
| Runs in | Rust, C, Python, JavaScript, browser (WebAssembly), command line | Transformers.js, ONNX Runtime, llama.cpp |
| Where | [`app/`](app/) and [Hugging Face](https://huggingface.co/Cyronius/titler) | [Hugging Face](https://huggingface.co/Cyronius/titler) |

- **Try v2 in your browser:** https://cyronius.github.io/model-experiments/titler/ (runs locally with WebAssembly, nothing is sent anywhere).
- **Use v2 in your own app:** [app/EMBEDDING.md](app/EMBEDDING.md).
- **Model on Hugging Face:** https://huggingface.co/Cyronius/titler (holds both v2 and v1.5).

## v2

The v2 model is a from-scratch decoder-only Llama with a custom Unigram tokenizer (4,352 tokens), quantized to Q4_K_M. The Rust engine in [`app/`](app/) adds the tokenizer, beam search (4 beams by default) and title cleanup around it. Same input gives the same output.

Judged by a blind pairwise judge against v1.5, the v2 title wins about **65%** of the pairs that have a winner.

| Speed (4 beams) | typical | slowest 5% (long prompts) |
|---|---:|---:|
| llama.cpp | 18 ms | 48 ms |
| Native Rust, one thread | 40–70 ms | 200–275 ms |
| Browser (WebAssembly) | 50–175 ms | |

It reads the first 256 tokens of the input, uses about 25 MB of memory, and supports **English only**. Known limits: it sometimes drops an apostrophe in a possessive, and hard inputs such as pasted logs give weaker titles. See [specs/title/spec.md](specs/title/spec.md) for the title rules and the measured limits.

### How v2 was built

Commands run from this folder with `uv run`.

1. **Data.** Scripts `01`–`05` download, clean, deduplicate and sample real first-turn prompts (WildChat, OpenAssistant, Chatbot Arena, LMSYS-Chat-1M). Later scripts add WildChat-4.8M, synthetic hard cases and FineWeb-Edu text, and clean the titles. Qwen3-14B wrote most of the titles. Data licenses are in [app/model/NOTICE.md](app/model/NOTICE.md).
2. **Tokenizer.** `scripts/12_train_tokenizer_v2.py` trains the Unigram tokenizer.
3. **Training.** `scripts/13_train_llama.py` runs two stages: `lm` (next-token training on raw prompts and general text, so the model learns English) and `title` (loss on title tokens only). Training titles have Title Case removed, and `clean_title` puts it back at the end.
4. **Titles for evaluation.** `scripts/15_gen_v2_titles.py` writes titles for the fixed eval set.
5. **Judging.** `scripts/11_pairwise_judge.py` judges two title sets blind. It runs each pair in both orders and counts only verdicts that survive the swap.
6. **Export.** `scripts/export_llama_gguf.py` writes the GGUF with a Unigram tokenizer that matches the training one.
7. **Rust parity.** `scripts/24_make_app_golden.py` writes the golden files that `cargo test -p titler-core` compares the Rust code against.

The history of v1.0 to v1.5 and a timeline of v2 are in [EXPERIMENTS.md](EXPERIMENTS.md).

## v1.5 (legacy)

The earlier model: a 7.61M-parameter T5 trained from scratch on about **481k prompt → title pairs** distilled from larger language models, with 3 encoder and 3 decoder layers, 256 hidden dimensions and a custom 8,192-token vocabulary. Getting there took five iterations. The full history, including what was tried and rejected, is in **[EXPERIMENTS.md](EXPERIMENTS.md)**. Live demo of v1.5 (server-side ONNX inference): https://titler-production.up.railway.app

| Metric | v1.2 | v1.4 | **v1.5** |
| --- | ---: | ---: | ---: |
| Parameters | 7.61M | 7.61M | 7.61M |
| int8 ONNX size | 10.3 MB | 10.3 MB | 10.3 MB |
| CPU latency | ~160–220 ms | ~160–220 ms | ~160–220 ms |
| ROUGE-L vs teacher | 0.484 | 0.473 | **0.513** |
| Format compliance | 98.0% | 97.8% | **99.8%** |

The target output is a useful **3–8 word title**. See [EXPERIMENTS.md](EXPERIMENTS.md) (v1.6) for the format-compliance fix behind the 95.0% → 99.8% jump.

### v1.5 runtimes

- `examples/python/`: ONNX Runtime
- `examples/node/`: Transformers.js
- `examples/web/`: Transformers.js with WASM
- `llama.cpp`: GGUF builds

WebGPU currently has upstream issues with quantized seq2seq models, so the browser example uses WASM.

```javascript
import { pipeline } from "@huggingface/transformers";

const titler = await pipeline("text2text-generation", "Cyronius/titler", { dtype: "q8" });
const result = await titler("how do I center a div horizontally with flexbox", {
  max_new_tokens: 32,
  num_beams: 2,
  no_repeat_ngram_size: 2
});
console.log(result[0].generated_text);
```

### v1.5 training pipeline

```bash
uv run python scripts/01_download.py --output data/01_raw_prompts.jsonl --source wildchat oasst2 arena lmsys
uv run python scripts/02_clean_dedup.py --input data/01_raw_prompts.jsonl --output data/02_deduped.jsonl
uv run python scripts/03_stratify.py --input data/02_deduped.jsonl --output data/03_sampled.jsonl --total 200000
```

Some datasets are gated on Hugging Face. Accept their terms and run `hf auth login`.

The teacher scripts use an OpenAI-compatible endpoint such as LM Studio:

```bash
uv run python scripts/bakeoff.py select --input data/03_sampled.jsonl
uv run python scripts/bakeoff.py run --model <model-id>
uv run python scripts/bakeoff.py report
uv run python scripts/04_generate_titles.py --input data/03_sampled.jsonl --output data/04_titled.jsonl --model <model-id>
uv run python scripts/05_validate_split.py --input data/04_titled.jsonl --outdir data
uv run python scripts/train_tokenizer.py --input data/train.jsonl --output model/tokenizer --vocab-size 8192
```

The v1.5 model was trained with `notebooks/train_titler.ipynb`. Then export and evaluate:

```bash
uv run optimum-cli export onnx --model model/titler-v1 --task text2text-generation-with-past model/titler-v1-onnx
uv run python scripts/quantize.py --input model/titler-v1-onnx --output model/titler-v1-web
uv run python scripts/eval.py --model model/titler-v1-web --test data/test.jsonl --judge <model-id> --hardcases data/hardcases.jsonl
uv run --extra gguf python scripts/export_gguf.py --model model/titler-v15-scratch --out model/titler-v15-gguf --name titler-v1.5
```

### v1.5 GGUF builds

Measured against the f16 baseline on the same 100-prompt held-out set (greedy, temp 0):

| Quant | Size | ROUGE-L vs teacher | Outputs differing from f16 |
| --- | ---: | ---: | --- |
| F16 | 15.4 MB | 0.484 | baseline |
| Q8_0 | 8.3 MB | 0.485 | 6/100, paraphrases only |
| Q6_K | 6.4 MB | 0.490 | 11/100, paraphrases only (recommended if size matters) |
| Q4_0 | 5.0 MB | 0.465 | 29/100, mostly paraphrases, a few real topic drifts |

Run them with `llama-completion -m titler-v1.5-q8_0.gguf -p "<prompt>" -n 32 --temp 0`. LM Studio and Ollama do not run this T5 model correctly, because their inference paths skip the encoder pass.

## Project structure

```text
app/           v2 engine: Rust core, C / Python / npm packages, command line, website, model
scripts/       data, training, export and evaluation tools (both versions)
specs/title/   the title rules and measured limits
examples/      v1.5 inference examples (Python, Node, browser)
hf-release/    the Hugging Face model card
notebooks/     v1.5 training notebook
data/          dataset build outputs
EXPERIMENTS.md history of v1.0 to v2: what was tried and rejected
```

## License

The code and both models (v2 and v1.5) are dual-licensed: you may use them under either the [MIT License](LICENSE-MIT) or the [Apache License 2.0](LICENSE-APACHE), at your option. Data and attribution notes: [app/model/NOTICE.md](app/model/NOTICE.md).
