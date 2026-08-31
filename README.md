# titler

A ~7.6M-parameter from-scratch T5 that turns a user's first AI-chat prompt into a 3-8 word title. CPU-only, ~10MB int8 ONNX, runs in Python (ONNX Runtime), Node, and the browser (transformers.js).

Built by distillation: real first-turn user prompts (WildChat, oasst2, Chatbot Arena, LMSYS-Chat-1M) are titled by a teacher LLM, and the tiny T5 is trained on those pairs. Published model card with the full data/license disclosure and benchmarks: `hf-release/README.md`.

## Pipeline

All commands run from the repo root with `uv run`.

```powershell
# 1. data (WildChat and lmsys are gated: accept terms on HF, then `hf auth login`)
uv run python scripts/01_download.py --output data/01_raw_prompts.jsonl --source wildchat oasst2 arena lmsys
uv run python scripts/02_clean_dedup.py --input data/01_raw_prompts.jsonl --output data/02_deduped.jsonl
uv run python scripts/03_stratify.py --input data/02_deduped.jsonl --output data/03_sampled.jsonl --total 200000

# 2. teacher bake-off (LM Studio running on :1234)
uv run python scripts/bakeoff.py select --input data/03_sampled.jsonl
uv run python scripts/bakeoff.py run --model <model-id>        # repeat per candidate
uv run python scripts/bakeoff.py report                        # -> data/bakeoff_report.md

# 3. bulk title generation (resumable; rerun to continue)
uv run python scripts/04_generate_titles.py --input data/03_sampled.jsonl --output data/04_titled.jsonl --model <winner>
uv run python scripts/05_validate_split.py --input data/04_titled.jsonl --outdir data

# 4. tokenizer + training
uv run python scripts/train_tokenizer.py --input data/train.jsonl --output model/tokenizer --vocab-size 8192
# upload train.jsonl, val.jsonl, model/tokenizer/ to Google Drive under titler/,
# run notebooks/train_titler.ipynb on a Colab T4, download titler-v1.zip -> model/titler-v1/

# 5. export + quantize (transformers.js layout)
uv run optimum-cli export onnx --model model/titler-v1 --task text2text-generation-with-past model/titler-v1-onnx
uv run python scripts/quantize.py --input model/titler-v1-onnx --output model/titler-v1-web

# 6. eval gates
uv run python scripts/eval.py --model model/titler-v1-web --test data/test.jsonl --judge <big-model-id> --hardcases data/hardcases.jsonl
```

## Results (v1.5, 2026-08-31 — current)

Same 3+3/d256/relu from-scratch architecture since v1.1 — every gain since v1.2 came from data,
not the model. v1.5 adds ~113k LMSYS-Chat-1M-derived prompts (targeting the short/vague/casual
distribution the wild pool was thin on) to the pool that v1.4's collapse-fix work already
established as the right architecture to build on.

| gate | v1.2 (339k) | v1.4-scratch (349k, collapse-fix) | **v1.5 (481k)** |
|---|---|---|---|
| ROUGE-L F1 vs teacher (test, n=500, beams=2) | 0.484 | 0.473 | **0.513** |
| Format compliance | 98.0% raw | 97.8% raw | **99.8%** |
| Hard-case suite >= 80% usable | pass | pass, generic-collapse still present | **pass, collapse resolved** |

**Architecture scaling was tried and rejected, twice:** a 21M-param variant (8+4 layers,
d_model 320, gated-gelu — 2.75x the params) trained on the *same* 209k set as v1.1 improved
ROUGE-L by only +0.004 (v1.1→v1.2, see `.claude/plans/titler-v2-8x4-d320-gated.md`). Later,
initializing from a pretrained `t5-efficient-tiny` checkpoint (v1.3) closed some of the ROUGE-L
gap but introduced a new failure mode — confident collapse to generic templates ("User Asks for
Assistance") on short/ambiguous inputs — that took real investigation to root-cause as genuine
distributional collapse, not a decoding bug (`.claude/plans/titler-v1.3-pretrained-init.md`,
`titler-v1.4-collapse-fix-dual-arch.md`). The from-scratch architecture matched or beat the
pretrained one on every axis at half the size, so pretrained init was dropped. **Adding data
(v1.5) is what actually resolved the collapse** — see `.claude/plans/titler-v1.5-lmsys-data-expansion.md`.

**Format compliance was reported at 95.0% through v1.4/early v1.5 and never cleared its own
>=99% gate — turned out to be mostly a measurement gap, not a model-quality one.** `eval.py`
never applied the 8-word clamp that every real consumer of this model already applies downstream
(`examples/`, `scripts/usability_judge.py`), so ~68% of the reported "failures" were titles
exactly 1-2 words over the cap — already-correct titles the deployed pipeline would have shipped
fine. Moved the clamp into the shared `clean_title()` (`scripts/common/validation.py`) as the
single source of truth, fixed `eval.py` to call it, and reported the one remaining ambiguous
check (near-verbatim prompt echoing, which manual review found mostly flags legitimately good
titles) as a separate informational metric rather than folding it into the gate. Real number now:
**99.8%** (one genuine short-output edge case, 0.2%, left as an accepted case — a `min_new_tokens`
decode constraint was tried and made things worse, forcing awkward padding onto already-good
short titles without even fixing the target case). Full investigation:
`.claude/plans/titler-v1.6-format-compliance-fix.md`.

Decode settings: `num_beams=2, max_new_tokens=32, no_repeat_ngram_size=2` + the clamp
(strip quotes, cap 8 words) shown in `examples/`. ~160-220ms per title (Python ORT,
idle machine) — unchanged since v1.1, architecture/size is identical. Artifact: 10.28 MB int8 ONNX.

Known limits: non-English input, long rambling/emotional prompts, and jargon the tokenizer
fragments oddly ("div") can still garble a word or occasionally hallucinate off-topic on
ambiguous long-form prompts.

## Runtimes

- `examples/node/` — transformers.js in Node (also validates JS/Python tokenizer parity)
- `examples/web/` — transformers.js in the browser (WASM; note: q8 seq2seq is broken on WebGPU upstream, use WASM)
- `examples/python/` — optimum ONNX Runtime
- `llama.cpp` (GGUF) — see below

## GGUF (llama.cpp)

```powershell
uv run --extra gguf python scripts/export_gguf.py --model model/titler-v15-scratch --out model/titler-v15-gguf --name titler-v1.5
```

Needs Docker Desktop running (quantization runs via the `ghcr.io/ggml-org/llama.cpp` image);
the llama.cpp conversion tooling itself is auto-fetched into `.cache/` on first run, not vendored.

Benchmarked f16/Q8_0/Q6_K/Q4_0 on the same 100-prompt test set, greedy decode, 2026-08-31 (v1.5):

| quant | size | ROUGE-L vs teacher | outputs differing from f16 |
|---|---|---|---|
| f16 (baseline) | 15.4MB | 0.484 | — |
| Q8_0 (recommended default) | 8.3MB | 0.485 | 6/100 — paraphrases only |
| Q6_K | 6.4MB | 0.490 | 11/100 — paraphrases only |
| Q4_0 | 5.0MB | 0.465 | 29/100 — mostly paraphrases, a few repeated-word artifacts and real topic drifts |

Q8_0 and Q6_K are both effectively free quality-wise (every diff from f16 is a paraphrase, not a
regression); Q6_K is the better default if file size matters, since it's smaller than Q8_0 at no
measured cost. Q4_0 has a real, modest cost (~1 in 3 outputs differ, occasionally changing what
the title is about, not just phrasing) — supported (`--quant Q4_0`) but ship it only if the extra
~1.3-3.3MB genuinely matters. Full per-prompt diffs: ad hoc benchmark script and results were
generated during the v1.5 release — see `.claude/plans/titler-v1.5-lmsys-data-expansion.md`.

llama.cpp's T5 converter requires a real SentencePiece `tokenizer.model` — there's no
`tokenizer.json` fallback — so `scripts/make_spiece.py` synthesizes one from our HF Unigram
tokenizer (same underlying piece+score representation). One learned vocab token is a literal
NUL byte (picked up from a binary/garbled paste during tokenizer training); SentencePiece's
loader rejects that, so the script renames just that one piece's display text, keeping its
id/score in place so the embedding matrix stays aligned.

Run with `llama-completion` (raw completion), not `llama-cli` — the latter applies a chat
template this non-instruction-tuned model was never trained on and silently returns empty output:

```
llama-completion -m titler-v1.5-q8_0.gguf -p "how do I center a div horizontally with flexbox" -n 32 --temp 0
```

`llama-completion`'s default sampler doesn't apply an ngram-repeat ban, so occasional word
repeats ("Chest Hurts After Gym After Gym") show up more than in the ONNX runtimes, which use
`no_repeat_ngram_size=2`; tune `--repeat-penalty` if that matters for your use.

**LM Studio and Ollama do not work with this model.** Both load the GGUF and tokenize correctly,
but their inference backends never run the required encoder pass for T5 (encoder-decoder)
architectures — they're built around decoder-only chat models — so the decoder generates from an
empty cross-attention state and immediately emits EOS: a silent empty response on every prompt,
from the chat UI *and* their raw completion APIs. Confirmed directly against LM Studio's API on
this exact model, and it's not a chat-template problem — a custom template that strips all
chat formatting still returns empty. Use `llama-completion`, Docker's
`ghcr.io/ggml-org/llama.cpp:light` image (`--entrypoint /app/llama-completion`), or another tool
that correctly calls `llama_encode()` before `llama_decode()` for encoder-decoder models.

## Notes

- Architecture and all design decisions: `.claude/plans/titler-v1.md`, `titler-v2-8x4-d320-gated.md`,
  `titler-v1.3-pretrained-init.md`, `titler-v1.4-collapse-fix-dual-arch.md`,
  `titler-v1.5-lmsys-data-expansion.md`, `titler-v1.6-format-compliance-fix.md` (local only).
- `feed_forward_proj: "relu"` and 3+3 layers are size-driven: the ONNX export duplicates the embedding matrix in encoder and decoder graphs, so 4+4/gated variants blow the ~10MB budget.
- `accelerate` must be installed for the optimum export or tied weights get duplicated (+8MB).
- Do not pass `decoder_file_name` to `ORTModelForSeq2SeqLM.from_pretrained` — use `use_merged=True` and let the pattern match find the `*_quantized.onnx` files.
