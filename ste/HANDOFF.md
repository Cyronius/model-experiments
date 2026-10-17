# Handoff: STE rewriter

Read `ste/PLAN.md` for the full design and `ste/pod/README.md` for the pod steps. This page is the short version.

## Goal

A tiny, fast, non-autoregressive model that rewrites English into ASD-STE100 Simplified Technical English. Its job is to normalize text going into or out of LLMs. It must never emit a word outside the STE vocabulary, and it must leave code, identifiers, paths, URLs, and numbers untouched. Targets, not gates: under 100 MB, under 500 ms per sentence on CPU, NPU preferred, GGUF format, no API spend.

## Design in six lines

1. Input is one sentence up to several paragraphs plus a `PROC` or `DESC` mode. Long text is split at paragraphs, then sentences, into 64, 128, and 256 token buckets.
2. The model edits a fixed canvas that starts as a copy of the source, over a few passes. Frozen positions stay put.
3. Output scores only a closed STE word list plus slot tokens, through an additive lexicon mask.
4. A deterministic checker runs between passes, feeds violations back, and decides when to stop.
5. Protected spans become slot tokens before the model runs and are restored after.
6. Build the iterative tagger first as the baseline. The canvas model comes second and must beat it.

## Decisions already made. Do not reopen.

- Format is GGUF. A new `ste` architecture goes into llama.cpp for CPU and Vulkan. An engine copied from OpenFlowLM-Next's `open_embedding` covers the NPU.
- Width is 512 by default, 6 layers. It must be a multiple of 256 for ggml block types. Phase 3 times 512, 768, and 1024. Width goes up only if quality falls short.
- Success is measured in Qwen3 tokens, with tiktoken `o200k_base` as a check. No compact numeric output format.
- The lexicon comes from the PyPI packages `biz-dfch-asdste100vocab` and `biz-dfch-asdste100rules`. They are AGPL. Derived files stay gitignored and are rebuilt with `build_lexicon.py --fetch`. Never commit them.
- Data generation and training run on a rented pod. Pod budget is $5. No paid API calls.

## State on branch `ccr-822a92e7-k01v1x`

Done and tested:
- `ste/lexicon/build_lexicon.py`: 2,200 rows, 1,838 aligned pairs, 66 rules.
- `ste/checker/checker.py`: tier A subset. On the dictionary's own pairs, 93.7% of STE passes and 81.5% of non-STE is caught. About 6% false rejects, mostly nouns spelled like rejected verbs.
- `ste/data/gen.py`, `fetch_sources.py`, `judge.py`: tested only against a fake teacher.
- `ste/pod/`: scripts with a hard stop at 85% of budget, tested with a fake clock.

Not started: noiser, pair builder, model code, GGUF writer, llama.cpp patch, NPU engine.

## Do next, in order

1. Run the pod smoke session, then the full session. The owner launches the pod. You cannot. Steps are in `ste/pod/README.md`. Verify the model file globs in `pod/common.sh` first; they were never checked against Hugging Face.
2. Write `data/noise_rules.py`: reverse the dictionary (swap approved words for rejected words that list them as alternatives), plus merges, nominalizations, dropped articles, perfect tense, passive. Write `data/build_pairs.py` to assemble, dedupe, align, and bucket.
3. Build a gold set of 300 sentences and 50 paragraphs of real messy text with hand-verified STE. The owner verifies.
4. Week-one measurement: run 200 of the owner's real prompts through the teacher and compare token counts before and after. If STE does not save tokens, tell the owner before building further.
5. Improve the checker: noun clusters (R2.1), one instruction per sentence (R5.2), articles (R4.5). Re-run `checker/tests/calibrate.py` after every change and keep both numbers from falling.
6. Train the tagger baseline on the pod. Then the canvas model.
7. Before writing the llama.cpp architecture, verify it can return logits at every position of a non-causal batch and accept a per-call additive bias.
8. Time random-weight models at three widths on llama.cpp CPU, llama.cpp Vulkan, and the NPU engine. ggml-xdna lost on every model narrower than width 2,048 and skips chunks under 1,024 tokens, so expect the NPU may lose. Report numbers before building NPU kernels.

## Rules for the agent

- Test against real data or a fake server and say which. Do not claim a run you did not do.
- Report checker numbers whenever you change the checker.
- Never commit AGPL-derived files or model weights.
- Stop and ask the owner before spending money, launching or stopping a pod, or changing a decision above.
