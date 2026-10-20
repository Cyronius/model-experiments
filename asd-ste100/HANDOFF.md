# Handoff: STE rewriter

Read `asd-ste100/PLAN.md` for the full design and `asd-ste100/pod/README.md` for the pod steps. This page is the short version.

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
- The goal is clear, simple language, not fewer tokens. Success is checker-clean STE that keeps the meaning. Token counts are reported, never targeted; STE often costs a few percent more tokens and that is fine. No compact numeric output format.
- The lexicon comes from the PyPI packages `biz-dfch-asdste100vocab` and `biz-dfch-asdste100rules`. They are AGPL. Derived files stay gitignored and are rebuilt with `build_lexicon.py --fetch`. Never commit them.
- Software words: `lexicon/tech_words.txt` lists about 400 words that coding text uses (main, master, branch, commit, default, run, build, and so on). We treat them as approved, and the model output vocabulary must include them. Edit the file by hand. Words the dictionary does not list are kept as they are. No glossaries. The aim is about 80 percent of STE, not all of it.
- Data generation and training run on a rented pod. Pod budget is $5. No paid API calls.

## State on `main`

Done and tested:
- `asd-ste100/lexicon/build_lexicon.py`: 2,200 rows, 1,838 aligned pairs, 66 rules.
- `asd-ste100/checker/checker.py`: tier A subset. On the dictionary's own pairs with strict STE, 93.7% of STE passes and 81.5% of non-STE is caught. With `tech_words.txt` the numbers are 94.9% and 72.0%. That drop is by design: the list approves words that the dictionary examples reject. `calibrate.py` prints both. About 6% false rejects, mostly nouns spelled like rejected verbs.
- `asd-ste100/data/gen.py`, `fetch_sources.py`, `judge.py`. Two real smoke runs (200 sources each, pruned Qwen3.6-27B-A2.8B on llama.cpp, L40S) accepted 38% and 46%. Most accepted PROC rows were answers to the source with invented facts, not rewrites. Fixes since then, tested on the saved smoke rows and a fake server only:
  - The source goes inside `<text>` tags, and the teacher is told not to answer it.
  - The first prompt lists the source's banned words with their alternatives.
  - The retry feedback uses plain words, not rule codes.
  - `sane()` rejects answers (a question or request that comes back as a statement) and invented names, numbers, or words. On the saved rows it rejects 22 of 31 PROC rows (all answers but 1) and 8 of 72 DESC rows (about 2 false rejects).
  - Sources are 1 to 3 sentences and at most 60 words. Output is capped at 300 tokens.
- Full data run `full1` (2026-10-20, secure L40S, $1.71): 12,000 sources, 11,464 clean rows (10,595 seed + 869 hard), 11,417 judged, 8,587 rated 5 (PROC 80%, DESC 73%), 8,587 messy back-translations. Output in `data/out/full1/` (gitignored). In the messy file, 555 rows are identical to the STE target and 7 start with a style label or preamble; `build_pairs.py` must drop them.
- `asd-ste100/pod/`: scripts with a hard stop at 85% of budget. Pod runs work on secure-cloud L40S; a community host had a CUDA 12.4 driver that cannot run the CUDA 12.8 image.

Not started: noiser, pair builder, model code, GGUF writer, llama.cpp patch, NPU engine.

## Do next, in order

1. Done 2026-10-20: the full session `full1` (see State). Notes for the next pod run follow. Ask the owner before you launch or stop a pod. Steps are in `asd-ste100/pod/README.md`. The teacher is now Qwen3.6-35B-A3B-FP8 on vLLM 0.32, and the judge is gpt-oss-20b on llama.cpp. Smoke results on a secure-cloud L40S (2026-10-19), 200 sources each:
   - v1 prompt (hints said "use X"): 186 accepted, judge mean 4.22, 134 rated 5, 22 rated 1.
   - v2 prompt (hints are examples, meaning first): 175 accepted, judge mean 4.37, 127 rated 5, 13 rated 1. This is the current prompt.
   - Generation took 36 s for 200 sources. The judge took about 40 s.
   - The remaining score-1 rows are mostly word-sense swaps from the dictionary's suggestions (backs to rear, arrest to stop). Keep only rows the judge rates 5 when you build pairs. The judge now scores every row by default.
   - 48 GB cards are often out of stock. Retry for a few minutes. Do not use port 8081 on RunPod; nginx owns it.
2. Write `data/noise_rules.py`: reverse the dictionary (swap approved words for rejected words that list them as alternatives), plus merges, nominalizations, dropped articles, perfect tense, passive. Write `data/build_pairs.py` to assemble, dedupe, align, and bucket.
3. Build a gold set of 300 sentences and 50 paragraphs of real messy text with hand-verified STE. The owner verifies.
4. Meaning check on real input: run 200 of the owner's real prompts through the teacher and have the judge score meaning preservation. Report token counts too, as information only.
5. Improve the checker: noun clusters (R2.1), one instruction per sentence (R5.2), articles (R4.5). Re-run `checker/tests/calibrate.py` after every change and keep both strict STE numbers from falling.
6. Train the tagger baseline on the pod. Then the canvas model.
7. Before writing the llama.cpp architecture, verify it can return logits at every position of a non-causal batch and accept a per-call additive bias.
8. Time random-weight models at three widths on llama.cpp CPU, llama.cpp Vulkan, and the NPU engine. ggml-xdna lost on every model narrower than width 2,048 and skips chunks under 1,024 tokens, so expect the NPU may lose. Report numbers before building NPU kernels.

## Rules for the agent

- Test against real data or a fake server and say which. Do not claim a run you did not do.
- Report checker numbers whenever you change the checker.
- Never commit AGPL-derived files or model weights.
- Stop and ask the owner before spending money, launching or stopping a pod, or changing a decision above.
