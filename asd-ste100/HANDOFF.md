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
- Teacher generation and model training use a rented pod, with a $5 budget and no paid API calls. For the current million-pair expansion, the owner explicitly selected **free downloads and local processing only** on 2026-10-23. No additional pod spend is authorized for this expansion.

## Current state

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

Pair preparation completed locally on 2026-10-23, with no spend:
- `data/noise_rules.py`, `text_utils.py`, `build_pairs.py`, and `validate_pairs.py` are implemented. `data/out/pairs-v1/` contains 28,411 aligned surface-token pairs: 25,552 train, 1,368 dev, 1,491 test. Slices: 7,489 original rewrites, 6,638 messy back-translations, 14,284 rule variants.
- Targets require score 5 **and** `changed: false`; 20 score-5 rows fail the latter. Dictionary evaluation texts are excluded. Origins and identical source/target texts stay in one split. The builder drops 682 normalized unchanged messy rows, 38 preambles, 837 messy and 287 original protected-content mismatches, plus length/duplicate failures. All targets pass the current checker error filter.
- Noise and back-translations remain unjudged for meaning. Tense/voice patterns are conservative; only 4 passive and 2 nominalization examples survive in this corpus. Surface tokens/canvases are prepared, but model vocabulary IDs, BPE fallback and unknown-word slotting still need implementation. Commands, schema and validation are in `data/README.md`.
- Validation on real `full1` artifacts: all 28,411 emitted pairs pass the full audit, including target provenance, protected-token equality, canvas reconstruction and split separation. Ten offline unit tests pass. A second full build produces byte-identical train/dev/test, rejection and summary files. No checker implementation changed.

Not started: model code, GGUF writer, llama.cpp patch, NPU engine. The owner-verified gold set and real-input meaning check are also still outstanding. The model tokenizer/slotter is now implemented; see the continuation below.

Latest work, 2026-10-23:
- The noiser now preserves capital `A` as a possible variable and limits perfect-tense changes to subject-pronoun clause positions. Rebuilt teacher-target pairs are in `data/out/pairs-v2/`: 28,381 pairs (25,523 train, 1,367 dev, 1,491 test), all audited. `pairs-v1` remains a historical snapshot and the exclusion set for the local expansion.
- Downloaded 17 pinned public files (2.62 GB) from Magicoder, selected AgentInstruct subsets, and the four new Apache-licensed SmolTalk subsets. Catalog and license scope: `data/source_catalog.json`. Raw data/cards/manifests: `data/out/million/` (ignored).
- The free local expansion is complete in `data/out/local-million-v1/`: **1,000,000 unique pairs from 403,989 distinct target texts**, split into 906,045 train, 48,332 dev and 45,623 test. There are 41 gzip shards totaling 607.9 MB. Raw checker-filtered text supplies targets; structural rules supply noisy inputs. These carry `target_review: checker_only` and `meaning_review: unjudged_rule`, with zero teacher or judge calls. This does not meet the hand-verified gold-set requirement.
- The full audit passed all 1,000,000 pairs and 180,754 recorded documents: no protected-content changes, split leakage by recorded text/document, invalid rule corruption, conflicting source labels or canvas reconstruction failures. `summary.json` lists the shards and hashes; `validation.json` records the audit. Nineteen unit tests passed, including resume and parallel processing, and the nine local-corpus tests passed again after audit refinements.
- Coverage is uneven: 895,240 article-deletion pairs, 71,338 sentence merges, 32,884 formal-phrase variants, 486 perfect-tense variants, and 26 each of passive and nominalization. Use this as a large augmentation corpus; hard rewrites and owner-verified real-input evaluation remain gaps. A 200-pair `review_sample.jsonl` oversamples rare operations and is not verified gold.
- Rebuild/resume with `data/parallel_local_stage.py`, then `export_local_million.py` and `validate_local_million.py`. Full commands, attribution and limits are in `data/LOCAL_CORPUS.md`. The early stage was curated to remove headings/code fragments and corrected-noise failures; history and code fingerprints are retained in the artifact directory. Keep the teacher-target corpus identified separately; recompute joint groups before constructing combined training/evaluation splits.

Tokenizer continuation, 2026-10-23:
- `tokenizer/ste_tokenizer.py` and `prepare.py` implement the closed output vocabulary, input-only byte BPE, source-built copy slots, additive output mask, case features, ID alignment, bucket reassignment, compressed export and independent reconstruction audit. Commands and schema are in `tokenizer/README.md`. All work is local, with no new spend and no checker changes.
- `data/out/tokenizer-v3/` has 2,568 output IDs and 9,458 input IDs, including a 4,096-token byte fallback fitted only on `pairs-v2` training sources. Protected code/identifiers, numbers/units, heuristic multiword names and unknown tokens are copied through request-local slots. Training uses 70% slots and 30% raw BPE spans with explicit slot-reference features; targets always use slots. The model must consume those features. Name/unit detection is heuristic, and the technical-verb slot pool remains reserved.
- `data/out/id-pairs-v3/` retains 13,414 of 28,381 teacher-corpus pairs (12,067 train, 636 dev, 711 test), all audited. Of these, only 1,876 are original teacher rewrites; 682 are messy back-translations and 10,856 are rule augmentations. Remaining pairs fail vocabulary, exact slot preservation or capacity gates. Do not silently expand the output vocabulary to recover them: checker-clean and representable under a closed vocabulary are different requirements. The original surface corpus remains unchanged.
- `data/out/id-local-million-v3/` retains 610,642 of the million local pairs: 553,586 train, 29,262 dev, 27,794 test. All retained rows pass the full ID reconstruction audit. There are 27 gzip shards totaling 348.4 MB; 384,887 of the 389,358 rejected rows contain a target word outside the output vocabulary. Review labels remain checker-only and unjudged-rule. Neither corpus has been merged or trained.
- Thirteen tokenizer tests pass, in addition to the nineteen data tests. Real teacher-corpus exports with one and four workers have byte-identical shards. Early `tokenizer-v1/v2` and `id-pairs-v1/v2` artifacts are exploratory; `id-local-million-v2` is an interrupted export, not a usable dataset. Use manifest-listed final `v3` artifacts.

## Do next, in order

1. Done 2026-10-20: the full session `full1` (see State). Notes for the next pod run follow. Ask the owner before you launch or stop a pod. Steps are in `asd-ste100/pod/README.md`. The teacher is now Qwen3.6-35B-A3B-FP8 on vLLM 0.32, and the judge is gpt-oss-20b on llama.cpp. Smoke results on a secure-cloud L40S (2026-10-19), 200 sources each:
   - v1 prompt (hints said "use X"): 186 accepted, judge mean 4.22, 134 rated 5, 22 rated 1.
   - v2 prompt (hints are examples, meaning first): 175 accepted, judge mean 4.37, 127 rated 5, 13 rated 1. This is the current prompt.
   - Generation took 36 s for 200 sources. The judge took about 40 s.
   - The remaining score-1 rows are mostly word-sense swaps from the dictionary's suggestions (backs to rear, arrest to stop). Keep only rows the judge rates 5 when you build pairs. The judge now scores every row by default.
   - 48 GB cards are often out of stock. Retry for a few minutes. Do not use port 8081 on RunPod; nginx owns it.
2. Done 2026-10-23: rule noiser, pair builder and model tokenizer/slotter; use `data/README.md` and `tokenizer/README.md` to rebuild and validate. Before combined training, recompute joint connected groups across teacher and local augmentation texts/documents, keeping their review labels and reporting the strict tokenizer rejection rate. Then implement the iterative tagger baseline locally. Do not treat the generated dev/test splits as the owner-verified gold set.
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
