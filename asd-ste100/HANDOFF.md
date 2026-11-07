# Handoff: STE rewriter

Read `asd-ste100/PLAN.md` for the full design and `asd-ste100/pod/README.md` for the pod steps. This page is the short version.

## Goal

A tiny, fast, non-autoregressive model that rewrites English into ASD-STE100 Simplified Technical English. Its job is to normalize text going into or out of LLMs. It must never emit a word outside the STE vocabulary, and it must leave code, identifiers, paths, URLs, and numbers untouched. Targets, not gates: under 100 MB, under 500 ms per sentence on CPU, NPU preferred, GGUF format, no API spend. **Changed 2026-11-06 (owner):** the size and speed targets are suspended. First show a model that rewrites well at any small size, then shrink it. See "November 6" at the end.

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

Joint/model continuation, 2026-10-24:
- Restored and hash-verified all 120 transfer files. Joint surface groups/splits and strict ID exports are built and fully audited. Current corpus: `data/out/id-joint-v1`, 624,072 pairs (565,599 train / 29,977 dev / 28,496 test), with `tokenizer-joint-v1` fitted only on joint train. The output vocabulary is unchanged. See [joint results](data/JOINT_SPLITS.md).
- `model_tagger/` now implements the offline baseline with raw-span pooling, slot/case/checker features, twenty transforms, masks and guarded iterative decoding. Default-size random-weight forward/loss/backward/action decode passed. No trainer, optimizer steps, trained weights or runtime export exists. See [model notes](model_tagger/README.md).
- 26 data, 13 tokenizer and 6 tagger tests pass. No checker change or paid compute. Next: full multi-pass label coverage/slot audits, gold and real-input meaning evaluation, then authorized training. Earlier "not started" model notes and "do next" joint grouping steps above are historical; use this continuation and the root handoff for current status.

Deterministic cleanup continuation, 2026-10-24:
- Current model-data subset: `data/out/id-joint-clean-v2`, 607,040 pairs (550,129 train / 29,263 dev / 27,648 test), with unchanged retained rows and parent joint splits/groups. 17,032 rows are excluded by deterministic gates; 3,035 review-only flagged rows remain. Original parent/source artifacts are intact.
- Future noising/extraction/builders now block the corresponding literal-copy, list-marker and damaged-target patterns. Checker and output vocabulary are unchanged. Data/ID/subset audits and code/input fingerprints are saved with the filtered corpus; 34 data, 13 tokenizer and six tagger tests pass. See [cleanup details](data/JOINT_SPLITS.md#deterministic-cleanup-october-24-2026).
- Semantic and broader grammar failures remain outside this pass. Owner-verified evaluation and full iterative tag-label coverage remain required before training. No paid compute or optimizer steps.

Local training continuation, 2026-10-24:
- Owner authorized finishing the rollout audit and training locally. `tagger-rollouts-v3` passes full trajectory/feature/slot/target replay and canonical-text split audits: 599,855 pairs, including 543,592 train pairs. Eighteen pairs in two groups with token-canonical train/dev overlap are quarantined; remaining splits/groups are unchanged.
- `train.py`, `packed.py`, `rollouts.py`, `audit_splits.py` and `quarantine_rollouts.py` implement preprocessing, audit and resumable CPU training. The decoder's zero-pass checker exit was corrected so article repairs can run. Twelve model/rollout tests pass; optimizer/checkpoint/resume and full pilot epoch were exercised. Resume matched a continuous run bit-for-bit.
- `local-train-v1` is running as user service `asd-ste100-local-v1.service`, three epochs with BF16, eight CPU threads, checkpoints every 500 updates and a 24-hour cap. Read `data/out/local-train-v1/status.json` and `local-train-v1.log`. No paid resources. See root handoff/model README for current commands; earlier statements that no trainer exists are historical.

Training/evaluation completed:
- Local training finished all three epochs / 101,937 updates in 5h43m, best dev loss 0.27277. Service is inactive after successful completion. Best and resumable checkpoints are saved.
- Frozen full-test evaluation is complete: 62.46% exact references on 27,648 noisy test inputs; 63.93% clean controls changed; 4.66% clean token edit rate. The baseline is **not ready for use**. See `data/out/evaluation-v2/assessment.json` and examples.
- Concrete blockers include encoder-only software words (file/delete/existing/otherwise), over-editing, weak original/messy rewriting and unsafe raw code/markup outside slot detection. No checkpoint/runtime/vocabulary changes or paid resources were used during evaluation. Next work needs deliberate policy review, inference guards and dev-only calibration; owner-verified meaning evaluation remains outstanding.

## 2026-10-25: vocabulary fixes and guarded retry loop implemented

User authorized the specific fixes followed by the loop. No paid resources or new
training run were used. Existing `local-train-v1` weights and `evaluation-v2`
results remain frozen. Snapshot of original tagger/tokenizer/whitelist is under
`data/out/fixes-v1/baseline/`.

Implemented:
- Seven explicit software forms: file/files/delete/deletes/deleted/existing/otherwise.
- Additional atomic literal detection: balanced raw calls, quotes, multi-backtick
  and unmatched code, HTML, C++/C#/.NET, formal math, paired bold, generic URI schemes.
- `migrate_vocabulary.py`: inherits train-only BPE; remaps embeddings, output heads,
  transforms and copy-reference rows by token identity. All inherited input/BPE,
  output emission and slot-reference parameters verified exactly equal. New
  emission rows suppressed; these need training for insertion/replacement. KEEP
  now works for promoted source words. No old optimizer/packed-ID reuse allowed.
- `safe_editor.py`: confidence margin, frozen copy anchors, literal validation,
  ordered content-word preservation (articles and three explicit utilize/use
  inflections excepted), last accepted state, rejected-action feedback and trace.
  Rejections count toward the five-attempt cap. Exhausted/stalled/capped results
  remain unresolved; callers must inspect that flag. Baseline rewrite is unchanged.
- CLI requires checkpoint/vocabulary/policy fingerprints. README has command.

Dev-only confidence calibration: SHA-sampled256 pairs, each noisy+clean, margins
0/2/4/6. Selected margin4 using <=5% clean changed constraint and maximum noisy
exact matches. At margin0: noisy exact139/256 and clean changed159/256(62.1%).
Margin4: noisy exact109/256, clean changed11/256(4.3%), clean token edits0.22%.
Separate next256 dev rows with margin locked: noisy exact107/256(41.8%), clean
changed11/256(4.3%), clean token edits0.22%. Identity noisy exact0/256. Safeguards
trade repair recall for conservatism; these synthetic-heavy samples are not gold.

Across the1024 selected-margin outputs (512 calibration +512 verification),
independent output validation found zero copy failures, zero accepted guard
failures (including every accepted intermediate trace), and zero resolved
encoder-only output words. A generic-URI detector bug found during validation
was fixed before the final rerun. Policy and predictions under fixes-v1/
calibration and verification; output-validation.json contains counts.

Diagnostics: preserve 'Do not delete `config.json`.', conditional instructions,
C++, raw calls and double-backtick literals. 'Utilize the existing cache.' now
reaches 'Use the existing cache.' on attempt5 after unsafe proposals are rejected.
'Open file.' remains unchanged: the partial checker misses the absent article.
'Set `timeout_ms` to500' still may get an unwanted article. This is safer
inference, not a claim of production readiness or a repaired grammar model.

35 tests pass (22 tagger/loop/copy +13 tokenizer). Checker calibration strict
unchanged1723/1838 STE-clean,1498/1838 non-STE caught; software1744/1838 clean,
non-STE caught1323->1321 due explicit whitelist changes. No test-set tuning.

Future training must regenerate/version tokenizer exports and rollouts, rerun
quality/split audits, and strengthen clean KEEP examples/balance real rewrites.
Current migration preserves learned weights; it does not teach new emission rows
or solve broad rewriting. Owner-verified meaning gold remains outstanding.

## 2026-10-25: user authorized revised local training

`run_local_v2.py` is running as `asd-ste100-local-v2.service`. This is the complete
rebuild/audit/train/dev-compare pipeline, not a statement that optimizer training
has started: consult `data/out/local-v2-pipeline/status.json` for the current stage.
Stage logs persist there; main optimizer status will be `local-train-v2/status.json`.
No paid resources are used.

Full original grouped surface corpus:1,028,381 rows. New deterministic filter:
998,805 retained,29,576 excluded, with every decision and retained payload independently
validated. Fresh ID export:649,386 rows (train588,840/dev31,387/test29,159); target
vocabulary/copy/capacity rejections are explicit, not silently promoted words.
ID audit, full rollout replay, canonical overlap quarantine and fresh audits gate
training. The improved source/tokenizer fingerprints include protected_regions,
quality_gates and text_utils dependencies. Keep old data/runs/evaluation frozen.

Revised trainer supports vocabulary-checked weight initialization with a fresh
optimizer, training-only kind weights, and deterministic capped local replay.
Planned main run:3 epochs, BF16 CPU8threads, LR6e-5/warmup200, KEEP1/terminal1,
weights8/12/3/1 original/messy/rule/local_rule, cap200,000 local examples per epoch,
checkpoint500, quick dev1000, full dev every epoch,8-hour cap. All accepted real
examples participate each epoch. An actual optimizer/RNG exact-resume pilot runs
before the main training stage.71 tests pass (34 data+13 tokenizer+24 model). No new weights should be described as
improved until the dev comparison exists.

Proposed software review benchmark:32 authored/assistant-reviewed cases,16clean
and16repair, with literal requirements and review rationales. Owner review pending.
Input/reference canonical overlap against all rebuilt train/dev/test texts must
pass before training. It stays out of training and confidence tuning. Baseline
score:13/16 clean exact,8/16 repairs exact,2unresolved,0literal/guard failures.
One clean case uses encoder-only 'zero'; unchanged unresolved fallback is correct
safety behavior, not closed-output normalization.31/32 references are representable;
all preserve ordered content and literal counts. Do not modify references to fit
model predictions. See benchmarks/README.md and local-v2-pipeline reports.

The pipeline automatically calibrates the trained best checkpoint on512dev rows,
verifies on the next512, compares the baseline on those same rows, and scores both
on the proposed software review cases. Final output:local-train-v2/comparison.json.
No automatic publication or paid compute. Current stage and eventual result must
be checked before reporting completion.

### Local v2 launch verified (2026-10-25 07:59 UTC)

All preprocessing gates finished successfully. Final dataset is
`tagger-rollouts-v4`:648,627 accepted pairs; train588,161/dev31,342/test29,124;
1,297,295 packed examples.759 whole trajectories rejected. Full replay passes,
including exact slots and terminal targets. Canonical split audit passes on all
pairs plus48 intermediate sources, with zero cross-split groups or conflicting
source targets. No quarantine/v5 was needed. Proposed benchmark isolation passes
against all649,386 ID pairs, with zero input/reference overlaps.

Actual resume pilot passed:4+4 vs continuous8 updates produce bit-identical model,
optimizer and Torch RNG state. The main run started successfully; first100updates
completed with finite loss. Config totals:27,870,263 parameters,6967batches/epoch,
20,901updates across3epochs. Train state is `local-train-v2/status.json`; pipeline
state is `local-v2-pipeline/status.json`. At initial speed, approximately1.5–2hours
for training, then automatic dev calibration/verification/baseline and software
review comparison. This is an estimate under current local CPU contention.

Loss weighting changed from v1, so raw v1/v2 dev losses are not comparable quality
scores. Use the decoded, matched dev and software review comparisons after completion.
Frozen source snapshot:local-v2-pipeline/source-v2/; source hashes,71test results,
benchmark/reference/isolation reports and replay/resume diagnostics persist beside
stage logs. A checkpoint is saved every500updates; the service continues after
this chat turn. No production-readiness claim, no paid resources, no deployment.

## Local v3 balance comparison authorized (October25 continuation)

User said 'do it' after v2 reduced unwanted edits but lost exact repairs. Dibs skill
was read and the shared measurement reservation respected; no compute was started
through the measurement. Work now runs under `dibs run cpu` in user service
`asd-ste100-balance-v3.service`, preserving the CODEX_THREAD_ID/Dibs identity.
No paid services/keys used. Read balance-v3/status.json for the current stage.

Important correction to the initial v2 assessment: the coarse confidence grid also
suppressed repairs. On the original512 calibration rows, margin0 had243 exact and
26 clean changes(5.08%), so the old grid jumped to2 with160exact. Finer margin0.25
has228exact and19clean changes(3.71%). No new verification results yet. Baseline
is recalibrated on the same slice before any new checkpoint comparisons.

`calibrate_editor.py --margins` added; `train.py --retain-epochs` saves immutable
model-only checkpoints every epoch. A new local run uses KEEP0.3/terminal0.5,
otherwise the same v2 initialization/data/cap/kind weights/LR/seed/3epoch schedule.
It starts from the migrated v1 weights, fresh optimizer, not from conservativev2.
Existing passed dataset audit/hash gates remain active. Model/tokenizer37tests
rerun and pass. Old runs, policies, and test evaluation remain frozen.

`run_balance_v3.py` orchestrates tests, existing checkpoint calibration, training,
all-three-epoch calibration, checkpoint selection, fresh1024-row dev verification,
and fixed software review scoring. Winner and margins are chosen solely on the
original512 calibration rows under <=5% clean changes, before new verification.
Fresh verification rows1024:2048 exclude earlier calibration0:512 and earlier
verification512:1024 by row ID; group independence within dev is not claimed.
Outputs:balance-v3/selection.json,comparison.json,software-review.json; model files
local-train-v3/epoch-1.pt through3, best.pt and last.pt. Snapshot under balance-v3/source.
Do not claim training or improvement complete until those status/reports exist.

Local v3 launch verified11:33UTC October25: optimizer running, first300/20,901
updates finite. Dibs CPU hold belongs to this Codex thread and follows the service.
Baseline finer calibration completed:220/512 exact,25clean changes at margin4;
v2:228exact,19clean changes at margin0.25. These calibration results replace no
frozen verification figures. Main model/tokenizer37tests passed. See balance-v3/
launch-validation.json for launch evidence. Full final comparison remains pending.

## October 26: reviewed rewrite data and local v4 run

The v3 comparison completed. On fresh 1,024 dev rows, v2 had 463 exact repairs
and 58 changed clean controls; the best v3 checkpoint had 441 and 51. We retained
v2 as the initialization for a data-focused experiment. The bounded diffusion
prototype is historical and is not part of this run.

The revised ID corpus `data/out/id-quality-v4b/` has 623,001 pairs:
562,535 train, 31,346 dev, 29,120 test. We quarantined 261 entire groups
containing unreviewed operator-changing or otherwise suspect original/messy/rule
examples; a 48-row nonlocal spot audit recorded 32 approve, 9 review, 7 exclude.
This sample does not certify all remaining meanings. We added 1,800 training
pairs from 36 assistant-reviewed software rewrite frames, identified separately
as `reviewed` examples. Their references are representable in the frozen
vocabulary. `data_revision_v4b.py` independently replayed all retained and added
payloads; `revision-validation.json` passed. The old corpus and model remain.

`reviewed_rewrites.py` defines six explicit phrase repairs and the inference
guard allows only these ordered, literal-preserving substitutions in addition to
its existing rules. The 31 model tests pass. The proposed frozen
`benchmarks/software-review-v4.jsonl` contains 80 cases: 32 earlier software
cases, 24 phrase repairs and 24 clean controls. Its references are assistant
reviewed; owner review and real-input semantic validation remain outstanding.

The free local pipeline started at 02:58 CDT as user service
`asd-ste100-quality-v4.service` under `dibs run cpu`. Check
`data/out/quality-v4-pipeline/status.json` for its actual stage. It runs full
rollout replay, canonical split audit and benchmark isolation before two CPU
training epochs, then selects a confidence policy on 512 dev rows and compares
v2 versus v4 on the next 1,024 dev rows and the frozen 80 cases. Training status
will appear in `data/out/local-train-v4/status.json`; final comparison in
`data/out/quality-v4-pipeline/comparison.json`. Do not claim an improved model or
promote it until those files and audit gates pass. No paid resources were used.

Launch gate results at 03:16 CDT: rollout export accepted 622,320 pairs and
rejected 681 whole trajectories; all 1,800 reviewed examples were accepted.
Independent replay verified exact intermediate slots, terminal targets and parent
splits/metadata. Canonical split audit passed. Benchmark isolation checked all
623,001 parent ID pairs against 80 cases and found zero overlaps. Optimizer
training started, with 100/4,719 updates and finite loss. The model has
27,870,263 parameters. These are launch facts, not final quality results.

The v4 run completed all 4,719 updates and both full dev passes in 32 minutes;
the service exited successfully. Its best full-dev loss was 0.1380 at epoch two,
but decoded calibration selected epoch one. The original coarse margin grid
selected margin2: on fresh dev rows1024:2048, v4 had445/1024 exact references
and42 changed clean controls versus v2 margin0.25 with502 and58. The 80-case
software review set instead improved: v4 had40/40 clean exact and13/31 required
repairs exact, versus v2 29/40 and9/31. Both had three duplicate-article source
cases returned unresolved; these were not accepted guard violations.

Because margins1 and2 straddled the calibration clean-change cap, a narrower
grid was recorded before inspecting fresh v4 verification outputs. On the same
512 calibration rows it selected epoch-one margin1.25:239 exact/23 changed clean,
near v2's242/19. Locked fresh verification then gave v4 493 exact/66 clean
changes versus v2 502/58. The software set gave v4 40/40 clean exact and14/31
required repairs exact. Files: `data/out/quality-v4-pipeline/{comparison,
finer-policy-plan,finer-comparison,nonlocal-dev-comparison}.json` and checkpoint
`data/out/local-train-v4/epoch-1.pt`. The narrow selection used calibration only;
it was not adjusted to the fresh result.

All nonlocal v4b dev ID rows were also diagnosed with locked policies:554 rules,
89 original teacher rewrites and34 messy pairs. Exact reference counts for v2
were53/554,0/89,0/34; v4 margin1.25 gave50/554,0/89,0/34. Exact mismatch
can be a valid alternative or a poor reference, so this is not a semantic error
rate. It does show that the software phrase examples did not solve open ended
rewrites. General v4b dev verification contains only three original and one
messy pair among1,024 sampled rows, so it mostly measures local rules.

Decision: retain v2 as the general checkpoint. Keep v4 as a research artifact
for software-style restraint and the six reviewed phrase families. No production
promotion, test-set tuning or paid service was used. The next quality gate is an
owner-verified meaning set with genuinely diverse rewrites and clean controls;
synthetic exact matching cannot establish safe STE normalization.

## October 28: documentation teacher pilot, first 12 passages

The October 26 queue (`data/doc_pilot/queue.py`, unit
`asd-ste100-doc-teacher-v1`) never started synthesis. It waited for the DFlash2
controlled timing run, which stopped as `interrupted_resumable` instead of
`complete`, and timed out after 12 hours. On October 28 the owner asked to
continue, and the first 12 passages ran directly, without the stale wait:
`dibs run cpu port-8765 -- run_local.py --limit 12`, unit
`asd-ste100-doc-teacher-first12`. Source validation, the 5 pilot tests and both
pinned SHA-256 checks passed. The run took 4.5 minutes on shared CPU, about 23 s
per request. No paid resources were used.

The deterministic gates accepted 2 of 12. The assistant semantic review is in
`data/out/doc-teacher-pilot-v1/first-12-assistant-review.json`; owner verdicts
are still pending in `first-12-review.md`. Both accepted targets change
meaning: one turns a description into an instruction ("Set the maximum ...
with --parallel-max"), and the other drops the scope word "requested". Of the
10 rejections, 2 change meaning, 3 keep meaning but are not STE, 2 return the
source unchanged, and 3 had an attempt that kept meaning and failed only on a
gate conflict or one word. Nothing from this batch is training-ready.

Findings that need an owner decision before the remaining 24 run:

- The modal and negation gates require identical token counts, but the checker
  rejects `may`, `should`, `would` and `never`. 8 of the 36 frozen inputs (5 of
  these 12) contain `may` or `should`, so no target can pass. The count gates
  also block valid rewrites: "it is necessary to" -> "must", "it is not
  possible" -> "cannot". How to keep the strength of `should` and "may wish to"
  in STE is a policy question.
- The teacher prompt has no STE vocabulary. Every rejection has R1.1
  unapproved words; `may` appears 10 times.
- Retry feedback gives rule codes only, not the flagged words or the checker's
  suggestions. None of the 10 retries produced an acceptance; 5 were worse
  than attempt 1.
- The checker flags `up to` inside "up to date". Changing the checker requires
  recalibration (rules above).

Running the other 24 with the same prompt and gates would mostly reproduce
these failures. The pilot code, prompt and gates were not changed. The Rust
holdout was not sent to the teacher.

## October 28 (later): documentation teacher pilot v2

The owner asked to give the teacher much more STE context. v2 reruns the same 12
passages with an STE system prompt of about 27,300 tokens: the task and the
modal policy, 9 worked examples, all rule summaries, the approved, software and
unapproved word lists, per-passage checker hints, retry feedback that names the
words, and meaning-class gates. Details are in `data/doc_pilot/README.md`
(Version 2). The system prompt is evaluated once and restored from a
llama-server slot before each request. Prefill took 24.5 minutes on shared CPU;
the 12 passages then took 15 minutes. 18 pilot tests pass. v1 code and
outputs are unchanged. No money spent.

| Same 12 inputs | v1 | v2 |
|---|---|---|
| Accepted by gates | 2 | 7 |
| Usable after assistant review | 0 | 5 |
| Accepted but meaning changed | 2 | 2 |

The two accepted v2 failures drop qualifiers that no gate sees ("requested"
again; "convenient and reliable"). Review is still essential. The rejected
passages show three faults to fix next:

- The teacher pastes sense-blind dictionary alternatives ("matching" -> "aligns
  with the index", "lost" -> "is not find").
- Negation feedback does not say what was added or removed, so 2 passages
  repeated the same target three times.
- "nothing else" -> "only" is a valid rewrite that the gates reject.

A review-only check for dropped source content words would flag both accepted
failures. The `should` policy ("we recommend that you" for advice, "must" for a
requirement) is the assistant's default; it needs owner review. Verdicts are in
`data/out/doc-teacher-pilot-v2/first-12-assistant-review.json`, and the owner
packet is `first-12-review.md`. Run the remaining 24 only after owner review.

## October 29: documentation teacher pilot v3 (all 36)

The v3 fixes for the v2 review are documented in `data/doc_pilot/README.md`
("Version 3"). The run covered all 36 teacher inputs. One run gives two
acceptance rules: gates only, and gates plus a same-model meaning check.

- First 12, meaning kept by assistant review: v1 0, v2 5, v3 gates only 7 of 10
  accepted, v3 with meaning check 6 of 6 accepted.
- All 36: gates only accepted 30, of which 21 keep the meaning. With the meaning
  check, 13 were accepted and all 13 keep it, but 9 good rewrites were lost.
  The check flagged all 9 meaning changes and 11 of 21 good rewrites.
- Decisions pending for the owner: the `should` policy, including the
  permission sense; whether the meaning check is a gate or a review flag; owner
  review of `data/out/doc-teacher-pilot-v3/first-36-review.md`. No pair is
  training-ready.
- All of `data/doc_pilot/` is still uncommitted.

## October 30: documentation teacher v5, production, and the model contract

Details and every number: `data/doc_pilot/README.md`, sections "Version 4
results" and "Version 5". In short:
- v5 runs the v4 method in batches of 48 through OpenRouter (owner-funded keys in
  `~/.config/openrouter/`, never in the repo): Qwen3.8-27B rewrites (Nemotron 3
  Ultra on the free tier for 1,795 passages), DeepSeek V4.1 Flash checks the
  meaning, gpt-oss-120b checks it a second time as a review flag.
- All 21,380 passages of `data/out/doc-pool-v5/pool.jsonl` (Python docs,
  PostgreSQL docs, curl docs; no Rust Book) are processed. The candidate set
  `data/out/doc-teacher-v5/final/accepted.jsonl` has 12,763 pairs; 556 of them
  were repaired after a second-check flag, and 1,403 stay flagged. The owner
  chose to leave the flagged ones out of training. Audits of 40 random pairs per
  slice: 40, 40 and 39 keep the meaning. Spend about $12 in all (both keys).
- The student model's vocabulary is `data/out/fixes-v1/vocabulary` (the v2/v4
  checkpoints), not `tokenizer-joint-v1`. `data/doc_pilot/model_gate_v5.py`
  checks a pair against it: 5,201 of the 11,360 unflagged pairs encode. The rest
  need tokenizer changes (contractions, "been", "being" and -ing forms are copied
  as names; a copied word may not repeat), owner-approved `tech_words.txt` lines
  for technical terms that the dictionary rejects in another sense (candidates
  with counts: `data/out/doc-teacher-v5/kept-source-words.tsv`), a repair pass,
  or a decision on dropping words that are not in the dictionary. Table in the
  README, "Fitting the student model".
- Changed today, tests pass (14 tokenizer, 34 data, tagger suite, 21 + 10 pilot):
  the tokenizer reads a sentence-initial "A" before a lowercase word as the
  article; the checker flags an -s form of a rejected verb after "this"
  (calibration: STE side unchanged; non-STE caught +1 in both settings).
- Run the tokenizer, gate and tagger code with `data/out/.venv/bin/python` (the
  pinned tokenizers, torch and numpy).
- Later the same day, with the owner's approval: vocabulary `data/out/fixes-v2/vocabulary`
  (37 output words from new `tech_words.txt` lines; 339 input-only grammar words),
  a model-gate repair pass (1,935 of 4,000 fixed, $1.92), 7,794 training-ready
  documentation pairs, and local v5 training (`model_tagger/run_doc_v5.py`). The
  owner chose to leave out pairs that drop a word the dictionary does not list.
  A copied word may still not repeat (not changed; the repair asks for the count).
- Result: v5 does not learn documentation rewriting. Documentation dev: 0 of 679
  resolved by the guarded editor, 2 exact by raw decoding, as the migrated v2;
  dev edit-tag accuracy on these pairs 3.4%. The model deletes the word to replace
  and does not write its replacement. Software review 39/40 clean and 13/31
  repairs (v2 29/40, 9/31); fresh dev 477 exact / 60 clean changes (v2 494 / 68).
  Keep v2 as the general checkpoint. Table: `model_tagger/README.md`, "local v5".
- Documentation fine-tune v5b (`model_tagger/run_doc_v5b.py`, owner-approved,
  free): doc rows batched six times per epoch (`train.py --kind-repeats`), LR 1e-4,
  three epochs from v5 epoch 2. Doc dev REPLACE positions 1.8% exact; it does not
  fit its own training pairs; fresh dev exact drops 477 -> 391. Probes show the
  trained encoder has lost plasticity: v5 cannot memorize 256 doc states, while a
  randomly initialized tagger memorizes a batch in 50 steps. A fresh tagger trained
  on the doc rows only reaches 33.5% REPLACE exact on doc dev and 88 of 679 exact
  raw rewrites (v2/v5: 2). Details: `model_tagger/README.md`, "v5b"; probe scripts,
  outputs and that checkpoint: `data/out/doc-v5b-pipeline/probes/`.
- October 31, tagger rebuild from random initialization (owner-approved, free).
  v6 (`model_tagger/run_rebuild_v6.py`) used KEEP 0.4 from the start and never
  learned to edit; 1,000-step probes traced it to the KEEP weight. v6b
  (`model_tagger/run_rebuild_v6b.py`) follows the v1 -> v2 recipe: three epochs at
  KEEP 0.1, then a one-epoch polish at KEEP 0.4, doc rows four times per epoch.
  Selected: `data/out/local-train-v6b-polish/epoch-1.pt`. Fresh dev 473 exact / 55
  clean changes (v5: 477 / 60); software review 36/40 clean, 24/31 repairs (v5:
  39/40, 13/31). Documentation: 17 of 679 exact raw rewrites (v5: 2, doc-only
  fresh tagger: 88), 1 exact through the guarded editor; the raw rewrites still
  damage meaning. The 4 clean software cases it changes all get "the" before a
  code literal, which 10.6% of the local-rule training pairs teach (`drop_article`
  noise before a literal). Details: `model_tagger/README.md`, "v6 and v6b".
- Nothing from `data/doc_pilot/` or the tagger/tokenizer changes since is committed.
  Open owner decisions: the `should` policy; whether v6b polish replaces v2 as the
  general checkpoint; whether `drop_article` noise should skip code literals; the
  next step for documentation rewriting (more documentation data, or the canvas
  model).
- October 31, canvas editor smoke test (`model_canvas/`, owner-approved, free):
  8M parameters, trained on the doc pairs only. 56 of 679 doc dev exact (doc-only
  tagger: 88), 13 clean references changed (tagger 23), 47 failed slot copies
  (tagger 13). It overfits after epoch 15 and writes repeated words, but it does
  split sentences. Details and next steps: `model_canvas/README.md`.
- Canvas smoke v2 (free): slot fitting removes all failed outputs; early stopping on
  held-out train groups gives 58 of 679 exact (tagger 88), 29 clean changes. The
  canvas editor does not beat the tagger on the doc pairs alone. Next data-side
  test: pretrain it on the local rule pairs. `model_canvas/README.md`, "Smoke v2".
- Canvas pretraining test (free, `data/out/canvas-pretrain-v1/`): rule-pair
  pretraining (article-before-literal pairs skipped) then the doc pairs, selected by
  held-out decode: 62 of 679 exact, 10 clean changes; the doc-only control with the
  same selection also 62 exact, 32 clean changes; tagger 88. Pretraining helps clean
  inputs and convergence, not exact rewrites. Canvas editor shelved.
  `model_canvas/README.md`, "Pretraining test".

## November 6: size target moved; pretrained small LM next

The owner agreed that neither the tagger nor the canvas editor rewrites
documentation well enough (best: 88 of 679 doc dev exact, from the doc-only
tagger). Decision: **suspend the 100 MB / 500 ms targets.** Get a model that
rewrites well first, then shrink it. The closed STE output vocabulary, protected
spans and GGUF format still apply.

Next model: fine-tune **LiquidAI/LFM3-1.7B** (released 2026-10-27; LFM Open
License 1.0, free under $10M annual revenue, which the owner confirmed applies)
on the 7,794 documentation pairs, with decoding restricted to STE words plus
copied source spans. Run **LiquidAI/LFM2.5-350M** beside it to see how quality
holds at a smaller size. Score on the 679 doc dev pairs of `data/out/id-doc-v5/`
against the 88 exact of the doc-only tagger, plus an assistant meaning review.

Facts checked 2026-11-06:
- LFM3 is `model_type: lfm3` with remote code (`configuration_lfm3.py`,
  `lfm3_conv.py`); transformers 5.14.1 in `target/manifold-rocm/env` has no
  native `lfm3`, so load with `trust_remote_code`. LFM2.5 is native (`lfm2`).
- The local llama.cpp builds under `/var/cache/lemonade/bin/llamacpp/`
  (b11393 Vulkan, b11580 ROCm) fail with `unknown model architecture: 'lfm3'`.
  Upstream has it (`LLM_ARCH_LFM3`); release b12177 is current. A newer build
  is needed for GGUF inference.
- No zero-shot or training run has been done yet.

## November 7: LFM few-shot baseline on the doc dev pairs (running)

- Committed the Oct 24 – Nov 6 work (8089510) after the tests passed: tokenizer 15, data 34,
  tagger 32, canvas 8, doc pilot 68 (unittest, `data/out/.venv/bin/python`).
- `model_lm/doc_pairs.py` exports the 7,794 training-ready doc pairs as surface text with the
  tagger's split: `data/out/lm-doc-v1/` train 6,897 / dev 679 / test 218. The dev ids equal
  `id-doc-v5` dev.
- Models downloaded to `data/out/lfm/` (LFM3-1.7B 3.3 GB, LFM2.5-350M 0.7 GB; both are the
  post-trained chat models, no separate -Instruct repo). LFM3's remote code needs a newer
  transformers than `target/manifold-rocm/env` has (5.14.1 lacks
  `use_kernel_func_from_hub_with_fallback`); `data/out/lm-overlay/` holds transformers 5.22.2 +
  tokenizers 0.23 (`PYTHONPATH=data/out/lm-overlay`). 5.22.2 still has no native `lfm3`.
- LFM3 with `enable_thinking=False` writes its answer, then `</think>`, then the answer again.
  Suppressing `</think>` made it think aloud after the answer instead (a first run, stopped at 512
  of 1,358: 55 of 247 sources hit the token cap). `generate.py` now stops at `</think>` or at the
  first line break (no doc source or reference has one) and keeps the text before it.
- `model_lm/ste_constraint.py`: character-level constraint on the top-k candidates. Every
  finished word must be an STE output word (fixes-v2 vocabulary, any case) or a source word that
  the checker accepts on its own (code, names, numbers, unlisted words; a rejected word such as
  "additional" cannot be copied). Punctuation and spacing are free. 4 unit tests.
- `model_lm/generate.py`: 4-shot (fixed train pairs), system prompt in plain STE, greedy,
  fp32 on the GPU. Rewrites each dev source and each reference (to count clean references
  changed). `model_lm/score.py`: the tagger probe's numbers (exact, unchanged, closer, farther,
  clean changed) plus checker-clean outputs, protected values kept, mean word similarity.
- `model_lm/run_zero_shot.sh` (detached, `dibs gpu`): LFM3 and LFM2.5, constrained and free,
  then a 24-row CPU canary against the GPU outputs (silent GPU faults since Nov 5), then
  scores. Output and `ALL.done` in `data/out/lm-zeroshot-v1/`. Bar: 88 of 679 exact.

Few-shot results (679 doc dev pairs; bar: doc-only tagger 88 exact / 23 clean references changed).
CPU canary: 0 of 24 rows differ from the GPU run. `data/out/lm-zeroshot-v1/scores.json`.

| 4-shot | exact | checker-clean | clean refs changed | protected kept | unchanged |
|---|---|---|---|---|---|
| LFM3-1.7B constrained | 32 | 432 | 400 | 595 | 34 |
| LFM3-1.7B free | 10 | 81 | 462 | 626 | 162 |
| LFM2.5-350M constrained | 3 | 542 | 642 | 533 | 5 |
| LFM2.5-350M free | 1 | 247 | 649 | 559 | 17 |

- The constraint makes LFM3 replace words (exact 10 -> 32, checker-clean 81 -> 432); free, it copies
  162 sources unchanged. Few-shot rewrites mostly keep the meaning in a 6-row look but reword
  freely, rarely split sentences, and change most clean references.
- Gap: the constraint uses the student output vocabulary, which holds words the checker rejects
  (may, any, either, should, every, would), and it cannot see rejected phrases (such as, according
  to, have to). R1.1 is 218 of the constrained LFM3 run's checker errors, R6.3 (sentence length) 62.
  Left unchanged so the fine-tune runs compare with these; a stricter constraint is a later rescore.
- `model_lm/finetune.py` + `run_finetune.sh` (queued, waits for the GPU): full fine-tune, 3 epochs,
  LR 3e-5 (350M) / 1e-5 (1.7B), epoch picked on 4% held-out train documents, zero-shot prompt,
  constrained and free dev rewrites, CPU canary, `data/out/lm-ft-v1/scores.json`, `ALL.done`.
