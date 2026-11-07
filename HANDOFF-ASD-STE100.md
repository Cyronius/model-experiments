# ASD-STE100: continue on another computer

Updated October 24, 2026. Start here, then read [the experiment handoff](asd-ste100/HANDOFF.md), [the design](asd-ste100/PLAN.md), and [the tokenizer notes](asd-ste100/tokenizer/README.md).

## Where we stopped

We are building a tiny non-autoregressive English-to-Simplified-Technical-English editor. The lexicon converter, deterministic checker, data preparation, tokenizer/slotter and offline iterative tagger are implemented. **The first local CPU baseline finished three epochs successfully in 5 hours 43 minutes. Frozen-checkpoint test evaluation is complete and the baseline is not ready for use; there is no evaluated release, GGUF writer, llama.cpp patch or NPU engine.** Joint surface/ID exports and a random-weight tagger forward/loss/backward/decode smoke are complete. No new money was spent.

The baseline is an iterative tagger: six bidirectional transformer layers, width 512, eight heads and FFN 2048. It comes before the canvas editor. Output is a closed STE/software vocabulary plus copy slots. Code, identifiers, numbers and detected names/units must be restored unchanged. The final model format is GGUF. Target size/latency are under 100 MB and under 500 ms per sentence on CPU; NPU work waits for measurements. Keep the existing decisions in the experiment handoff.

## Completed data work

| Artifact under `asd-ste100/data/out/` | State |
|---|---|
| `full1/ste-out/` | Saved teacher/judge run; 8,587 score-5 targets before final pair filtering. Original pod cost was $1.71. Do not rerun paid generation to recover local files. |
| `pairs-v2/` | Corrected surface corpus: 28,381 pairs, fully audited. `pairs-v1` is historical and was the exclusion set for local augmentation. |
| `local-million-v1/` | One million unique checker-only rule augmentations from 403,989 target texts; full surface audit passed. Article restoration dominates. |
| `tokenizer-v3/` | Final tokenizer: 2,568 output IDs, 9,458 input IDs, including a 4,096-token byte fallback fitted on teacher-corpus training sources only. |
| `id-pairs-v3/` | 13,414 retained teacher-corpus pairs: 12,067 train / 636 dev / 711 test; full ID audit passed. Only 1,876 are original teacher rewrites; the rest are augmentations. |
| `id-local-million-v3/` | Historical separate local export: 610,642 retained pairs; full ID audit passed. |
| `joint-surface-v1/` | All 1,028,381 surface pairs grouped jointly and audited: 931,571 train / 49,788 dev / 47,022 test. |
| `tokenizer-joint-v1/` | Joint-train-only byte fallback; exact same output vocabulary and input/output sizes as v3. |
| `id-joint-v1/` | Historical parent model corpus: 624,072 retained pairs; 565,599 train / 29,977 dev / 28,496 test. Full ID audit passed. |
| `id-joint-clean-v2/` | Current deterministic-filtered corpus: 607,040 pairs; 550,129 train / 29,263 dev / 27,648 test. Parent rows/splits preserved. |
| `tagger-rollouts-v3/` | Current audited tag-training data: 599,855 pairs; 543,592 train / 28,954 dev / 27,309 test. Includes terminal KEEP states; both audits passed. |
| `local-train-v1/` | Active three-epoch CPU run; checkpoint/status/config/metrics here, service `asd-ste100-local-v1.service`. |
| `tagger-smoke-v1.json` | Default 27,859,497-parameter random-weight forward/loss/backward and action-decode diagnostic passed. |

The strict tokenizer rejects many checker-clean targets: the checker permits unknown technical nouns and some rejected words in noun-like contexts, while the output vocabulary is closed. Do not silently approve those words to recover pair counts. Original surface datasets remain unchanged. Local augmentation labels remain `target_review: checker_only` / `meaning_review: unjudged_rule`; they are not meaning-verified gold.

Training uses 70% slotted / 30% raw input exposure for detected spans. Raw spans use byte BPE and carry `source_slot_refs`; the model must consume slot-reference and case features. Targets always use copy slots. Inference slots every detected span. Mask inactive slots and control tokens, and separately check exact slot counts before returning decoded output. Detection is heuristic. Technical-verb slots are reserved; no glossary implementation exists.

Thirteen tokenizer tests and nineteen data tests passed. Both retained ID corpora passed full reconstruction audits. Repeated tokenizer fits were byte-identical; one-worker and four-worker teacher exports produced identical compressed shards. The checker was not changed in this continuation.

## What Git does and does not transfer

The `main` commit contains this root handoff, experiment documentation, data builders, pinned public source catalog, tokenizer, and tests. Other experiments' uncommitted changes are outside this handoff.

**Git does not contain the data or fitted tokenizer.** Everything under `asd-ste100/data/out/` and the dictionary-derived files under `asd-ste100/lexicon/` are ignored. The dictionary packages are AGPL; the existing project decision is to keep derived artifacts out of Git. A fresh clone alone supports code work and synthetic tokenizer tests, but it cannot reproduce the exact teacher corpus without the saved teacher output.

A local transfer archive is prepared at:

```text
asd-ste100/data/out/asd-ste100-transfer-20261023.tar
asd-ste100/data/out/asd-ste100-transfer-20261023.manifest.json
asd-ste100/data/out/asd-ste100-transfer-20261023.tar.sha256
```

Copy those files separately using your own drive or file transfer. They are not uploaded with the Git commit. The archive is 1,207,521,280 bytes (about 1.21 GB), contains 120 files, and has SHA-256 `4ff323b7acacfd46961395573fd66a7c6d3600b98a4e77b25900de0d2b46b331`. Every archived file was read back and checked against its source hash.

The archive uses repository-relative paths and contains the exact fitted vocabulary, audited surface/ID exports, saved teacher outputs, small provenance/audit files, and exact derived lexicon. The companion manifest records SHA-256 for every included file. It excludes the 2.62 GB raw source downloads, the 2 GB staging SQLite database, exploratory tokenizer/ID versions and the interrupted `id-local-million-v2` export. It supports continuing model/data-split work, not resuming the old staging database.

## Set up the other computer

```powershell
git clone https://github.com/Cyronius/model-experiments.git
cd model-experiments
git switch main
git pull --ff-only
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS instead: source .venv/bin/activate
python -m pip install -r asd-ste100/tokenizer/requirements.txt pyarrow
```

Use Python 3.10 or newer. No paid account or GPU is needed for this setup. Check the transferred archive's SHA-256 against the `.sha256` file, then extract it **from the repository root**:

```powershell
Get-FileHash /path/to/asd-ste100-transfer-20261023.tar -Algorithm SHA256
tar -xf /path/to/asd-ste100-transfer-20261023.tar
python -m unittest discover -s asd-ste100/data/tests -v
python -m unittest discover -s asd-ste100/tokenizer/tests -v
python asd-ste100/tokenizer/prepare.py validate --vocab asd-ste100/data/out/tokenizer-v3 --out asd-ste100/data/out/id-pairs-v3
python asd-ste100/tokenizer/prepare.py validate --vocab asd-ste100/data/out/tokenizer-v3 --out asd-ste100/data/out/id-local-million-v3
```

On Linux/macOS, use `sha256sum` in place of `Get-FileHash`. Do not use Python `-O` for audits: they depend on assertions. Read compressed shards from `summary.json`, not a directory glob. The original local-surface manifest uses Windows separators; normalize `\\` to `/` when reading it on another OS, as `tokenizer/prepare.py` already does.

If the archive is unavailable, the exact AGPL-derived lexicon can be rebuilt with `python asd-ste100/lexicon/build_lexicon.py --fetch`; compare it with the hashes recorded in existing artifact manifests before assuming identical results. The public raw datasets can be downloaded for free with `harvest_sources.py`; see [the local corpus notes](asd-ste100/data/LOCAL_CORPUS.md). Teacher outputs cannot be recreated by those downloads. A fresh local-million build may select a different valid subset from the interrupted-and-curated historical run. Do not present that as a byte-identical rebuild.

## Do next

1. Read `data/out/evaluation-v2/assessment.json` and its examples. The baseline is not ready for use. Review the software-term/POS copy policy explicitly: the current output vocabulary forbids `file`, `delete`, `existing` and `otherwise`, causing unsafe substitutions/deletions. Do not silently expand the vocabulary or reuse old token IDs after changing its policy.
2. Apply literal-fidelity guards at inference; calibrate edit confidence on dev only and investigate the KEEP/edit weighting before another run. The test checkpoint was frozen; no thresholds, weights or runtime were tuned against test results. The completed run must not be restarted as though still active.
3. Address broader rewrite data/meaning coverage and obtain owner-verified gold and real-input evaluation. No deployment-quality claim or paid run is authorized.

Current authorization is free downloads and local processing. Ask before spending money, launching/stopping a pod, or changing established design decisions. Never commit derived dictionary/vocabulary artifacts or weights. If the checker changes, rerun calibration and report strict and software-word-list results.

Suggested next-agent prompt:

> Read this handoff and model_tagger/README.md. The full tag-trajectory replay and canonical-text split audits passed for tagger-rollouts-v3, and local-train-v1 is running as user service asd-ste100-local-v1.service. Check status/logs and checkpoints; do not launch another copy. No paid resources are authorized. Evaluate the trained baseline honestly against generated dev/test and keep owner-verified meaning evaluation separate.

## Local continuation, October 24, 2026

Restored the original archive from the owner's FTP server. Archive SHA-256 and all 120 file hashes match the recorded manifest. The restored archive is still the historical October 23 transfer; it does not contain today's new joint artifacts or model code. Do not overwrite it or assume it transfers today's work.

Joint groups use normalized source/target text, corpus-scoped origins and recorded document IDs before filtering, including removed duplicates/conflicts. The build/audit found 166,913 components, zero duplicates and zero conflicting targets. The largest component (37,206 records) stays in train. Metadata, attribution and original review labels are retained. See [the joint workflow and results](asd-ste100/data/JOINT_SPLITS.md).

The current ID corpus has 610,659 local rules, 10,855 teacher rules, 1,876 original teacher rewrites and 682 messy back-translations. Only 1,688 original teacher rewrites are in train; dev/test contain 103/85. The byte fallback was refitted on joint train. Historical `tokenizer-v3` must not be used to claim an uncontaminated joint holdout.

`model_tagger/tagger.py` implements the specified default encoder, slot/case/checker features, pooled raw copy spans, closed action masks, twenty transforms, a case head, provisional per-pass labels and a five-pass guarded decode loop. The full-size smoke used four training examples covering all four kinds, computed finite loss/gradients and exactly decoded supervised actions. A random prediction was rejected for slot fidelity, as intended. There are no optimizer steps or saved weights. Details and limitations: [tagger README](asd-ste100/model_tagger/README.md).

Twenty-six data tests, thirteen tokenizer tests and six tagger tests pass. Environment: Python 3.14, tokenizers 0.22.2, pyarrow 26.0.0 and CPU PyTorch 2.14.1 in ignored `data/out/.venv/`. Run scripts with that environment's Python. The checker was not changed. New corpus manifests/audits, logs and smoke metrics remain under ignored `data/out/`; no paid compute was used.

## Quality spot check, October 24, 2026

At the owner's request, reviewed 150 distinct retained pairs: 60 uniform and 90 deliberately oversampled diagnostic examples. Uniform judgments: 54 with no obvious issue, 3 needing review/context, 3 recommended for exclusion/repair. Diagnostic judgments: 59 / 18 / 13; these are not population rates. Reports and per-pair notes are in ignored `data/out/spot-check-20261024-v1/`; see [spot-check findings](asd-ste100/data/JOINT_SPLITS.md#assistant-spot-check-october-24-2026).

Reproduced noiser/extractor blind spots: lowercase math variable treated as an article, quoted text edited inside unfenced code, a list marker treated as a sentence boundary, inline HTML removed inside backticks. Some teacher/messy examples change question intent/referents/modality or retain damaged grammar. No dataset rows, checker, tokenizer or review labels were changed. Next local data work should fix these gates and measure affected records before training, alongside multi-pass tag coverage and owner-verified evaluation.

## Deterministic cleanup, October 24, 2026

Completed the owner-requested deterministic cleanup. Use `data/out/id-joint-clean-v2` with `tokenizer-joint-v1` for further model-data work. It keeps 607,040 of 624,072 parent ID pairs and removes 17,032 (2.73%). The 3,035 retained review-flagged rows are not silently excluded. All original teacher rewrites remain; this pass does not resolve their semantic errors from the spot check.

Hard gates cover recognized code/math/markup changes, synthetic list-marker merging, empty tag references, explicit truncation, obvious unfinished copula clauses and target contractions outside recognized code/quoted text. Review-only gates cover generic empty literals, ellipses, dangling headings and potentially complete relative clauses. Per-rule samples were inspected and overly broad patterns narrowed. `id-joint-clean-v1` is preliminary: v2 restores five legitimate passages using curly quotation marks. Use only the final v2 quality report, not exploratory scan totals.

Future extraction preserves code while stripping HTML and normalizing prose; noising rejects unsafe literal changes/list markers; teacher/local builders reject the hard target defects. The checker/output vocabulary were not changed. The existing tokenizer fit still uses parent joint train, including subsequently excluded training sources, with no holdout leakage introduced. No regrouping occurred after filtering.

Reports/rejections and code/input hashes are in the filtered directory. Its `quality-validation.json` checks every parent decision and exact retained payloads; `validation.json` checks the full ID/slot/case/alignment reconstruction. See [cleanup details and commands](asd-ste100/data/JOINT_SPLITS.md#deterministic-cleanup-october-24-2026). The original parent and all source artifacts remain intact. New current-subset model diagnostic: `data/out/tagger-smoke-clean-v2.json`; no optimizer steps or weights saved. Tests: 34 data, 13 tokenizer, six tagger. No paid compute.

Next: inspect review-only flags and address meaning/grammar problems outside these gates, audit full multi-pass tag-label coverage on the filtered corpus, and prepare owner-verified evaluation before authorized training. Do not treat “clean” as meaning-verified gold.

## Rollout audit and local training, October 24, 2026

The owner requested completion of the training audit followed by local training. The default 27,859,497-parameter model is now training on CPU using `model_tagger/train.py`. Data: `tagger-rollouts-v3`, 599,855 accepted pairs / 1,199,751 examples including terminal KEEP states. Splits: 543,592 / 28,954 / 27,309 pairs, with 1,087,219 / 57,911 / 54,621 examples. The full replay verifies every source pair, action, input/case/slot/checker/transform feature, intermediate slot count and terminal target. The canonical audit covers punctuation-spacing-normalized text and all 48 intermediate sources; both audits passed.

Of 607,040 deterministic-clean parent pairs, 7,167 fail rollout/literal safety, early checker stopping or representation gates. A further 18 pairs from two entire groups are quarantined because canonical text overlapped train/dev. No remaining split/group was reassigned. `tagger-rollouts-v1` is pre-quarantine; `v2` is superseded by final `v3`. Original parent corpora remain intact. Accepted kinds: 587,067 local rules, 10,802 teacher rules, 1,407 original rewrites and 579 messy pairs. Original rewrites lose 361 pairs to early checker stopping, 87 to intermediate slot fidelity, 18 to detokenization/span changes and three to literal changes. These are representation/exit-policy exclusions, not new semantic judgments.

Fixed a decoder bug: it previously returned checker-clean input before any model edit, skipping article repairs. It now runs at least one pass, then applies the established checker/KEEP/repetition/five-pass stopping rules. All states retain 70% slotted / 30% raw training exposure; slotted inference must give identical editing units. KEEP-token loss weight is 0.1 and terminal-example weight is 0.25. The architecture, vocabulary and checker are unchanged. These are initial baseline optimization settings, not validated quality choices.

The pilot completed forward/backward/optimizer/evaluation/best-checkpoint saving. A two-step checkpoint resumed to four steps produced bit-identical model weights to a continuous four-step run. Twelve model/rollout unit tests pass, including reference/packed-collator equivalence and raw/slotted-copy checks. `local-train-v1` runs as user systemd service `asd-ste100-local-v1.service` with eight CPU threads, two data workers, BF16, batch 32, LR 3e-4, 500 warmup steps, three epochs (101,937 updates), checkpoint every 500 updates and a 24-hour cap. It is detached from this chat. Status: `data/out/local-train-v1/status.json`; log: `data/out/local-train-v1.log`; exact args/hashes: `config.json`; resumable state: `last.pt`; best epoch-dev weights: `best.pt` when available.

All artifacts and weights remain ignored; no paid APIs or pods were used. Initial generated dev loss is 10.0652. Training remains in progress; do not present pilot or early loss as final compliance, meaning quality, latency or GGUF-size results. The existing owner-verified evaluation and semantic-data gaps remain.

## Training completed; evaluation underway

`local-train-v1` completed all 101,937 updates / three epochs with service exit status 0 in 20,561 seconds (5h43m). Final/best full-dev weighted loss: 0.27277. `best.pt` is the epoch-3 weights (about 107 MiB); `last.pt` includes optimizer/RNG state (about 319 MiB). The service is inactive after successful completion; do not describe it as still running.

At the owner's request, `model_tagger/evaluate_model.py` is evaluating the frozen best checkpoint on all 27,648 deterministic-clean test pairs, including the 339 excluded from training-rollout preparation, plus all clean target controls. It verifies no canonical-text overlap with actual training sources/targets/intermediate states before scoring; no overlapping test group was found. Its batch decoder agreed with the reference loop on 24/24 seeded checks. Current artifacts/log: `data/out/evaluation-v1/` and `evaluation-v1.log`. This is an evaluation, not a retrain or decoder-tuning run. The checkpoint/runtime and test references remain unchanged.

## Frozen model evaluation completed

Current assessment: **NOT READY FOR USE**. Best checkpoint is epoch 3/update 101,937. `evaluate_model.py` evaluated every one of the 27,648 clean-corpus test pairs (including 339 excluded from rollout training) and 27,648 clean-target controls, with no canonical overlap against actual training text/intermediates. Batch/reference decoding agreed on 24/24 seeded cases. The checkpoint, vocabulary, tokenizer and decoder runtime stayed frozen; no test tuning or additional training was done.

Noisy test exact reference matches: 17,269 / 27,648 (62.46%); identity baseline 0. Clean controls: 17,675 / 27,648 (63.93%) had token changes, with 4.66% surface-token edit rate (goal <=3%). Only 1/85 original teacher rewrites and 0/37 messy targets matched exactly, versus 17,021/26,947 local rule pairs. These generated references are imperfect: exact matching is not semantic correctness, and some unchanged originals preserve meaning better than their reference. The qualitative examples nevertheless show broken grammar, deleted predicates and unnecessary articles.

Additional literal guards flag 315 returned noisy outputs and 315 clean outputs for raw code/markup changes; all are outside the rollout-representable subset. Detected copy-slot surfaces are preserved in returned text; the model refused seven noisy and five clean requests. The four-thread FP32 PyTorch p95 on this Ryzen AI Max+ 395 is 9.8 ms for 100 sampled sentences and 15.1 ms for 100 sampled 2–6-sentence paragraphs. This is not a GGUF/NPU/Strix Point benchmark. Raw checkpoint is 111,467,306 bytes; quantized deployment size is unmeasured.

Diagnostic prompts expose a vocabulary-policy problem: `file`, `delete`, `existing` and `otherwise` are encoder-only. Example outputs: Open file. -> Open the.; Do not delete `config.json`. -> Do not the. `config.json`.; Utilize the existing cache. -> Give the cache. The model cannot simply KEEP these tokens. Their handling needs deliberate software-term/POS policy review rather than silently approving all rejected words. Separately, the edit loss weights and article-heavy corpus are plausible contributors to aggressive editing, but causality has not been isolated.

Final score artifacts are `data/out/evaluation-v2/{assessment,summary,manual-notes,diagnostic-prompts}.json`, `predictions.jsonl.gz` and `examples.txt`. `evaluation-v1` contains original inference outputs and source snapshot. Eight spurious slot-fidelity flags from re-tokenizing adjacent names/backticks were corrected by `copy_fidelity.py` / `rescore_copies.py`; all 55,296 frozen predictions were verified identical. v2 is the authoritative score report. Three copy-count regression tests pass; no checker change or paid resource was used. The local training service exited successfully and is inactive.

2026-10-25: Implemented the authorized vocabulary corrections and guarded
five-attempt feedback loop. See the final section of `asd-ste100/HANDOFF.md` and
`model_tagger/README.md` for results and invocation. Local artifacts are under
`asd-ste100/data/out/fixes-v1/`.35 tests pass. Separate256-pair dev verification:
107 noisy exact repairs,11 clean changes(4.3%,0.22% token edits);1024 combined
calibration/verification outputs pass independent copy/closed-output/guard checks.
Existing checkpoint/test evaluation frozen. New model is a token-remapped copy,
not a new training run; promoted emission rows still require training. No money spent.

2026-10-25 local v2 training authorized and pipeline started as
`asd-ste100-local-v2.service`. Consult `asd-ste100/data/out/local-v2-pipeline/status.json`
for the actual stage; optimizer status appears at `data/out/local-train-v2/status.json`
when training starts. Full filtering audit passed(998,805/1,028,381 retained); fresh
ID export and full ID audit passed(649,386 pairs). Whole trajectory/split audits,
quarantine if necessary, benchmark isolation and an exact4+4-vs8 optimizer resume
pilot gate the main run. Main settings:3epochs CPU BF16, initialize migrated weights,
fresh optimizer, KEEP1/terminal1, capped/weighted local replay (details in model README).
71 tests pass.32 proposed owner-pending software review cases added and baseline
scored(13/16clean exact,8/16repair exact,2unresolved,0literal/guardfailures). All original
train/dev/test group assignments preserved for surviving rows. More retained rows
reflect representability changes, not independent meaning approval. Final automated
dev/baseline/benchmark comparison will be `data/out/local-train-v2/comparison.json`.
No paid resources. Check status before claiming the new training has completed.

Launch verified07:59UTC: all rebuild/replay/split/benchmark-isolation audits passed.
Final `tagger-rollouts-v4` has648,627 pairs; zero canonical overlaps/conflicts; no
quarantine needed. Resume pilot bit-identical for model/optimizer/RNG. Main local
v2 optimizer is running (first100/20,901updates verified,3epochs). Rough initial
ETA1.5–2hours plus automatic comparison. Check service/status files for current
state; do not describe the job as finished until its completion evidence exists.
Raw loss is not directly comparable to v1 because KEEP/terminal weights changed.

Local v3 loss-balance/checkpoint experiment authorized and running under Dibs CPU
coordination as `asd-ste100-balance-v3.service` (see balance-v3/status.json).
KEEP0.3/terminal0.5; otherwise v2 settings/data and migrated-v1 initialization.
Every epoch retained. Finer v2 confidence calibration recovers228/512 exact repairs
at19/512 clean changes(3.71%) versus old margin2's160/512 on the same calibration
rows. This is tuning data, not fresh verification. Pipeline selects on original512
calibration rows only, then reports fresh dev1024:2048 and fixed software review
scores without tuning on them.37model/tokenizer tests passed. Old artifacts frozen;
new optimizer files under local-train-v3. No paid compute; respect Dibs holds.

V3 optimizer launch verified (first300/20,901updates finite), under the Dibs CPU
service above. Moderate KEEP0.3/terminal0.5 experiment and per-epoch retention are
active. Final comparison pending; training status:asd-ste100/data/out/local-train-v3/status.json.

October 26 continuation: v3 finished without beating v2 on the fresh dev slice
(v2 463/1024 exact repairs and 58 clean changes; best v3 441 and 51). The owner
directed data improvement after dropping diffusion. The independently validated
`id-quality-v4b` corpus contains 623,001 pairs, including 1,800 examples from 36
assistant-reviewed rewrite frames, with 261 suspect groups quarantined. A new
80-case software benchmark is frozen for comparison; owner review is pending.
See the latest section of `asd-ste100/HANDOFF.md` for the audit and scope.

`asd-ste100-quality-v4.service` now runs the free local audit/train/evaluation
pipeline under Dibs CPU coordination. Read `asd-ste100/data/out/quality-v4-pipeline/status.json`
for the current stage and `comparison.json` there for the result when complete.
The optimizer status will be `asd-ste100/data/out/local-train-v4/status.json`.
It must pass rollout replay, canonical split and benchmark-isolation gates before
training. Do not launch a duplicate or claim improvement before the matched
fresh-dev/software comparison. No paid services were used or authorized.

V4 is complete. All gates passed, 622,320 accepted rollout pairs, both CPU
epochs and 4,719 updates finished. A predeclared finer margin check selected
epoch-one margin1.25 on calibration rows; fresh dev had493/1024 exact and66
changed clean controls versus v2's502 and58. The80-case software review favored
v4 (40/40 clean controls and14/31 required repairs versus v2 29/40 and9/31),
but all89 original and34 messy dev pairs still had zero exact matches for both
models. Exact references are not meaning gold. Keep v2 as the general checkpoint;
v4 is a specific software-phrase research result, not production ready. Read
the final section of `asd-ste100/HANDOFF.md` and the comparison JSON files for
the audit, protocol and limitations. No money spent.

October 28: the documentation teacher pilot's first 12 passages ran locally
(the October 26 queue had timed out waiting for DFlash2 timing). The gates
accepted 2 of 12, and assistant review found that both accepted targets change
meaning. Rejections come mostly from words the STE vocabulary does not approve,
and from a gate conflict: the modal and negation count gates require `may` and
`should` to stay, while the checker rejects them, which blocks 8 of the 36
frozen inputs. The pilot code was not changed. An owner decision on the gate
and prompt policy is needed before the remaining 24 run. See the final section
of `asd-ste100/HANDOFF.md` and
`data/out/doc-teacher-pilot-v1/first-12-assistant-review.json`. No money spent.

October 28, later: the doc teacher pilot v2 gave the teacher the full STE
material (a 27k-token system prompt, cached as a llama-server slot) and
meaning-class gates. On the same 12 passages, 7 pass the gates and 5 hold up
under assistant review; v1 had 2 and 0. Two accepted rewrites still drop
qualifiers. See the last section of `asd-ste100/HANDOFF.md`. Owner review of
both packets and of the `should` policy is pending. No money spent.

October 29: the documentation teacher pilot v3 ran on all 36 inputs. Gates
alone accept 30, of which 21 keep the meaning (assistant review). Adding a
same-model meaning check gives 13 accepted, all 13 keeping the meaning, but
loses 9 good rewrites. See `asd-ste100/HANDOFF.md` and
`asd-ste100/data/out/doc-teacher-pilot-v3/all-36-assistant-review.json`. The
owner decisions are still pending.

October 30: the v5 documentation teacher processed all 21,380 pool passages
through OpenRouter (owner-funded, about $12): 12,763 candidate pairs, 11,360
unflagged. Only 5,201 of those fit the student model's vocabulary and copy rules
today; the rest wait on tokenizer and `tech_words.txt` decisions. See the last
section of `asd-ste100/HANDOFF.md`. Nothing committed.
Later: vocabulary fixes-v2, a repair pass ($1.92) and local v5 training on 7,794
documentation pairs. v5 does not learn documentation rewriting (0 of 679 dev
inputs resolved, as v2); it improves the software review set (39/40 clean,
13/31 repairs). Keep v2. Details in `asd-ste100/model_tagger/README.md`.
Then a documentation fine-tune of v5 (v5b, free): still no documentation rewriting,
because the trained encoder has lost plasticity (it cannot memorize 256 doc
examples). A randomly initialized tagger trained on the doc pairs alone gets 88 of
679 doc dev rewrites exact (v2: 2). Next test: rebuild the tagger from scratch with
the doc pairs mixed in. Details in `asd-ste100/model_tagger/README.md`, "v5b".
October 31: a from-scratch tagger rebuild with the doc pairs mixed in (free). The
first try (v6) never learned to edit (KEEP weight too high from the start); v6b,
two stages as v1 -> v2, works. It matches v5 on the general dev rows (473 exact vs
477) and nearly doubles the software repairs (24/31 vs 13/31; clean 36/40, all four
misses add "the" before a code literal). Documentation: 17 of 679 exact raw
rewrites (v5: 2, doc-only tagger: 88), still damaging meaning; 1 through the
guarded editor. Not good enough for documentation. Details in
`asd-ste100/model_tagger/README.md`, "v6 and v6b".
Later: a first canvas editor (8M parameters, doc pairs only, free) gets 56 of 679
doc rewrites exact against the doc-only tagger's 88, with fewer clean inputs changed
(13 vs 23). It overfits and repeats words, but it splits sentences. See
`asd-ste100/model_canvas/README.md`.

Canvas pretraining test (2026-10-31, free): pretraining the 8M canvas editor on the
cleaned rule pairs before the doc pairs gives 62 of 679 doc dev exact (control
without pretraining: 62; doc-only tagger: 88), with clean changes down from 32 to 10.
The canvas editor is shelved; details in `asd-ste100/model_canvas/README.md`.
