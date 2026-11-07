# Offline iterative tagger baseline

This is the local baseline from design section 3.5. The first local training run has completed; there is no evaluated release, GGUF writer or deployment runtime. Run it after the [joint surface split and tokenizer workflow](../data/JOINT_SPLITS.md). Derived vocabulary, data and future weights remain ignored.

Install `tokenizer/requirements.txt`, then CPU PyTorch with:

```bash
python -m pip install -r asd-ste100/model_tagger/requirements.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s asd-ste100/model_tagger/tests -v
python asd-ste100/model_tagger/smoke.py --vocab asd-ste100/data/out/tokenizer-joint-v1 --pairs asd-ste100/data/out/id-joint-clean-v2 --out asd-ste100/data/out/tagger-smoke-v1.json
```

The default encoder is six bidirectional Transformer layers, width 512, eight heads and FFN 2048, with learned absolute positions, LayerNorm and GELU. It embeds token IDs, four input case classes, copy-slot references and ten checker rule features. The mode token is prepended. Checker features mark affected sentence spans; paragraph-length feedback applies across the input. The checker implementation is unchanged.

The encoder consumes every BPE piece. Mean pooling over `source_spans` yields one editing position per original surface span, so a raw code/name/number span stays atomic. Its copy-slot reference supplies the output unit ID while its raw bytes, case and slot-reference embeddings still enter the encoder. A prefix action anchored at the mode token permits insertion before the first word.

Actions are `KEEP`, `DELETE`, `REPLACE_w`, `APPEND_w` and twenty deterministic transforms: three case operations, sixteen suffix operations and merge-with-next. Transforms are available only when their result exists in the closed output vocabulary; they are candidate edits, not assertions of grammatical correctness. Merge consumes two ordinary spans and rejects overlapping edits. Case prediction for a replaced or appended ordinary token uses a separate three-class head. Slots ignore output case and always restore their original surface.

Per-request masks block inactive slots and every control token, including `NULL`, from replacement/append emissions. Encoder-only words cannot use KEEP or APPEND. Every decoded pass separately enforces exact slot counts. The iterative loop reruns the shared tokenizer/checker, stops at checker-clean output, all KEEP, repetition, or the five-pass cap, and reports unresolved status. Slot-invalid predictions return no output text. It also checks original protected surfaces across retokenized passes. Input chunking and capacity recovery belong to the future host loop; tokenizer capacity failures remain explicit errors.

`supervision()` derives one pass from a minimum-cost alignment of canonical output/copy units and case. Each anchor performs one action; extra insertions or insertion/replacement conflicts are deferred. The completed rollout exporter audits convergence within five passes, intermediate literal/slot fidelity, inference-equivalent copy units, terminal targets and exit behavior. It excludes failed pairs as a whole and preserves slice/review metadata. See the audited rollout and training sections below for current results; the small smoke alone does not establish corpus coverage.

The smoke uses the full default architecture, computes a real tag/case loss on selected local records, runs backward and checks finite gradients. It verifies exact teacher-forced action decoding and exercises random-weight iterative decoding. It performs no optimizer step and saves no weights. Its elapsed forward/backward time is a diagnostic, not the CPU inference latency benchmark. Random output, checker-clean output and generated holdouts do not establish meaning preservation. Owner-verified gold, real-input evaluation, training authorization and quality/latency/size measurement remain outstanding.

Use `id-joint-clean-v2` after the deterministic cleanup. The original parent and earlier random-weight smoke are historical. This filtered subset preserves the parent splits and all retained labels; it is not meaning-verified gold.

## Audited rollout data and local training

The current training export is `data/out/tagger-rollouts-v3`, built from `id-joint-clean-v2` with `tokenizer-joint-v1`. It retains **599,855 pairs**: 543,592 train, 28,954 dev and 27,309 test. Packed examples include each edit state plus a terminal KEEP state: **1,087,219 train / 57,911 dev / 54,621 test examples**. Every state uses the existing deterministic 70% slotted / 30% raw exposure, including target/terminal states; inference remains fully slotted. The audit checks that raw/slotted editing units agree.

The first rollout export accepted 599,873 pairs and rejected 7,167: 6,470 intermediate literal changes (including formatting of unprotected raw calls), 433 trajectories that the checker would terminate before the target, 143 detokenization/span changes, 112 slot-fidelity failures, and nine source-capacity failures. It reaches 599,837 targets in one edit, seven without an edit, 17 in two edits, six in three, five in four and one in five. The final export additionally quarantines two complete groups (18 local pairs, nine train and nine dev) found to overlap after token/punctuation-spacing canonicalization. No remaining group was reassigned. `v1` is the pre-quarantine diagnostic export; `v2` is superseded by `v3`'s current audit implementation fingerprints.

`rollouts.py` audits actions, masks, intermediate exact copy counts, original protected surfaces, code/math/markup guards, re-encoding, capacity, checker stopping and terminal targets. Failed trajectories contribute no partial training states. Independent replay rechecks all 607,040 parent decisions and all stored input/case/slot/checker/transform features. `audit_splits.py` checks canonical source/target text and all 48 generated intermediate source texts, using real protected surfaces rather than opaque slot IDs. It also checks conflicting canonical sources. The runner requires both audits and verifies their hashes before loading data. Metadata and review labels remain tied to the original pairs.

The decoder now always runs one initial model pass. Checker-clean input previously returned before editing, which incorrectly skipped missing-article repairs. Subsequent checker stopping, all-KEEP, repetition and the five-edit cap remain as designed. Raw source features are pooled into the same copy units as fully slotted inference. Masks used by the packed collator are tested against the reference collator; single selected-action checks avoid allocating a full vocabulary mask during the audit.

`packed.py` stores little-endian int32 feature arrays plus int64 indexes. The trainer memory-maps them, batches similar lengths and uses precomputed checker/transform features. It keeps the default 6-layer, width-512 architecture. Baseline optimization: AdamW, LR 3e-4, 500 warmup updates then cosine decay, batch 32, BF16 CPU autocast, eight CPU threads and two data workers. KEEP-token loss weight is 0.1; terminal-example weight is 0.25, to reduce the overwhelming KEEP imbalance. These are training hyperparameters, not new meaning-review labels. Dev loss and edit-tag/terminal-KEEP metrics are logged by data slice. Quick dev checks use a fixed seeded sample across length buckets; epoch-end checks use all dev examples and a small iterative decode sample.

```bash
python asd-ste100/model_tagger/rollouts.py export --pairs asd-ste100/data/out/id-joint-clean-v2 --vocab asd-ste100/data/out/tokenizer-joint-v1 --out asd-ste100/data/out/tagger-rollouts-v1 --workers 8
python asd-ste100/model_tagger/rollouts.py validate --pairs asd-ste100/data/out/id-joint-clean-v2 --vocab asd-ste100/data/out/tokenizer-joint-v1 --out asd-ste100/data/out/tagger-rollouts-v1 --workers 8
python asd-ste100/model_tagger/audit_splits.py --data asd-ste100/data/out/tagger-rollouts-v1 --vocab asd-ste100/data/out/tokenizer-joint-v1
# If the canonical audit finds implicated groups, quarantine whole groups:
python asd-ste100/model_tagger/quarantine_rollouts.py --parent asd-ste100/data/out/tagger-rollouts-v1 --pairs asd-ste100/data/out/id-joint-clean-v2 --out asd-ste100/data/out/tagger-rollouts-v3
python asd-ste100/model_tagger/rollouts.py validate --pairs asd-ste100/data/out/id-joint-clean-v2 --vocab asd-ste100/data/out/tokenizer-joint-v1 --out asd-ste100/data/out/tagger-rollouts-v3 --workers 8
python asd-ste100/model_tagger/audit_splits.py --data asd-ste100/data/out/tagger-rollouts-v3 --vocab asd-ste100/data/out/tokenizer-joint-v1
python asd-ste100/model_tagger/train.py --data asd-ste100/data/out/tagger-rollouts-v3 --vocab asd-ste100/data/out/tokenizer-joint-v1 --out asd-ste100/data/out/local-train-v1 --epochs 3 --threads 8 --workers 2 --precision bf16 --checkpoint-every 500 --max-hours 24
```

Use the existing ignored `.venv` Python or activate it first. Outputs must be new/empty, except an explicitly resumed run. To resume training, repeat the exact training command with `--resume`; paths are normalized, and data/code/hyperparameter mismatches fail closed. Runtime step/time limits can be extended without changing optimization. `last.pt` atomically saves model, optimizer, RNG and exact epoch/batch position; `best.pt` saves the best epoch-end dev-loss weights. A pilot's two steps plus resume to four produced bit-identical weights to a continuous four-step run, and a separate full pilot epoch completed evaluation and best-checkpoint saving.

`status.json`, `metrics.jsonl`, `config.json`, checkpoints and per-epoch dev diagnostics live in the run directory. SIGTERM/SIGINT to the main trainer requests a checkpoint at the batch boundary. Local runs are capped at three epochs or 24 accumulated hours; no paid resources or API calls are used. Checkpoints and all corpus-derived artifacts stay ignored. Full quality, owner-verified meaning, GGUF export and runtime latency are still unmeasured.

## Active local run — October 24, 2026

The owner authorized local training after the audit. The full replay and canonical split audits for `tagger-rollouts-v3` passed before launch. `local-train-v1` is running under the user service `asd-ste100-local-v1.service`, at Nice=10, with three epochs, BF16, batch 32, eight CPU threads, two workers, checkpoints every 500 updates and a 24-hour accumulated time cap. Total planned updates: 101,937. Initial fixed-sample dev loss: 10.0652. This is training in progress, not a quality-qualified model.

```bash
systemctl --user status asd-ste100-local-v1.service
cat asd-ste100/data/out/local-train-v1/status.json
tail -f asd-ste100/data/out/local-train-v1.log
```

The service uses `KillMode=mixed` and a 300-second stop timeout, allowing the main trainer to checkpoint before workers are killed. `systemctl --user stop asd-ste100-local-v1.service` requests a graceful stop. Do not launch a second copy against the same run directory. A machine reboot still requires an explicit resume; this is not an enabled boot service.

Current proof artifacts: `tagger-rollouts-v3/validation.json`, `tagger-rollouts-v3/split-validation.json`, `local-train-resume-verification.json`, `local-training-launch.json`, and `local-train-v1/config.json`. The latter records exact training arguments and hashes. No RunPod resources or paid services were used. The 1,407 retained original teacher rewrites and 579 messy pairs remain a small minority; broad rewrite quality and meaning preservation must still be evaluated.

## Completed baseline evaluation

The initial three-epoch CPU run completed successfully in 5h43m, with all 101,937 updates saved and best full-dev weighted loss 0.27277 at epoch 3. The service is inactive after successful completion. Read `best.pt` for evaluation; `last.pt` is resumable state. Earlier active-run status paragraphs are historical.

`evaluate_model.py` performs frozen-checkpoint actual decoding on the full clean-corpus test split, clean-target over-editing controls, literal/slot checks, unchanged-input baseline comparisons, exact-reference and edit-distance diagnostics, and four-thread individual-request timing. It first checks test isolation against actual training data including intermediate states and checks batched/reference decoder agreement. It does not use test results to tune weights, thresholds or checkpoint selection. All results stay under ignored `data/out/`.

```bash
python asd-ste100/model_tagger/evaluate_model.py --checkpoint asd-ste100/data/out/local-train-v1/best.pt --vocab asd-ste100/data/out/tokenizer-joint-v1 --pairs asd-ste100/data/out/id-joint-clean-v2 --rollouts asd-ste100/data/out/tagger-rollouts-v3 --out asd-ste100/data/out/evaluation-v1 --threads 4 --batch-size 32
```

Use a new/empty evaluation output directory. Generated references and checker-clean outputs are not a meaning-quality certification. The owner-verified gold set and downstream real-input meaning evaluation are still outstanding.

## Evaluation result: not ready for use

Final score report: `data/out/evaluation-v2/assessment.json` and `summary.json`; per-case frozen predictions and human-readable samples are in the same directory. On all 27,648 clean-corpus test pairs, the model matches 62.46% of references, but it changes 63.93% of clean controls (4.66% token edit rate). Original/messy exact-reference counts are 1/85 and 0/37. Manual checks show lost predicates and broken grammar; exact-reference mismatches are not all errors because generated targets can be poor or have valid alternatives.

The current output vocabulary excludes common words including file/delete/existing/otherwise, which the runtime cannot KEEP. This is a concrete policy/coverage gap; no vocabulary changes have been made. Existing literal guards also flag 315 noisy outputs and 315 clean outputs on unprotected code/markup outside the supported rollout subset. Detected copy-slot characters were preserved, and invalid slot-count predictions were refused. Before another run: review software noun/verb copy policy, enforce literal guards in inference, and calibrate edit confidence on dev only. Broader meaning evaluation is still required.

Four-thread single-request FP32 PyTorch p95 was 9.8 ms for sampled sentences and 15.1 ms for short paragraphs on the Ryzen AI Max+ 395. This does not measure a GGUF/NPU deployment. Checkpoint size is 111.5 MB; deployment/quantized size has not been measured.

`copy_fidelity.py` counts exact original slot surfaces without treating newly merged name tokens as missing copies. `rescore_copies.py` corrected eight retokenization-only fidelity flags while preserving every frozen prediction; v1 inference/source evidence remains intact. It does not modify model/runtime behavior. Three regression tests cover merged names, nested code literals, word boundaries, case and duplicate copies.

### Guarded inference and retries (2026-10-25)

`safe_editor.py` is the guarded inference entry point. The older `tagger.rewrite`
remains the frozen baseline policy for comparison. The guarded editor uses a
calibrated KEEP logit margin, freezes copy-slot anchors, validates literal and
content-word preservation, and rejects unsafe proposals without advancing the
state. Rejected action/position pairs are banned on the next proposal for that
same state. Accepted states clear positional feedback. All attempts, including
rejections, count toward the hard five-attempt limit; the result includes a trace.
Unresolved results retain the last accepted text (possibly the original input),
which must not be presented as a successful normalization.

The content gate is deliberately conservative: it permits article changes and
three explicitly reviewed utilize/use inflection mappings. It does not establish
semantic equivalence, grammar correctness, or full STE compliance. `resolved`
means checker-clean and accepted by these partial guards. Broader paraphrases
usually abstain. The checker can still accept missing or extra articles.

`migrate_vocabulary.py` promotes the seven explicitly approved software forms
file/files/delete/deletes/deleted/existing/otherwise. It inherits the original
train-only byte BPE and remaps input embeddings, output emissions, transforms,
and slot-reference embeddings by token identity. Existing parameters are retained;
new emission rows are suppressed until trained. This makes the promoted words
legal to KEEP, but does not train their insertion/replacement behavior. Do not
resume the old optimizer or reuse old packed IDs with the new vocabulary.

`calibrate_editor.py` selects confidence on SHA-sampled DEV rows only. Use
`--offset 256 --locked-policy .../policy.json` for a disjoint dev verification
sample. Calibration writes predictions, traces and a checkpoint-bound policy.
The existing test evaluation is unchanged and is not used to choose the margin.

Example (paths are local ignored artifacts):

```bash
asd-ste100/data/out/.venv/bin/python asd-ste100/model_tagger/safe_editor.py \
  'Do not delete `config.json`.' \
  --vocabulary asd-ste100/data/out/fixes-v1/vocabulary \
  --checkpoint asd-ste100/data/out/fixes-v1/vocabulary/model.pt \
  --policy asd-ste100/data/out/fixes-v1/calibration/policy.json
```

Tests: `python -m unittest discover -s asd-ste100/model_tagger/tests` and
`python -m unittest discover -s asd-ste100/tokenizer/tests` in the local venv.

### Revised local training pipeline (2026-10-25)

`run_local_v2.py` runs the full rebuild and audit chain before training. It starts
from `joint-surface-v1` (the original whole-corpus grouping), applies and independently
validates deterministic filtering, exports fresh IDs with the migrated vocabulary,
replays every source/case/slot/edit field, exports/audits trajectories, and checks
canonical source/target/intermediate split overlap. Implicated whole groups are
quarantined and both trajectory and split audits must pass again.

Training uses `--initialize-from` with a vocabulary-matched checkpoint and a fresh
optimizer. The local v2 experiment uses three epochs, LR6e-5/warmup200, KEEP1.0 and
terminal1.0, kind weights8/12/3/1 (original/messy/rule/local_rule), and a deterministic
200,000 local-example cap per epoch. All accepted original/messy/rule examples are
retained each epoch; synthetic sampling rotates by epoch without replacement.
Dev loss excludes kind weights and uses the full dev split at epoch boundaries.
The decoder sample uses the guarded editor. These settings are experimental,
not a claim that every real rewrite has independently verified meaning.

Before the main run, an eight-update pilot compares resumed4+4 updates against
continuous8 updates, including model/optimizer/RNG state. The main run has an
eight-hour local time cap, checkpoints every500 updates, and a fresh run directory.
Resume fails closed if initialization, settings, source, vocabulary or data change.

On completion, the pipeline calibrates the guarded editor on512 dev pairs, checks
the locked threshold on a disjoint512, evaluates the frozen migrated baseline on
the same verification rows, and scores both against the32-case proposed software
review set. No test-set tuning or deployment occurs. A failed audit or missing
acceptable confidence policy stops the pipeline and records the stage/error.

Status and stage logs: `data/out/local-v2-pipeline/`. Training status/checkpoints:
`data/out/local-train-v2/`. The background service is `asd-ste100-local-v2.service`.
The review cases and their owner-review status are in `benchmarks/`.

### Loss-balance and checkpoint comparison (local v3)

`run_balance_v3.py`, under a Dibs CPU reservation, compares the frozen migrated
baseline, v2, and all three retained v3 epoch checkpoints. `calibrate_editor.py`
now accepts `--margins` for smaller dev-only confidence increments. The old
0/2/4/6 grid rejected v2 margin0 on26/512 clean changes (5.08%) and jumped to2;
margin0.25 instead yields228/512 noisy exact and19/512 clean changes (3.71%) on
that calibration slice. These are calibration results, not new verification scores.

V3 changes only KEEP/terminal loss weights to0.3/0.5 from v2's1/1, keeping its
initial model, data, local replay cap, kind weights, seed, LR and three-epoch
schedule. The initial model is the migrated v1 weights, with a fresh optimizer.
`--retain-epochs` preserves model-only epoch checkpoints as well as best/last.
No repeat corpus rebuild is needed: train.py still requires the existing complete
trajectory and split audits and checks all hashes before training.

Each candidate's confidence policy and the overall checkpoint winner are selected
on the original512-row dev calibration slice under the <=5% clean-change limit.
Selection is written before examining a fresh1024-row dev slice at offset1024 or
the fixed software review set. These are disjoint rows; this does not claim an
independent owner-verified meaning evaluation. Verification is reporting-only.
Old test scores and production defaults remain unchanged.

Service:`asd-ste100-balance-v3.service`. Experiment status/reports:
`data/out/balance-v3/`; optimizer status/checkpoints:`data/out/local-train-v3/`.
Dibs CPU ownership follows the service child and releases automatically. No paid
resources. A four-hour training cap stops rather than bypasses a failed/partial run.

### Reviewed-data local v4 result (2026-10-26)

V3 did not beat v2 on fresh dev: its best epoch had441/1024 exact references and
51 clean changes, versus v2's463 and58 on the prior v3 comparison slice. The
v4 data experiment therefore starts from v2, with a fresh optimizer, two CPU
epochs, LR3e-5, KEEP0.4/terminal0.6, and kind weights8/12/3/1/12 for
original/messy/rule/local_rule/reviewed. The local-rule cap is50,000 examples
per epoch. The model remains27,870,263 parameters; this is an iterative tagger,
not a diffusion model.

`data_revision_v4b.py` built 623,001 pairs from the existing joint ID corpus:
261 suspect groups were quarantined, and1,800 examples derived from36
assistant-reviewed software rewrite frames were added as a distinct kind. The
six exact phrase equivalences in `reviewed_rewrites.py` are allowed by the
content guard when ordered content and protected literals stay unchanged. All
retained and added payloads passed independent replay. Rollout export accepted
622,320 whole pairs. The full rollout replay and canonical split audit passed;
the frozen80-case software benchmark had zero input/reference overlap against
all623,001 parent pairs. Model/tokenizer suites passed31+13 tests.

The two-epoch run completed all4,719 updates. `data/out/quality-v4-pipeline/`
contains the source snapshot, audit logs, selected policies, predictions and
comparison reports. Coarse calibration selected epoch-one margin2; fresh dev
rows1024:2048 yielded445/1024 exact and42/1024 changed clean controls, versus
v2 margin0.25 with502/1024 and58/1024. The80-case software review improved:
v4 got40/40 clean exact and13/31 required repairs, versus v2 29/40 and9/31.
Three duplicate-article inputs remained unresolved for both; their source
defects were not accepted as valid output. "Clean exact" measures unchanged
text and includes five unresolved clean cases in each model; it is not a claim
that all forty were successfully normalized.

The margin grid had jumped from1 to2 across the calibration safety cap. A
narrower grid was specified before examining fresh verification outputs and
selected epoch-one margin1.25 on calibration rows0:512. Locked verification
then gave493/1024 exact and66/1024 clean changes; software cases gave40/40
clean exact and14/31 required repairs. The v2 general baseline still leads on
fresh dev. `finer-policy-plan.json` records the protocol, and
`finer-comparison.json` records the result.

The general dev slice is almost entirely synthetic local rules. A separate
diagnostic over all nonlocal dev ID rows found exact reference counts of
53/554 rule,0/89 original,0/34 messy for v2, and50/554,0/89,0/34 for v4 at
margin1.25. These imperfect references and exact matching do not measure
meaning quality. V4 shows a specific improvement on reviewed software phrase
families and clean-control restraint, but it is not a general replacement for
v2. Keep both as research artifacts; no production promotion or paid compute.

### Documentation-teacher local v5 (2026-10-30)

`extend_vocabulary.py` builds `data/out/fixes-v2/vocabulary` from fixes-v1 and the v2
checkpoint: 37 new output words (new `tech_words.txt` lines) and 339 input-only
grammar words (contractions, been/being/am, -ing forms of approved and software
verbs; see the tokenizer README). New input words start from the mean of their BPE
piece embeddings; new emission rows start suppressed. `packed.KINDS` gains
`doc_teacher` (appended, so older packed data keeps its indices); `--kind-weights`
now takes six values. `data_revision_v5.py` is the v4b revision on `id-joint-v5`
(surface-clean-v3 exported with fixes-v2) plus 7,794 documentation pairs from
`data/out/doc-teacher-v5/final/accepted.jsonl`, split by source document;
`id-doc-v5/` holds their 679 dev and 218 test rows. `run_doc_v5.py` runs the whole
chain under a Dibs CPU hold: 584,179 pairs, 581,022 accepted as rollouts (5,423 of
the documentation pairs: 2,371 are rejected because the checker would stop the
loop before the target), split audit passed with no quarantine, two epochs from
the migrated v2 (LR3e-5, KEEP0.4/terminal0.6, kind weights 8,12,3,1,12,8, local cap
50,000), 5,304 updates in 31 minutes.

Selection on dev rows0:512 chose epoch two at margin1.0 (226 exact, 24 clean
changes; the migrated v2 baseline 239 and 23 at margin0.25). Frozen results:

| | Migrated v2 | v5 epoch 2 |
|---|---|---|
| Fresh dev rows1024:2048: exact / clean changed | 494 / 68 | 477 / 60 |
| Software review: clean exact / required repairs exact | 29/40, 9/31 | 39/40, 13/31 |
| Documentation dev (679): resolved / clean changed (guarded) | 0 / 92 | 0 / 61 |
| Documentation dev, raw decode: exact / unchanged | 2 / 87 | 2 / 90 |

Neither model rewrites documentation. The dev edit-tag accuracy of the
documentation pairs is 3.4% after both epochs (local rules 63%, original teacher
rewrites 7%). The proposals delete the word to replace and do not insert the
replacement ("can modify the version" -> "can the version"); the guarded editor
rejects them as content changes and returns the input. v5 matches v4 on the
software set and changes fewer clean inputs, at 17 fewer exact references on the
synthetic-heavy dev rows. `doc-v5-pipeline/comparison.json` and
`doc-dev-raw-decode.json` have the numbers. Keep v2 as the general checkpoint.

### Documentation fine-tune v5b and the plasticity probes (2026-10-30)

Kind weights did little in v5 because the loss is normalized per batch and most
batches hold a single kind. `train.py --kind-repeats` batches a kind's training rows
again each epoch (training only; tested). `run_doc_v5b.py` starts from v5 epoch 2 on
the same audited `tagger-rollouts-v7` data: documentation rows six times per epoch
(54% of the rows), local rule cap 25,000, peak LR 1e-4, three epochs, 10,246 updates
in 59 minutes. Every epoch is calibrated on dev rows 0:512 and verified with its
frozen policy. `edit_confusion.py` records, teacher-forced, what a checkpoint emits
on documentation edit states (gold vs predicted category, exact tag, top five, mass).

| | v5 epoch 2 | v5b epoch 2 | v5b epoch 3 |
|---|---|---|---|
| Fresh dev rows 1024:2048: exact / clean changed | 477 / 60 | 390 / 55 | 391 / 49 |
| Software review: clean exact / repairs exact | 39/40, 13/31 | 40/40, 16/31 | 38/40, 16/31 |
| Documentation dev (679), guarded: resolved / clean changed | 0 / 61 | 0 / 35 | 0 / 29 |
| Doc dev REPLACE positions: exact tag / gold in top 5 | 0% / 13% | 1.0% / 19% | 1.8% / 20% |
| Doc train REPLACE positions: exact tag | 0.1% | 0.9% | 1.8% |

The fine-tune does not fit even its own training pairs, and it gives up the rule
edits (local-rule dev edit-tag accuracy 63% to 50%). Probes (scripts and outputs in
`data/out/doc-v5b-pipeline/probes/`):

- A source-word lookup table (most frequent REPLACE target for each source word in
  doc train) gets 44.5% of doc dev REPLACE positions when told where they are. The
  targets are learnable; most are short function words ("it", "this", "is", ".").
- v5 epoch 2 trained on 256 doc edit states for 40 epochs at LR 3e-4: REPLACE exact
  0.4%. On one batch of 16 for 200 steps: 47% (fp32 the same; LR 1e-3 collapses to
  all KEEP).
- A randomly initialized tagger memorizes the same batch in 50 steps (100%). With
  the trained tag head and a fresh encoder it also memorizes; with the trained
  encoder and a fresh head it does not (7% after 175 steps). The trained encoder has
  lost plasticity.
- A fresh tagger trained on the 9,801 doc rows only (12 epochs, LR 3e-4, 22 minutes):
  doc dev REPLACE exact 33.5% (top five 52%), TRANSFORM 51%, DELETE 27%, APPEND 11%,
  KEEP kept 98.1%; it fits train (99.6%) and stops improving on dev after epoch 6.
  Raw decode of the 679 doc dev pairs: 88 exact (v2 and v5: 2, v5b epoch 3: 7), 11
  unchanged, 79 farther from the reference, 23 clean references changed, 13 not
  encodable. The exact ones are one- to five-word substitutions (every->each,
  listed->shown, modify->change, indicates->shows); sentence splits come out broken
  ("It the frontend must send ..."). It has never seen the rule data, so it is not a
  general checkpoint.

The architecture can learn documentation rewriting from these pairs; the v2 lineage
cannot absorb it. A rebuild from random initialization with the documentation rows
mixed in from the start is the next test of the tagger. Keep v2 as the general
checkpoint until then.

### Tagger rebuild v6 and v6b (2026-10-31)

`run_rebuild_v6.py` trained a randomly initialized tagger on `tagger-rollouts-v7`
with the documentation rows batched four times per epoch and the local rule rows
capped at 200,000 per epoch, but with the fine-tuning loss settings (KEEP 0.4,
terminal 0.6, kind weights 8,12,3,1,12,8). It never learned to edit: local-rule dev
edit-tag accuracy 0.2% at step 2,000 (v1: 63%), best dev loss 0.81, no epoch
calibrates below margin 6. 1,000-step probes (`data/out/rebuild-v6-pipeline/probes/`)
put it on the KEEP weight: KEEP 0.4 with kind weights 1 reaches 14%, the v1 setting
KEEP 0.1 / terminal 0.25 reaches 67%. The v6 outputs stay as a record.

`run_rebuild_v6b.py` follows the v1 -> v2 recipe on the same data and sampling.
Stage 1: fresh, three epochs, KEEP 0.1 / terminal 0.25, kind weights 1, LR 3e-4
(24,799 updates, 85 minutes). Stage 2 (polish): one epoch from the stage-1 end at
LR 6e-5, KEEP 0.4 / terminal 0.6 (8,266 updates, 38 minutes). Selection (max exact on
calibration rows 0:512 with at most 5% clean changes) picks the polish checkpoint,
`data/out/local-train-v6b-polish/epoch-1.pt`, at margin 1.5.

| | v2 (migrated) | v5 epoch 2 | v6b epoch 3 | v6b polish |
|---|---|---|---|---|
| Fresh dev rows 1024:2048: exact / clean changed | 494 / 68 | 477 / 60 | 437 / 49 | 473 / 55 |
| Software review: clean exact / repairs exact | 29/40, 9/31 | 39/40, 13/31 | 37/40, 25/31 | 36/40, 24/31 |
| Documentation dev (679), guarded: exact / clean changed | 0 / - | 0 / 61 | 0 / 56 | 1 / 54 |
| Doc dev raw decode: exact / farther / clean changed | 2 / - / - | 2 / 14 / 136 | 9 / 322 / 580 | 17 / 141 / 183 |
| Doc dev REPLACE positions: exact tag / gold in top 5 | - | 0% / 13% | 10.9% / 30% | 12.1% / 32% |
| Local-rule dev edit-tag accuracy / exact edit precision | - | 63% / - | 79% / 37% | 68% / 65% |

The fresh doc-only tagger (v5b probes) reaches 33.5% REPLACE exact and 88 exact /
79 farther / 23 clean changed on the same raw decode; mixed with the rule data, the
documentation rows are learned about a third as well. The raw rewrites still damage
meaning ("must now send" -> "must not send"; "indicates" and "needed" deleted) and
break sentence splits. The guarded editor resolves 5 of the 679 doc dev sources.

The four clean software cases the polish checkpoint changes all insert "the" before
a code literal ("Set the `poll_delay_ms` to 275 ..."). The source is the local rule
data: 53,622 of 507,399 local-rule training pairs (10.6%) are `drop_article` noise
before a backtick literal, so they teach that insertion; the doc pairs add it in 19
of 6,897. Not changed.

v6b polish is a fresh-lineage checkpoint that matches v5 on the general dev rows
and nearly doubles the software repairs (24/31 vs 13/31); it does not rewrite
documentation well enough for use. Adopting it as the general checkpoint, and the
`drop_article`-before-literal rule, are owner decisions.
