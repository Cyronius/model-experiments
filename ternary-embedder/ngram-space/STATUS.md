# Plan B status (n-gram lookup embedder in Qwen space)

Plan: `ngram-space/PLAN.md` revision 3. Shared contract: `shared-qwen8b/README.md`.

## Spend (hosted API, Plan B cap $5)

| Date | Stage | What | Tokens | Dollars |
|---|---|---|---|---|
| 2026-11-12 | planning | one test call (before shared.py) | 11 | 0.00000011 |
| 2026-11-12 | 0 | `shared.embed` check, 2 words | 4 | 0.00000004 |
| 2026-11-12 | 2 | 200-item seed probe | 577 | 0.0000058 |
| 2026-11-12 | 2 | T1 seeds: 29,523 vocab + 250k bigrams + 250k trigrams | 1,908,900 | 0.0190890 |
| 2026-11-12 | 3 | T2 seeds: 750k bigrams + 750k trigrams | 5,604,402 | 0.0560440 |
| 2026-11-12 | 4 | probe pairs: 1,753 texts embedded by the teacher | 95,811 | 0.0009581 |
| **total** | | shared spend table, plan B | **7,609,705** | **$0.0761** |

Counts come from `shared.embed(..., plan="B")` (uncached calls only).

## Log

### 2026-11-12 Stage 0 (done)

- 10:27 Started. Plan A had claimed `code`, `jaswanth` and `reserved` at
  10:27:46 (owner file "Plan A"); the Jaswanth download is under Plan A's
  `dibs disk claim` (8 GB, ETA 12:27). Plan B waits for those and does not
  build them.
- Created `ngram-space/{stage0..stage4}`, `training-v2/work/ngram-space/`.
- Wrote `ngram-space/common.py` (tokenization without special tokens,
  bigram/trigram key packing, `embed` via `shared.embed(plan="B")`).
- Started Stage 1's local tokenization (CPU only, no teacher) while waiting:
  `dibs run cpu -- .venv/bin/python -I ngram-space/stage1/tokenize_local.py`.

- 10:31-10:45 Shared items: `code`, `jaswanth`, `reserved` DONE by Plan A;
  `protocol-check` written by Plan A (Jaswanth **accepted**, median 0.9975);
  `leakage-jaswanth` + validation ids **claimed and built by Plan B**
  (268 rows removed; 10,084 validation rows), DONE + manifest entry.
- One hosted call through `shared.embed(plan="B")` succeeded (4 tokens).
- Training set: 1,173,363 Jaswanth texts (96.4M tokens), median 33 tokens.
- Stage 0 results: `stage0/RESULTS.md`, `stage0/result.json`.

### 2026-11-12 Stage 1 (done)

- Counting corpus 461M tokens (local corpora 364M + Jaswanth train 96M).
- Tiers: T1/T2 full for both orders; bigram T3 only 1,582,536 rows (all
  bigrams with count >= 20). Validation coverage bigram T1/T2/T3 49.5/62.4/65.7%,
  trigram 21.1/31.5/37.4%. T2 adds >= 10 points for both orders.
- Results: `stage1/RESULTS.md`, `stage1/result.json`.

### 2026-11-12 Stage 2 (done)

- PCA (300k training vectors): 512 / 1024 dims explain 77.2 / 89.0% of variance.
- Seeds embedded: 29,523 vocabulary strings, 250k bigram and 250k trigram
  surfaces; 1.89M tokens, $0.0189 (probe of 200 items first).
- Linear ceiling (504 contract, stored dim 1024, core nDCG@10): TT 59.76;
  d=512 57.76; d=1024 59.03. Seed assertion passed (max rel. error 1e-7).
- Results: `stage2/RESULTS.md`, `stage2/result.json`.

### 2026-11-12 Stage 3 (in progress)

- Training on GPU under `dibs run gpu` (ROCm env, memguard 12 GB), 23 s per
  epoch at 300k texts. GPU was held by other sessions 11:17-12:32 and again
  from ~12:50; runs queue behind them.
- L0 (val loss / core TS / dev agreement, 504 contract, stored dim 1024):
  S0 seeds only 0.4434 / 11.78 / 23.2; S1 random (lam 1e-4) 0.2482 / 37.60 / 48.2;
  S2 d512 (lam 1e-3) 0.2285 / 41.36 / 51.5; **S2 d1024 (lam 1e-3) 0.2138 / 42.22 / 53.0**.
- Width: 1024 passes the B1 test against 512 (+1.45 agreement, CI 1.15-1.73;
  +0.86 TS): keep 1024. Seeding: S2 beats S1 by +4.8 agreement, +4.6 TS.
- Observation: any lam >= 1e-4 acts as a strong pull because Adam normalizes
  the pull gradient on rows that get no other gradient in a batch; lam 1e-4
  and 1e-3 behave alike, lam 0 is clearly worse (S2 d1024 val 0.2464).
- Mixed index (best L0): G2 fails raw (recall ratio 0.73, provisional top-10
  share 26% at p = 10%) and calibrated (calibration from the round-2 split
  over-corrects: provisional share ~0%).
- L1 bigram arms warm-started from S2 d1024 lam 1e-3 (16:08-16:17, 12 runs):
  none improved on its start. Epoch 1 always raised validation loss (S3 lam 0:
  0.2138 -> 0.2400; seeded scale 0.25 starts at 0.2117 and stays there), so early
  stopping kept epoch 0. Best: S4 scale 0.25 (any lam) 0.21172 = untrained seeds
  on top of L0. Likely causes: a fresh Adam state takes near-full-size steps on
  every touched row, and L0 had already reached its validation minimum (its
  epoch 7 was worse than epoch 6), so one more epoch overfits.
- Rule fixed before the next result (16:25): rerun the same S3/S4 grid trained
  from the seeds, no warm start (`stage3/run_l1a_cold.sh`, arms `*_cold`).
  Within S3 and within S4, pick over {warm, cold} x lam (x scale) on validation
  loss; between S3 and S4 by dev agreement and TS as before. S5 uses the
  winner's start (warm or cold).
- From-seeds arms (16:17-16:31): all worse, val 0.2259-0.2380 (best epoch 1-2):
  the bigram tables overfit 300k texts. Within S3 and S4 the warm arms win:
  S3 = L0 unchanged (0.21383); **S4 = scale 0.25 (0.21172)**, i.e. untrained
  seeded bigrams on top of L0. Its B1 test against L0: agreement +0.63 (rule
  fails), TS 42.52 (+0.30).
- S5 (S4 winner + T1 trigrams seeded x 0.25, warm): epoch 1 worse for every lam
  (lam 0 went NaN in epoch 1, not yet explained); epoch 0 kept, val 0.21368
  (all lam identical). Eval: core TS 43.43, agreement 54.1, c128 41.87,
  d4096 44.28.
- **B1 PASS** (best L1 = S5 by agreement and TS): agreement +1.13 (paired 95%
  CI 0.74-1.51), core TS +1.22 over the best L0. The gain comes entirely from
  untrained seeded n-gram rows; training them overfits at 300k texts.
- S6 condition holds (B1 pass, T2 coverage +12.9 / +10.4 points); seeding won,
  so the T2 items get embedded first (`seed_embed.py --tier T2`).
- Data curve (best L1 recipe; val loss L0 -> L1 seeds): 300k 0.21383 -> 0.21368;
  600k 0.20955 -> 0.20888; all 1,173,363 0.20631 -> 0.20665. Training the n-gram
  rows still worsens epoch 1 at all rows (0.2067 -> 0.2267), and the epoch-1
  *training* loss (0.237) is above the starting loss: this is not overfitting
  (correcting the note above) but the update size. lr_sparse 1e-3 (my choice,
  not the plan's) is large next to n-gram row coordinates (~0.02).
- Rule fixed before the result (16:56): S5 recipe (warm from S4 winner, seeds
  x 0.25, lam 1e-3, 300k) with lr_sparse 1e-4 and 3e-5; these join S5 and are
  picked on validation loss. If one trains (val below 0.21368), the data curve's
  L1 step is rerun with that lr.
- Curve dev agreement (old lr, n-grams untrained): 300k 54.1, 600k 55.0, all 55.5
  (core TS 43.43 / 43.83 / 43.91; L0 at all rows 42.67 / 54.0).
- lr check: lr_sparse 1e-4 trains (S5 val 0.21368 -> 0.19872 at epoch 3).
- Rule fixed before the result (16:59): the L0 recipe (S2 d1024 lam 1e-3) is
  rerun with lr_sparse 1e-4 and 3e-5 so B1 compares equally tuned rungs; best L0
  and best L1 are each picked on validation loss over all their lr variants,
  then B1 is recomputed. Later rungs and the curve use the selected lr.
- lr results: L1 (S5 recipe) lr_sparse 3e-5 val **0.19771** (epoch 6), 1e-4
  0.19872. L0 lr_sparse 1e-4 0.21442, 3e-5 0.22029 (both stopped by the 10-epoch
  cap): best L0 stays S2 d1024 lam 1e-3 at lr 1e-3 (0.21383).
- **Best L1 = S5_d1024_lam1e-3_lrs3e-5**: core TS 46.12, agreement 55.7, c128
  44.57, d4096 46.37 (Plan A H2 at 300k: 46.37). **B1 recomputed: PASS**,
  agreement +2.75 (CI 2.35-3.17), TS +3.91. (lrs1e-4: TS 46.30, agreement 55.5.)
- T2 seeds: 750k bigram + 750k trigram surfaces, 5.60M tokens, $0.0560 (Plan B
  total $0.0751). Coordinates `c_n{2,3}_T2.npy` built.
- S6 (T2 tables, warm from best L1, new rows seeded x 0.25, lam 1e-3,
  lr_sparse 3e-5) queued after the data-curve rerun.
- S6 (19:33): val 0.19176 (epoch 1). Core TS **47.25**, agreement 56.6, c128 45.90,
  d4096 47.65. Against S5: agreement +0.87 (CI 0.60-1.14), TS +1.13. Selected as
  best L1 (the plan picks among S3-S6 by agreement and TS); note for the owner:
  on the B1 test itself the T2 extension would fail (+0.87 < 1.0), and T2 adds
  ~0.8 GB of int8 rows (T1 0.26 GB -> T2 1.03 GB).
- S7 queued: S6 + readout MLP (lam 1e-3, lr_sparse 3e-5, lr_mlp 1e-3).
  S8 = same recipe at d 512: S2 d512 -> T2 n-grams seeded x 0.25 -> + MLP.
- Data curve rerun (S5 recipe, lr_sparse 3e-5; L0 step per row count as before):
  300k val 0.19771 / agreement 55.7 / core TS 46.12; 600k 0.19160 / 56.3 / 46.57;
  all 0.18777 / 56.8 / 45.90. Rise 600k -> all = +0.5 agreement < 1.0: by the
  plan's rule the 12.5M set is not worth having for this recipe (val loss still
  falls). 300k -> all = +1.1 agreement: Stage 4 arms use all rows.
- S7 (lr_mlp 1e-3): epoch 1 worse (0.19314 vs 0.19176), early stop -> S7 = S6.
- Rule fixed before the result (20:00): S7 also at lr_mlp 1e-4 (arm S7_d1024_lrm1e-4);
  pick on validation loss; S8 uses the winning lr_mlp (rerun if 1e-4 wins).
- S7 lr_mlp 1e-4: epoch 1 worse too (train 0.182, val 0.19222): S7 = S6.
  **B2 at L2 FAILS: keep L1.** S8a (S6 recipe at d 512): val 0.20494, core TS
  46.18, agreement 55.5; S8 (+ MLP) = S8a. Width at the final recipe: 1024 vs 512
  +1.09 agreement (CI 0.82-1.37), +1.07 TS: keep 1024.
- Mixed index S6 at p = 10%: raw recall ratio 0.87 (pass), provisional top-10
  share 27.5% (fail); calibrated 0.009 / 0.0% (fail). G2 fails like Plan A's.
- **Stage 3 done**: `stage3/RESULTS.md`, `stage3/result.json`. Selected model:
  S6 (d 1024, unigrams + T2 bigram and trigram tables, linear readout), core TS
  47.25 at 504/1024, 45.90 at 128/1024, agreement 56.6.
- T2 seed vectors (`seedvec_n{2,3}_T2.npy`, 12 GB) deleted after building the
  coordinates; the API cache still holds them. Disk claim released.
- Disk: Plan B runs had grown to 93 GB (each T2-size checkpoint is 4 GB). Deleted
  all `last.pt` (resume-only; every run finished) and the best.pt of S7, S7 lrm
  and S8 (identical to their epoch-0 starts S6 / S8a): runs now 35 GB, /home
  81 GB free.
- **Checkpoint: reported to the owner, waiting before Stage 4.**

### 2026-11-12 Stage 4 (in progress)

- 20:30 Owner (in the session): "If it sounds like a good idea, just knock
  yourself out." Taken as a go-ahead for Stage 4 with the plan's defaults:
  keep T2 (the plan's selection rule), all 1,173,363 rows. Not taken as approval
  for the 12.5M set, Stage 5, or any spend beyond the cap.
- `train.py --resume` added (runs on all rows exceed 20 min): weights from
  last.pt, history from train.json, fresh optimizer state.
- Rules fixed before results: the base is re-fit on all rows (F_L1 = S6 recipe;
  F_L2 = + MLP at lr_mlp 1e-4), B2 decides each rung against the rung below on
  all rows; mixer arms use lr_dense 3e-4 (as fixed in Stage 3) and random grid
  offsets in training.
- F_L1 (all rows): val 0.18244, core TS 47.11, agreement 57.3, c128 46.29.
  F_L2: MLP never improved again (epoch 0 kept, = F_L1). B2 at L2 fails.
- 2026-11-13 ladder done (validation loss): L3 c16 0.18223; L4 c16 0.16217
  (epoch 2 blew up to 0.402); **L4 c32 0.15229** (7 good epochs, spike at
  epoch 8); L5 c32 diverged in epoch 1 (0.467), epoch 0 kept.
- Evaluated: L3 c16 TS 47.38 / agree 57.7; L4 c16 49.96 / 62.7;
  **L4 c32 TS 50.40 / agree 63.6** (c128 48.12, d4096 51.12).
- B2: L3 vs L1 +0.44 (0.20, 0.69) FAIL; L4 c32 vs L3 +5.88 (5.41, 6.33),
  TS +3.03 PASS; L4 c32 vs L1 +6.32, TS +3.29; c32 vs c16 +0.93, TS +0.44.
  Selected rung so far: L4 c32. L5 fails as measured (diverged).
- Deviation: `--clip` added to train.py (off by default). Variants queued in
  `stage4/run_variants.sh`: d512 (warm from S8a), control from Stage 2 seeds in
  one run, and L5 retried with grad clip 1.0 (counts only if it beats L4 c32 on
  validation loss). Deleted finished last.pt files (disk at 99%).
- Selected rung L4 c32, gates: B3 PASS (50.40 >= 42.28); B5 PASS (504 50.40 vs
  128 48.12); B6 PASS (TS 50.40 / 50.26 / 50.21 at offsets 0 / 5 / 11, spread
  0.19; offset-variant cosine mean 0.998, p5 0.996); B4 FAIL (raw: recall ratio
  0.846, share 17.7%; calibrated: 0.21, 0.9%). BM25 50.55 is 0.15 above TS,
  under the 3-point line. Probes (report only): kept separation negation 0.13,
  swap 0.12, reorder 0.29; entity gap kept 1.11.

- 2026-11-13 15:02 F_L4_c32_d512 done: 10 epochs (hit the cap, still improving),
  val 0.16363 (d1024 F_L4_c32: 0.15229). Not yet evaluated.
- 15:33 Paused at another session's request (the owner needs the GPU to
  benchmark): killed the variants job 30 min into the control's epoch 1 (lost;
  it restarts from scratch). run_variants.sh now skips finished arms. Resume:
  `dibs run gpu --label "Plan B stage 4 variants (control, L5 clip)" --eta 12h --
  ngram-space/stage4/run_variants.sh >> training-v2/work/ngram-space/stage4_variants.log 2>&1`
  Holding CPU-heavy evaluation too while the benchmark runs.

## Open questions for the owner (plan section 8)

1. API key / $5 cap: settled 2026-11-12.
2. Model file budget: default 2 GB on disk, memory mapped.
3. Training on the 12.5-million-pair set: not approved; not streamed.
4. Stored dimension: default 1024.
5. Order of work with Plan A: both running; shared items built by whoever claims first.
6. Stage 5 (Qwen3.8 donor): off unless asked.

7. (new, Stage 3) T2 tables: +0.87 agreement / +1.13 TS over T1 for +0.8 GB of
   int8 rows; selected by the plan's rule, but they would fail a B1-style test.
   Keep T2?
8. (new, Stage 3) The 12.5M set: the curve rises +0.5 agreement from 600k to all
   (< 1.0), so by the plan's rule it is not worth having for Plan B.
9. (new, Stage 3) Deviations to confirm: smaller sparse learning rate
   (3e-5, chosen by pre-registered checks), cold-start arms, MLP lr check.

## Next action

Stage 4: variants training on the GPU; mixed index, offsets and probes for L4 c32 running on CPU; then RESULTS and the checkpoint report.
