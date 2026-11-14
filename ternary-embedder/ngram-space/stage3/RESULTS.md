# Stage 3: rungs L0-L2, gates B1-B2, data curve (2026-11-12)

All arms train on the Plan B Stage 0 training set (300,000 random texts unless
noted), GPU, ROCm env, `dibs run gpu`, memguard 12 GB. Validation: the 9,204
Plan B Jaswanth validation texts. Evaluation: dev tasks, teacher queries, core
= mean nDCG@10 of SciFact and NFCorpus; dev agreement@10 pooled over SciFact,
NFCorpus, ArguAna queries; paired bootstrap seed 0, 4,000 resamples.

References (504 contract, stored dim 1024): TT 59.76; linear ceiling d = 1024
59.03 (d = 512 57.76); SS384 Q8 windowed 48.28 (published 128-contract Q8 47.48,
the B3 reference); BM25 50.55; Plan A TS 47.12 (full set), 46.37 (300k).

## Main table

| Arm | what | val loss | train (sample) | core TS 504/1024 | agree | 504/4096 | 128/1024 | 128/4096 |
|---|---|---|---|---|---|---|---|---|
| S0 | seeds only, no training | 0.4434 | – | 11.78 | 23.2 | 14.21 | 15.44 | 17.38 |
| S1 lam 1e-4 | L0 random init (selected lam) | 0.2482 | 0.2296 | 37.60 | 48.2 | 36.78 | 36.37 | 35.86 |
| S2 d512 lam 1e-3 | L0 seeded, d 512 | 0.2285 | 0.2127 | 41.36 | 51.5 | 40.86 | 39.86 | 39.65 |
| **S2 d1024 lam 1e-3** | **best L0** | 0.2138 | 0.1974 | **42.22** | 53.0 | 42.14 | 40.17 | 40.29 |
| S4 s0.25 | + T1 bigram seeds x 0.25 (untrained) | 0.2117 | 0.1980 | 42.52 | 53.6 | 42.81 | 41.28 | 41.45 |
| S5 (lr 1e-3) | + T1 trigram seeds x 0.25 (untrained) | 0.2137 | 0.2009 | 43.43 | 54.1 | 44.28 | 41.87 | 42.89 |
| S5 lrs 1e-4 | S5 trained, lr_sparse 1e-4 | 0.1987 | 0.1701 | 46.30 | 55.5 | 46.46 | 44.18 | 45.22 |
| S5 lrs 3e-5 | S5 trained, lr_sparse 3e-5 | 0.1977 | 0.1719 | 46.12 | 55.7 | 46.37 | 44.57 | 45.15 |
| **S6** | **S5 lrs 3e-5 extended to T2 (best L1)** | **0.1918** | 0.1663 | **47.25** | **56.6** | 47.65 | 45.90 | 46.64 |
| S7 | S6 + readout MLP (lr_mlp 1e-3 and 1e-4) | 0.1918 (epoch 0 kept) | | = S6 | | | | |
| S8a | the S6 recipe at d 512 | 0.2049 | 0.1813 | 46.18 | 55.5 | 47.03 | 45.64 | – |
| S8 | S8a + MLP | 0.2049 (epoch 0 kept) | | = S8a | | | | |

Every arm's per-task numbers, all four contract x dimension cells, and the
training histories are in `arms/<arm>.json` and `result.json`.

## Gates and rules

| Test | agreement delta (95% CI) | core TS delta | result |
|---|---|---|---|
| Width at L0: 1024 vs 512 | +1.45 (1.15, 1.73) | +0.86 | **keep 1024** |
| **B1**: best L1 (S5 lrs 3e-5) vs best L0 | +2.75 (2.35, 3.17) | +3.91 | **PASS** |
| (B1 with the first, untrained S5) | +1.13 (0.74, 1.51) | +1.22 | pass |
| S6 (T2) vs S5 (T1), for information | +0.87 (0.60, 1.14) | +1.13 | would fail the B1 test |
| Width at the final L1 recipe: S6 vs S8a | +1.09 (0.82, 1.37) | +1.07 | keep 1024 |
| **B2 at L2**: S7 vs S6 | 0 (identical model) | 0 | **FAIL: keep L1** |

S6 was picked as the best L1 because the plan selects among S3-S6 by agreement
and TS, and S6 is ahead on both. The T2 tables would not pass the B1 test on
their own (+0.87 < 1.0) and grow the int8 n-gram rows from 0.26 GB to 1.03 GB.
That trade is flagged for the owner.

## Seeding (reported plainly, as the plan asks)

- S1 vs S2 (unigrams): seeding helps after training: +4.8 agreement, +4.6 TS
  (S2 d1024 lam 1e-3 vs S1 lam 1e-4).
- S3 vs S4 (bigrams): every trained S3 and S4 arm at lr_sparse 1e-3 got worse
  than its start, so these arms compare *starting points*: zero bigram rows
  (= the L0 model, 0.2138) vs seeded rows x 0.25 (0.2117, +0.6 agreement). Seeds
  help; the scale 0.25 won over 0.5 and 1.0 on validation loss. The seeded
  n-gram start was kept for every later arm.

## What went wrong on the way, and the deviations

1. **Warm-started L1 arms never trained** (16:08, 12 runs): epoch 1 always
   raised validation loss, and early stopping kept epoch 0.
2. **Cold L1 arms** (trained from seeds, no warm start; added by a rule written
   before their result): all worse, val 0.226-0.238.
3. I first called this overfitting. The data curve disproved that: at all
   1.17M texts the epoch-1 *training* loss (0.237) was above the starting loss.
   The cause was the **sparse learning rate**: 1e-3 (my choice, the plan fixes
   none) is large next to n-gram row coordinates (~0.02), and Adam takes
   near-full-size steps on every row it touches.
4. **Learning-rate check**, rules written before results: lr_sparse 1e-4 and
   3e-5 for L1 (S5 recipe) and for L0, best of each rung on validation loss,
   then B1 recomputed. L1 trains well at 3e-5 (0.2137 -> 0.1977); L0 is worse
   at the smaller rates (0.2144, 0.2203, both stopped by the 10-epoch cap), so
   the best L0 is unchanged. B1 was recomputed on equally tuned rungs.
5. **MLP (S7, S8) never improved** at lr_mlp 1e-3 or 1e-4: training loss falls
   (0.182) while validation loss rises. S6 itself was best at epoch 1 of its
   run, so the model is already at its validation minimum on 300k texts and any
   further epoch overfits. B2 fails as measured. Retesting the MLP on all rows
   is an option for Stage 4, but it is not what the plan's S7 measures.
6. S5 at lam 0 (lr 1e-3) went NaN in epoch 1. Not explained; it did not affect
   any selection. A likely cause is the weight sum W collapsing toward zero for
   some document without a pull term.
7. Two GPU jobs overlapped once (the curve rerun and S6, 19:33), because the
   curve's hold showed free while its process was still training. Both finished
   normally; the GPU footprint stayed far under the 60 GB limit.

## Data curve (best L1 recipe, T1 n-grams, lr_sparse 3e-5)

| texts | val loss | dev agreement | core TS |
|---|---|---|---|
| 300,000 | 0.1977 | 55.7 | 46.12 |
| 600,000 | 0.1916 | 56.3 | 46.57 |
| 1,173,363 (all) | 0.1878 | 56.8 | 45.90 |

The rise from 600k to all is **+0.5 agreement, under the plan's 1.0-point
threshold**: by the plan's rule the 12.5-million-pair set is not worth having
for this recipe. Validation loss is still falling. Core TS is noisy (it drops at
all rows while agreement and loss improve). Stage 4 arms use all rows (+1.1
agreement over 300k). (With the first, untrained-n-gram recipe the curve was
54.1 / 55.0 / 55.5.)

## Mixed index (the G2 test of Plan A, gate B4's definition), p = 10%

| Arm | variant | recall ratio (>= 0.85) | provisional top-10 share (<= 15%) | G2 |
|---|---|---|---|---|
| S0 | raw | 0.02 | 1.8% | fail |
| best L0 | raw | 0.73 | 26.4% | fail |
| best L0 | calibrated (a 0.528, b 0.056) | 0.003 | 0.0% | fail |
| **S6** | raw | **0.87** | 27.5% | **fail** |
| S6 | calibrated (a 0.522, b 0.070) | 0.009 | 0.0% | fail |

S6 keeps relevant provisional documents in the top 10 (recall ratio passes)
but its vectors score too high against unrelated queries and take 27.5% of
top-10 slots. The calibration fitted on the round-2 validation split (short
passages, < 127 tokens) over-corrects on dev documents. Same failure as Plan A.

## Files and commands

Scripts in this folder: `run_l0.sh`, `run_l1a.sh` (warm S3/S4),
`run_l1a_cold.sh`, `run_l1b.sh` (S5), `run_lr.sh`, `run_l0_lr.sh`,
`run_curve.sh`, `run_curve_lr.sh`, `run_s6.sh`, `run_s7_s8.sh`, `run_s7_lr.sh`;
evaluation `../eval_arm.py ARM [--mixed]`, paired tests `../gate.py SMALL LARGE`.
Runs: `training-v2/work/ngram-space/runs/<arm>/` (args, best/last checkpoints,
train.json, eval rows). Logs: `training-v2/work/ngram-space/stage3_*.log`.
