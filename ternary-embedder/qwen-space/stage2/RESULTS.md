# Stage 2: head-only Qwen-space test (2026-11-12)

Frozen Q8 backbone (`model-q8.pt` via `load_reference`), new head into the 4096-d space of hosted
Qwen3-Embedding-8B, fit on public Jaswanth vectors (protocol check: accepted). No backbone training.
Scripts: `prep_index.py`, `prep_data.py`, `run.py` (arms, 300k), `prep_full.py` + `full_fit.py`
(full-set extension), `heads.py`. Rules fixed before results: `qwen-space/STATUS.md`.

**Verdict: G1 pass, G3 pass, G2 fail (narrow-fail fallback: marked at risk, continue to Stages 3-4).
BM25 beats TS by 3.4 points: a lexical or hybrid provisional tier is competitive.**

## Data

- Jaswanth: 1,300,000 rows. Dropped: 117,168 over the 504 contract, 266 rows (197 distinct texts)
  in the shared leakage list (dev tasks + NQ + Touché corpora; built by Plan B), 1 empty.
- Validation (contract rule, `is_validation`): 9,202 texts that fit the contract. Never trained on.
- Training: seed-0 sample of 300,000 for arms 1-4 and the curve; the full pool of 1,173,363 for the
  extension. Kept lengths: median 33 content tokens, p95 344, p99 468; only 22.5% exceed 126 tokens.
  Distribution in `data-stats.json`.

## Arms (core = mean of SciFact and NFCorpus nDCG@10; 504 contract)

Validation loss = mean over prefixes 256/512/1024/4096 of 1 - cosine.

| Arm (300k) | reading | val loss | core TS @1024 | core TS @4096 | core TS 128 @1024 |
|---|---|---:|---:|---:|---:|
| H1 ridge | windowed | 0.2658 | 40.50 | 43.74 | 38.71 |
| H1 ridge | native-512 | 0.2646 | 41.66 | 44.98 | 38.55 |
| H2 MLP | windowed | 0.2152 | 45.04 | 46.44 | 43.50 |
| **H2 MLP** | **native-512** | **0.2148** | **46.37** | 48.03 | 43.74 |
| H3 MLP + margin | windowed | 0.2193 | 44.49 | 45.75 | 43.15 |
| H3 MLP + margin | native-512 | 0.2188 | 45.99 | 47.27 | 43.06 |
| H4 ridge on float features | windowed | 0.2657 | 40.26 | 43.67 | 38.57 |

- Reading mode by validation loss: native-512 for every arm (unexpected: the plan expected the
  untrained positions 128-511 from bge-micro-v2 to be worse; they win by a little on loss and by
  1.2-1.5 points of TS). So a native long path does not need backbone training to beat windowing here.
- Selected arm (best core TS at 504/1024 among H1-H3): **H2, native-512**.
- H3's margin term (round-2 triplets) did not help. H4: the quantized backbone costs nothing (Q8
  ridge 40.50 vs float ridge 40.26).
- ridge penalty: every lambda from 1e-5 to 1e-2 gives the same loss (n >> parameters).

**Data curve** (H2 native): 50k 41.71 -> 100k 44.58 -> 300k 46.37 -> full 1,173,363 **47.12**
(val loss 0.256, 0.236, 0.215, 0.201). Still rising at 300k, so it was extended (plan arm 5); the
full-set fit has the lower validation loss and is the selected head
(`training-v2/work/qwen-space/stage2/head_selected.npz`, copy `head_H2_q8_native_full.npz`).
The curve is still rising at 1.17M; more data (the 12.5M set) would likely add more (not approved).

## Selected head on the dev tasks (nDCG@10)

| | SciFact | NFCorpus | ArguAna | core |
|---|---:|---:|---:|---:|
| TT 504 @1024 (ceiling) | 78.27 | 41.24 | 76.63 | 59.76 |
| **TS 504 @1024** | 64.42 | 29.83 | 51.56 | **47.12** |
| TS 504 @4096 | 67.00 | 30.26 | 52.10 | 48.63 |
| TS 128 @1024 | 59.73 | 29.61 | 50.52 | 44.67 |
| TS 128 @4096 | 60.89 | 30.13 | 51.82 | 45.51 |
| SS384 Q8 504 windowed | 65.29 | 31.26 | 45.69 | 48.28 |
| BM25 504 | 68.93 | 32.17 | 49.37 | 50.55 |

Paired TS - SS384 at 504 / 1024 (95% bootstrap, `diagnose.paired`): SciFact -0.87 [-4.25, +2.57],
NFCorpus -1.43 [-3.57, +0.65], ArguAna +5.86 [+4.14, +7.60]. At 4096: SciFact +1.70 [-1.57, +5.08],
NFCorpus -1.00 [-3.00, +1.01], ArguAna +6.41 [+4.73, +8.07]. All cells in `result.json`.

Cosine provisional vs refined (504, 1024 dims): mean 0.776 / 0.790 / 0.743 (SciFact / NFCorpus /
ArguAna), 5th percentile 0.719 / 0.739 / 0.670. At 4096: mean 0.758 / 0.775 / 0.733.

## Mixed index (504 contract, 1024 dims, teacher queries, five seeds)

Raw provisional scores (no calibration):

| p | | SciFact | NFCorpus | ArguAna |
|---|---|---:|---:|---:|
| 0 (all refined) | nDCG@10 | 78.27 | 41.24 | 76.63 |
| 1% | nDCG@10 / recall ratio / top-10 share | 78.05 / 1.20 / 4.2% | 41.07 / 0.67 / 2.5% | 76.34 / 0.82 / 2.3% |
| 10% | | 75.72 / 0.86 / 25.2% | 39.36 / 0.72 / 20.5% | 73.43 / 0.83 / 17.6% |
| 50% | | 68.90 / 0.88 / 65.0% | 34.45 / 0.72 / 60.1% | 62.22 / 0.82 / 57.7% |

Recall ratio = recall@10 of relevant provisional documents / recall@10 of the same documents when
refined. Top-10 share = share of top-10 slots held by provisional documents.

**Calibration** (`a*s + b` on the round-2 cache validation split, 385 queries x 770 passages):
a = 0.850, b = -0.023. It over-corrects badly on the dev tasks: at p = 10% the recall ratio falls to
0.17 and the share to 0.5%; nDCG drops (SciFact 72.38, ArguAna 70.45). The fit data are short
(< 128 Qwen tokens) MS MARCO / NLI / Quora texts, unlike these documents. Provisional vectors
score too high against unrelated queries (they are pulled toward the mean of the space) and slightly
too low for their own relevant queries, and a single affine map cannot fix both.

## Gates (fixed in advance; core; 1024 dims; 504 contract)

| Gate | Value | Threshold | Result |
|---|---|---|---|
| G1 alignment cost | TS 47.12 vs SS384 48.28: **-1.15** | >= -2.0 | **pass** |
| G2a recall ratio, p = 10% | raw **0.790** (calibrated 0.170) | >= 0.85 | **fail** |
| G2b provisional top-10 share, p = 10% | raw **22.9%** (calibrated 0.5%) | <= 15% | **fail** (raw) |
| G3 long inputs | TS 504 47.12 vs TS 128 44.67: **+2.45** | >= -0.5 | **pass** |
| Stop rule | TS < SS384 - 6 | | not triggered |
| BM25 vs TS | BM25 50.55, TS 47.12: **+3.43** | > 3 means "say so" | **lexical / hybrid tier competitive** |

G2 uses the raw variant: the rule was "calibrated if it passes, else raw"; neither passes.
(`run.py` first picked the calibrated variant by a coding error; corrected in code and in
`result.json`, and noted in STATUS.md.)

Fallback applied: G2 fails narrowly, so the result is **at risk**: in a mixed index at 10%
provisional documents, provisional documents crowd about 23% of the top-10 slots and lose about
21% of their relevant hits. Stages 3 and 4 continue.

## Spend

Stage 2 made no hosted calls. Plan A total so far: **$0.302** (30,170,225 tokens; includes the
Touché confirmation-task vectors fetched early, $0.23) plus $0.00008 planning.

## `result.json` keys (read by Plan B; do not rename)

`arms.{H1,H2,H3}_q8_{windowed,native}`, `arms.H4_float_windowed`, `arms.H2_q8_native_full`
(each `fit` + `eval`); `eval` keys `{task}|TS_{128,504}_{1024,4096}` (`{ndcg10, recall10}`),
`{task}|cos_prov_ref_{128,504}`, `core|TS_{c}_{k}` (float). `mode_by_val_loss`, `selected_arm`,
`data_curve`, **`final_eval`** (the selected head: Plan B's `TS`), `paired_TS_minus_SS384`,
`calibration`, `mixed_index_504_1024.{task}.p{1,10,50}_{raw,cal}`, `gates`, `selected_head`,
`leakage_refilter_dropped`, `spend_plan_A_after`. The 300k-only snapshot is `result-300k.json`.
