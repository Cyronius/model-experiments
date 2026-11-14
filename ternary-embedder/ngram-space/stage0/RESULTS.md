# Stage 0: setup (Plan B), 2026-11-12

## Shared items

| Item | State found | Plan B action |
|---|---|---|
| `code` (`shared-qwen8b/shared.py`) | claimed and DONE by Plan A 10:31 | used, not modified |
| `jaswanth` | claimed and DONE by Plan A (1,300,000 rows, revision `ef5ade98…`; the plan's 1,275,000 was the card's figure) | used, read only |
| `reserved` (`qwen-space/RESERVED.md`) | DONE by Plan A: BEIR NQ and Touché-2020 | used for the leakage list only |
| `protocol-check` | claimed and written by Plan A | read: **Jaswanth accepted** (median cos 0.99748, p5 0.99658 at 4096; 0.99767 / 0.99664 at the 1024 prefix, n = 200). Query protocol: full string `Instruct: {instruction}\nQuery:{query}`, `custom_instruction` empty |
| `leakage-jaswanth` + validation ids | unclaimed | **claimed and built by Plan B** (`stage0/build_leakage.py`), DONE marker `leakage-jaswanth/DONE`, recorded in `MANIFEST.json` |

Leakage rule as implemented (fixed before looking at matches): `shared.leak_key`
of the benchmark document `(title + " " + text).strip()` and of the training
text; equal keys match; a prefix match either way counts when the shorter key
has at least 30 characters; a training text whose key equals a document
title's key (titles of 10+ characters) matches. Benchmarks: SciFact, NFCorpus,
ArguAna (dev) and the **full** NQ and Touché-2020 corpora (confirmation).

| Benchmark | documents | Jaswanth rows matched |
|---|---|---|
| SciFact | 5,183 | 31 |
| NFCorpus | 3,633 | 3 |
| ArguAna | 8,674 | 1 |
| NQ | 2,681,468 | 218 (190 via exact title, e.g. book titles) |
| Touché-2020 | 382,545 | 20 |
| **removed rows** | | **268** |

Validation split (contract rule on the text): 10,084 rows, ids in
`training-v2/work/shared-qwen8b/splits/jaswanth-validation-ids.npy`.

## Hosted call

One call through `shared.embed(["hello", "world"], plan="B")`: 4 tokens,
$0.00000004, unit-norm 4096-d float16 vectors. Key loaded only into that
process's environment.

## Training set (`result.json`)

| | rows |
|---|---|
| Jaswanth rows | 1,300,000 |
| fit the 504 contract in full (1 to 504 content tokens) | 1,182,831 |
| dropped: longer than 504 tokens | 117,168 |
| dropped: empty text | 1 |
| validation rows that fit 504 (Plan B validation set) | 9,204 |
| leakage rows among the fitting, non-validation rows | 264 |
| **training rows** | **1,173,363** (96,441,219 tokens) |

Token length of training rows: median 33, p75 114, p90 216, p95 344, p99 468.
77.8% have at most 128 tokens. Histogram (0-16, 16-32, 32-64, 64-128, 128-256,
256-504): 399,743 / 178,420 / 149,437 / 183,581 / 171,279 / 90,904.

The set is dominated by short texts. Long-document behaviour (gate B5) is
trained on comparatively few examples; noted as a risk.

## Files

- `ngram-space/common.py`: tokenization (no special tokens), n-gram key
  packing (bigram `a<<15|b`, trigram `a<<30|b<<15|c` in uint64), shared imports.
- `ngram-space/evalkit.py`: dev evaluation (uses `round2_metrics.evaluate`,
  `diagnose.paired`; agreement@10 from the per-query top-10 lists).
- Work folder `training-v2/work/ngram-space/`:
  `jaswanth/` (codes.npy int8 [1.3M, 4096], absmax, texts, ntok, ids.u16),
  `stage0/{train,val}/ids.u16 + starts.npy + rows.npy`, `stage0/{train,val}_rows.npy`.

Commands (from `ternary-embedder/`):

```
dibs run cpu -- .venv/bin/python -I ngram-space/stage0/convert_jaswanth.py
dibs run cpu -- .venv/bin/python -I ngram-space/stage0/build_leakage.py
.venv/bin/python -I ngram-space/stage0/build_train.py
.venv/bin/python -I ngram-space/stage0/train_memmap.py
```

`result.json` keys: rows, fit_504, dropped_over_504, dropped_empty,
validation_split, validation_fit_504, leakage_rows,
leakage_and_fit_and_not_validation, train_rows, train_tokens,
train_token_length_percentiles, train_share_le_128, train_share_le_126,
all_rows_token_length_percentiles, histogram_edges, train_histogram.
