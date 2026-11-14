# Stage 2: seeds from the hosted teacher (2026-11-12)

## Teacher calls (Plan B, through `shared.embed(plan="B")`, no instruction, bare strings)

| Group | items | sent (non-empty) | unique strings |
|---|---|---|---|
| vocabulary ("##" stripped; special and `[unused*]` skipped) | 30,522 | 29,523 | 26,814 |
| T1 bigram surfaces | 250,000 | 250,000 | 247,122 |
| T1 trigram surfaces | 250,000 | 250,000 | 249,355 |

- Probe first: 200 items, 577 tokens, $0.0000058 (2.9 tokens per item), as
  the plan asks; usage matched the estimate, so the rest went ahead.
- Full run: 1,885,478 tokens, **$0.0189**. (The first full run used the
  default 4 parallel requests and was stopped after ~8,400 items to restart
  with 12; nothing was paid twice: the cache kept what was done.)
- Plan B total after Stage 2: 16,271 calls, 1,909,481 tokens, **$0.0191** of
  the $5 cap.

## PCA basis (step 2)

300,000 random training vectors (seed 0; the same rows start Stage 3),
dequantized and re-normalized. `|mu|` = 0.520. The first 512 directions
explain 77.2% of the centered variance, 1024 explain 89.0%. Validation
vectors projected to rank d and mapped back (`normalize(V V^T (t - mu) + mu)`):
mean cosine 0.912 (p5 0.858) at d = 512, 0.959 (p5 0.932) at d = 1024.

## Linear ceiling (step 3)

Dev documents' teacher vectors replaced by their rank-d projection; teacher
queries. nDCG@10, core = mean of SciFact and NFCorpus; agreement@10 over all
queries of the three tasks.

| | contract | stored dim | SciFact | NFCorpus | ArguAna | **core** | dev agreement |
|---|---|---|---|---|---|---|---|
| TT | 504 | 1024 | 78.27 | 41.24 | 76.63 | **59.76** | 100 |
| TT | 504 | 4096 | 77.90 | 41.44 | 76.57 | 59.67 | 100 |
| TT | 128 | 1024 | 74.96 | 38.36 | 73.17 | 56.66 | 100 |
| linear d = 512 | 504 | 1024 | 75.58 | 39.94 | 75.65 | **57.76** | 87.0 |
| linear d = 512 | 504 | 4096 | 77.15 | 40.24 | 76.00 | 58.70 | 90.1 |
| linear d = 512 | 128 | 1024 | 73.96 | 38.33 | 71.95 | 56.14 | 87.0 |
| linear d = 1024 | 504 | 1024 | 77.13 | 40.93 | 76.16 | **59.03** | 92.4 |
| linear d = 1024 | 504 | 4096 | 77.90 | 41.02 | 76.41 | 59.46 | 95.4 |
| linear d = 1024 | 128 | 1024 | 75.24 | 38.46 | 72.56 | 56.85 | 92.3 |

TT matches Plan A's Stage 1 table exactly (59.76 core at 504/1024). No model
of this plan with a linear readout can beat the d rows: at the 504 contract
and stored dimension 1024 the ceilings are **57.76 (d = 512)** and **59.03
(d = 1024)** core, 87.0 and 92.4 dev agreement.

## Seeds (step 4) and the numerical assertion

`c(x) = V^T (teacher(x) - mu)`; `u[t] = c(t)`; `w[t] = a / (a + p[t])`,
`a = 0.001` (median 0.995, minimum 0.020 for "the"); bigram and trigram
residuals as in the plan (trigram residuals subtract the T1 bigram residuals
already in the table). Mean row norms: u 0.662, bigram residual 0.429,
trigram residual 0.432.

Assertion on 1,000 random seeded items (400 unigrams, 300 T1 bigrams, 300 T1
trigrams), each fed alone through the model code (`ngram-space/bmodel.py`):
the representation equals the seed target `c(x)` to a maximum relative error
of 9.7e-8 at d = 512 and d = 1024 (for d = 1024 the n-gram residual carries
the first 512 coordinates, so n-gram items are compared there; unigrams on
all 1024). Minimum output cosine against the teacher's rank-d projection:
0.9999999. **Passed.**

## Files and commands

Work folder `training-v2/work/ngram-space/stage2/`: `mu.npy`, `V.npy`
(4096 x 1024), `eigvals.npy`, `train300k_rows.npy`, `seedvec_*.npy` (teacher
vectors, float16), `c_vocab.npy`, `c_n2_T1.npy`, `c_n3_T1.npy`, `w_seed.npy`,
`seeds.json`, `pca.json`.

```
.venv/bin/python -I ngram-space/stage2/pca.py
DEEPINFRA_API_KEY=... .venv/bin/python -I ngram-space/stage2/seed_embed.py --probe 200
DEEPINFRA_API_KEY=... .venv/bin/python -I ngram-space/stage2/seed_embed.py --tier T1 --workers 12
.venv/bin/python -I ngram-space/stage2/seeds.py
.venv/bin/python -I ngram-space/stage2/ceiling.py      # cache hits only
```

`result.json` keys: pca, seed_items, seed_spend, spend_B_total_after_stage2,
seeds, ceiling (per arm TT / linear_d512 / linear_d1024, per contract and
stored dim: core_ndcg10, dev_agreement10, SciFact, NFCorpus, ArguAna).
