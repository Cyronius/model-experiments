# Stage 3: which documents to refine (2026-11-12)

Script: `run.py` (`dibs run cpu -- .venv/bin/python qwen-space/stage3/run.py`). Selected Stage 2 head
(H2 MLP, native-512 reading, full-set fit), dev tasks, 504 contract, stored dimension 1024, raw
provisional scores (the G2 variant; calibration hurt). A fraction f of documents is refined (teacher
vector) according to each policy; the rest stay provisional; nDCG@10 against teacher queries.

All policies are scored on the same half of the queries (random permutation, seed 0, odd positions:
150 SciFact, 161 NFCorpus, 703 ArguAna); the other half is the history of the query-driven policy.
Random is the mean of five seeds.

## Curves (nDCG@10 at f = 0, 5, 10, 25, 50, 100%)

Core mean (SciFact, NFCorpus):

| Policy | 0 | 5 | 10 | 25 | 50 | 100 | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 46.92 | 47.59 | 47.73 | 49.84 | 52.77 | 60.79 | 53.28 |
| lowest head confidence (norm) | 46.92 | 46.64 | 46.59 | 48.52 | 51.37 | 60.79 | 52.33 |
| longest first (token count) | 46.92 | 46.96 | 47.55 | 47.67 | 49.86 | 60.79 | 51.70 |
| highest [UNK] share | 46.92 | 46.43 | 46.66 | 47.87 | 52.89 | 60.79 | 52.76 |
| highest share of words split into >= 3 pieces | 46.92 | 46.78 | 47.56 | 49.84 | 53.05 | 60.79 | 53.33 |
| **query-driven** (provisional top-20 of past queries) | 46.92 | 47.67 | 47.79 | 50.79 | 52.77 | 60.79 | **53.48** |
| oracle (largest 1 - cos(prov, ref)) | 46.92 | 46.36 | 46.66 | 49.76 | 54.23 | 60.79 | 53.64 |

All three tasks: random 48.71 / 49.36 / 49.73 / 52.20 / 56.17 / 66.06; query-driven 48.71 / 49.94 /
50.38 / 54.09 / 58.10 / 66.06 (the query-driven gain is clearest on ArguAna: +5.8 at 25%).
Per-task curves are in `result.json`.

Fraction refined to reach 95% of the fully refined score (core, linear interpolation): random 0.81,
query-driven 0.81, split3 0.80, [UNK] 0.81, confidence 0.84, longest 0.86, oracle 0.77.
To close 95% of the gap between all-provisional and all-refined: about 0.96 for every policy.

## Findings

- **No practical policy is clearly better than random on the core tasks.** Query-driven is best by
  AUC (+0.2 core, +1.1 over all three tasks), within noise on SciFact/NFCorpus and clearly ahead on
  ArguAna. Head confidence and length are worse than random.
- **The "oracle" is not an upper bound here**, and refining the first 5% can lower nDCG (SciFact
  63.58 -> ~61-63 for most policies). Cause: provisional vectors score too high against unrelated
  queries (Stage 2, G2), so a refined relevant document can fall behind inflated provisional ones.
  Choosing which documents to refine matters less than this score inflation.
- Head confidence correlates only weakly with provisional-refined cosine (Pearson 0.29-0.36).
- Reaching 95% of the refined score needs about 80% of documents refined, under any policy.

## Recommendation

**Query-driven refinement:** log, per provisional document, how often it enters the top-20 of real
queries, and refine the most frequent first; break ties (and handle documents never retrieved) in
arrival order or at random. **The runtime does not need to return any signal for this policy**;
it comes from the search log. The runtime still returns the head-output norm and the content-token
count for diagnostics. Given the score inflation, the bigger lever is to refine as quickly as the
budget allows (the hosted cost is about $0.01 per million tokens, about $0.004 per 1,000 documents
of 400 tokens).

## `result.json` keys

`tasks.{task}.curves.{policy}` (list over `fractions`), `tasks.{task}.conf_corr_with_cos`,
`core_mean_curves`, `all3_mean_curves`, `auc_core`, `reach_core.{policy}.{gap95,score95}`,
`best_practical_by_auc_core`, `recommended_policy`, `recommended_signal`, `calibration_variant`.
