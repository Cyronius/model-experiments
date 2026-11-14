# Broad-data controlled comparison

All four training arms completed. The paid pod is deleted with HTTP404 verification. Estimated total cost: **$3.52**, within the approved **$10** cap.

Architecture: three layers, 384 dimensions, 17,241,984 parameters. Each arm saw 3,771,503 pair presentations. Q4/Q8 are fixed exports from the independent float branch.

| Candidate | SciFact | NFCorpus | FiQA | SCIDOCS | STSBenchmark | Fixed12 mean |
|---|---:|---:|---:|---:|---:|---:|
| broad-float | 61.17 | 31.56 | 27.03 | 17.73 | 82.87 | 70.70 |
| broad-q4 | 61.17 | 31.84 | 26.45 | 17.57 | 82.68 | 70.62 |
| broad-q8 | 61.55 | 31.67 | 26.92 | 17.72 | 82.82 | 70.74 |
| broad-ternary | 60.12 | 31.17 | 25.12 | 16.90 | 81.85 | 69.61 |
| parent-float | 62.07 | 31.98 | 26.81 | 17.74 | 83.99 | 71.96 |
| parent-ternary | 59.87 | 30.59 | 24.84 | 16.79 | 82.45 | 70.48 |
| repeat-float | 61.81 | 31.83 | 26.15 | 17.96 | 82.79 | 70.98 |
| repeat-q4 | 60.61 | 31.78 | 25.74 | 17.92 | 82.65 | 70.69 |
| repeat-q8 | 61.66 | 32.05 | 26.38 | 17.86 | 82.78 | 70.97 |
| repeat-ternary | 60.07 | 30.56 | 24.17 | 17.05 | 81.42 | 69.67 |
| bge_micro | 61.33 | 31.10 | 23.52 | 17.52 | 82.50 | 70.90 |
| minilm | 62.39 | 31.45 | 36.63 | 21.72 | 82.03 | 71.48 |

## Broader data versus repeated old data

| Precision | Core retrieval delta (95% query CI) | FiQA/SCIDOCS delta (95% query CI) | STS delta | Fixed12 delta |
|---|---:|---:|---:|---:|
| float | -0.456 [-1.139, +0.202] | +0.320 [-0.054, +0.696] | +0.079 | -0.274 |
| ternary | +0.333 [-0.411, +1.075] | +0.399 [-0.091, +0.906] | +0.428 | -0.062 |
| q4 | +0.309 [-0.391, +1.001] | +0.179 [-0.174, +0.541] | +0.024 | -0.064 |
| q8 | -0.249 [-0.894, +0.403] | +0.199 [-0.181, +0.568] | +0.044 | -0.225 |

These contrasts compare complete data recipes at matched pair presentations. Source composition, token lengths and compute differ. Query intervals do not establish training-seed stability or full-MTEB superiority.

No trained candidate passed every predeclared development gate. Full new reserved judgments were left unopened. The recorded Touche query1 preview was excluded from the planned confirmation.

All artifacts remain experimental; deployment was not changed. Detailed selections, scores and intervals: [development-summary.json](results/development-summary.json).

Free follow-up diagnostics and interpretation: [ANALYSIS.md](ANALYSIS.md).
