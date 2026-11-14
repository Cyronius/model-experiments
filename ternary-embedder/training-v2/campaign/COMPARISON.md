# Frozen campaign comparisons

Observed development results,128-token cap. The12-task mean is not full MTEB.

| Candidate | Core retrieval | SciFact | NFCorpus | STSBenchmark |12-task mean | Both core references |
|---|---:|---:|---:|---:|---:|---|
| human-mix/float | 46.63 | 61.84 | 31.42 | 83.92 | 71.99 | no |
| final-float-blends/float-50 | 47.02 | 62.07 | 31.98 | 83.99 | 71.96 | no |
| delivery/float | 47.02 | 62.07 | 31.98 | 83.99 | 71.96 | no |
| final-float-blends/float-25 | 46.96 | 62.10 | 31.83 | 83.97 | 71.85 | no |
| float-blends/float-25 | 47.02 | 62.13 | 31.91 | 83.94 | 71.78 | no |
| float-blends/float-50 | 47.05 | 62.12 | 31.98 | 83.78 | 71.70 | no |
| float-blends/float-75 | 46.93 | 61.74 | 32.12 | 83.38 | 71.46 | no |
| nli-aux/float-0.25 | 45.34 | 59.34 | 31.34 | 82.96 | 71.15 | no |
| science-float/3-float-fresh | 46.67 | 61.87 | 31.47 | 82.68 | 71.11 | no |
| scales/float-low-lr | 46.43 | 61.49 | 31.38 | 82.61 | 71.09 | no |
| nli-aux/float-0.1 | 45.71 | 60.06 | 31.36 | 83.00 | 71.06 | no |
| science-float/float-fresh | 45.99 | 60.92 | 31.06 | 82.65 | 70.86 | no |
| science-float/float-repeat | 45.18 | 59.49 | 30.87 | 82.70 | 70.65 | no |
| batch1024/float-1e-5 | 45.78 | 60.06 | 31.51 | 82.47 | 70.65 | no |
| batch1024/predata-float-4e-5 | 46.03 | 60.24 | 31.83 | 82.37 | 70.59 | no |
| nli-aux/float-0.0 | 46.03 | 60.24 | 31.83 | 82.37 | 70.59 | no |
| hardneg/float-pairs | 46.03 | 60.24 | 31.83 | 82.37 | 70.59 | no |
| hardneg/float-hard | 45.23 | 59.44 | 31.02 | 82.22 | 70.43 | no |
| batch1024/float-4e-5 | 46.14 | 60.42 | 31.86 | 82.07 | 70.42 | no |
| depth3/2-float | 45.28 | 59.38 | 31.18 | 82.28 | 70.41 | no |
| depth3/3-float | 43.06 | 56.27 | 29.84 | 81.86 | 69.57 | no |
| delivery/q4 | 46.67 | 61.33 | 32.01 | 83.64 | 71.67 | no |
| q4-groups/q4-g16 | 46.43 | 60.91 | 31.94 | 83.58 | 71.46 | no |
| q4-groups/q4-g32 | 46.31 | 60.74 | 31.88 | 83.56 | 71.46 | no |
| q4-mse/q4-mse | 45.41 | 59.30 | 31.52 | 83.54 | 71.28 | no |
| delivery/q8 | 47.48 | 62.97 | 31.98 | 83.98 | 72.02 | yes |
| human-mix/ternary-low-lr | 44.54 | 58.85 | 30.22 | 82.69 | 70.75 | no |
| human-mix/ternary | 43.70 | 57.55 | 29.85 | 82.62 | 70.56 | no |
| ternary-blends/ternary-2-25 | 44.59 | 58.60 | 30.58 | 81.09 | 70.53 | no |
| ternary-blends/ternary-3-25 | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| role-projection/shared-1 | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| role-projection/shared-10 | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| role-projection/query-1 | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| role-projection/query-10 | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| delivery/ternary | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| projection-blend/projection-0.1 | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| projection-blend/projection-1 | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| projection-blend/projection-10 | 45.23 | 59.87 | 30.59 | 82.45 | 70.48 | no |
| ternary-blends/ternary-3-50 | 45.02 | 59.46 | 30.59 | 82.41 | 70.40 | no |
| ternary-blends/ternary-2-50 | 44.52 | 58.34 | 30.70 | 81.20 | 70.38 | no |
| ternary-blends/ternary-3-75 | 45.00 | 59.52 | 30.48 | 82.11 | 70.29 | no |
| projection/projection-1 | 44.53 | 59.37 | 29.68 | 82.27 | 70.29 | no |
| projection/projection-0.1 | 44.34 | 59.13 | 29.56 | 82.30 | 70.27 | no |
| ternary-blends/ternary-2-75 | 44.98 | 59.31 | 30.64 | 81.41 | 70.27 | no |
| projection/projection-10 | 44.61 | 59.42 | 29.80 | 82.08 | 70.22 | no |
| science-ternary/3-ternary-fresh | 44.68 | 59.34 | 30.02 | 81.55 | 70.08 | no |
| scales/3-scales-only | 44.74 | 59.52 | 29.96 | 81.51 | 70.07 | no |
| scales/fixed | 44.72 | 59.24 | 30.20 | 81.36 | 70.06 | no |
| scales/3-learned | 44.21 | 58.65 | 29.78 | 81.45 | 70.06 | no |
| scales/learned | 44.97 | 59.54 | 30.41 | 81.34 | 70.02 | no |
| nli-aux/ternary-0.25 | 44.10 | 57.71 | 30.49 | 81.59 | 70.01 | no |
| science-ternary/ternary-fresh | 44.63 | 58.79 | 30.46 | 81.43 | 69.97 | no |
| nli-aux/ternary-0.1 | 44.20 | 58.01 | 30.39 | 81.66 | 69.81 | no |
| science-ternary/ternary-repeat | 44.36 | 58.67 | 30.06 | 81.46 | 69.77 | no |
| depth3/2-ternary | 44.35 | 58.12 | 30.58 | 81.09 | 69.39 | no |
| batch1024/ternary-1e-5 | 44.28 | 57.87 | 30.68 | 81.06 | 69.39 | no |
| hardneg/ternary-pairs | 44.79 | 58.36 | 31.23 | 80.82 | 69.38 | no |
| batch1024/ternary-4e-5 | 44.72 | 58.52 | 30.91 | 80.92 | 69.30 | no |
| hardneg/ternary-hard | 43.75 | 57.38 | 30.13 | 80.79 | 69.25 | no |
| nli-aux/ternary-0.0 | 44.56 | 58.41 | 30.71 | 80.84 | 69.23 | no |
| batch1024/predata-ternary-4e-5 | 44.56 | 58.41 | 30.71 | 80.84 | 69.23 | no |
| depth3/3-ternary | 43.58 | 57.27 | 29.89 | 80.92 | 68.98 | no |

Reference core means: bge-micro46.21, MiniLM46.92. Wider means: bge-micro70.90, MiniLM71.48. The strict core gate requires a higher score on each task individually, not only their average.

This script does not score FiQA/SCIDOCS judgments or establish seed stability. The separate frozen Q8 confirmation is recorded in [delivery results](../delivery/RESULTS.md). Default deployed weights are unchanged.
