# Where the data-first changes occurred

Category means from the same fixed12-task suite and128-token cap.

| Category | Float before | Float after | Float change | Ternary before | Ternary after | Ternary change |
|---|---:|---:|---:|---:|---:|---:|
| similarity | 80.31 | 78.95 | -1.37 | 79.25 | 77.74 | -1.51 |
| retrieval | 44.44 | 45.47 | +1.04 | 42.13 | 43.76 | +1.63 |
| classification | 77.01 | 78.20 | +1.19 | 75.85 | 77.13 | +1.28 |
| reranking | 61.11 | 60.67 | -0.43 | 60.38 | 60.45 | +0.08 |
| paraphrases | 67.65 | 65.57 | -2.08 | 66.64 | 64.60 | -2.04 |

Both precisions show similar losses in similarity and paraphrases alongside retrieval/classification gains. The broad forgetting is not exclusively a ternary-quantization effect. Test richer human supervision separately from ternary-specific scale adaptation. Category weighting and task composition matter; these are observed development results, not new held-out confirmation.
