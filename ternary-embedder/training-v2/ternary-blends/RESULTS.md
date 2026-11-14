# Ternary checkpoint blends

Six fixed local blends were frozen before scoring. Every candidate retains
ternary Transformer matrices and int8 activation arithmetic. The experiment
averages QAT master parameters from the same pretrained lineage, then applies
the ordinary ternary code/row-scale derivation. These are not float models.
No new paid allocation was used.

| Layers | Fine-tuned fraction | Retrieval mean | STSBenchmark | 12-task mean |
|---|---:|---:|---:|---:|
| 2 | 25% | 44.586 | 81.090 | **70.531** |
| 2 | 50% | 44.522 | 81.197 | 70.384 |
| 2 | 75% | 44.978 | 81.408 | 70.273 |
| 3 | 25% | **45.227** | **82.452** | 70.481 |
| 3 | 50% | 45.022 | 82.412 | 70.402 |
| 3 | 75% | 44.999 | 82.105 | 70.289 |
| Original two-layer ternary | — | 42.128 | 81.583 | 70.156 |
| bge-micro-v2 | — | 46.214 | 82.497 | 70.904 |
| MiniLM-L6-v2 | — | 46.920 | 82.032 | 71.481 |

The three-layer 25% blend improves retrieval by about **3.10 points** over the
original model, sentence similarity by **0.87**, and the wider mean by **0.33**.
Its SciFact/NFCorpus scores are **59.868/30.586**. It remains behind both
reference retrieval targets. The two-layer 25% blend offers a slightly higher
broader mean and the cheaper two-layer architecture, at lower retrieval quality.

Compared with the full three-layer science fine-tune, the 25% blend recovers
about **0.548 retrieval**, **0.900 sentence-similarity**, and **0.401 broader**
points. This suggests the fine-tune was overwriting useful pretrained behavior;
it does not establish that this fraction or blending method generalizes to
unseen tasks. The candidates have different historical pretraining across
depths, so this is not a controlled causal comparison of layer count.

These are observed development results across the fixed 12-task subset, not
full MTEB. The blend grid is diagnostic: the original fine-tuning classifier
selection guard has not been re-established for these interpolations. No full
pipeline seed replication or reserved FiQA/SCIDOCS evaluation has been performed.

[Frozen choices](results/frozen.json), [full metrics and query intervals](results/summary.json).
