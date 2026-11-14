# Local float checkpoint blends

A fixed three-point interpolation grid improves the quality balance without a
new paid job. The 25% fine-tuned blend reaches **47.022 retrieval**,
**83.942 STSBenchmark**, and **71.777 on the fixed 12-task subset**. MiniLM's
corresponding scores are 46.920, 82.032, and 71.481. The blend still misses the
per-task retrieval gate: **SciFact 62.134 versus MiniLM 62.386**.

These are observed development results, not full-MTEB or independent
generalization claims. The query-bootstrap interval for the blend's retrieval
difference against MiniLM is **[-1.384,+1.654]**, so the small mean advantage
is not statistically established.

| Fine-tuned weight fraction | SciFact | NFCorpus | Retrieval mean | STSBenchmark | 12-task mean |
|---|---:|---:|---:|---:|---:|
| 0%, pretrained three-layer float parent | 60.387 | 31.349 | 45.868 | 83.851 | 71.529 |
| 25% | **62.134** | 31.910 | 47.022 | **83.942** | **71.777** |
| 50% | 62.121 | 31.981 | **47.051** | 83.778 | 71.697 |
| 75% | 61.739 | **32.121** | 46.930 | 83.376 | 71.464 |
| 100%, fresh-science fine-tuned endpoint | 61.868 | 31.473 | 46.670 | 82.677 | 71.110 |
| bge-micro-v2 | 61.330 | 31.098 | 46.214 | 82.497 | 70.904 |
| MiniLM-L6-v2 | 62.386 | 31.453 | 46.920 | 82.032 | 71.481 |

The blend averages every float parameter between the exact pretrained parent
and its fine-tuned descendant. It remains one ordinary three-layer float model
with 17,241,984 parameters; no ensemble or extra inference pass is introduced.
Fractions 0.25/0.5/0.75 were all frozen before any blend quality score. The
25% candidate is a development choice for broader balance, not a claim that a
quarter is universally optimal.

Relative to the original two-layer float control, the 25% blend gains **2.586**
retrieval points, query95% interval **[+1.475,+3.716]**. That comparison includes
a different historical pretraining budget, a third layer, new data, and blending;
it does not isolate the effect of any one change. Relative to its own three-layer
pretrained parent, it gains about 1.154 retrieval points and 0.248 broader points.

Compared with MiniLM, the 25% blend is stronger on most STS tasks and slightly
stronger in average retrieval, but weaker on Banking77 classification,
AskUbuntu duplicate ranking, SICK-R, and SciFact. The average therefore conceals
remaining practical weaknesses.

The blend grid is marked diagnostic in its frozen manifest: the original
training recipe's classifier/source selection guards have not been re-established
for these interpolations, and training-seed confirmation is outstanding. The
original default model is unchanged. Reserved FiQA/SCIDOCS judgments remain
unopened. Packed Q4/Q8 exports are being evaluated separately and have no
established native packed-kernel speed.

Evidence: [frozen grid](results/frozen.json),
[all quality metrics and intervals](results/summary.json).
Method inspiration: [weight-space averaging](https://proceedings.mlr.press/v162/wortsman22a.html);
that paper's results do not establish gains for this embedding model.
