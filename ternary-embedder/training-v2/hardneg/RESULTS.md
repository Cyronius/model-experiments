# Hard-negative comparison

Two-layer original parents, same batch1024/micro128, LR4e-5, two complete relevance epochs. All candidates frozen before scoring. Observed development tests; the fixed12-task mean is not full MTEB.

| Arm | Retrieval | STSBenchmark |12-task mean |
|---|---:|---:|---:|
| float-pairs | 46.035 | 82.366 | 70.593 |
| float-hard | 45.230 | 82.221 | 70.432 |
| ternary-pairs | 44.793 | 80.820 | 69.376 |
| ternary-hard | 43.755 | 80.795 | 69.245 |

Paired recipe contrasts (query sampling, not training-seed uncertainty):

- float-hard minus float-pairs: retrieval -0.804,95% query interval[-1.584,-0.048];12-task mean-0.161.
- ternary-hard minus ternary-pairs: retrieval -1.039,95% query interval[-1.808,-0.321];12-task mean-0.130.

This particular unjudged BM25-negative set hurts both precisions. It does not establish that all hard negatives are harmful. Both conditions use the same expanded held negative corpus for checkpoint selection. Omit this recipe from the next data comparison.

The older relevance base includes1645 input overlaps with already-observed nonretrieval tests; these are development diagnostics. Core SciFact/NFCorpus exclusions remain clean. New experiments use benchmark-clean base pairs. Historical public pretraining exposure remains unknown.

All arms completed; local checkpoints SHA-verified, owned pod deletion verified HTTP404. Estimated cost$1.1490. Reserved FiQA/SCIDOCS judgments remain unopened. Default deployment unchanged.
