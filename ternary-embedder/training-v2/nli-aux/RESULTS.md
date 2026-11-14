# Human NLI comparison

Two-layer original parents, same batch1024/micro128, LR4e-5, two complete relevance epochs. All candidates frozen before scoring. Observed development tests; the fixed12-task mean is not full MTEB.

| Arm | Retrieval | STSBenchmark |12-task mean |
|---|---:|---:|---:|
| float-0.0 | 46.035 | 82.366 | 70.593 |
| float-0.1 | 45.708 | 82.996 | 71.061 |
| float-0.25 | 45.342 | 82.957 | 71.145 |
| ternary-0.0 | 44.559 | 80.841 | 69.234 |
| ternary-0.1 | 44.199 | 81.661 | 69.809 |
| ternary-0.25 | 44.104 | 81.590 | 70.009 |

Paired recipe contrasts (query sampling, not training-seed uncertainty):

- float-0.1 minus float-0.0: retrieval -0.326,95% query interval[-0.636,-0.025];12-task mean+0.468.
- float-0.25 minus float-0.0: retrieval -0.693,95% query interval[-1.248,-0.176];12-task mean+0.552.
- ternary-0.1 minus ternary-0.0: retrieval -0.360,95% query interval[-0.976,+0.267];12-task mean+0.575.
- ternary-0.25 minus ternary-0.0: retrieval -0.456,95% query interval[-1.141,+0.247];12-task mean+0.774.

Choose weight.1 for float and.25 for ternary in the next data comparison. Float.25 buys little broader improvement over.1 for a larger retrieval loss; ternary.25 adds.200 broader mean over.1 for only.095 retrieval cost. Keep the disposable classifier out of inference. The next comparison increases auxiliary batch64→192 to cover all300k available human pairs.

This run exposes each encoder to98176 auxiliary pairs at nonzero loss weight, not all300k. It restores much of the broad quality lost during retrieval continuation, while leaving reference targets unmet.

The older relevance base includes1645 input overlaps with already-observed nonretrieval tests; these are development diagnostics. Core SciFact/NFCorpus exclusions remain clean. New experiments use benchmark-clean base pairs. Historical public pretraining exposure remains unknown.

All arms completed; local checkpoints SHA-verified, owned pod deletion verified HTTP404. Estimated cost$1.5413. Reserved FiQA/SCIDOCS judgments remain unopened. Default deployment unchanged.
