# Larger-batch comparison

Six frozen two-layer candidates, batch1024 with128-example gradient-cache chunks, two complete epochs. Quality is measured on observed development tasks; the12-task mean is not full MTEB.

| Candidate | SciFact | NFCorpus | Retrieval mean | STSBenchmark |12-task mean |
|---|---:|---:|---:|---:|---:|
| float-1e-5 | 60.06 | 31.51 | 45.78 | 82.47 | 70.65 |
| float-4e-5 | 60.42 | 31.86 | 46.14 | 82.07 | 70.42 |
| ternary-1e-5 | 57.87 | 30.68 | 44.28 | 81.06 | 69.39 |
| ternary-4e-5 | 58.52 | 30.91 | 44.72 | 80.92 | 69.30 |
| predata-float-4e-5 | 60.24 | 31.83 | 46.03 | 82.37 | 70.59 |
| predata-ternary-4e-5 | 58.41 | 30.71 | 44.56 | 80.84 | 69.23 |

Best retrieval is float4e-5 at46.14, below MiniLM46.92 and bge-micro46.21. Best ternary retrieval is44.72. None passes the reference gate. The best wider mean70.65 remains below the original float71.40 and MiniLM71.48. Larger batches and more continuation did not resolve broad forgetting.

The first four arms start from the data-first parents; the predata arms reset to the independent original parents. Cached loss logits use FP32, whereas the earlier small-batch recipe used BF16 autocast; this is a combined arithmetic/batch recipe comparison, not a pure batch-size causal estimate.

A subsequent audit found1645 training pairs with visible text also present in already-observed nonretrieval tests. SciFact/NFCorpus exclusions remain clean; broader scores are development diagnostics. Future fine-tuning uses the benchmark-clean corpus. Exact input exclusion does not establish that historical pretrained checkpoints never saw public test text.

All six arms completed. Off-pod checkpoints verified; owned pod deletion confirmed HTTP404. Estimated total cost $1.5172. Reserved FiQA/SCIDOCS labels remain unopened.
