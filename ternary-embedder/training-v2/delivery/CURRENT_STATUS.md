# Ternary embedder campaign status

Training is complete. Estimated GPU spending was **$9.5843 of the approved $10**. All nine campaign RunPod allocations were deleted, with deletion verified. Checkpoints and the four final model artifacts are stored locally.

Each final model has **17,241,984 parameters**, with three transformer layers and 384-dimensional embeddings.

| Model | SciFact nDCG@10 | NFCorpus nDCG@10 | Retrieval mean | STSBenchmark | 12-task development mean |
|---|---:|---:|---:|---:|---:|
| Original ternary | 54.85 | 29.40 | 42.13 | 81.58 | 70.16 |
| Final ternary | 59.87 | 30.59 | 45.23 | 82.45 | 70.48 |
| Final float | 62.07 | 31.98 | 47.02 | 83.99 | 71.96 |
| Final Q4 | 61.33 | 32.01 | 46.67 | 83.64 | 71.67 |
| Final Q8 | **62.97** | **31.98** | **47.48** | **83.98** | **72.02** |
| BGE-micro-v2 | 61.33 | 31.10 | 46.21 | 82.50 | 70.90 |
| MiniLM-L6-v2 | 62.39 | 31.45 | 46.92 | 82.03 | 71.48 |
| Qwen 0.6B, prompted queries | 67.73 | 35.50 | 51.62 | 91.20 | Not measured |

These are development measurements at a common 128-token cap, using full retrieval corpora. They are not full-MTEB scores or proof of broad superiority. Repeated development selection and lack of end-to-end seed replication limit the conclusions. In particular, the small Q8 advantage over float could reflect quantization effects on these samples.

The final Q8 weights were frozen before opening the FiQA and SCIDOCS judgments. Their confirmation retrieval mean is **22.25**, versus **20.52** for BGE-micro and **29.17** for MiniLM. No further training or model selection used these judgments. See [full results](RESULTS.md) for both tasks and paired intervals.

The ternary model gained **3.10 retrieval points** over the original, but has not caught the two compact reference models. Q8 passes every development gate against both references. Qwen 0.6B retains a 4.14-point advantage on the development retrieval mean.

Final, isolated batch-one CPU measurements include tokenization and use two pinned physical cores:

| Model | 32 tokens, ms | 128 tokens, ms |
|---|---:|---:|
| Float | 2.53 | 8.05 |
| Ternary CPU | 3.08 | 8.61 |
| BGE-micro-v2 | 3.37 | 7.68 |
| MiniLM-L6-v2 | 5.81 | 15.15 |
| Qwen 0.6B, query | 61.63 | 157.03 |
| Ternary native NPU | 1.69 | 1.79 |

The NPU result measures one actual input per dispatch; unused slots are padded. Its retrieval mean is 45.16, versus CPU ternary's 45.23. CPU and NPU rows represent different hardware. Q4 and Q8 files are genuinely packed, but the reference loader reconstructs dense float weights; optimized packed execution has not been implemented or timed.

Final artifacts: `../../models/campaign-20261025/`. Source evidence: `family-results/summary.json`, `cost-closeout.json`, `cpu-speed.json`, `native-speed.json`, and `latency-dibs.log`.
