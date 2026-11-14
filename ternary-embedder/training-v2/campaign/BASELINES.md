# Campaign baseline findings

All models use CPU evaluation, a128-token input cap, and the same fixed12-task MTEB1.39.7 harness. The12-task mean is not the full MTEB; seven of its tasks measure similarity.

| Model | SciFact | NFCorpus | Core retrieval mean | STSBenchmark |12-task mean |
|---|---:|---:|---:|---:|---:|
| Original ternary | 54.85 | 29.40 | 42.13 | 81.58 | 70.16 |
| Data-first ternary | 57.14 | 30.37 | 43.76 | 81.46 | 69.49 |
| Original float | 57.40 | 31.47 | 44.44 | 82.44 | 71.40 |
| Data-first float | 59.44 | 31.51 | 45.47 | 82.59 | 70.67 |
| bge-micro-v2 | 61.33 | 31.10 | 46.21 | 82.50 | 70.90 |
| MiniLM-L6-v2 | 62.39 | 31.45 | 46.92 | 82.03 | 71.48 |

The positive-pair data-first round improved retrieval but lowered the broader mean for both precisions. STSBenchmark alone missed losses on SICK-R, older STS tasks and Twitter paraphrases. The original float model is close to MiniLM on this subset but trails it in retrieval.

Follow-up design: retain the matched depth controls, test larger contrastive batches at two learning rates, and include original-parent reset arms. Prepare filtered hard negatives and trainable ternary row scales. New experiments should seek balanced quality rather than assuming more positive-pair updates help.

These task scores are now observed development evidence. FiQA/SCIDOCS relevance judgments remain unopened; preserve their predeclared conditional confirmation gate. Training-seed replication is still required for a promising final recipe.
