# Fresh science data and pretrained third-layer results

All six arms completed, were copied locally with SHA-256 checks, and were frozen
before external scoring. Both RunPod allocations were deleted; GET returned 404.
Estimated combined compute and temporary-disk cost: **$2.4412**. Cumulative
campaign cost before the scale-calibration followup: **$7.4503 of $10**.

The pretrained third-layer float model comes close to MiniLM on retrieval.
Ternary retains a larger gap, and the additional layer barely changes its
two-task retrieval mean. None of these candidates passes the declared reference
gate on both retrieval tasks.

## Quality

Scores are percentages. Retrieval uses the full SciFact and NFCorpus corpora,
128-token inputs, raw graded relevance, and the same fixed evaluator as the
references. The 12-task mean is a development subset, not full MTEB.

| Candidate | SciFact | NFCorpus | Retrieval mean | STSBenchmark | 12-task mean |
|---|---:|---:|---:|---:|---:|
| Original two-layer ternary | 54.85 | 29.40 | 42.13 | 81.58 | 70.16 |
| Two-layer float, repeated data | 59.49 | 30.87 | 45.18 | 82.70 | 70.65 |
| Two-layer float, fresh science | 60.92 | 31.06 | 45.99 | 82.65 | 70.86 |
| Three-layer float, fresh science | 61.87 | 31.47 | **46.67** | 82.68 | **71.11** |
| Two-layer ternary, repeated data | 58.67 | 30.06 | 44.36 | 81.46 | 69.77 |
| Two-layer ternary, fresh science | 58.79 | 30.46 | 44.63 | 81.43 | 69.97 |
| Three-layer ternary, fresh science | 59.34 | 30.02 | **44.68** | 81.55 | **70.08** |
| bge-micro-v2 | 61.33 | 31.10 | 46.21 | 82.50 | 70.90 |
| MiniLM-L6-v2 | 62.39 | 31.45 | 46.92 | 82.03 | 71.48 |
| Qwen3-Embedding-0.6B, prompted | 67.73 | 35.50 | 51.62 | 91.20 | unmeasured |

The three-layer float candidate exceeds bge-micro's point estimates on both
retrieval tasks, sentence similarity, and the wider mean. Its retrieval advantage
is not statistically established: paired-query 95% interval **[-0.63, +1.59]**.
Against MiniLM, its retrieval difference is **-0.249**, interval
**[-1.79, +1.42]**; SciFact is 0.518 points lower, and the wider mean is 0.371
lower. These are observed development comparisons across many experiments.

Relative to the original ternary model, three-layer ternary gains **2.551**
retrieval points, query interval **[+1.08, +4.10]**, while its wider mean changes
by about **-0.076**. This is a retrieval gain, not established broad superiority.

## What the controlled comparison says

Fresh science versus an equal-size repeated-old-science control changes float
retrieval by **+0.808**, query interval **[+0.305, +1.331]**, and the wider mean
by **+0.206**. The corresponding ternary changes are **+0.265**,
interval **[-0.721, +1.192]**, and **+0.201**. New data helps float retrieval more
clearly than ternary in this experiment; a larger useful effect for ternary is
not established.

Three-layer versus two-layer fresh-data candidates differ by **+0.682** float
retrieval points, interval **[-0.424, +1.842]**, and **+0.050** ternary points,
interval **[-1.097, +1.190]**. Their historical pretraining differs: the recovered
three-layer float checkpoint saw 75M tokens, while the original two-layer float
parent saw 150M. These contrasts compare candidates, not a controlled causal
effect of adding a layer. The separate clone-and-append experiment lost quality.

The same-depth float/ternary gaps are about **1.36 points** at two layers and
**1.99 points** at three. They include separate training histories and are not
pure post-training quantization losses. They motivate the next learned-row-scale
test; they do not prove it will close the gap.

## Data, training, and selection

Each arm sees 1,023,999 training pairs per epoch for two epochs, batch 1024,
microbatch 128, learning rate 4e-5, seed 29. The matched repeat and fresh mixtures
share source counts, 21,011 held pairs, and human NLI supervision. Fresh science
adds 244,901 training pairs selected across all 62 pinned S2ORC shards; titles and
abstracts are weak relevance associations, not human relevance judgments.

All arms use a disposable NLI classifier, auxiliary batch 192, and 100 frozen
encoder head-warmup steps. Encoder NLI weight is 0.1 for float and 0.25 for
ternary, chosen from the earlier fixed comparison. The encoder receives 385,440
NLI exposures covering **287,456 distinct pairs**; the head sees the complete
300,000-pair pool. This corrects the original plan's full-pool encoder wording;
see [exposure audit](../campaign/science-exposure-audit.json).

Checkpoint selection uses only the held training-source retrieval macro and
held NLI classifier accuracy, with per-source retrieval and NLI floors. No
external development score selects a checkpoint within an arm. All six
checkpoints were frozen before their external scores were opened. Campaign
recipe selection nevertheless uses observed development tasks; query intervals
do not measure training-seed stability or correct for multiple comparisons.

New fine-tunes remove all known exact text overlaps with the observed benchmark
inputs. Historical public-text pretraining exposure is unknown. Reserved
FiQA/SCIDOCS judgments remain unopened.

## CPU and actual NPU inference

Loaded-model, one-text latency includes tokenization. CPU runs use the same two
pinned physical cores, thread count, padding, and at least 0.75 seconds of warmup.
Inputs are padded to the stated cap. Dibs reported CLEAN before and after the
series. These measurements belong to the frozen science checkpoints, before
the later calibration followup.

| Model/backend | 32-token cap, ms | 128-token cap, ms |
|---|---:|---:|
| Two-layer float, CPU | 1.505 | 5.265 |
| Three-layer float, CPU | 2.532 | 7.868 |
| Two-layer ternary reference, CPU | 1.898 | 5.833 |
| Three-layer ternary reference, CPU | 3.123 | 8.762 |
| bge-micro-v2, CPU | 2.913 | 7.624 |
| MiniLM-L6-v2, CPU | 5.808 | 14.841 |
| Qwen 0.6B documents, CPU | 60.984 | 156.038 |
| Qwen 0.6B instructed queries, CPU | 60.919 | 155.968 |
| Two-layer ternary, native NPU | 1.264 | 1.264 |
| Three-layer ternary, native NPU | 1.676 | 1.697 |

The NPU measurements dispatch one actual text, with unused compiled batch slots
padded. They are not batch-four throughput divided by four. Full-corpus native
retrieval means are **44.803** for two layers and **44.823** for three, versus
CPU reference 44.629 and 44.679. Small arithmetic changes can alter near-tied
rankings; native quality remains close to the CPU reference.

Two layers have **15,467,520 parameters**, three **17,241,984**: 11.47% more
total parameters and 50% more Transformer-layer work. Token embeddings dominate
the total size. Native two- and three-layer graphs support both caps; the default
deployed checkpoint is unchanged.

Machine-readable evidence: [quality and query intervals](results/summary.json),
[CPU timing](results/cpu-speed.json),
[Dibs timing audit](results/latency-dibs-pinned.log), and native
[two-layer](results/native-2/npu-quality.json) /
[three-layer](results/native-3/npu-quality.json) quality.
