# Experimental embedding family

Estimated campaign cost **$9.58 of $10**. All nine owned RunPod allocations were deleted; their controllers recorded GET404. No retained paid volumes.

**Quality improved, but the goal of beating both references broadly remains unmet.** Q8 leads the development point estimates, and trails MiniLM on the reserved two-task mean. The complete confirmation and intervals are below.

Three Transformer layers, width 384, 12 heads, FFN 1536, mean pooling, 384-dimensional unit vectors, maximum 128 tokens. Each backbone has 17,241,984 parameters. The ternary Transformer arithmetic is retained. Float is an independently trained branch; Q4/Q8 derive from the exact selected float checkpoint.

| Artifact | SciFact | NFCorpus | Retrieval mean | STSBenchmark | 12-task mean | File MiB |
|---|---:|---:|---:|---:|---:|---:|
| [model-ternary.pt](../../models/campaign-20261025/model-ternary.pt) | 59.868 | 30.586 | 45.227 | 82.452 | 70.481 | 65.79 |
| [model-float.pt](../../models/campaign-20261025/model-float.pt) | 62.067 | 31.977 | 47.022 | 83.990 | 71.956 | 65.79 |
| [model-q4.pt](../../models/campaign-20261025/model-q4.pt) | 61.327 | 32.012 | 46.669 | 83.639 | 71.667 | 48.71 |
| [model-q8.pt](../../models/campaign-20261025/model-q8.pt) | 62.967 | 31.984 | 47.475 | 83.977 | 72.023 | 50.77 |
| Original two-layer ternary |54.853|29.403|42.128|81.583|70.156| — |
| bge-micro-v2 |61.330|31.098|46.214|82.497|70.904| — |
| MiniLM-L6-v2 |62.386|31.453|46.920|82.032|71.481| — |
| Qwen0.6B, prompted |67.732|35.503|51.618|91.199| unmeasured | — |

Retrieval is the equal-task nDCG@10 mean over full SciFact/NFCorpus corpora. All quality uses a 128-token cap. Model selection used observed development results across many experiments. The 12-task mean is not full MTEB. Q8 exceeds both references on each core task, STSBenchmark, and the 12-task mean, as point estimates. Full-pipeline seed replication is outstanding. The original project default is not overwritten.

Q4 uses real four-bit dense codes with group 16 BF16 scales and int8 activations; Q8 uses eight-bit dense codes with group 64 scales and int8 activations. Embeddings, output projection, biases, and norms remain float32. These packed files have been quality-tested through the unpacked reference loader; optimized packed CPU/NPU kernels for Q4/Q8 are not provided. Ternary checkpoints store floating master/reconstruction tensors, while the native engine packs their three-valued dense weights at load time.

The selected ternary model was checked on the real NPU against CPU vectors at both 32/128-token buckets. Native full-corpus retrieval mean is **45.164**, versus CPU **45.227**.

Development Q8-minus-reference retrieval differences (paired query intervals, without correction for repeated model selection):

- bge_micro: **+1.261** points, 95% interval **[+0.251, +2.286]**.
- minilm: **+0.556** points, 95% interval **[-0.971, +2.126]**.

## Reserved retrieval confirmation

After Q8 passed the development gates, its exact file hash was frozen before opening FiQA and SCIDOCS test judgments. Only that candidate and the two pinned reference models were tested. No further training or selection used these judgments; these datasets are now opened and cannot serve as future reserved confirmation. Public historical pretraining exposure is unknown.

| Model | FiQA nDCG@10 | SCIDOCS nDCG@10 | Equal-task mean |
|---|---:|---:|---:|
| q8 | 26.805 | 17.697 | 22.251 |
| bge_micro | 23.523 | 17.521 | 20.522 |
| minilm | 36.626 | 21.716 | 29.171 |

Paired query-bootstrap differences, Q8 minus reference (5,000 resamples; seed 20261025):

- bge_micro: **+1.729** points, 95% interval **[+1.066, +2.377]**. Higher on both individual tasks: True.
- minilm: **-6.920** points, 95% interval **[-7.939, -5.920]**. Higher on both individual tasks: False.

Intervals quantify query sampling within these two fixed datasets. They do not measure general domain superiority, training-seed variation, or development-selection uncertainty. See the [protocol](RESERVED_PROTOCOL.md) and [activation receipt](reserved/activation.json).

## Loaded-model latency

Same two pinned physical CPU cores, batch 1, tokenization included, fixed padding, minimum 0.75 seconds of warmup. Initial and final Dibs checks were CLEAN. Qwen uses BF16; the small models use float32 CPU inference. Ternary CPU timing uses cached effective weights and dense kernels; NPU timing uses actual native ternary kernels.

| CPU model |32-token cap, ms |128-token cap, ms |
|---|---:|---:|
| float | 2.532 | 8.051 |
| ternary | 3.080 | 8.610 |
| bge_micro | 3.372 | 7.682 |
| minilm | 5.812 | 15.146 |
| qwen_0_6b_document | 61.465 | 156.825 |
| qwen_0_6b_query | 61.630 | 157.033 |

Native NPU dispatches one actual text, with unused compiled slots padded. These are single-text latencies, not batch throughput divided by four.
At the 32-token cap: **1.689 ms**.
At the 128-token cap: **1.791 ms**.

## Use in this repository

Use the existing project environments and tokenizer. From the repository root:

```python
import sys
sys.path.insert(0, "ternary-embedder/training-v2")
from evaluate_round2_mteb import Encoder
encoder = Encoder("ternary-embedder/models/campaign-20261025/model-ternary.pt")
documents = encoder.encode_documents(["A passage to index."])
queries = encoder.encode_queries(["A search query."])
# For packed variants: Encoder(path, precision="q4") or precision="q8".
```

The winning models use the common query/document head. The separately tested optional query-head variants selected the unchanged parent and are not part of this family. Native inference accepts model-ternary.pt through FusedEncoder(cap, checkpoint=...). The prebuilt three-layer 32/128 graphs and local NPU runtime are required.

## Findings

Pretrained-weight retention produced the strongest balanced gains. Fresh science helped float retrieval more clearly than ternary. The tested clone-and-append layer, unjudged BM25 negatives, additional continuation, learned scales, and output/query projection changes did not close the remaining reference gap. Human entailment/contradiction ranking raised broader quality but hurt retrieval; logical contradiction is not a human judgment of passage irrelevance. More epochs alone did not solve the problem.

The current direction is better relevance supervision with stronger protection of pretrained behavior. This is an inference from this campaign, not a proven optimal recipe. Q8 leads the development point estimates, while the confirmation table above determines how well that carries to the two reserved domains. Ternary still trails the references on development retrieval. The small Q8 gain over float may be quantization sensitivity on these samples. See [experiment interpretation](EXPERIMENT_NOTES.md).

Evidence: [frozen file hashes](../../models/campaign-20261025/frozen.json), [delivery summary](summary.json), [billing closeout](cost-closeout.json), [all development comparisons](../campaign/COMPARISON.md), [native quality](native/npu-quality.json), [timing audit](latency-dibs.log).

Full optimizer, source-sampler, classifier where used, and RNG backups remain under training-v2/work/*-run/backups with SHA-verified receipts. The scale continuation recovered a saved step 68 checkpoint and completed its original training budget. No credentials or private SSH keys are included in this model directory.
