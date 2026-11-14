# Qwen3-Embedding-0.6B comparison — 2026-10-24

Fresh local evaluation of the trained ternary 2a_384 model on the NPU versus
Qwen3-Embedding-0.6B, bf16, on the Ryzen AI MAX+ 395 CPU. No paid resources.

## Retrieval quality

Full-corpus exact cosine ranking, linear-gain TREC nDCG@10, 128-token caps for
both models using their respective tokenizers. Qwen outputs 1024 dimensions;
our model outputs 384. All 623 judged queries are evaluated against all 5183
SciFact and 3633 NFCorpus documents. Scores are percentages; higher is better.

| Model / query format | SciFact | NFCorpus | Equal-task mean |
|---|---:|---:|---:|
| Ternary encoder, actual Linux NPU vectors | 54.39 | 29.51 | 41.95 |
| Qwen3-Embedding-0.6B, no instructions | 64.81 | 30.42 | 47.62 |
| Qwen3-Embedding-0.6B, task instructions | 67.73 | 35.50 | 51.62 |

Qwen's prompted advantage is 13.34 points on SciFact (paired query-bootstrap
95% CI: 9.83–16.90), and 6.00 on NFCorpus (4.17–7.84). The equal-task mean
advantage is 9.67 points. This establishes a retrieval advantage on these two
tasks, not an overall MTEB ranking. Qwen's full long-context capability is not
evaluated here. Prompt tokens count against Qwen's 128-token query cap.

Query instructions:

- SciFact: `Given a scientific claim, retrieve documents that support or refute the claim`
- NFCorpus: `Given a question, retrieve relevant documents that answer the question`

Qwen uses `Instruct: {instruction}\nQuery:{query}`, left padding, last-token
pooling and L2 normalization, following its [official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Documents have no instruction. Plain-query and prompted-query results share
the same document vectors. The NPU path uses its trained pooling and output
projection. `result.json` retains metrics, per-task paired intervals, dataset
hashes, model identities and timing samples; `*_rows.json` retain query rankings.

## Semantic similarity

STSBenchmark test split, 1379 human-scored sentence pairs, Spearman correlation
multiplied by 100. Both sides of each pair use the same input formatting.

| Model / input format | STSBenchmark |
|---|---:|
| Ternary encoder, actual Linux NPU vectors | 81.55 |
| Qwen, plain sentences | 84.63 |
| Qwen, instructed sentences | 91.20 |

The symmetric instruction is `Retrieve semantically similar text.` in the same
`Instruct: ...\nQuery:...` template. Every model uses a 128-token cap. This is
one task, not the seven-task STS average previously reported for the student.
The instruction substantially improves Qwen here as well as in retrieval.

## Speed

All figures include tokenization and normalization, exclude model loading,
and use warm models. The machine was shared with other work. These compare
specific implementations: Qwen CPU versus our custom NPU encoder. They are
not a claim about optimized Qwen GPU/NPU inference.

| Input / implementation | Median text-to-vector latency |
|---|---:|
| One short sentence, our NPU, fixed cap 128, 2 host threads | 1.37 ms |
| Same sentence, Qwen CPU, dynamic padding, 2 threads | 45.20 ms |
| Same sentence, Qwen CPU, dynamic padding, 8 threads | 26.72 ms |
| Same sentence, Qwen CPU, fixed padding to 128, 2 threads | 181.56 ms |

The dynamically padded Qwen input is 12 tokens; our compiled graph always
processes 128 token rows. The NPU path is 19.4x faster than the faster of the
two dynamic-padding CPU configurations measured. This is latency on the
stated short sentence, not a corpus-average throughput comparison.

At fixed batch 4 / cap 32, our NPU path takes 1.71 ms per batch (2342 emb/s)
versus 162.51 ms for Qwen CPU with two threads (24.6 emb/s). Fixed padding
penalizes Qwen relative to ordinary dynamic-padding use, as the single-text
measurements above demonstrate. Quality encoding used eight CPU threads
for throughput; that thread count is separate from the fixed-padding speed run.

## Direct CPU comparison

A fresh paired run places **both models on the same CPU**, alternates their
execution order, and uses five warmups plus 30 retained samples per model.
Each timing includes tokenization. Both use the same texts, caps, padding
policy, batch sizes and thread counts. The CPU was shared with other work.

| Threads | Padding / batch / cap | Our CPU median | Qwen CPU median | Ratio |
|---|---|---:|---:|---:|
| 2 | Dynamic / 1 / 128 | 2.40 ms | 45.32 ms | 18.9x |
| 8 | Dynamic / 1 / 128 | 1.67 ms | 28.13 ms | 16.8x |
| 2 | Fixed / 1 / 128 | 6.56 ms | 165.24 ms | 25.2x |
| 8 | Fixed / 1 / 128 | 3.09 ms | 60.55 ms | 19.6x |
| 2 | Fixed / 4 / 32 | 6.33 ms | 157.14 ms | 24.8x |
| 8 | Fixed / 4 / 32 | 3.02 ms | 56.33 ms | 18.6x |

Batch-4 times are for all four vectors, not per vector. Dynamic padding uses
each model's own tokenizer; the common short sentence need not tokenize to
the same number of tokens. The large speed difference persists on the CPU;
most of the prior advantage over Qwen comes from the small model. Our CPU
short-text latency is close to the previously measured NPU latency, although
that earlier measurement was a separate run and uses fixed padding.

The student CPU path precomputes its trained ternary effective weights once,
then uses float32 PyTorch dense linears with int8 activation quantization
simulated between layers. It does not requantize static weights each call,
and does not implement a specialized packed-ternary CPU kernel. Qwen uses
bf16 SDPA. Thus this is a comparison of these CPU inference implementations,
not an assertion of equal arithmetic formats or maximal CPU performance.
`cpu-speed.json` retains all samples and identities.

Fresh CPU quality evaluation on the same full corpora gives **42.13 mean
retrieval** (54.85 SciFact / 29.40 NFCorpus) and **81.58 STSBenchmark**, versus
prompted Qwen's 51.62 and 91.20. The corresponding NPU scores are 41.95 and
81.55, confirming that moving the student to CPU does not change the quality
conclusion. `cpu-quality.json` retains these CPU results. Both quality
runs use the same 128-token cap and the same evaluator.

## Reproduce

Use the Linux setup in [native-stage/LINUX.md](../../native-stage/LINUX.md).
The preferred checkpoint, local tokenizer and cap-32/cap-128 NPU builds must
be present. Public datasets and Qwen weights are downloaded without paid
services. Qwen is pinned to revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

```bash
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
ternary-embedder/.venv/bin/pip install pyarrow scipy
ternary-embedder/.venv/bin/python ternary-embedder/comparisons/qwen06b/fetch.py
ternary-embedder/.venv/bin/python ternary-embedder/comparisons/qwen06b/compare.py
ternary-embedder/.venv/bin/python ternary-embedder/comparisons/qwen06b/supplement.py
ternary-embedder/.venv/bin/python ternary-embedder/comparisons/qwen06b/cpu_speed.py
ternary-embedder/.venv/bin/python ternary-embedder/comparisons/qwen06b/cpu_quality.py
```

Local model/data caches and `.npy` vectors are ignored by git. Scripts reuse
those vectors and prior speed measurements if present: remove the relevant
comparison `.npy`/result files before changing models or inputs or requesting
a fresh run. The supplement evaluates STSBenchmark's 1379 test pairs, with
symmetric plain or instructed inputs, and saves scores in `supplement.json`.
