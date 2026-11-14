# Packed Q4/Q8 quality on the float blend

All exports derive from the exact three-layer25% float blend. Q4 and Q8 store
real packed dense codes with BF16 group scales and simulate int8 activations in
the reference loader. Embeddings, output projection, biases, and norms remain
float32. This establishes packed-format accuracy, not native packed-kernel speed.

| Variant | Retrieval mean | STSBenchmark | 12-task mean |
|---|---:|---:|---:|
| Float | **47.022** | **83.942** | **71.777** |
| Q8, group64 | 46.886 | 83.928 | 71.740 |
| Q4, group64 | 45.962 | 83.532 | 71.367 |
| Q4, group32 | 46.309 | 83.555 | 71.456 |
| Q4, group16 | **46.425** | **83.577** | **71.461** |
| Q4, group64, weight-MSE scales | 45.412 | 83.545 | 71.277 |

Q8 loses only about **0.136 retrieval points** and **0.037 broader points**.
Q4 group16 reduces the retrieval loss to **0.597**, compared with **1.059** for
group64. Its51,075,659-byte artifact is about0.47MiB larger than the group64
export. Q8 is53,232,139 bytes. Whole-file savings are modest because the large
token-embedding table remains float32.

The weight-MSE scale grid guarantees no worse reconstruction error than its
included max-absolute scale candidate, after BF16 scale rounding. Nevertheless,
it loses another0.550 retrieval points against ordinary group64 quantization.
Weight reconstruction error alone was a poor proxy for embedding quality here;
that export is rejected. Smaller max-absolute groups retain outliers and provide
the better tested Q4 tradeoff.

Each fixed export grid was frozen before its new external scores. Choosing the
final group size uses observed development results; it is not independent
confirmation. The12-task average is not full MTEB. No reserved FiQA/SCIDOCS
judgments or training targets were used to choose individual quantization scales.

[Float/Q8/basic-Q4 comparison](results/summary.json),
[Q4 group-size experiment](../q4-groups/results/summary.json),
[rejected MSE-scale experiment](../q4-mse/results/summary.json).
