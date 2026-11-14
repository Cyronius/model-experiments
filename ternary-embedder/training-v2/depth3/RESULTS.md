# Additional-layer results

Four matched arms each saw two complete passes over780,743 audited pairs. Three-layer models cloned the final parent block and ramped from identity over500 updates. All choices were frozen before observed-task scoring.

| Variant | Parameters | Core retrieval mean | STSBenchmark |12-task mean |
|---|---:|---:|---:|---:|
| 2-float | 15,467,520 | 45.28 | 82.28 | 70.41 |
| 3-float | 17,241,984 | 43.06 | 81.86 | 69.57 |
| 2-ternary | 15,467,520 | 44.35 | 81.09 | 69.39 |
| 3-ternary | 17,241,984 | 43.58 | 80.92 | 68.98 |

Appending the cloned third block failed to improve quality. Relative to continued two-layer controls, core retrieval changed by-2.23 points for float and-0.78 for ternary; wider quality also fell. This tests one growth/initialization recipe, not the impossibility of improving a three-layer architecture.

The current data-first float parent remains better than its continuation (45.47 versus45.28 core retrieval;70.67 versus70.41 wider mean). Ternary continuation adds0.59 retrieval points (43.76→44.35) while the wider mean slips69.49→69.39. More updates alone have a trade-off.

Decision: retain two layers for the next larger-batch/data experiments. Include original-parent reset arms because the pre-data-first float model has the stronger wider mean71.40. Do not promote these depth candidates or add a three-layer NPU implementation on this evidence.

Estimated cost including temporary disk: **$0.8017**. All final/best/resumable artifacts were recovered and checksummed; pod44q0zq3vqn9bc3 was deleted and GET returned404. No reserved FiQA/SCIDOCS judgments were scored.

Matched CPU text-to-vector latency with two threads and already-loaded weights:

| Model |32-token bucket |128-token bucket |
|---|---:|---:|
| Two-layer float |1.76ms |5.88ms |
| Three-layer float |2.97ms |8.46ms |
| Two-layer ternary reference |1.86ms |5.70ms |
| Three-layer ternary reference |3.01ms |8.53ms |
| bge-micro-v2 |2.89ms |7.54ms |
| MiniLM-L6-v2 |6.09ms |16.36ms |

Tokenization is included, model loading excluded. Short text is padded to32; the
longer text fills128. Ternary CPU measurements cache effective ternary weights
and simulate int8 activations with dense floating-point kernels. They are not
native packed-ternary CPU benchmarks. No three-layer NPU graph was built.
Raw timings are in`results/cpu-speed.json`.
