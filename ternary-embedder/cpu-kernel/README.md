# Packed ternary CPU kernels (AVX-512 VNNI)

Real ternary linears for the final 3-layer model (`models/campaign-20261025/model-ternary.pt`), 2026-11-01.
`tern_cpu.cpp` (build: `g++ -O3 -march=native -fopenmp -shared -fPIC -o libtern_cpu.so tern_cpu.cpp`),
`fast.py` (swaps the Student's linears), `check_bench.py` (correctness + timing), `prof.py` (time split).

Each kernel fuses int8 quantization (power-of-two scale per 256 channels) and ternary matmul via `vpdpbusd`
on codes+1 (unsigned) with the activation sum subtracted. Attention, layer norm, GELU, embeddings and pooling remain torch float.

| variant | weights | layout |
|---|---|---|
| dense_sim | float, codes*scale | the earlier "ternary CPU" row (dense float kernels) |
| i8 / p2 | int8 / 2-bit packed | v1: reduce along K (horizontal sums) |
| i8v2 / p2v2 | int8 / 2-bit packed | v2: outputs across lanes, 8x32 register tile, no horizontal sums |

Batch 1, tokenization included, fixed padding, 2 physical cores (cpus 0,1), median of 200 after >=0.75 s warmup:

| variant | 32-token cap, ms | 128-token cap, ms |
|---|---:|---:|
| dense_sim (earlier ternary CPU) | 3.06 | 8.82 |
| i8 v1 | 0.96 | 3.50 |
| p2 v1 | 1.20 | 4.47 |
| i8 v2 | 0.73 | 2.58 |
| **p2 v2 (2-bit packed)** | **0.67** | **2.34** |

Min cosine versus the torch reference path over 3 texts per cap: 1.0 (cap 32), 0.9999998 (cap 128). The timings above
were taken on a shared machine without a Dibs CLEAN check (raw numbers kept in `bench-shared-20261101.json`).

## Full-corpus retrieval with p2v2 (2026-11-02, `eval_retrieval.py`, results in `retrieval.json`)

Same data, tokenizer, 128-token cap and metrics as the NPU harness (`training-v2/evaluate_round2_npu.py`); batches of 64
sorted by length, padded to the longest. nDCG@10:

| task | torch reference | packed p2v2 | p2v2 - ref (paired 95% CI) | p2v2 - NPU (paired 95% CI) |
|---|---:|---:|---:|---:|
| SciFact | 59.868 | 59.865 | -0.003 [-0.009, 0.000] | +0.18 [-0.23, +0.64] |
| NFCorpus | 30.586 | 30.592 | +0.006 [-0.037, +0.057] | -0.05 [-0.25, +0.16] |
| ArguAna | 46.570 | 46.559 | -0.011 [-0.033, 0.000] | +0.16 [-0.06, +0.39] |
| **core mean (SciFact, NFCorpus)** | **45.227** | **45.228** | | NPU: 45.164 |

The torch reference reproduces the published CPU score (45.227) exactly. Vector cosine p2v2 versus reference over every
query and document: min 0.99951, mean 0.9999987. Only 1-3 queries per task change nDCG at all. The packed CPU kernel
therefore carries no measurable quality loss.

Time split for p2 v2: linears 0.30 of 0.59 ms (cap 32) and 0.83 of 1.98 ms (cap 128); the rest is torch attention/norm/GELU
overhead. A lookup-table kernel could only shave part of the linear share, so it was not built.
