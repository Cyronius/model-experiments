# Native-384 NPU encoder

> **Moved 2026-10-31.** fused2, the three-layer encoder, now lives in phlegm at `open_kernels/designs/trit_embed` (`trit_design.py`, `trit_npu.py`), which builds the identical program. Use that copy; the files here are kept as history.

## fused2: packed three-layer encoder (2026-10-28)

A parallel, faster path for the three-layer campaign checkpoint. The original `fused` files and builds are unchanged.

Profiling with stubbed kernels showed the original 1.62 ms dispatch was mostly compute, not weight streaming: attention 0.62 ms (a scalar per-lane mask loop), projections 0.43, data movement and sync 0.48, quantization 0.07, LayerNorm 0.02. Changes, in order of effect:

- **Segment-ID masks.** The mask packet holds one segment ID per token (0 = padding). Each core expands it once per dispatch into an additive score bias, replacing the per-lane mask loop. Any mix of texts can share one 128-token dispatch; `pack()` fills slots best-fit by length.
- **Host overlap.** `host/td_npu2.cpp` adds `tdn_bind`/`tdn_start`/`tdn_wait`; two buffer sets alternate so the host prepares and pools while the NPU runs. Two separate hardware contexts were slower, because the driver swaps full-array contexts.
- **Double-buffered weight broadcast.** To fit the 16 KiB program memory, each weight packet now carries its 32 channels' bias and scale at byte 3072 (170 packets per layer instead of 186), and attention finishes inside `attend`.
- **Cheaper exp and reciprocal.** GELU outputs, attention probabilities and the attention normalizer are all rounded to bf16 next, so `fexp` (degree-4 polynomial, 2.6e-6 relative, 2^n applied through the exponent bits) and `frecip` (hardware estimate plus one Newton step) replace vecmath.h's ~1e-7 versions. Vectors did not change measurably.

Measured on the KJV (31102 verses, 1.07M tokens): text to vector including tokenization, two host threads, `dibs` CLEAN.

| Path | Dispatch | 31102 verses | 9360 passages |
|---|---:|---:|---:|
| Original fused, cap 32/128 buckets | 1.62 ms | 36.5 s | 18.6 s |
| + segment masks and packing | 1.09 ms | 12.8 s | 12.8 s |
| + host overlap | 1.09 ms | 9.6 s | 9.9 s |
| + double-buffered broadcast | 0.975 ms | 8.0 s | 8.1 s |
| + cheaper exp and reciprocal | 0.83 ms | **7.5 s** (4163/s) | **7.5 s** |
| CPU reference, 16 threads | — | 19.3 s | 27.6 s |

Dispatch times drift by about 10% between sessions; compare rows measured together. Minimum vector cosine to the CPU model over all verses is 0.99894 (mean 0.99968), against 0.99901 for the original NPU path. Aligning texts to 8-token block-floating groups did not change accuracy and cost 9% throughput. Only the three-layer cap-128 graph is built.

**fused2 retrieval and matched throughput (2026-11-03, `eval_f2.py`, `fused2-eval.json`).** Full-corpus nDCG@10, fused2 vs
CPU torch reference: SciFact 59.79 vs 59.87, NFCorpus 30.55 vs 30.59, ArguAna 46.40 vs 46.57; core mean **45.17** (CPU
45.23, v1 NPU 45.16). Every paired 95% interval against the CPU reference and against v1 includes zero; vector cosine to
the reference min 0.9986, mean 0.99978. Same repeated texts as `gpu-bench/` (tokenization included, 2 host threads,
shared machine, no Dibs CLEAN): batch 1 1.09 ms (13-token text) / 1.38 ms (128 tokens); batch 64 of the short text
8053 emb/s packed at real length (8 dispatches) or 4446 emb/s with each text given a full 32-token span (16 dispatches);
batch 64 at 128 tokens 907 emb/s (64 dispatches, 1.34 ms median dispatch).

Remaining profile at 0.84 ms (each part stubbed in turn): data movement and sync 0.45 ms (54%), projections 0.27 (of which GELU 0.15), quantization 0.065, attention 0.04, LayerNorm 0.02. Tried without gain: separate even/odd MAC accumulators (within noise, +224 bytes) and vectorized quantizer scales (+336 bytes). Program text is 15952 of 16384 bytes. Per-kernel `-Os`, unroll pragmas and aiecc `-O` do not change core code size, because IRON links kernels as bitcode before one whole-program code generation. The data-movement floor is mostly fixed 4096-byte packets: weight blocks fill 56–81% of them.

```bash
source /home/cyrus/code/phlegm/target/npu-dev/env.sh
python build_fused2.py --cap 128 --layers 3 --bdepth 2
python build_host.py td_npu2
```

`Fused2Encoder(checkpoint).encode(token_id_lists)` returns vectors. `bench_f2.py` times raw dispatches for any build directory; `build_fused2.py --stub attend,linear` or `--defs F2_PROF_NOGELU` builds profiling variants. NPU runs from a Claude Code session need `ulimit -l` raised first (the user systemd manager caps locked memory at 8 MiB).

## Earlier results

Optional three-layer execution was added on2026-10-25. `build_fused.py --layers 3` creates separate cap32/128 artifacts; `FusedEncoder` chooses the compiled depth from checkpoint dimensions and validates its manifest. Both depths use16080 bytes of program text per core, within16KiB. Historical2/3-layer checkpoints pass single-text and full-bucket vector/intermediate-state checks: minimum vector cosine0.999543 and minimum valid-token cosine0.998830 against CPU. See [verification artifacts](../training-v2/native-depth/trace-verification.json) and [full buckets](../training-v2/native-depth/trace-verification-full.json). These verify the implementation; no three-layer latency or new-candidate retrieval result is claimed here. Use the compiler environment Python for builds and the project Torch venv for verification.

Linux port and paired output-transfer optimization completed on 2026-10-24:
see [Linux results and reproduction](LINUX.md). The results below are the
retained October 23 Windows measurements.

The complete two-layer encoder body now runs on 32 Strix NPU cores in one XRT
dispatch. Measured locally on 2026-10-23; no new training or cloud work.

## Complete encoder results

| Path | Batch / sequence cap | Median text-to-vector time | Embeddings/s |
|---|---|---:|---:|
| Fused NPU body + host input/output | 4 / 32 | 6.587 ms | 607.3 |
| Same path, pretokenized forward | 4 / 32 | 2.229 ms | 1794.9 |
| Fused NPU body + host input/output | 1 / 128 | 4.135 ms | 241.8 |
| Same path, pretokenized forward | 1 / 128 | 2.283 ms | 438.0 |
| PyTorch CPU MiniLM, same texts / cap | 4 / 32 | 45.634 ms | 87.7 |
| PyTorch CPU MiniLM, same texts / cap | 1 / 128 | 45.325 ms | 22.1 |

Each row has one excluded warm-up and 15 retained wall-clock samples; model
loading, compilation and initial weight upload are excluded. CPU operations
use two PyTorch threads. The NPU was leased with `dibs`; whole-machine CPU
quietness was not enforced. Its recorded power mode was `Default`. The first
four SciFact corpus texts, truncated to the stated cap, form the microbenchmark.
The 32-token row does **not** have the retrieval quality measured at 128 tokens.
Bulk 128-token retrieval encoding, tokenizing 100 texts together and dispatching
one text at a time, reached 332.8/s on SciFact and 326.1/s on NFCorpus with the
final padding-isolated build. Earlier builds reached 361–484/s in complete
passes and 856/s / 292/s in the short / long microbenchmarks, alongside much
faster CPU MiniLM times. `bulk-timing-history.json` and `timing-history.json`
preserve these observations and code identities. CPU load was 38% after the
second corpus pass. Treat these speeds as observations on a shared machine,
not a stable throughput ceiling.

These fresh CPU MiniLM timings are not the older `open_npue` NPU benchmark.
Its historical 1305/s used batch 64 and a different text workload. The current
checkout no longer contains that backend or its packed MiniLM model; no fresh
apples-to-apples NPU MiniLM speed comparison is claimed here.

| 128-token full-corpus retrieval, nDCG@10 | SciFact | NFCorpus | Equal-task mean |
|---|---:|---:|---:|
| Actual fused NPU vectors | 54.3910 | 29.4909 | **41.9409** |
| Same trained QAT checkpoint, CPU reference | 54.8555 | 29.4089 | 42.1322 |
| Matched-budget float student, prior CPU diagnosis | 57.4027 | 31.4696 | 44.4361 |
| MiniLM, prior CPU diagnosis | 62.3860 | 31.4534 | 46.9197 |

The NPU arithmetic adds a 0.1913-point mean loss (paired query-bootstrap 95%
CI for NPU minus CPU: -0.4332 to +0.0248). This is a separate, much
smaller effect than the trained model's existing gap. All 623 judged queries
are ranked against the complete corpora (5183 and 3633 documents), with the
same tokenizer, no prompts, cosine similarity and linear-gain TREC nDCG.
All 9439 NPU embeddings are finite. Their minimum cosine to the CPU vectors
is 0.9996390; maximum element difference is 0.005195. Small numerical changes
can reorder nearly tied documents. This is approximate equivalence, not an
integer-exact claim for the fused encoder.

`fused-result.json` retains timing samples and build identities;
`fused-retrieval.json` retains corpus/vector hashes, paired query-bootstrap
intervals, full-corpus scores and bulk timings. Ignored
`results/fused-retrieval/` holds actual vectors, per-query rankings and cache
identities. The older hybrid path is retained in `encoder-result.json`:
12 separate NPU projections with CPU nonlinear operations took 30–34 ms per
128-token-row forward. Fusion removed that bottleneck.

## Complete encoder layout and numerical scope

`fused_design.py` assigns four token rows to each of 32 cores, eight columns
by four rows. Both trained layers execute in one dispatch. The NPU performs
all Q/K/V/O and FFN projections, activation quantization/dequantization,
12-head attention with head dimension 32, masked online softmax, tanh GELU,
residuals and both post-residual LayerNorms with eps=1e-12. FFN-down retains
six independent 256-channel activation scales.

Tokenization, word/position/type embedding lookup, input LayerNorm, pooling,
the trained float output projection and final L2 normalization remain on the
CPU and are included in text-to-vector timings. This is a callable offline
engine, not a serving backend. `embed.py` accepts arbitrary text batches.

The kernels use packed ternary weights and int8 projection inputs. Q/K/V,
attention probabilities and the GELU output round to bf16. Float vector math
uses split-bf16 MACs because AIE2P lacks a native fp32 vector multiplier.
`vecmath.h` and `ironutil.py` come from the archived tiny-planner; `fused.h`
adapts its arithmetic to this model's post-norm architecture. The persistent
XRT C ABI currently uses its existing `host/out/td_npu.dll` at the absolute
path in `encoder.py`; the reports record the loaded DLL's SHA256.

The routed program is 16096 B per core, within the 16384 B instruction budget.
Per-core allocation ends at byte 34688, including the 5120 B stack: 6144 B
float state, 8192 B output/KV packet, 4096 B broadcast packet, 3072 B bf16
query/FFN buffer, 4096 B shared parameter/attention workspace, two 1664 B
quantizer buffers, 512 B mask and 128 B attention metadata. Column input splits
and output joins use five inbound and five outbound memtile DMA channels.
K/V packets pass through DDR and are broadcast by head; weights stream each
dispatch and are padded into 4096-byte packets (1523712 bytes for both layers).
Layer traces are drained for both layers. Weights are not memtile-resident.

Masked token IDs are canonicalized to pad ID 0 before embedding lookup, and
padded query rows are excluded from the shared softmax maximum. This prevents
caller-supplied masked values from affecting block-floating rounding for valid
tokens. `masking-result.json` verifies exact invariance to changed masked IDs,
exact batch permutation, and partial-batch error below 1e-7. Empty sequences
are rejected. Masked query states themselves intentionally differ from the CPU
BERT trace; they are never pooled or used as valid keys. The separate
`fused-verification.json` reports valid-token state errors explicitly.

```powershell
. C:/dev/mlir-aie/iron_env.ps1
foreach ($cap in @(32,128)) {
    & C:/dev/mlir-aie/ironenv/Scripts/python.exe ternary-embedder/native-stage/build_fused.py --cap $cap
    if ($LASTEXITCODE) { throw 'Encoder build failed' }
}
dibs run npu --label 'ternary full encoder benchmark' -- C:/Python311/python.exe C:/code/model-experiments/ternary-embedder/native-stage/bench_fused.py --caps 32 128
dibs run npu --label 'ternary full retrieval check' -- C:/Python311/python.exe C:/code/model-experiments/ternary-embedder/native-stage/eval_fused.py
dibs run npu --label 'ternary batch mask check' -- C:/Python311/python.exe C:/code/model-experiments/ternary-embedder/native-stage/verify_masking.py
dibs run npu --label 'ternary embed text' -- C:/Python311/python.exe C:/code/model-experiments/ternary-embedder/native-stage/embed.py 'A sentence to embed.' --output C:/code/model-experiments/ternary-embedder/native-stage/results/example.npy
```

Builds require the same local toolchain/packager as the projection gate below.
Runtime and eval require the original local checkpoint and cached tokenizer /
MiniLM / MTEB files described in `../retrieval-diagnostics/README.md`.
Builds and large fixture/vector files are ignored. Resident weights, pooled
NPU output, DMA-only timing, larger request batching and serving integration
are future optimization work. The ~750k tok/s planning estimate remains
unvalidated; this implementation is about 56–57k padded token rows/s before
tokenization, using the warm pretokenized forward time.

## Earlier projection gate

Measured on the local Strix NPU on 2026-10-23. All 36 trained projection
fixtures passed exact integer checks. This extends the native-width arithmetic
probe to full output matrices, real trained weights and real activation codes.
The measurements below describe the earlier projection-only implementation.

## What runs on the NPU

One core, four token rows, all output columns, one dispatch per dense projection.
Packed ternary weights stream from host memory in 32-output-channel blocks;
activation codes stay in core memory throughout the projection. There is no
artificial repetition and no width padding.

| Projection K -> N | Scale groups per row | Packed weights per dispatch | Median of fixture warm medians |
|---|---:|---:|---:|
| 384 -> 384 (Q/K/V/O) | 1 | 36 KiB | 0.1460 ms |
| 384 -> 1536 (FFN up) | 1 | 144 KiB | 0.1995 ms |
| 1536 -> 384 (FFN down) | 6 | 144 KiB | 0.1920 ms |

These are the harness's reported run intervals, not embedding latency or kernel
cycle counts. Each of 36 cases has one excluded warm-up and nine retained calls
(324 retained calls total), in fixed order under `dibs run npu`. This was not a
whole-machine quiet benchmark. Do not extrapolate these small streaming kernels
to a resident-weight, batched encoder's throughput.

The cases cover Q/K/V/O/FFN-up/FFN-down in both trained layers at sequence caps
8, 32 and 128. Each NPU fixture takes the first four flattened token rows of
the corresponding projection input; this is **not every row of a full sequence**.
Every output channel and every activation-scale group is checked against an
independent int64 dot product. CPU dequantization of the NPU integers is also
bit-exact against the exported integer oracle's dequantized values.

## Numerical contract

`arithmetic.py` defines the layout and an independent NumPy implementation:

- Weight codes {-1, 0, 1}, two-bit two's complement, four codes per byte.
- A block's weights are `[K/16, 128 bytes]`; each 128-byte vector expands into
  two int4 matrices with entries `4*w`.
- Int8 activations are `[K/16, 4 rows, 16]`, using ties-to-even rounding and
  power-of-two scales as in `tern-embed/model.py`.
- Width 384 uses one activation scale per token. Width 1536 uses **six separate
  256-channel scales**. A single unscaled 1536-wide dot product would be wrong.
- Integer output is `[N/32, groups, 2 halves, 4 rows, 16]`, holding `4*dot`.
  Apply each group's activation scale before summing groups, then the output
  row's bf16-rounded weight scale and its bias.

The CPU oracle independently decodes all 12 packed matrices and runs all token
rows of both layers for four corpus texts at each sequence cap, with mixed
padding lengths. Attention, GELU, both post-residual LayerNorms, pooling and the
float output projection retain their PyTorch reference implementations. All
these operations, activation quantization and dequantization remain on the CPU
at this gate. There is **no claim of NPU final-vector or MTEB equivalence yet**.

Against the trained PyTorch model, the worst CPU oracle final-vector element
error is 8.65e-7; the largest state error is 1.90e-4 at sequence cap 128. Small
linear summation differences can cross subsequent quantizer thresholds, so the
full CPU oracle is tolerance-checked, not declared bit-identical to PyTorch.

## Compiled resource budget

The generated images route successfully on one column, core `(0,2)`. Each uses
two shim MM2S channels and one S2MM channel. There is no memtile allocation or
resident full-layer weight layout in this graph.

| K -> N | Program `.text` | Input buffer | Weight buffer | Output buffer | Stack |
|---|---:|---:|---:|---:|---:|
| 384 -> 384 | 1408 B | 1536 B | 3072 B | 512 B | 4096 B |
| 384 -> 1536 | 976 B | 1536 B | 3072 B | 512 B | 4096 B |
| 1536 -> 384 | 1504 B | 6144 B | 12288 B | 3072 B | 4096 B |

Program sizes come from the compiled ELF's `.text`; buffer sizes and channels
are confirmed in `final.prj/input_with_addresses.mlir`. These are the resources
of the streaming projection graph, not estimates for a complete stage.

## Reproduce

Run from `C:/code/model-experiments`. No network or new training is required.
The original checkpoint and tokenized smoke corpus must be present locally.
Build/runtime dependencies are the same as the adjacent native-width probe:
mlir-aie 1.4.2, Peano 21.0.0.2026080301+c9c5ecb7, its locally built XRT packager,
and `openflowlm-next/open_kernels/harness/out/run_kernel.exe`.

```powershell
C:/Python311/python.exe -m unittest discover -s ternary-embedder/native-stage -p test_arithmetic.py -v
C:/Python311/python.exe ternary-embedder/native-stage/export.py
. C:/dev/mlir-aie/iron_env.ps1
foreach ($shape in @(@(384,384), @(384,1536), @(1536,384))) {
    & C:/dev/mlir-aie/ironenv/Scripts/python.exe ternary-embedder/native-stage/build.py --k $shape[0] --n $shape[1]
    if ($LASTEXITCODE) { throw 'Projection build failed' }
}
dibs run npu --label 'ternary trained native-384 projections' -- C:/Python311/python.exe C:/code/model-experiments/ternary-embedder/native-stage/bench.py
```

`oracle-result.json` records the checkpoint/corpus/reference hashes and CPU
comparisons. `projection-result.json` retains all 36 hardware checks, all timing
samples, artifact/source hashes, toolchain versions and compiled budgets.
Ignored `fixtures/` holds packed matrices, per-matrix float parameters, projection
inputs/references, and full CPU intermediate traces. Ignored `results/` retains
timestamped harness configs, logs and output dumps. Builds stay under `build_*`.
The benchmark rejects stale kernel sources, changed build artifacts, changed
packed weights and changed input/integer-reference fixtures.

## Earlier gate's follow-up (now completed above)

The subsequent fused implementation moved quantization/dequantization, attention, GELU and the two post-residual
LayerNorms onto the NPU and compared intermediate states against the saved
traces. It keeps eps=1e-12 and 12 heads of 32; the older tiny-planner
kernel instead used pre-norm, eps=1e-5 and heads of 64, so it cannot be copied
unchanged. Full-array routing and end-to-end quality/throughput are now measured
above. Resident weights, replicas, pooled NPU vectors and a DMA-only control
remain unmeasured.
