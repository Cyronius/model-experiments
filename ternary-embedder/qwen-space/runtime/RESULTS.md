# Stage 4: CPU runtime (2026-11-12)

A torch-free C library (`libtde.so`, C ABI) plus a small Python package (`tde/`: ctypes, numpy,
`tokenizers`) that turns text into provisional vectors in the hosted Qwen3-Embedding-8B space.

## Build and portability

- `build.sh`: `-O3 -march=x86-64-v3` baseline (AVX2, FMA, F16C), the AVX-512 VNNI kernel in its own
  translation unit (`src/lin_vnni.cpp`, `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni`),
  chosen once at load with `__builtin_cpu_supports`; `TDE_FORCE_AVX2=1` forces the AVX2 path.
  No OpenMP; single-threaded per call; the Python package runs a worker pool (default 2) and ctypes
  releases the GIL during calls.
- No libstdc++ was available for static linking here, so the library avoids the C++ runtime
  entirely (plain C allocation, no exceptions/RTTI); libgcc is linked statically.
  `ldd`: only libc, libm, the loader. `objdump -T`: the newest symbol version needed is
  **GLIBC_2.16** (2.2.5, 2.14, 2.16), under the 2.28 limit. (`sqrtf@GLIBC_2.43` appeared before
  `-fno-math-errno`; fixed.) AVX-512 instructions appear only in the VNNI object (0 `zmm` uses in
  the baseline objects).
- The library refuses to load (returns null) on a CPU without AVX2/FMA/F16C.

## Model format (`pack.py` -> `model.tde`, 61 MB)

One file: `TDE1` magic, a JSON header (config, metadata, tensor table), 64-byte aligned tensors.
Metadata records the source checkpoint hash (`model-q8.pt`, 38e400de...), the head file hash, the
weight format, the reading mode and the stored dimension.

- Linears: 7-bit codes in [-64, 63], one float scale per output row, stored as code + 64 in the v2
  tiled layout [N/32][K/4][32][4] (outputs across SIMD lanes, weights as the unsigned operand,
  signed int8 activations, offset removed with the activation sum). The AVX2 sequence
  `vpmaddubsw` + `vpmaddwd` cannot saturate (2 x 127 x 128 = 32512) and computes the same integers
  as `vpdpbusd`; the float epilogue is identical on both paths.
- Activations: int8, power-of-two scale per token per 256 channels (whole row for K = 384), as
  `quant_act`.
- Embeddings: int8 word table with a per-row scale; positions, LayerNorms, biases float32.
  Attention, LayerNorm and tanh-GELU run in float32 (vectorised exp for softmax and tanh).
- Reading: the selected head uses the **native** path, one sequence of up to 504 content tokens
  plus [CLS]/[SEP] with the 512-row position table (rows 128-511 from bge-micro-v2). The library
  also supports the windowed path (4 x 126) through the config.
- Head: the MLP (384 -> 2048 GELU -> 4096 plus linear skip) with the feature standardisation folded
  in, float32; then L2 normalisation. The package cuts to the stored dimension (default 1024) and
  re-normalises; `return_confidence=True` also returns the head-output norm.

Python pre-checks (`format_check.py`, `format-check.json`; SS384, 128 contract, core):
int8 embeddings cost **0.077** (limit 0.1: use int8); 7-bit per-row codes derived from the Q8
reconstruction cost **-0.03** (limit 0.3: use them; derived from the float checkpoint: 0.017).
On the final head the whole runtime format costs 0.19 core TS against the Stage 2 float head
(46.93 vs 47.12 at 504/1024).

## Gates (`check.py` -> `check.json`; final model `f6dc1bf0...`)

| Check | Result | Threshold |
|---|---|---|
| Token ids vs the transformers tokenizer, every dev text (19,519) | 0 mismatches | identical |
| 1,000 mixed-length docs (6 length bins up to 504), min cos vs Python reference, VNNI path | 0.99981 (median 0.999994) | >= 0.999 |
| same, AVX2 path | 0.99981 | >= 0.999 |
| Dev TS (504/1024) runtime minus reference, core, VNNI / AVX2 | -0.031 / -0.031 (max per task 0.11) | within 0.15 |
| VNNI vs AVX2, min cos | 0.9999998 | >= 0.9999 |

All pass. Repeated from a fresh virtual environment in Stage 5 (`stage5/fresh-env.json`): same.

## Q4 check (30 minutes, hard stop): dropped

The probe (`q4probe.cpp`: the 384 -> 1536 FFN linear over 504 rows, weights one per byte vs two per
byte unpacked in the loop, AVX2 first) was built at about 15:10. A clean measurement slot never
opened within the box: `dibs run --measure` waited behind another session's build reservation
(until about 18:06). Per the plan, when the 30 minutes run out Q4 is dropped. No timing numbers
exist, so none are reported.

## Timings (dev box, not server numbers)

Not measured: no clean `dibs run --measure` slot opened on 2026-11-12 (other sessions' build
reservations from about 15:00 to past 19:10). `bench.py` is ready; the command is in STATUS.md and
the release README. Engineering targets (504 tokens, one thread, dev box): <= 12 ms VNNI, <= 40 ms
AVX2. Note the selected head reads a 504-token document as one 506-token sequence (native path),
so attention costs about 4x more than the windowed path the targets were estimated for.

## Azure probe

`azure_probe.py` (also in the release folder): CPU model, core/thread counts, flags, glibc, memory;
then, if the release is beside it, a 20-second benchmark at 64 and 504 tokens with 1, 2 and 4
workers; prints one JSON block. Commands are in the release README.
