# Native 384 versus 512 on XDNA2

Measured on the local Strix NPU on 2026-10-23. **Keep hidden width 384 for the
custom embedding kernel. There is no compulsory rounding to 512.**

The same native ternary primitive compiled and ran at K = 384, 512, 1536 and
2048. All outputs matched an independent int64 CPU dot-product reference
exactly. The generated 384 image allocates an actual 1536-byte activation
buffer (4 x 384 int8) and 3072-byte weight buffer (384 x 32 x 2 bits), and its
DMA transfers exactly those lengths. The input is not secretly widened to 512.

## Measurements

One compute core, four input rows, 32 output channels, 2-bit ternary weights
expanded into int4 operands, int8 activations, int32 accumulation. Operands stay
in core memory inside the repeated arithmetic loop. Each dispatch accumulates
1024 repeated products; these times are **not per-embedding latencies**.

| Actual input width K | Native K/16 blocks | Warm median per dispatch | P10–P90 | Samples |
|---|---:|---:|---:|---:|
| 384 | 24 | 0.4365 ms | 0.4003–0.4954 ms | 164 |
| 512 | 32 | 0.5290 ms | 0.4950–0.5964 ms | 164 |
| 1536 | 96 | 1.2640 ms | 1.2000–1.3574 ms | 164 |
| 2048 | 128 | 1.6200 ms | 1.5650–1.6910 ms | 164 |

512 took 1.212x the time of 384; 2048 took 1.282x the time of 1536. Four
ABBA/BAAB groups were used for each pair; four warm-up calls were excluded per
block. The host was loaded (about 69% CPU before the run); this was a `dibs run
npu` reservation, not a whole-machine quiet benchmark. Every corresponding
block-median comparison favored the smaller native width. Raw logs/fixtures
are in `results/`; the retained aggregate is `width-result.json`.

## What this settles

- The primitive uses 16 input channels and 32 output channels at a time. Both
  model widths and both FFN widths are exact multiples. No K or N tail is needed
  for their dense projections with this blocking.
- The 512-wide primitive does more work and ran slower. Width 384 is not charged
  for 512 channels by this custom implementation.
- Calculated from the model shapes, a full 384/1536 encoder layer has 1,769,472
  ternary weights (432 KiB packed); a 512/2048 layer has 3,145,728 (768 KiB).
  The 512 model needs 1.778x as many dense MACs and packed weight bytes. These
  full-layer counts are arithmetic, not measured full-layer timings.
- The older `dit_gemm` padding rules belong to that software design. They are
  not a reason to widen this model when writing a dedicated ternary kernel.

## What remains unmeasured

This is a projection arithmetic/packing probe, not a complete embedding layer.
It does not include activation quantization/scales, biases, GELU, LayerNorm,
attention, pooling, inter-core reductions, or a resident memtile weight layout.
In particular it does not reproduce the different per-256 activation scales of
the complete trained 512 model. It establishes exact native integer products
and their cost, not full-model numerical equivalence or throughput. The
four-replica layout and the earlier ~750k tok/s estimate are still unvalidated.

Next: implement/profile a complete native-384 stage and validate its states and
vectors against the trained CPU model. Keep the quality gate separate from the
hardware width decision.

## Reproduce

Source: `probe.cc`, `design.py`, `build.py`, `bench.py`. The build pins `npu2`
with one column and uses mlir-aie 1.4.2 and Peano
21.0.0.2026080301+c9c5ecb7 from `C:/dev/mlir-aie/ironenv`.

The old WSL distributions refer to missing VHDX files. The native compiler
worked, but `xclbinutil` was absent. It was built without modifying the installed
runtime, from Xilinx/XRT commit
`676a145e6a083d9f7fdf58b31c387e14f880e98a`, under ignored `tools/XRT`.
Boost 1.92 dependencies are under ignored `tools/deps`; `tools/xclbinutil-build`
contains the packager. The installed C++ `run_kernel.exe` harness opens the
local device as `NPU Strix`; its runtime DLL comes from `C:/dev/xrtNPUfromDLL`.
No GPU training, power-mode changes, or driver changes were performed.

PowerShell (from repository root):

```powershell
. C:/dev/mlir-aie/iron_env.ps1
foreach ($k in @(384, 512, 1536, 2048)) {
    & C:/dev/mlir-aie/ironenv/Scripts/python.exe ternary-embedder/npu-width-probe/build.py --k $k
    if ($LASTEXITCODE) { throw "Build failed: $k" }
}
dibs run npu --label 'ternary native width comparison' -- C:/Python311/python.exe C:/code/model-experiments/ternary-embedder/npu-width-probe/bench.py
```

The packager was configured with the upstream standalone CMake project,
`XRT_SOURCE_DIR=<tools/XRT>/src`, `XRT_EDGE=ON`,
`CMAKE_PREFIX_PATH=<tools/deps>/x64-windows`, `Boost_USE_STATIC_LIBS=OFF`, and
the Visual Studio 17 2022 x64 generator; build target `xclbinutil`, Release.
The probe builder automatically adds the resulting executable and Boost DLLs
to its child process's search path.
