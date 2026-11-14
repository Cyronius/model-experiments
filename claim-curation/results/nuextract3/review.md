# Proposed claims for review

Exact citations are checked mechanically. Semantic support has not been certified.

## C-adcf49a361f860f47f40

Every core has 2 input and 2 output DMA channels.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 14-14.

```text
Every core: 2 input + 2 output DMA channels.
```

Citation valid: True. Review status: proposed.

## C-e0de685b9fbc0447dbf4

A build error was encountered for a core requiring 4 input and 2 output DMA channels.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 14-14.

```text
Build error that names it: `requires 4 input/2 output DMA channels, but only 2 input/2 output available`
```

Citation valid: True. Review status: proposed.

## C-65493768739e98b40dc6

A build error was encountered for the DeltaNet glue core.

Conditions: ["DeltaNet glue core"]

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 14-14.

```text
Another seen on the DeltaNet glue core: `tile (2, 3) requires 3 input/1 output DMA channels, but only 2 input/2 output available`
```

Citation valid: True. Review status: proposed.

## C-9cbd5237bf88d4c39f60

A core cannot take a third independent input stream.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 14-14.

```text
A core therefore cannot take a third independent input stream
```

Citation valid: True. Review status: proposed.

## C-0ac03d6c04fbbe0dc6df

The fix used was to ride a header record at the front of the weight stream.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 14-14.

```text
the fix used was to ride a header record at the front of the weight stream (`[xm bf16 | router output f32[1024]]`)
```

Citation valid: True. Review status: proposed.

## C-96a68139827126f628e1

A shim BD's outermost dimension wraps silently above 64.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 27-27.

```text
A shim BD's outermost dimension wraps silently above 64
```

Citation valid: True. Review status: proposed.

## C-0d23f69650cd4c79c8e5

A [72, 2, 64, 64] K fill read chunks 0-7 nine times.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 27-27.

```text
a `[72, 2, 64, 64]` K fill read chunks 0-7 nine times
```

Citation valid: True. Review status: proposed.

## C-60dff508a5987a61f1d2

The read operation produced garbage output.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 27-27.

```text
garbage output
```

Citation valid: True. Review status: proposed.

## C-a61c082119749e0e2f80

Splitting into several fills per pass caused a hang.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 27-27.

```text
Splitting into several fills per pass hung instead
```

Citation valid: True. Review status: proposed.

## C-6f1828253c2928b60064

Keeping large counts in a middle dimension prevents the hang.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 27-27.

```text
Keep large counts in a middle dimension
```

Citation valid: True. Review status: proposed.

## C-7c1359775dd3eb63b4c4

The memtile performs the head-dimension split.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 27-27.

```text
let the memtile do the head-dim split
```

Citation valid: True. Review status: proposed.

## C-cb98dc806e631bbf5a24

The dit_fa trap 1 was triggered.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 27-27.

```text
dit_fa trap 1
```

Citation valid: True. Review status: proposed.

## C-0443e3f7c09987b991fc

The issue was measured on hardware.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 27-27.

```text
Measured on hardware.
```

Citation valid: True. Review status: proposed.

## C-2d76fbaa80b22dbd1640

The stack used was mlir-aie 1.4.2.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 5-5.

```text
mlir-aie 1.4.2
```

Citation valid: True. Review status: proposed.

## C-b7ad40014579ef8fef34

The Peano version used was 21.0.0.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 5-5.

```text
Peano 21.0.0
```

Citation valid: True. Review status: proposed.

## C-3260f4c48e6338947d47

Later runs of Peano were version 21.0.0.2026080301.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 5-5.

```text
later runs 21.0.0.2026080301
```

Citation valid: True. Review status: proposed.

## C-4ad9dcde1069634a2f87

aiecc build errors were quoted.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 5-5.

```text
aiecc build errors quoted below
```

Citation valid: True. Review status: proposed.

## C-4d12deb1a69fcea0e396

XRT/driver versions were not stated for the early items.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 5-5.

```text
XRT/driver versions not stated for the early items
```

Citation valid: True. Review status: proposed.

## C-7a62ea097c4b54c87c97

Granite attention workers with 0x1800 builds, runs, and reads back NaN from the first blocked position (pos >= 4).

Conditions: ["0x1800 builds", "pos >= 4"]

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 41-41.

```text
Granite attention workers: 0x1800 builds, runs and reads back NaN from the first blocked position (pos >= 4);
```

Citation valid: True. Review status: proposed.

## C-9d4b414e8ebf1c9e4865

Granite attention workers with 0x3000 builds work.

Conditions: ["0x3000"]

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 41-41.

```text
0x3000 works (the unrolled tree spills).
```

Citation valid: True. Review status: proposed.

## C-86fe4e094d61740f8793

The firmware version was not stated.

Conditions: []

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 5-5.

```text
NPU driver and firmware not stated
```

Citation valid: True. Review status: proposed.

## C-048457a2d3441f90379a

The `attn_fa` softmax with four interleaved exponentials has a 5 KB frame.

Conditions: []

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 42-42.

```text
`attn_fa` softmax with four interleaved exponentials has a 5 KB frame
```

Citation valid: True. Review status: proposed.

## C-4e5d3d02ef1cc958c4e7

At a 4 KB stack, the array hung (state 8) or returned NaN in approximately 5% of rows.

Conditions: ["4 KB stack"]

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 42-42.

```text
at a 4 KB stack the array hung (state 8) or, slightly over, returned NaN in ~5 % of rows.
```

Citation valid: True. Review status: proposed.

## C-7006d5eb9c4befbc0d1a

A stack of 0x2000 works.

Conditions: []

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 42-42.

```text
`STACK = 0x2000` works.
```

Citation valid: True. Review status: proposed.

## C-1c0a243e534b8d21870a

The firmware version was not stated for the runs.

Conditions: []

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 5-5.

```text
version not stated for these runs
```

Citation valid: True. Review status: proposed.

## C-789f124a62442947725b

The NPU driver and firmware were not stated.

Conditions: []

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 5-5.

```text
NPU driver and firmware not stated
```

Citation valid: True. Review status: proposed.

## C-f2ef358ce783daac2b05

A 128-byte shim transfer silently delivers zeros.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 24-24.

```text
A 128-byte shim transfer silently delivers zeros.
```

Citation valid: True. Review status: proposed.

## C-d76704d505c8eb1f97be

Attention's q (64 bf16 = 128 B) and its 128 B result both arrived empty, with no error.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 24-24.

```text
Attention's q (64 bf16 = 128 B) and its 128 B result both arrived empty, with no error
```

Citation valid: True. Review status: proposed.

## C-2b6c7a26ab8c6e7d6bfe

Every score was wrong but in a plausible range.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 24-24.

```text
every score was wrong but in a plausible range
```

Citation valid: True. Review status: proposed.

## C-26c46924e66b4d41d743

Padding both to 1024 B fixed it.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 24-24.

```text
Padding both to 1024 B fixed it.
```

Citation valid: True. Review status: proposed.

## C-487d89abb791d6824bfc

The same design's 8192 B weight stream and 264 B state readback worked.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 24-24.

```text
The same design's 8192 B weight stream and 264 B state readback worked.
```

Citation valid: True. Review status: proposed.

## C-2090b1d92be384e80abe

The GEMV designs have 128-byte `y` elements and are exact.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 25-25.

```text
The GEMV designs have 128-byte `y` elements and are exact
```

Citation valid: True. Review status: proposed.

## C-5f4e57e577a134428927

The shim transfer size matters, not the ObjectFifo element size.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 25-25.

```text
the shim transfer size matters, not the ObjectFifo element size
```

Citation valid: True. Review status: proposed.

## C-320ef24f24de3c2ced81

The NPU firmware was not stated.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 5-5.

```text
NPU firmware not stated
```

Citation valid: True. Review status: proposed.

## C-c7016f086667efb7284b

The budget used was 64 KB minus the worker stack 0xD00, resulting in 62208 B.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 31-31.

```text
Budget used: 64 KB minus the worker stack 0xD00 = 62208 B.
```

Citation valid: True. Review status: proposed.

## C-a1b0bdc71e9dbe46bc0b

Batch 2 with 5 tiles per call needed 61952 B of buffers and it built and ran.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 31-31.

```text
Batch 2 with 5 tiles per call needs 61952 B of buffers; it built and ran.
```

Citation valid: True. Review status: proposed.

## C-9d92c5307d7cce9e3af9

The "63 KB minus stack" rule (61184 B) would have rejected the batch.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 31-31.

```text
The "63 KB minus stack" rule (61184 B) would have rejected it
```

Citation valid: True. Review status: proposed.

## C-c54814243a0302524625

Falling back silently halved the DMA element and cost approximately 20% throughput.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 31-31.

```text
falling back silently halved the DMA element and cost ~20% throughput.
```

Citation valid: True. Review status: proposed.

## C-07b419b171e3cf1c0735

The NPU firmware version was not stated.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 5-5.

```text
NPU firmware not stated
```

Citation valid: True. Review status: proposed.

## C-916bab4fc88b00e262f5

The ROT13 path completed in 0.602 ms.

Conditions: ["ROT13", "2026-09-01"]

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
classic `run k [2 bufs] -> state 4 (0.602 ms)`,
```

Citation valid: True. Review status: proposed.

## C-ff2c6b70bb467a673af3

The ELF path completed in 0.447 ms.

Conditions: ["ROT13", "2026-09-01"]

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
ELF `0.447 ms`,
```

Citation valid: True. Review status: proposed.

## C-0491af3dfc2efb9603b9

The output was byte-exact.

Conditions: ["ROT13", "2026-09-01"]

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
output byte-exact.
```

Citation valid: True. Review status: proposed.

## C-9f1c1b2100defe981dae

State 4 indicates completion.

Conditions: []

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
State 4 = completed.
```

Citation valid: True. Review status: proposed.

## C-1141d79c4cd56e714ac2

The XRT runtime version is 32.00.20102.3931.

Conditions: []

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 5-5.

```text
XRT runtime 32.00.20102.3931
```

Citation valid: True. Review status: proposed.

## C-0318b3c4e94a90fd0a27

The XRT master version is 2.26.0.

Conditions: []

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 5-5.

```text
XRT master 2.26.0
```

Citation valid: True. Review status: proposed.

## C-d9eb4cfd1573ec8130c8

The mlir-aie version is 1.4.2.

Conditions: []

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 5-5.

```text
mlir-aie 1.4.2
```

Citation valid: True. Review status: proposed.

## C-3ec5fbf7b157d57bb473

The Peano version is 21.0.0.

Conditions: []

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 5-5.

```text
Peano 21.0.0
```

Citation valid: True. Review status: proposed.

## C-2ad1f6c6b29694695528

Driver and firmware versions were not stated for the early items.

Conditions: []

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 5-5.

```text
driver/firmware versions not stated for the early items
```

Citation valid: True. Review status: proposed.

## C-d4437bc64cfb68f16db2

Masked softmax scores are bf16 'lowest', not -inf.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 41-41.

```text
Masked softmax scores are bf16 `lowest`, not -inf
```

Citation valid: True. Review status: proposed.

## C-9fb538aca6bc531309be

The exp argument is S*c - m*c in fp32.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 41-41.

```text
the exp argument is `S*c - m*c` in fp32
```

Citation valid: True. Review status: proposed.

## C-141ca283bc86631cd440

-inf would stay -inf on a path that expects finite values.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 41-41.

```text
-inf would stay -inf on a path that expects finite values
```

Citation valid: True. Review status: proposed.

## C-a88ad2015d4391d80954

The dit_fa trap 5 is triggered when masked softmax scores are -inf.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 41-41.

```text
dit_fa trap 5
```

Citation valid: True. Review status: proposed.

## C-c1b8489f8de9cadf7761

The software stack includes mlir-aie 1.4.2 (dev16 build named for the whisper runs), Peano 21.0.0.2026080301, and Ryzen AI 9 HX 370 in turbo power mode.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 5-5.

```text
mlir-aie 1.4.2 (dev16 build named for the whisper runs), Peano 21.0.0.2026080301, Ryzen AI 9 HX 370 in turbo power mode
```

Citation valid: True. Review status: proposed.

## C-e5bdc970a55d8a816d3a

The XRT/driver version is not stated in these READMEs.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 5-5.

```text
XRT/driver not stated in these READMEs
```

Citation valid: True. Review status: proposed.

## C-9092ea15d1f3aa6c5f7c

The core starts in floor rounding.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
The core starts in floor rounding
```

Citation valid: True. Review status: proposed.

## C-bb86001d0a468457b7d2

In vae_ew, converting to bf16 before any data kernel ran meant the first dispatch after a context load was 2x less accurate than every later one.

Conditions: ["vae_ew", "converting to bf16 before any data kernel ran"]

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
In vae_ew, converting to bf16 before any data kernel ran meant the first dispatch after a context load was 2x less accurate than every later one
```

Citation valid: True. Review status: proposed.

## C-33c058cbabce1ce62e5b

The accuracy of the first dispatch after a context load was 1.8e-4 vs 8.5e-5 rel_fro from identical inputs.

Conditions: ["vae_ew", "converting to bf16 before any data kernel ran"]

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
1.8e-4 vs 8.5e-5 rel_fro from identical inputs
```

Citation valid: True. Review status: proposed.

## C-14e98eebe6a37f583374

The fix is to set the rounding mode in the first kernel of every dispatch (`vew_begin`), not only in the data kernels.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
Fix: set the rounding mode in the first kernel of every dispatch (`vew_begin`), not only in the data kernels.
```

Citation valid: True. Review status: proposed.

## C-9919fd42e203aa73163b

The measurements were taken on mlir-aie 1.4.2 (dev16 build named for the whisper runs), Peano 21.0.0.2026080301, Ryzen AI 9 HX 370 in turbo power mode for the dit_* numbers.

Conditions: ["mlir-aie 1.4.2 (dev16 build named for the whisper runs)", "Peano 21.0.0.2026080301", "Ryzen AI 9 HX 370 in turbo power mode"]

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 5-5.

```text
mlir-aie 1.4.2 (dev16 build named for the whisper runs), Peano 21.0.0.2026080301, Ryzen AI 9 HX 370 in turbo power mode for the dit_* numbers
```

Citation valid: True. Review status: proposed.

## C-2c70993ddf956b95af09

The XRT/driver was not stated in these READMEs.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 5-5.

```text
XRT/driver not stated in these READMEs
```

Citation valid: True. Review status: proposed.

## C-1fb5929bea89749e835a

The ash kernel took 7 ms on device Cedar with compiler 1.2 at batch 2.

Conditions: ["device Cedar", "compiler 1.2", "batch 2"]

Source: [synthetic://synthetic-negation](../../data/sources/synthetic-negation.md), lines 2-2.

```text
On device Cedar with compiler 1.2, the ash kernel took 7 ms at batch 2.
```

Citation valid: True. Review status: proposed.

## C-bf8537632869d6bffac1

The firmware version was not recorded.

Conditions: []

Source: [synthetic://synthetic-negation](../../data/sources/synthetic-negation.md), lines 2-2.

```text
The firmware version was not recorded.
```

Citation valid: True. Review status: proposed.

## C-9a669d03a96569afc647

This measurement does not establish the latency at other batch sizes.

Conditions: []

Source: [synthetic://synthetic-negation](../../data/sources/synthetic-negation.md), lines 2-2.

```text
This measurement does not establish the latency at other batch sizes.
```

Citation valid: True. Review status: proposed.

## C-860a53f087ac93e67b32

The birch kernel required stack 0x1800 with compiler 1.2.

Conditions: ["compiler 1.2"]

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 2-2.

```text
For the birch kernel, stack 0x1800 was sufficient only with compiler 1.2.
```

Citation valid: True. Review status: proposed.

## C-67833840233aec952694

Compiler 1.3 was not tested.

Conditions: []

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 2-2.

```text
Compiler 1.3 was not tested.
```

Citation valid: True. Review status: proposed.

## C-b827c12ef0df09d2b491

The result must not be generalized to compiler 1.3.

Conditions: []

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 2-2.

```text
The result must not be generalized to compiler 1.3.
```

Citation valid: True. Review status: proposed.

