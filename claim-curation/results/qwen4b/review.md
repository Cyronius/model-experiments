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

## C-3d9cb66c8bac40111dd9

A build error indicates that a core requires 4 input and 2 output DMA channels but only has 2 input and 2 output available.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 14-14.

```text
Build error that names it: `requires 4 input/2 output DMA channels, but only 2 input/2 output available`.
```

Citation valid: True. Review status: proposed.

## C-165a6a0ce06ab09dd226

A build error on the DeltaNet glue core indicates that tile (2, 3) requires 3 input and 1 output DMA channel but only has 2 input and 2 output available.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 14-14.

```text
Another seen on the DeltaNet glue core: `tile (2, 3) requires 3 input/1 output DMA channels, but only 2 input/2 output available`.
```

Citation valid: True. Review status: proposed.

## C-cc1bd2b80254246bff04

A core cannot take a third independent input stream due to DMA channel limitations.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), lines 14-14.

```text
A core therefore cannot take a third independent input stream;
```

Citation valid: True. Review status: proposed.

## C-3436113376104d886baa

The fix for the DMA channel limitation involved riding a header record at the front of the weight stream.

Conditions: []

Source: [raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md](../../data/sources/array.md), unresolved.

```text
The fix used was to ride a header record at the front of the weight stream (`[xm bf16 | router output f32[1024]]`).
```

Citation valid: False. Review status: proposed.

Validation: quote is not verbatim in supplied text; quote has no allowed original-source span

## C-15106b16f687cf0115b7

Granite attention workers using 0x1800 builds read back NaN from the first blocked position (pos >= 4).

Conditions: ["0x1800 builds"]

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 41-41.

```text
Granite attention workers: 0x1800 builds, runs and reads back NaN from the first blocked position (pos >= 4);
```

Citation valid: True. Review status: proposed.

## C-a5c761d53036235eee62

Granite attention workers using 0x3000 builds experience unrolled tree spilling.

Conditions: ["0x3000 works"]

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), lines 41-41.

```text
0x3000 works (the unrolled tree spills).
```

Citation valid: True. Review status: proposed.

## C-e258dfbb4df1fa97a827

The `attn_fa` softmax with four interleaved exponentials has a 5 KB frame and hangs (state 8) or returns NaN in ~5% of rows when using a 4 KB stack.

Conditions: ["attn_fa softmax with four interleaved exponentials", "5 KB frame", "4 KB stack", "hung (state 8)", "returned NaN in ~5% of rows"]

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), unresolved.

```text
attn_fa softmax with four interleaved exponentials has a 5 KB frame: at a 4 KB stack the array hung (state 8) or, slightly over, returned NaN in ~5 % of rows.
```

Citation valid: False. Review status: proposed.

Validation: quote is not verbatim in supplied text; condition fragments are not verbatim: ['attn_fa softmax with four interleaved exponentials', 'returned NaN in ~5% of rows']; quote has no allowed original-source span

## C-29223bc91c1e73d19788

A stack size of `STACK = 0x2000` works for the `attn_fa` softmax with four interleaved exponentials.

Conditions: ["STACK = 0x2000"]

Source: [raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md](../../data/sources/granite.md), unresolved.

```text
STACK = 0x2000 works.
```

Citation valid: False. Review status: proposed.

Validation: quote is not verbatim in supplied text; quote has no allowed original-source span

## C-4f63a65829d359669af7

A 128-byte shim transfer silently delivers zeros, and Attention's q (64 bf16 = 128 B) and its 128 B result both arrived empty with no error, with every score being wrong but in a plausible range.

Conditions: ["shim transfer size 128 bytes", "Attention's q (64 bf16 = 128 B)", "128 B result", "no error", "every score wrong but in plausible range"]

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 24-24.

```text
A 128-byte shim transfer silently delivers zeros. Attention's q (64 bf16 = 128 B) and its 128 B result both arrived empty, with no error; every score was wrong but in a plausible range.
```

Citation valid: False. Review status: proposed.

Validation: condition fragments are not verbatim: ['shim transfer size 128 bytes', 'every score wrong but in plausible range']

## C-830a8e72bccf820a6efa

Padding both to 1024 B fixed the issue with the 128-byte shim transfer.

Conditions: ["padding to 1024 B"]

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 24-24.

```text
Padding both to 1024 B fixed it.
```

Citation valid: False. Review status: proposed.

Validation: condition fragments are not verbatim: ['padding to 1024 B']

## C-7b36075c14f22846861b

The same design's 8192 B weight stream and 264 B state readback worked.

Conditions: ["same design", "8192 B weight stream", "264 B state readback"]

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 24-24.

```text
The same design's 8192 B weight stream and 264 B state readback worked.
```

Citation valid: True. Review status: proposed.

## C-df13a721d93dacea937d

GEMV designs have 128-byte `y` elements and are exact because the shim moves the whole drain at once, and the shim transfer size matters, not the ObjectFifo element size.

Conditions: ["GEMV designs", "128-byte y elements", "shim moves whole drain at once", "shim transfer size matters, not ObjectFifo element size"]

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 25-25.

```text
The GEMV designs have 128-byte `y` elements and are exact, because there the shim moves the whole drain at once: the shim transfer size matters, not the ObjectFifo element size.
```

Citation valid: False. Review status: proposed.

Validation: condition fragments are not verbatim: ['128-byte y elements', 'shim moves whole drain at once', 'shim transfer size matters, not ObjectFifo element size']

## C-8ca5348b16082a710134

The information about NPU firmware is not stated.

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

## C-99906924ec10198f4467

A batch of 2 with 5 tiles per call required 61952 B of buffers.

Conditions: ["batch 2", "5 tiles per call"]

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 31-31.

```text
Batch 2 with 5 tiles per call needs 61952 B of buffers;
```

Citation valid: False. Review status: proposed.

Validation: condition fragments are not verbatim: ['batch 2']

## C-3a642ff5f52c241d4b73

The configuration built and ran successfully.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 31-31.

```text
it built and ran.
```

Citation valid: True. Review status: proposed.

## C-63fafa85af826d0c00b6

The '63 KB minus stack' rule (61184 B) would have rejected the configuration.

Conditions: ["63 KB minus stack", "61184 B"]

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), unresolved.

```text
The '63 KB minus stack' rule (61184 B) would have rejected it,
```

Citation valid: False. Review status: proposed.

Validation: quote is not verbatim in supplied text; quote has no allowed original-source span

## C-60235858be6d4fdca588

Falling back silently halved the DMA element and reduced throughput by approximately 20%.

Conditions: []

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 31-31.

```text
falling back silently halved the DMA element and cost ~20% throughput.
```

Citation valid: True. Review status: proposed.

## C-46ff5255ee8566e75c7e

The issue was reported in granite_gemv.py lines 61-66.

Conditions: ["granite_gemv.py", "61-66"]

Source: [raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md](../../data/sources/llm.md), lines 31-31.

```text
(granite_gemv.py:61-66)
```

Citation valid: True. Review status: proposed.

## C-00b5244ded16fcd83dd3

Both paths loaded ROT13 on 2026-09-01.

Conditions: ["2026-09-01"]

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
Both paths loaded ROT13 (2026-09-01)
```

Citation valid: True. Review status: proposed.

## C-1279c317670c0fee9fae

The classic path took 0.602 ms to reach state 4.

Conditions: ["classic", "run k [2 bufs] -> state 4"]

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
classic `run k [2 bufs] -> state 4 (0.602 ms)`
```

Citation valid: True. Review status: proposed.

## C-c4155ed67fadcdbbce77

The ELF path took 0.447 ms to reach state 4.

Conditions: ["ELF"]

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
ELF `0.447 ms`
```

Citation valid: True. Review status: proposed.

## C-6a7484b2bad050d23754

Both paths produced output that was byte-exact.

Conditions: []

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
output byte-exact
```

Citation valid: True. Review status: proposed.

## C-fb2fb3dfe4aabdf14d0d

State 4 represents a completed state.

Conditions: []

Source: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md](../../data/sources/abi.md), lines 16-16.

```text
State 4 = completed.
```

Citation valid: True. Review status: proposed.

## C-e87acbf3b299e2b2bf2e

The core starts in floor rounding by default, as indicated by the vegah trap list.

Conditions: ["vegah trap list"]

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
The core starts in floor rounding (vegah trap list says the same: "floor rounding default").
```

Citation valid: True. Review status: proposed.

## C-93ec2f14ac09d37ee8b1

In vae_ew, converting to bf16 before any data kernel ran resulted in the first dispatch after a context load being 2x less accurate than every later one, with relative Fro norms of 1.8e-4 vs 8.5e-5 from identical inputs.

Conditions: ["vaew_ew", "converting to bf16 before any data kernel ran", "first dispatch after a context load", "identical inputs"]

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
converting to bf16 before any data kernel ran meant the first dispatch after a context load was 2x less accurate than every later one: 1.8e-4 vs 8.5e-5 rel_fro from identical inputs.
```

Citation valid: False. Review status: proposed.

Validation: condition fragments are not verbatim: ['vaew_ew']

## C-5b4b0d23d857beb47c95

The fix for the accuracy issue was to set the rounding mode in the first kernel of every dispatch (`vew_begin`), not only in the data kernels.

Conditions: ["set the rounding mode in the first kernel of every dispatch", "vew_begin", "not only in the data kernels"]

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
Fix: set the rounding mode in the first kernel of every dispatch (`vew_begin`), not only in the data kernels.
```

Citation valid: True. Review status: proposed.

## C-3970c3b9f978c81540fc

The accuracy issue was measured.

Conditions: []

Source: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md](../../data/sources/arithmetic.md), lines 28-28.

```text
Measured.
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

## C-45349410a2bfb2289cb9

Stack 0x1800 was sufficient for the birch kernel only with compiler 1.2.

Conditions: ["birch kernel", "compiler 1.2"]

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 2-2.

```text
For the birch kernel, stack 0x1800 was sufficient only with compiler 1.2.
```

Citation valid: True. Review status: proposed.

## C-e67149d0e67ac3187b74

Compiler 1.3 was not tested for the birch kernel.

Conditions: ["birch kernel"]

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 2-2.

```text
Compiler 1.3 was not tested.
```

Citation valid: True. Review status: proposed.

## C-47297e1bf67d7f65febf

The result must not be generalized to compiler 1.3.

Conditions: ["compiler 1.3"]

Source: [synthetic://synthetic-version-scope](../../data/sources/synthetic-version-scope.md), lines 2-2.

```text
The result must not be generalized to compiler 1.3.
```

Citation valid: True. Review status: proposed.

