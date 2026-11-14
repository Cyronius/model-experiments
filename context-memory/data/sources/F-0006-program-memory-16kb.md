---
id: F-0006
evidence: measured
last_verified: 2026-09-03
stack: mlir-aie 1.4.2, Peano, XRT master, NPU driver version not recorded
sources: [phlegm-open-kernels, raw/phlegm/2026-09-03-open-kernels-notes.md, raw/phlegm/2026-09-02-array-resource-limits-dma-l1-program-memory.md, S-am027-aie-ml-v2-tile, S-mlir-aie-target-model, raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md, raw/OpenFlowLM-Next3/2026-10-19-oflm-core-memory-arithmetic-and-compiler-traps.md]
---

# F-0006: program-memory-16kb

**Claim:** Core program memory is 16 KB. Python-unrolled loops, per-K template copies, and static per-TU copies of shared kernel bodies overflow it (`XAIE_INVALID_ELF`).

**What to do:** Use `range_`, pass K at runtime, mark shared bodies `inline` + `noinline`, build with `-Os`.

**Where it was found:** phlegm open_kernels/designs (commit history to 2026-09-03).
