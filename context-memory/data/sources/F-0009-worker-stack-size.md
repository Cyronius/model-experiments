---
id: F-0009
evidence: measured
last_verified: 2026-09-02
stack: mlir-aie 1.4.2, Peano, XRT master, NPU driver version not recorded
sources: [phlegm-open-kernels, raw/phlegm/2026-09-03-open-kernels-notes.md, raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md]
---

# F-0009: worker-stack-size

**Claim:** A worker stack of 0xD00 hangs with fp32 vector code. 0x1800 works.

**What to do:** Set the stack to at least 0x1800 for fp32 vector kernels.

**Where it was found:** phlegm open_kernels/designs (commit history to 2026-09-03).

**Disputed:** Q-0028: LLMNpuTest GEMV and elementwise workers ran at 0xD00, and Granite attention workers read NaN at 0x1800 and needed 0x3000 (raw/LLMNpuTest/2026-09-01-dma-l1-program-memory-limits-and-silent-faults.md, raw/OpenFlowLM-Next3/2026-10-19-oflm-iron-peano-aiecc-dma-traps-and-hangs.md).
