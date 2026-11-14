---
id: F-0038
evidence: measured
last_verified: 2026-09-02
stack: XRT runtime 32.00.20102.3931 (FLM drive work), XRT master 2.26.0 xclbinutil/aiebu-asm, mlir-aie 1.4.2, Peano 21.0.0; driver and firmware not stated
sources: [raw/phlegm/2026-09-02-instruction-stream-ddr-patch-and-kernel-abi.md, raw/fastflowlm/2026-09-02-oflm-kernel-measurements-and-requirements.md]
---

# F-0038: elf-vs-classic-load

**Claim:** ROT13 on one xclbin completed with state 4 on both load paths, the classic mlir-aie flow in 0.602 ms and the `xrt::elf` flow in 0.447 ms, with byte-exact output.

**What to do:** Keep ROT13 as a smoke test. The ELF path has the lower fixed cost here.

**Where it was found:** 2026-09-01 smoke test.
