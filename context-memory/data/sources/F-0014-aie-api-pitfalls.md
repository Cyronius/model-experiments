---
id: F-0014
evidence: measured
last_verified: 2026-09-02
stack: mlir-aie 1.4.2, Peano, XRT master, NPU driver version not recorded
sources: [phlegm-open-kernels, raw/phlegm/2026-09-03-open-kernels-notes.md]
---

# F-0014: aie-api-pitfalls

**Claim:** On device: aie_api int32->float `to_float` on 16 lanes gives NaN. int16 `bit_and` scalarises. `get_cycles()` is unlinkable under Peano. Static vars in core code are not zeroed.

**What to do:** Mask through a uint8 view. Zero statics yourself. Avoid `get_cycles()`.

**Where it was found:** phlegm open_kernels/designs (commit history to 2026-09-03).
