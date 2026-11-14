---
id: F-0066
evidence: measured
last_verified: 2026-09-26
stack: mlir-aie 1.4.2, Peano 21.0.0.2026080301, Ryzen AI 9 HX 370 (dit_* numbers in turbo power mode); XRT and driver not stated
sources: [raw/phlegm/2026-09-26-aie2p-arithmetic-and-stack-traps.md]
---

# F-0066: masked-softmax-lowest

**Claim:** Masked softmax scores must be bf16 `lowest`, not -inf, because the exp argument is `S*c - m*c` in fp32 and -inf stays -inf on a path that expects finite values.

**What to do:** Mask with `lowest`.

**Where it was found:** dit_fa trap 5.
