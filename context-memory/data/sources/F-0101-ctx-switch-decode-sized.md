---
id: F-0101
evidence: measured
last_verified: 2026-09-13
stack: mlir-aie 1.4.2, Peano 21.0.0; XRT 2.26.0, amdxdna 2.26.0_20260817, firmware 1.1.2.64 for the 2026-09-24 runs only; other runs not stated; the XRT 2.26.0 / amdxdna 2.26.0_20260817 runs were on a Linux Ryzen AI 9 365 box (F-0178)
sources: [raw/phlegm/2026-09-13-context-switch-costs-and-submit-ahead.md, raw/fastflowlm/2026-09-02-oflm-kernel-measurements-and-requirements.md]
---

# F-0101: ctx-switch-decode-sized

**Claim:** With decode-sized kernels (floor.cfg, 2026-09-02) a dispatch costs 0.23-0.25 ms back to back in one context and 0.65-0.79 ms when the previous dispatch used another xclbin context; silu_mul after a switch costs 0.29-0.37 ms.

**What to do:** Plan ~0.4-0.5 ms per switch between xclbins at decode sizes. 132 dispatches times ~0.45 ms was ~60 ms of a 208 ms 27B step. See Q-0006 for other figures.

**Where it was found:** decode_chain/floor.cfg, driver `runlist` and `runx`.
