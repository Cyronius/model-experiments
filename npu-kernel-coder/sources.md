# npu-kernel-coder — documentation sources

Corpus for retrieval at inference time and for a small slice of training text. "Read" = someone has actually gone through it for this project; everything else is just listed.

## Local / our own

| source | what | status |
|---|---|---|
| `C:\code\openflowlm-next\.claude\skills` | aie-code-creator, aie-dataflow-opt, aie-dataflow-presim, aie-hw-bringup, aie-kernel-opt, peano-kernel-codegen, + engine/server skills | inventoried |
| `C:\code\openflowlm-next\.agents\skills` | per-model-family kernel skills (granite, phi3/nanbeige, qwen36, qwen38-27b, deltanet, dit-*, attention row groups) | inventoried |
| `C:\code\FastFlowLM` | local checkout of ROCm/FastFlowLM | not read |

## Community / third party

| source | what | status |
|---|---|---|
| [mistrjirka/pytorch-XDNA2 `xdna_tools`](https://github.com/mistrjirka/pytorch-XDNA2/tree/main/xdna_tools) | XDNA2 tooling | not read |
| [artlite](https://github.com/artlite-project/artlite) (whole repo, GPL-3.0) | XDNA2-only LLM + SD runtime; kernels generated at run time; skills, docs, tests, compile-analysis tools. Inventory in [plan.md](plan.md#data-we-have) | **inventoried** (snapshot `4699b93`, 2026-10-18; single squashed commit, so no history to mine) |
| [ROCm/FastFlowLM #636](https://github.com/ROCm/FastFlowLM/issues/636) | issue thread | not read |
| [Daniel Estévez: Getting peak TOPS on a Ryzen AI 7 350 NPU](https://destevez.net/2026/05/getting-peak-tops-on-a-ryzen-ai-7-350-npu/) | compute roofline on XDNA2 | not read |
| [atassis/xdna-engine](https://github.com/atassis/xdna-engine) | reverse-engineered engine | not read |
| [AngryLoki: DRAM bandwidth on Strix Halo (SVG)](https://gist.githubusercontent.com/AngryLoki/2cbd8be31138b5ba2dc08538c2a512fc/raw/94b642fd19b1e7160c08669323655a2f8082e6f9/dram_bandwidth.svg) | memory roofline for CPU / iGPU / NPU | **read**, see digest below |

## AMD official

| source | what |
|---|---|
| [AM027 AIE-ML v2 architecture](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2): [memory tile](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/AIE-ML-v2-Memory-Tile-Overview-and-Features), [memory tile DMA](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/AIE-ML-v2-Memory-Tile-DMA), [compute tile memory](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/AIE-ML-v2-Memory-Module), [vector unit](https://docs.amd.com/r/en-US/am027-versal-aie-ml-v2/Vector-Unit) | hardware reference for XDNA2's tile array |
| [UG1603 AIE-ML kernel & graph programming](https://docs.amd.com/r/en-US/ug1603-ai-engine-ml-kernel-graph) | kernel programming guide (Vitis flow, but the AIE API content carries over) |
| [Linux kernel amdxdna docs](https://docs.kernel.org/accel/amdxdna/amdnpu.html) | driver / firmware model |
| [Xilinx/llvm-aie (Peano)](https://github.com/Xilinx/llvm-aie) | the compiler our kernels build with |
| [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) + [IRON programming guide](https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/README.md) | dataflow layer; the largest body of example designs |
| [mlir-aie PR #3588](https://github.com/Xilinx/mlir-aie/pull/3588) | specific PR flagged as relevant |
| [amd/IRON](https://github.com/amd/IRON) | optimized kernel library; reference solutions |
| [amd/xdna-driver](https://github.com/amd/xdna-driver), [Xilinx/XRT](https://github.com/Xilinx/XRT) | runtime |
| [ROCm/FastFlowLM](https://github.com/ROCm/FastFlowLM) | NPU inference engine |
| [amd/triton-xdna](https://github.com/amd/triton-xdna) (issues [#82](https://github.com/amd/triton-xdna/issues/82), [#83](https://github.com/amd/triton-xdna/issues/83)) | Triton backend; a possible higher-level target language |
| [Ryzen AI docs](https://ryzenai.docs.amd.com/) | product-level docs, mostly ONNX/Vitis AI EP; low value for kernels |

Search leads, not corpora: [arXiv "XDNA"](https://arxiv.org/search/?query=XDNA&searchtype=all), GitHub code search for `aie2p`, repo search for `xdna2`.

Licensing: the GitHub repos are mostly Apache-2.0 (mlir-aie and llvm-aie carry the LLVM exception). docs.amd.com pages are AMD copyright. Fine for retrieval; check before redistributing them as part of a published dataset.

## Digest: DRAM bandwidth on Strix Halo (AngryLoki SVG)

Probably from the artlite team: the numbers and styling match artlite's README and `docs/npu-vs-igpu.md`.

Hardware: ASUS ROG Flow Z13, Ryzen AI MAX+ 395, 128 GB LPDDR5X-8000, 256-bit bus (16 ch × 16 bit, 256 GB/s peak). Measured October 2026.

**The NPU's DRAM ceiling is one port, not the DRAM.**
- All 8 shims share one 32 B/clk port: 32 B × 1.8 GHz = **57.6 GB/s**. Measured: read 56.5, write 57.3, mixed read+write 75.6.
- That's 22% of the bus. The iGPU reaches 243.6 (95%), both CPU CCDs 120.1.
- LLM decode on the NPU reads 53.0 GB/s, 92% of the port ceiling.

**Kernel design rules that follow:**
1. **More than 4 DMA channels (2 shims) doesn't raise bandwidth.** Each MM2S channel is 8 B/clk: 1 channel = 13.8 GB/s, 4 channels hit the port limit. Use more shims for layout or parallelism, not for bandwidth.
2. **Avoid gathers with a 4 KB-multiple stride and small rows.** A 64 KB stride puts every row on one DRAM channel: rows ≤ 256 B get ~14–16 GB/s, 512 B ~30, 1 KB recovers to ~56. Stride 2R is fine (~45 GB/s at 64 B rows, full rate from 128 B up). Pad leading dimensions off powers of two.
3. **NPU bandwidth scales linearly with NPU clock** (≈ 31.7 B × clock). Power mode sets the clock: powersaver 25.3, balanced 40.2, default 56.6 GB/s. FCLK and DRAM clocks don't matter.

**Sharing the bus:**
- Total reads stop at 236–240 GB/s whatever the mix. The iGPU wins arbitration.
- NPU keeps 88% of its solo rate next to the CPU, 73% next to the iGPU, 67% with all three.
- NPU + iGPU decoding at once: 204 GB/s total, only 7% above the iGPU alone.

**What this means for the project:**
- **The verifier must pin the NPU power mode.** Otherwise perf scores swing 2.2× between powersaver and default.
- **Score speed as a fraction of the roofline**, not only against our reference kernel. A memory-bound kernel's ceiling is bytes moved ÷ 57.6 GB/s at default clock. That catches the case where our reference is itself slow.
- Rules 1–3 belong in the training data and the skills as worked examples (bad stride → fix → measured gain), not just as prose.
- Caveat: measured on Strix Halo. The 32 B/clk NPU port comes from AMD's Strix Point slide, so the NPU ceiling should hold on Ryzen AI 300 too. The stride and contention numbers depend on the memory system, so re-measure there.
