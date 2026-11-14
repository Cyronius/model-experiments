---
from: phlegm
date: 2026-09-02
evidence: measured
stack: mlir-aie 1.4.2, Peano 21.0.0 (later runs 21.0.0.2026080301), aiecc build errors quoted below; XRT/driver versions not stated for the early items
---

# aie2p array resource limits: DMA channels, memtile, L1 and program-memory budgets, with the exact build errors

Sources: C:/code/phlegm/xdna-dump/sources/open-kernels-phase2-fused-linear-layer.md ("The constraint that shapes the design"); C:/code/phlegm/open_kernels/README.md ("designs/moe_experts", "Item 5, second half"); designs/dit_conv/README.md, dit_ew/README.md, dit_fa/README.md, dit_gemm/README.md; C:/code/phlegm/specs/open-engine/plans/archive/gemm-context-collapse.md ("The gate, honestly"); C:/code/phlegm/specs/open-engine/plans/wide-deltanet-bringup.md and wide-deltanet-a7.md; C:/code/phlegm/specs/open-engine/spec.md (lines ~884, ~2567, ~2581, ~2668, ~5199-5203). F-0001..F-0007 already cover the shim start queue, 16 BDs per column shim, zero strides, BD repeat/len/wrap, the 16 KB program memory in general, and the objectfifo depth+1 rule; this file does not repeat them.

## DMA channel counts (stated as the budget in the plans, confirmed by build errors)

- Every core: 2 input + 2 output DMA channels. Build error that names it: `requires 4 input/2 output DMA channels, but only 2 input/2 output available`. Another seen on the DeltaNet glue core: `tile (2, 3) requires 3 input/1 output DMA channels, but only 2 input/2 output available`. A core therefore cannot take a third independent input stream; the fix used was to ride a header record at the front of the weight stream (`[xm bf16 | router output f32[1024]]`).
- Every column shim: 2 fills + 2 drains, so 16 fills + 16 drains for the 8-column array. The DeltaNet step alone (8 S fills + 8 vec fills, 8 S_out + 8 o drains) uses the whole budget.
- Memtile: 6 S2MM + 6 MM2S channels; the join of 4 core outputs plus A, B and C streams uses all 6+6 in dit_conv and dit_ew.
- Memtile BD pools (aie2p): channels 0/2/4 own BDs 0-23, channels 1/3/5 own BDs 24-47. A join's output took 8 BDs in dit_conv; readers placed on MM2S 0/2/4/1 left even 24/24, odd 22/24 used.
- Memtile L2: 512 KB; dit_conv used 469 KB of 512 (2 x 97.5 KB band + 2 x 9 KB B + 2 x 128 KB C).
- Shared L1 hand-over: an AIE2 core can read its west neighbour's memory. Used to pair odd cores into even cores so the memtile join only needs 4 producers.
- Compiler error for fifo depth: `Number of elements to acquire 3 must be smaller than depth 2` (a main-core xn fifo, H=5120 whole-layer `lx` build).
- Raw shim tasks (`shim_dma_single_bd_task` + `dma_start_task`) inside an IRON runtime sequence must be freed by hand (appended to the TaskGroup's free actions).
- An ObjectFifo's MM2S transfer length is always its L2 object size (mlir-aie `AIEObjectFifoStatefulTransform.cpp:1435-1510` as cited by dit_conv), so a window that re-reads a band 9 times needs explicit `TileDma` + `Lock` + `Flow`.
- A join's `dims_to_stream` applies per joined part, not to the joined buffer (dit_fa trap 2).

## Shim BD limits beyond F-0004

- A shim BD's outermost dimension wraps silently above 64: a `[72, 2, 64, 64]` K fill read chunks 0-7 nine times (right byte count, no hang, garbage output). Splitting into several fills per pass hung instead. Keep large counts in a middle dimension (`[n_chunks, 64, 128]`) and let the memtile do the head-dim split (dit_fa trap 1). Measured on hardware.
- A BD stride is at most 2^20 32-bit words (4 MB), so scratch for an unused half cannot sit a buffer's length away (dit_gemm). K/V fills for a 33792-wide buffer broke at 64 rows; use `[lk/rs, rs, 128]` blocks with `rs * ld` under 2^20 words.
- A shim BD d0 is at most 1023 words (dit_ew).

## L1 (64 KB = 65536 B per core) budgets that decided designs

- q4 prefill GEMM core: ~60 KB of 64 KB (weight fifo 20480, activation 8192, output 16384, nibble scratch 4096, bf16 scratch 8192, stack 4096). moe_batch: ~45 KB (weight fifo 20480, activation 8192, output 4096, up/gate accumulators 8192, stack 4096). They cannot share a core, and the two kernels also need different topologies (GEMM broadcasts weights per row down an L3->L2->L1 chain and joins C at the shim; moe_batch splits a 4-row weight element per column at `Tile(c,1)`), so the 80 GEMM<->expert context changes per block cannot be fused (gemm-context-collapse.md).
- bfp16 GEMM core (`gemm_bfp16`, glu epilogue): data 64520 of 65536 B (A 2 x 4608, B 2 x 9216, C 32768, stack 4096), 1016 B free; program 4848 of 16384 B (1872 B without the glu epilogue).
- `attn_fa` core: program 9968 B before the prep stream, 14128 B with it; data ~59.5 of 64 KiB (Q hi/lo 2 x 4096, scratch 8192, state 2112, O element 8192, A fifo 2 x 8192, B 8192, stack 8192).
- dit_fa core: ~59 KB of 64 KB (A/B fifos 2 x 2 x 8 KB, output 8 KB, Q 8 KB, scores 4 KB, running sum 1 KB, .bss 1.7 KB, stack 4 KB).
- DeltaNet glue core (tile 2,3): fixed allocations 59,904 B leave 5,632 B for the private xn copy, i.e. hidden width <= 2816; at hidden 4096 the build died with `'aie.tile' op allocated buffers exceeded available memory` (68,096 B against 65,536). Fixed by walking xn in 4 KB elements; the glue core then allocated 64,000 of 65,536 B (stack 6144, qk 16384, side 3 x 4096, gact 6 x 2048, gout 4 x 2048, vt 4096, xnb 4096, f32[32] accumulators 512). Placed buffer address map is read from `final.prj/input_with_addresses.mlir`.
- The stack lives at address 0x0 with the fifo buffers after it (open_kernels/README.md dn_glue notes), and a too-small stack was seen to hang the core (state 8). A separate plan (qwen35-remaining-sizes.md) suspects, without a measurement, that a stack overflow on another core corrupts `qk` silently.

## Program memory (16384 B per core): numbers and compiler behaviour

- Build errors seen: `_XAie_LoadProgMemSection: Overflow of program memory`; `XAie_LoadElf failed with XAIE_INVALID_ELF` (F-0006 has this one).
- Soft-float library is 2.6 KB and is pulled in by just three scalar float operations; the main-core kernels avoid scalar float arithmetic entirely (vector lanes, broadcast, split, extract lane 0). Whole-layer main core reached 15.7 KB with every trick; the merged one-context image sits at 16256 of 16384 B.
- aiecc runs Peano `opt` at `default<O2>` with fixed flags. An `--unroll-full-max-count` given to aiecc is accepted and never reaches `opt`. `opt` fully unrolls loops with small trip counts: the Qwen 4B's bands are 20 weight elements (9B: 32), so its main core carried 40 call sites of `gemv_q4_gms` and 28 of `gemv_q4_gy` (9B: 8 and 12): control program 8,208 B vs 5,568 B, core linked to 16,480 B of 16,384 (2026-10-01). Keeping band loops rolled for the 4B fixed it.
- all-q8 35B: the DeltaNet pass-2 update grew the MoE main core to 16,608 B vs 16,384. Compiling only `gemv_q4_gup` at `-Oz` gives 16,304 B; `-Oz` on `gemv_q4_gdown` as well gives 16,448 B (over). A per-row pass-2 variant fit (16,080 B) but ran `lx0` at 2.48 ms vs 2.02 ms.
- A mixed q4/q8 `lx` main core (Qwen3.5 4B, hidden 2560) overflows even after `-Oz` on the two GEMV TUs and then on the whole core.
- Program-memory growth is data-dependent on trip count: making K a runtime loop bound cost +64 B per core (+2048 B over 32 cores, 1.4% of ~4.7 KB); see raw/phlegm/2026-09-13-iron-dataflow-hangs-and-runtime-loop-bounds.md.

## Resource-driven hangs (all ERT state 8)

- Qwen3.5-2B and 0.8B (16 value heads): all four kernel sets built, then both timed out on the first `lx` dispatch because `dn_glue.h` hardcoded `kNHead = 32` as a constexpr. Making it a knob (`DNGLUE_NHEAD`) fixed it (bit-exact conv state, cos 1.0 on k/q/v/decay/beta, maxrel <= 1.1e-5).
- `dx_attn.py` (Gemma 3, Phi-4-mini): emitted `NHL // HPO` og elements per core, which is zero when a core owns fewer heads than an og element carries (Gemma 3: NHL 2, HPO 4; Phi-4-mini: 4, 8). First `dxB` dispatch timed out (ERT state 8) with or without the sliding-window kernel. Fixed by `kOGH = min(NHL, HPO)`.
