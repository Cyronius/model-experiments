---
from: OpenFlowLM-Next3 (C:/code/OpenFlowLM-Next3-decB, branch perf/qwen3-decode-b, HEAD 69e1b98 2026-10-19; same skills on perf/qwen3-decode-a)
date: 2026-10-19
evidence: measured
stack: Windows 11, Ryzen AI 9 HX PRO 370 (Strix Point), aie2p; IRON / aiecc / Peano from the mlir-aie Windows build, version not stated for these runs (last stated 1.4.2.dev16+g7e00b57 / Peano 21.0.0.2026080301, 2026-09-06); XRT from C:\Xilinx\XRT, version not stated; NPU driver and firmware not stated
---

# DMA, stream-routing and dispatch-ordering traps, and the hangs they caused

Sources (C:/code/OpenFlowLM-Next3-decB): `.opencode/skill/open-dxl-rows/SKILL.md:21-23, 64-79, 208-209, 274-289`, `.agents/skills/open-granite-kernels/SKILL.md:264-265, 296-305, 362-389, 466-470, 503-504`, `.opencode/skill/open-gemm-bfp16/SKILL.md:65-67, 114-115, 157-161`, `.opencode/skill/open-qwen3-small/SKILL.md:14-24, 86-88`, `.opencode/skill/open-k2-kernels/SKILL.md:121-124`, `specs/open-engine/spec.md:209-213, 677-681, 4786-4788, 4898-4904, 4938-4941`. Each item was hit in a build or on hardware. Core-arithmetic and compiler traps are in `2026-10-19-oflm-core-memory-arithmetic-and-compiler-traps.md`. None of this is on the openflowlm-next main checkout.

## Streams, BDs, ports and channels

- Every stream to DDR leaves the compute rows through row 2's south ports, 4 a column, 32 in all. Granite's L-row pass needed 33 (og, KV, y and norm outflows), and aiecc reported it only as "Unable to find a legal routing".
- A memtile join takes at most 6 inputs (6 S2MM channels), or 5 when a weight ring pins S2MM channel 0.
- aie2p memtiles split their 48 BDs into even / odd pools of 24, and a ring takes one BD per buffer on each side. With the ring's in and out both on channel 0 it overflows past depth 12; pin them to channels 0 and 1 (repeats F-0047 with a new case).
- A 24-deep weight ring on `dxl` does not route ("no space for this BD": the og joins share the memtiles).
- A window placeholder must be at least two rows. The lowering drops the stride of a size-1 dimension, and a BD whose length is later patched past it re-reads row 0.
- A window fill whose row dimension is not the outermost BD dimension wraps after RB rows. Keep `[1, RB, 2, run]`.
- A shim drain with four dimensions treats the fourth as a repeat count: `Buffer descriptor length does not match` at `npu_dma_lowered`. A bf16 run of 2048 is split at 1024 on the way, so a 3-D pattern with a 2048 inner run becomes 4-D (repeats F-0004 / F-0053 with the error text).
- The shim cannot split one stream into "A tile here, junk elsewhere" with one affine pattern. A tap in `v8bfp16ebs8` units needs offsets in whole 36-byte groups; 32 KB slots do not qualify, 8199-word slots do.
- A shim fill that reorders rows in 16-byte runs is ~3.5x slower than a contiguous one. The Qwen3 prep stream took 5.9 ms with such a pattern, 1.6 ms head-major, 0.80 ms contiguous.
- A stream takes at most five data buffers. A sixth fails with XRT "invalid bitset position" (consistent with F-0011 and F-0035).
- XRT accepted an `xrt::bo` sub-buffer of a layer's state buffer on an `ext::bo` when the offset was 4 KiB aligned (pos0 x 2048 B). Prompts whose `pos0 * state_row` is not a multiple of 4096 take a host path instead.
- `Pipeline(3)` (the 4-BD shim queue, F-0001) awaits a channel's oldest fill before queueing a fourth. A fill issued early whose await needs a later transfer deadlocks.

## Hangs (ERT state 8) and dispatch ordering

- An RTP word read right after the runtime barrier holds the previous dispatch's value. On `dx` + `dxh`, layer -> layer and head -> head ran, but layer -> head hung (ERT state 8, serial and queued alike). Fix: read it after the dispatch's first data lands (repeats F-0061 with a new case).
- A mode written to the cores' RTP word in-context did not take effect on `dxl`: a draft layer right after verify layers, and the head after them, ran the previous mode's table and hung (ERT state 8). It had only ever switched across a context reload. Delaying the barrier behind a DMA round trip did not help. Fix: the first x element of each dispatch carries the mode.
- A core that waits on the runtime barrier must never release it: a release adds 1, and the next dispatch would start on the old mode.
- In the LoRA draft, the z-fill wait has to come before that tile's own drain is issued, or it waits on a drain that needs the fill it blocks (timeout, state 8).
- `attn_fa`: four passes' BDs in flight (`AF_INFLIGHT=4`) hang; 2 works.
- `dit_fa` prep with a bfp16 cache: two steps of three drains a column in flight hang (state 8, awaited or not); one step runs (+~0.3 ms a layer).
- `attn_fa` prep: two whole passes in flight hang the array (state 8, 7 s timeout) when a second input stream is consumed; Q passes alone ran 10 deep at 2 in flight (repeats F-0062).
- An older engine passing three buffers to the new four-buffer `af*` streams hangs.
- One xclbin, many instruction streams: the context's xclbin is the first step's build. After a core-program change, rebuild every stream, or new streams run on the old program. A `dit_fa` set that mixes old and new streams hangs.

## Stack

- Granite attention workers: 0x1800 builds, runs and reads back NaN from the first blocked position (pos >= 4); 0x3000 works (the unrolled tree spills).
- `attn_fa` softmax with four interleaved exponentials has a 5 KB frame: at a 4 KB stack the array hung (state 8) or, slightly over, returned NaN in ~5 % of rows. `STACK = 0x2000` works. Read the frame from `llvm-objdump -d` (`paddxm [sp], #N`).

## Host buffer trap

- Never write into a device buffer's host map without syncing it to the device. On this driver a FROM_DEVICE sync is a CLFLUSH loop, and it wrote the dirty lines back over the next dispatch's output: every kv group after the first saw the previous group's scores (K2 prefill). This is a third case of the trap in raw/openflowlm-next/2026-10-17-hangs-wrong-output-and-cache-coherence-traps.md (q8 route, Whisper).

## Doubtful

- The CLFLUSH write-back trap bears on Q-0016: a FROM_DEVICE sync over a range the host had dirtied clobbered device output.
</content>
</invoke>
