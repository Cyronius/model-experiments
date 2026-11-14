---
from: phlegm
date: 2026-09-02
evidence: measured
stack: XRT runtime 32.00.20102.3931 for the FLM-kernel drive work (npu-open-engine.md); XRT master 2.26.0 xclbinutil/aiebu-asm for the IRON flow; mlir-aie 1.4.2, Peano 21.0.0; driver/firmware versions not stated for the early items
---

# Kernel ABI, instruction-stream/transaction contents and how the ELF/classic loads differ

Sources: C:/code/phlegm/xdna-dump/sources/npu-open-engine.md ("M0", "Step 2c", "DECODE PATH interpreted"); open-kernels-feasibility.md ("How FLM fetches routed experts"); open-kernels-phase2-moe-first.md ("Step 2 design note"); C:/code/phlegm/open_kernels/README.md ("Running through phlegm", "Phase 2 ... moe_experts Step 2a"); open_kernels/harness/README.md; xdna-dump/tools/seq-capture/README.md. F-0012 (DDR patch on args >= 5 carry +0x80000000) and F-0015 (array->shim control) are already recorded; this file adds the surrounding facts.

## Kernel ABI (both FLM's closed kernels and mlir-aie flows)

- Every xclbin kernel enumerates as `MLIR_AIE(opcode, instr, ninstr, bo0..bo4)`: opcode, an instruction-buffer BO and its length, then data BOs. Reported by the first standalone probe (device[0] named "NPU Strix", 2026-08-29).
- ELF path (FLM's): transaction blob -> `aiebu` assembler -> ELF; `xrt::elf(blob)` -> `xrt::module(elf)` -> `xrt::ext::kernel(hw_ctx, module, "MLIR_AIE")`; `xrt::run r(kernel)`, `r.set_arg(0, 3)` (opcode 3), `r.set_arg(1, 0)`, `r.set_arg(2, 0)`, data BOs at args 3, 4, 5, ... FLM runs bind 5 data BOs at args 3-7 (confirmed live by the proxy capture). IRON's `insts.elf` loads through this path unchanged.
- Classic path (mlir-aie): `xrt::kernel(ctx, "MLIR_AIE")` with a cacheable instruction BO bound at args 1/2 (word count at arg 2), data BOs from arg 3. Both paths loaded ROT13 (2026-09-01): classic `run k [2 bufs] -> state 4 (0.602 ms)`, ELF `0.447 ms`, output byte-exact. State 4 = completed.
- XRT patches the device addresses of arg BOs into the control code itself: a run on a wrong xclbin still wrote into the caller's buffer. No manual address patching needed for the first 5 args (see F-0012 for later args).
- Only the matching xclbin completes a given control ELF; wrong ones time out (state 8). The project used this as a free "which kernel is this ELF for" oracle (`map_kernels.py`).
- Many FLM buffers are in place (arg preloaded with input, kernel overwrites it); classify by comparing pre-exec and post-exec hashes, not by D2H presence. A replay attempt matched a zeroed un-run BO by hash (op#269), so test only ops with non-zero output.
- FLM's `buffer.hpp` has commented-out logic that avoids device addresses `0x6000_0000-0x7FFF_FFFF` for BOs (source suspected the control code might not be fully address-independent; later work found XRT auto-patching made that unnecessary).

## Transaction contents (FLM's fused decode layer program)

Census of `elf_000005.bin` (49 KB of aie2 txn ops): 315 blockwrite / 315 ddr-patch / 284 write / 278 maskwrite / 276 tct. 240 DDR-patched descriptors reference 240 distinct pool offsets, all in the shared-expert/attention region (>= 503316480). Exactly 32 reference the routed-expert region, all at pool offset 0, and none reference the down-expert region; 32 = 8 experts x 4 gate/up stripes, in shim DMAs of columns 0, 1, 6, 7, BD ids {4,5,6,7,12,13,14,15}, written once and never enqueued by the stream (every task-queue write names BDs 1, 2, 9, 10 only).

Shim-DMA register map confirmed from FLM's own stream: BD n at `0x1D000 + 0x20*n`; MM2S task queues at `0x1D214` / `0x1D21C`; control at `0x1D210` / `0x1D218` (AIE-ML NOC-module DMA regs).

DDR-patch op in an mlir-aie instruction stream (harness README and open_kernels/README.md moe_experts): opcode `0x81`, 12 words, register at word +6, buffer arg index at +8, byte offset at +10. Re-pointing a DMA is therefore one word per patched fill plus one instruction-BO sync; measured ~0.04 ms per layer for 144 (later 144 or 216) fills. Each weight fill in the MoE stream is a `blockwrite` of BD words plus an `address_patch` that adds the BO base at submit. `attnpos` rewrites three words per token (the window fill's BD length word in the BD blockwrite before the patch op, the drain offset, the record offset), again one instruction-BO sync.

FLM decode ELF template: 480 B per token with four seqlen u32 at byte offsets 160/184/208/232 which are AIE tile addresses (`0x06202400`, `0x06309000`, `0x06402400`, `0x06509000`); poking them had no effect on decode (see raw/phlegm/2026-08-31-flm-hw-context-3-layer-budget.md).

## One xclbin, many instruction streams

IRON's xclbin for a shape family carries the static array configuration; shapes differ only in `insts.bin` / `insts.elf`. Examples with byte counts in raw/phlegm/2026-09-13-iron-dataflow-hangs-and-runtime-loop-bounds.md. Two kernel objects (two instruction streams) on one hw_context are fine (verified with rot13), and the driver `kernelx` per stream on one xclbin works. A multi-part sequence on one xclbin works only back to back (F-0013).

## Capture tooling facts that bear on the driver

- FLM's engine DLL imports exactly one NPU-touching non-system DLL, `xrt_coreutil.dll`; every control-code blob passes through `xrt::elf::elf(const void*, size_t)`. A proxy `xrt_coreutil.dll` forwarding all 541 exports to a renamed original, with Detours hooks on that ctor, `run::start`/`run::wait`, `bo::map`, `bo::sync`, `~bo`, captured live sequences without rebuilding the engine. There is no `bo::write` import; map + sync is the whole data path.
- The NPU stages tensors through fixed 1 MB DMA staging tiles, so activations appear in 1 MB syncs at offset 0 and fp32 logits (vocab 248320, ~970 KB) fit one tile (Tier-0b).
- Structural fingerprinting of AIE ctrlcode collapses 821 (interval-4, 8 layers) / 433 (interval-3, 6 layers) ops from ~489 byte-unique to 16 structural op types; weight-offset patching makes raw byte diffs useless.
- Hashing every kernel-arg buffer (including 512 MB pools) on each submit stalled 40-layer prefill ~100x (a 12-token prefill did not finish in 10 minutes); capped via `FLM_BO_RUNARG_MAX`. Host tooling, but a reminder that arg buffers here can be 512 MB.
- Object churn is cheap: FLM builds a fresh 480 B ELF plus a 45,072 B full-layer program, module and kernel objects per token at ~7 tok/s.
- Process exit after 19 xclbin contexts segfaults in XRT teardown after all work and dumps are done; harmless (noted in three places in the docs, also "access violation in XRT teardown" in the harness README).
- A process running 5 layers + lm_head in one process timed out on "in-process XRT resource accumulation" (M3 note, npu-open-engine.md), worked around by one process per layer; later handled by proper context lifecycle. Cause not diagnosed.
