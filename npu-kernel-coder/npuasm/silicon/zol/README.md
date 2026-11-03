# Hardware-loop probe, and the first program edited as npuasm text

Two questions, both answered on the NPU on 2026-11-03:

1. Do Peano's hardware zero-overhead loops (`movxm ls`, `movxm le`, `add.nc lc`) run on XDNA2 silicon? hipfire forbids them because in its own encoder the loop body ran once (`xdna2-notes/raw/hipfire/2026-10-03-aie2p-hand-encoded-core-and-dma-traps.md`, hearsay). Yet 2,120 of the 2,267 distinct Peano programs on this machine use them.
2. Does a core program edited in npuasm text and rebuilt without the compiler run correctly?

**Answers: yes and yes.**

## Setup

Strix Halo (`NPU Strix Halo`, aie2p, 6x8), NPU firmware 1.1.2.65 (the same firmware as hipfire's evidence), XRT 2.27.0, amdxdna in Linux 7.2.13. Built with mlir-aie 1.4.2 and Peano llvm-aie 21.0.0.2026080301+c9c5ecb7 from `openflowlm-next/ironvenv`.

- `zol.cc`: one counted scalar loop over 1,024 int32, `out[i] = in[i]*3 + i`, with unrolling and vectorization disabled.
- Peano compiles it into a software-pipelined hardware loop: `movxm r2, #0x400`, then `add.nc lc, r2, #-0x1` (reads r2 before the same bundle's `mova r2, #0x2` writes it, so `lc` = 1023), `movxm ls, #0x1a0`, and `movxm le, #0x200`. `le` is set one bundle before the loop start. The loop's last bundle, at `le`, is an aligned 16-byte all-NOP bundle.
- `zol.py`: one worker, a 1,024-word ObjectFifo in and out (structure copied from `openflowlm-next/open_kernels/designs/rot13`).
- `run.py`: system python3 with pyxrt. Random input, output pre-filled with `0xDEADBEEF`, 10 s timeout, checks every word.
- `edit_build.sh`:
  1. lifts the build with npuasm and replaces one bundle line in `core_c0r2.s`;
  2. rebuilds the CDOs, PDI and `insts.bin` with npuasm (cores assembled by `npuasm-core`);
  3. packages an xclbin with `xclbinutil` exactly as aiecc does, under a fresh PDI UUID (the firmware skips a PDI load it has just done).

Reserve the NPU with dibs first (shared machine): wrap each `run.py` call as `dibs run npu --label "zol probe" --eta 5m -- python3 run.py ...`. The runs recorded below were made without a hold, while the NPU was free.

```sh
cd ~/code/openflowlm-next/open_kernels && ../ironvenv/bin/python build_design.py \
    ~/code/model-experiments/npu-kernel-coder/npuasm/silicon/zol/zol.py \
    ~/code/model-experiments/npu-kernel-coder/npuasm/silicon/zol/build
cd ~/code/model-experiments/npu-kernel-coder/npuasm/silicon/zol
python3 run.py                                   # Peano build
./edit_build.sh $PWD/edit-mul5 'movx r0, #0x3' 'movx r0, #0x5'
python3 run.py edit-mul5 --mul 5
./edit_build.sh $PWD/edit-count512 'nopa; nopb; nops; movxm r2, #0x400; nopv' 'nopa; nopb; nops; movxm r2, #0x200; nopv'
python3 run.py edit-count512 --count 512
```

## Results

| Build | Runs | Result |
|---|---|---|
| Peano, unmodified | 3, then 2 more after the edits | 1,024/1,024 words correct every run; no word left at the sentinel |
| npuasm text, multiplier `#0x3` → `#0x5` (the PDI differs from Peano's in that one byte) | 3 + 1 | 1,024/1,024 words equal `in*5 + i` |
| npuasm text, loop source `#0x400` → `#0x200` | 3 | words 0–511 correct; none of words 512–1023 hold the full-loop value, so the loop stopped where the text says |

Every run completed (`ERT_CMD_STATE_COMPLETED`) in 0.08–0.56 ms, the first run of each process being the slowest. The device stayed healthy throughout.

The DMA copies the whole L1 tile, so words the loop skips show L1 contents rather than the DDR sentinel. The count check therefore asks that words past `count` do *not* hold this run's full-loop value. They also don't hold the previous process's identical-input result, so the tile does not keep leftover data across processes.

## What this settles, and what it doesn't

- Peano-form hardware loops run on this silicon and firmware, and their trip count follows `lc` exactly. hipfire's `ZeroOverheadLoop` rule is a limit of its own encoder, not of the hardware. What hipfire did differently is still unknown: its loop code was removed (`report.md`, "Decision: no hardware ZOL").
- A core program can be edited as npuasm text, assembled by `npuasm-core`, packed into CDOs and a PDI by npuasm, and run correctly. The packaging step still uses XRT's `xclbinutil`, and running uses XRT. Our own xclbin writer or a raw-ioctl loader is the rest of gate D.
- One loop shape, one core, one firmware. Untested here: vector loops, nested hardware loops, and the other XDNA2 parts.
