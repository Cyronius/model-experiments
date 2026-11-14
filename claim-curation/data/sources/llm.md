---
from: LLMNpuTest (C:/code/LLMNpuTest, main checkout, branch main, HEAD d94836a)
date: 2026-09-01
evidence: measured
stack: Windows; AMD Ryzen AI 9 HX 370, NPU Strix, 8 columns, AIE2P; XRT 2.21.0; NPU driver 32.0.20102.3930; mlir-aie 1.4.2.dev16+g7e00b57; Peano 21.0.0.2026080301 (Peano only, no Chess); NPU firmware not stated
---

# IRON/aie2p limits, build errors and silent faults hit while building granite kernels

Sources (C:/code/LLMNpuTest): commits `01ff971`, `625c6ea`, `e552d38`, `a520103`, `660048a`, `c712130`, `36befc8`, `3501bc4` (2026-09-01); `designs/granite_gemv/granite_gemv32.py:5-9`, `granite_gemv.py:50-81`, `granite_gather.py:46-52`, `granite_elementwise.py:58-75`, `granite_gemv.h:99-108`, `lm_head/lm_head.h:87-95`; `README.md:152-158`, `README.md:194-231`, `README.md:253-257`, `README.md:272-288`, `README.md:338-353`. Each item below happened in this repo's own builds or runs.

## DMA channels, placement and BDs

- One shim stream per core costs n+1 MM2S and n S2MM channels, with 16 of each on the device, so such a design tops out at 8 cores. 16 cores fail at placement with `no ShimNOCTile has sufficient DMA capacity`. (granite_gemv32.py:5-7; consistent with F-0046)
- npu2 is 8 columns x 6 rows: row 0 shim, row 1 memtile, rows 2-5 compute, 32 compute cores. (granite_gemv32.py:8-9)
- A memtile has its own channel budget: an 8-way `join()` fails with `no MemTile has sufficient DMA capacity for 8 input/1 output channels`. Four inputs per memtile work. (commit 660048a)
- A fifo cannot be joined into and forwarded out of: `objectfifo cannot be in more than one ObjectFifoLinkOp`. So an all-gather as join plus forward cannot be written at any width. (commit 660048a)
- A fused core needs four DMA streams where a plain GEMV needs three. On direct shim that is `2n <= 16`; through memtiles a column would need 10 in / 10 out against the ~6/6 it has. So fusion and the 24-core memtile leg do not combine. (README.md:253-257)
- A compute tile has 2 input DMA channels. cos/sin had to ride at the end of the x buffer, and the fused MLP sends x and then h through one fifo. (commits a520103, 3501bc4)
- BD sizes cap at 1023 per dimension: a gate-then-up strided tap failed with `Size 0 exceeds the [0:1023] range`; the host interleaves the two matrices instead. (commit 36befc8; the limit is F-0053, the error text is new)

## Silent faults and deadlocks

- A 128-byte shim transfer silently delivers zeros. Attention's q (64 bf16 = 128 B) and its 128 B result both arrived empty, with no error; every score was wrong but in a plausible range. Padding both to 1024 B fixed it. The same design's 8192 B weight stream and 264 B state readback worked. (commit e552d38)
- The GEMV designs have 128-byte `y` elements and are exact, because there the shim moves the whole drain at once: the shim transfer size matters, not the ObjectFifo element size. (README.md:274-280)
- One weight fill that spans both phases of a two-phase core program deadlocks: the fill cannot finish until the core consumes its phase-2 objects, which waits on `tg1.finish()`, which waits on the fill. "A deadlock, not an error." (commit 3501bc4)
- A drain and a refill of the same DDR buffer in one dispatch, with two TaskGroups and `finish()` between them, ran in order in 20 of 20 runs (each core read 5120 = 512 x (1+2+3+4)). One payload size, idle machine; the source calls this evidence, not proof. (commit c712130)

## L1 (64 KB per core)

- Budget used: 64 KB minus the worker stack 0xD00 = 62208 B. Batch 2 with 5 tiles per call needs 61952 B of buffers; it built and ran. The "63 KB minus stack" rule (61184 B) would have rejected it, and falling back silently halved the DMA element and cost ~20% throughput. (granite_gemv.py:61-66)
- Batch 4 at K = 2560 (20 KB of activations) pushes 5 tiles (51200 B double-buffered) over the budget; the build fails with `allocated buffers exceeded available memory` at MLIR level. (granite_gemv.py:76-81)
- An 8192-element SwiGLU object needs 3 buffers x 16384 B x depth 2 = 98304 B; the build fails with `Basic sequential allocation failed` and prints the MemoryMap. Streaming 2048-element chunks (24576 B) fixed it. (granite_elementwise.py:71-75)
- In q_proj+RoPE, depth 1 on two fifos that are acquired once and held fixed an L1 overflow. (commit a520103)
- GEMV and elementwise workers ran with `stack_size=0xD00`. Compare F-0009 (0xD00 hangs with fp32 vector code).

## Program memory (16 KB per core)

- A GEMV body templated on the K-tile index instantiated a 4736 B `.text` copy per entry point; even one entry point plus the runtime overflowed. (granite_gemv.h:103-107)
- With the index as a runtime argument and `__attribute__((noinline)) inline`, the body is emitted once (2528 B) and each entry point is a 208 B wrapper. `inline` gives vague linkage, so per-file copies merge into one COMDAT; that let down_proj (K = 8192) have eight entry points. (README.md:224-231)
- Vectorising the scalar sum of x halved the body again to 1248 B. The kernel is about 1.2 KB of `.text`.
- A Python-unrolled loop of 2 x 32 x 5 = 320 kernel calls overflowed the core; as a `range_` hardware loop it is 10 call sites. A copied comment claiming the row index "must be compile-time" caused it. (commit 36befc8)
- Reading `llvm-objdump -h` on the compiled object found the overflow cause in seconds, against minutes per full build. (README.md:462-466)

## Arithmetic on the device

- `aie::mul(vector<float>, vector<float>)` compiles and returns zero, silently. The kernels split each fp32 partial into bf16 hi and lo and MAC both; 8 + 8 mantissa bits land exactly in the fp32 accumulator. (README.md:216-222, lm_head.h:87-95)
- `aie::exp2<bfloat16>` has about 5.5e-02 relative error; the source calls it undocumented in the aie_kernels sources and the floor of any softmax built on it. (commit e552d38)
- `aie::tanh` sets the SwiGLU error floor: max rel err 1.224e-02 (gate+up+SwiGLU) and 2.114e-02 (fused MLP). Narrowing the accumulators to bf16 before the nonlinearity moved the error only from 1.227e-02 to 1.224e-02. (commits 36befc8, 3501bc4)
- The mlir-aie `aie_kernels/aie2p` elementwise references fail open: `rms_norm.cc` hardcodes `gamma = 1.0f` and never applies the weight (granite's weights range -15 to +23); `rope.cc` uses interleaved pairs where Llama and Granite use half-split `rotate_half` (results differ by 1.506 on the test); `swiglu.cc`'s entry point hardcodes length 1024. (commit 01ff971)

## FastFlowLM compatibility

- FLM's llama engine whitelists `hidden_size` to {2048, 3072, 4096}; the gate sits inside `llama_npu.dll`. Granite (2560) padded to 3072 passes the gate and then fails: it needs head_dim 64, and every shipped design at hidden >= 2560 is head_dim 128. On hardware, borrowed xclbins hang and matched donors segfault. (README.md:152-158)
- In the `.q4nx` file `conv1d` is stored transposed, in addition to the conventions in F-0029. `gate_proj` and `down_proj` are both stored `[2560, 5120]` (256 tile-rows x 10 K-tiles against 80 x 32), so (N, K) must come from `config.json`; the wrong factoring is a silent wrong matmul. (README.md:338-353)

## Doubtful

- "fp32 vector multiply returns zero" conflicts with F-0070, which says the aie2p fp32 vector multiply is emulated (nine bf16 passes) and slow, not zero. The two may use different `aie_api` calls or toolchain builds; not reconciled.
- exp2 error of ~5.5e-02 differs from F-0071 (+3.8% average, +6.1% at half-integers); the input ranges are not stated here.
- The 62208 B budget verified at 61952 B bears on Q-0020: it sits between the 61,440 B figure and the 65,536 B builds, and depends on the stack size.
