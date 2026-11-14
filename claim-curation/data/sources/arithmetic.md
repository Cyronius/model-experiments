---
from: phlegm
date: 2026-09-26
evidence: measured
stack: mlir-aie 1.4.2 (dev16 build named for the whisper runs), Peano 21.0.0.2026080301, Ryzen AI 9 HX 370 in turbo power mode for the dit_* numbers; XRT/driver not stated in these READMEs
---

# aie2p arithmetic units, approximations and core-stack/rounding traps (beyond F-0008, F-0009, F-0014, F-0016)

Sources: C:/code/phlegm/open_kernels/designs/dit_gemm/README.md, dit_fa/README.md ("Accuracy", "Traps hit getting here"), dit_ew/README.md ("Arithmetic"), vae_ew/README.md ("Trap: rounding mode"), gemm_q4/NOTES.md, C:/code/phlegm/xdna-dump/sources/open-kernels-feasibility.md (vegah trap list). Already in findings: hardware tanh/inv/invsqrt ~1e-2 LUTs (F-0008), stack 0xD00 hang vs 0x1800 (F-0009), `to_float` NaN / int16 `bit_and` / `get_cycles` / static zeroing (F-0014), bf16 GEMV compute-bound (F-0016).

## Which operands reach the fast matrix unit

- dit_gemm README: on aie2p only 8-bit operands reach the fast MAC array. int8 and bfp16ebs8 (8-bit mantissa sharing one exponent per 8 values) are the fast paths. A bf16 matmul either converts to bfp16 in-core or falls back to the fp32 vector unit, measured ~2.8 TFLOPS. FP8 (E4M3/E5M2) and MX formats have no matrix unit on this generation; `aie_api` has those only for aie2ps. (Documented in the README; the 2.8 TFLOPS is the only measured figure.)
- Dense int8 mmul `mmul<8,8,8,int8,uint8>` does 512 MACs per issue against 256 for `mmul<4,8,8,int16,uint8>` (gemm_q4 NOTES.md). It gained only 3%: eight tokens x 32 rows of live float accumulator is twice what the register file holds and the spilling eats the wider mmul.
- aie2p has no int32 x int32 elementwise mac: `mac_elem_64_conf` has no 32-bit overload, and int32 x int16 only lands in acc64, which at four live 32-lane accumulators is twice the register file (gemm_q4 NOTES.md, Q4_K work). Workaround used: narrow the int32 partial to int16 with a 9-bit SRS (13 for odd rows) first, then int16 x int16 mac.
- Float: aie2p's fp32 vector multiply is emulated. dit_ew's first fp32 16-lane version ran 4-6x slower than its own DMA (ln_mod 6.6-9.4 ms). Rewritten as bf16 x bf16 -> fp32-accumulator MACs on 32 lanes; a float scalar multiplying a vector (1/std, dt, cos/sin) is split into bf16 hi + lo and applied as two MACs, exact to fp32 rounding. vegah's trap list (documented, other repo) says an fp32 vector multiply "returns zero silently" and scalar float is 1617x slow.

## Approximate transcendental hardware (new numbers)

- `aie::exp2<bfloat16>` on aie2p is not bf16-accurate: it returns `2^n * (1 + f)` with f the input fraction truncated to 7 bits, a linear mantissa (Mitchell's approximation). Exact at integers, +6.1% at half-integers, +3.8% mean (dit_fa README; `utilities/aie-probes/exp2_probe.py`). AMD's own aie2p softmax kernel uses the same instruction. The mean cancels in softmax normalisation, the variation does not. A quadratic mantissa fix (`FA_EXP_FIX`) cost +9% kernel time (30.5 -> 33.2 ms) and a 3 KB stack frame.
- Vector tanh: 1.4-3.5% rms error below |x| = 2 (`utilities/aie-probes/vecmath_probe.py`), which makes SiLU written as `0.5x(1 + tanh(x/2))` ~1e-2 rel_fro (vae_ew) against exact; the leading 1 keeps the error under ~0.5% of the result.
- bf16 `inv`/`invsqrt` are bf16-accurate but scalar loops, so only used per row (dit_ew).
- Image-level effect measured by the project (512^2, 4 steps, 8 prompts, LPIPS vs bf16 mean/max): fp32 noise floor 0.0129/0.031; dit_gemm arithmetic 0.0150/0.031; attention with exact exp 0.0164/0.033; attention with hardware exp 0.0220/0.067; GEMM + attention hardware exp 0.0288/0.081. dit_fa shipped without the exp fix.

## Core rounding mode

The core starts in floor rounding (vegah trap list says the same: "floor rounding default"). In vae_ew, converting to bf16 before any data kernel ran meant the first dispatch after a context load was 2x less accurate than every later one: 1.8e-4 vs 8.5e-5 rel_fro from identical inputs. Fix: set the rounding mode in the first kernel of every dispatch (`vew_begin`), not only in the data kernels. Measured.

## Stack sizes that hang or overflow (additions to F-0009)

All hangs are ERT state 8 unless noted.

| design | too small | works | note |
|---|---|---|---|
| gemm_q4 `GEMM_INT8=1` | 0x1000 hangs silently | 0x2000 | needs the wider stack |
| dit_gemm with Peano | 0xD00 (the example's) overflowed into C with its own kernel | 0xF00 | Peano frames larger than chess |
| dit_fa `softmax_step` with `FA_EXP_FIX` | 3 KB stack hangs the array (frame is 3 KB) | larger | check `st ..., [sp, #-N]` in `llvm-objdump -d` when a kernel grows |
| gemm_q4 Q4_K, M = 32 | overflows `.bss` | M = 28 on a 4 KB stack | 2.5 KB sub-scale scratch |

- Masked softmax scores are bf16 `lowest`, not -inf, because the exp argument is `S*c - m*c` in fp32 and -inf would stay -inf on a path that expects finite values (dit_fa trap 5).
- `-DDC=...` on the kernel command line breaks `aie_api` (`AIE_RegFile::DC`); use other macro names such as `FA_*` (dit_fa trap 3).
- IRON compiles each ExternalFunction's source file separately: two `extern "C"` entry points in one `.cc` link as duplicate symbols (open_kernels/README.md moe_experts).
- Qk helpers must be `noinline`: inlined three times each they overflow the 16 KB program memory (dit_ew text-encoder modes).

## Accuracy figures that depend on the arithmetic (for reference)

- dit_gemm: rel_fro vs exact 1.25e-2 to 1.88e-2, grows with K because the accumulator re-rounds to bf16 every 64 of K. gate: finite, rel_fro <= 3e-2, row cosine > 0.999.
- dit_fa: rel_fro vs fp64 2.6e-2 (as shipped), 1.88e-2 with `FA_EXP_FIX`; `fa_emul.py` models the kernel and matches hardware to ~7e-3.
- mmul q4 GEMV, activation block-quantised to int16: exact for bf16 values within 2^7 of the block max (open_kernels/README.md item 5).
