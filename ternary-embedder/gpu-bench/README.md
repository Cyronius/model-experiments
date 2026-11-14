# iGPU (Radeon 8060S, gfx1151) and CPU batch results, 2026-11-02

`bench_gpu.py` (run with a ROCm torch venv: torch 2.9.1+rocm7.13.0a20260513 from rocm.nightlies.amd.com/v2/gfx1151),
raw numbers in `results-cuda.json`. `bench_cpu_b64.py` gives the CPU batch rows. Tokenization included, fixed padding,
median after >=1 s warmup, GPU synchronised by copying vectors to host. Shared machine, no Dibs check.
"ternary" on GPU is the dense float simulation (effective weights + int8 activation simulation); no packed ternary GPU kernel exists.
Float/bge/MiniLM are 3-layer/384 (float, bge-micro) and 6-layer/384 (MiniLM).

| batch 1 latency, ms (cap 32 / 128) | GPU fp32 | GPU bf16 | CPU |
|---|---:|---:|---:|
| ternary (GPU: dense sim; CPU: packed 2-bit, 2 cores) | 1.81 / 2.19 | - | 0.66 / 2.39 |
| float (ours) | 0.75 / 1.15 | 0.66 / 0.95 | 2.53 / 8.05 (earlier) |
| bge-micro-v2 | 1.10 / 1.35 | 1.18 / 1.30 | 2.93 / 8.21 |
| MiniLM-L6-v2 | 1.65 / 1.98 | 1.76 / 1.85 | 5.81 / 15.1 (earlier) |

| batch 64 throughput, emb/s (cap 32 / 128) | GPU fp32 | GPU bf16 | CPU 2 cores | CPU 8 cores |
|---|---:|---:|---:|---:|
| ternary | 6449 / 1252 (dense sim) | - | 1679 / 214 (packed) | 1130 / 323 (packed) |
| float (ours) | 9722 / 1965 | 13602 / 3747 | - | - |
| bge-micro-v2 | 8644 / 1898 | 12539 / 2514 | 638 / 143 | 1910 / 366 |
| MiniLM-L6-v2 | 5392 / 1134 | 8047 / 1474 | - | - |

The packed CPU kernel is tuned for batch 1; batch 64 gets no gain from batching and 8 threads are slower than 2
(12 column tiles for N=384 balance poorly, and torch attention at padded length dominates).

## Re-measure under `dibs run --measure`, 2026-11-03

`measure-noisy-20261103.log` (CPU, iGPU, NPU) and `../cpu-kernel/measure-clean-cpu.log` (CPU only). Both started CLEAN and
ended NOISY: another session's unattended Spice86 emulator (about 2 cores, not under Dibs) and a rustc build ran alongside.

- iGPU: every row reproduced the table above within about 4% (`results-cuda.json` now holds the re-run; the first run is
  `results-cuda-shared-20261102.json`).
- NPU fused2 (`../native-stage/fused2-speed-clean.json`): batch 1 1.10 / 1.44 ms; batch 64 8205 emb/s packed, 4453 padded
  to 32 tokens, 938 at 128 tokens. Within about 4% of the shared-machine run.
- CPU: not trustworthy yet. Batch-1 packed p2v2 on cpus 0,1 ranged 0.69-1.50 ms (cap 32) and 2.49-4.61 ms (cap 128)
  across the two runs, and within the second run check_bench and bench_cpu_b64 disagreed (1.11 vs 0.69 ms). 8 threads
  (cpus 0-7) were stable: 0.50 / 1.37 ms at batch 1, about 1920 / 247 emb/s at batch 64.
