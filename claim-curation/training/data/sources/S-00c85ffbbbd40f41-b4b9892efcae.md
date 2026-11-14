---
from: fastflowlm (C:/code/FastFlowLM, main checkout; upstream docs at 1fc5ce1 of 2026-08-25)
date: 2026-08-11
evidence: documented
stack: FastFlowLM v0.9.30 to v1.0.2 per page; benchmark systems Ryzen AI 7 350 (Kraken Point, 32 GB) and Ryzen AI 9 370 (Strix Point, 32 GB); default power mode performance; NPU driver, firmware and XRT not stated
---

# FLM docs: prefill chunk size, preemption, NPU request queue, release timeline, a hidden power table and extra benchmark rows

Sources (C:/code/FastFlowLM): `docs/docs/instructions/cli.md:280-313`, `src/runner/runner.cpp:83-92`, `src/server/rest_handler.cpp:334-337` and `:414-415`, `src/model_list.json` (35 `max_prefill_len` entries; Qwen3.6-35B-A3B at lines 286-287), `src/server/server.cpp:837-852`, `src/src/main.cpp:632-636`, `docs/docs/benchmarks/llama3_results.md:39-77` (HTML-commented blocks), `docs/docs/benchmarks/phi4_results.md`, `nanbeige4.1_results.md`, `smolvla_results.md`, `docs/docs/models/smolvla.md:15`, `flm_story.md:29-39`, `README.md:81-112`; `git tag --list`. Vendor text, not re-measured. Pages already covered by the other fastflowlm raw notes are not repeated.

## Prefill chunk size

- `--prefill-chunk-len` sets how many tokens one prefill chunk holds; the docs give a default of 4096. (cli.md:299-313)
- In code a user value below 512 is raised to 512. Without the flag FLM takes the model's `max_prefill_len` from `model_list.json`, which is 4096 for all 35 entries, Qwen3.6-35B-A3B included; the last fallback is also 4096. (runner.cpp:83-92, rest_handler.cpp:334-337, 414-415)
- This answers the chunk size the 2026-09-01 engine captures could not find ("Prefill chunk 1/1 with N tokens").

## Preemption

- `--preemption 1` on `flm run` or `flm serve` lets high-priority tasks interrupt NPU jobs. The docs warn: "Preemption is for **engineering testing/optimization** only. It requires a special driver + toolkit and is **not for public use**." (cli.md:280-294)
- FLM's code disables it on Linux hosts (see `2026-08-18-flm-source-driver-floor-validate-limits-hrx.md`).

## One NPU, one queue (flm serve)

- The server queues requests for the NPU; `--q-len` (default 10) caps the queue. Beyond it the reply is `NPU is in use and request queue is full (limit: 10). Please try again later.` (server.cpp:837-852, vm_args.hpp:101-102)
- The server's request timeout is 600 s. (main.cpp:636)

## Hidden power table (HTML comment in llama3_results.md:67-77, not shown on the site)

- "Average Power Consumption (Watts) during decoding": NPU (FLM) CPU 0.5 + NPU 1.6 = 2.1 W total; NPU (Ryzen AI Software) 11 + 2.0 = 13 W; iGPU 11.3 CPU + 9.00 iGPU = 20.3 W; CPU only 18.0 W. Model, FLM version and machine are not stated in the block.
- Another commented note: the Ryzen AI Software release limited context to 2,048 tokens in both NPU-only and hybrid mode. (llama3_results.md:59-61)

## Benchmark rows not copied before (vendor-reported)

- Phi-4-mini-instruct (v0.9.30, Ryzen AI 7 350, 32 GB): decode 21.8 / 21.2 / 19.9 / 18.1 / 14.9 / 11.2 tok/s at 1k / 2k / 4k / 8k / 16k / 32k; prefill 643 / 787 / 857 / 809 / 644 / 447 tok/s.
- Nanbeige4.1-3B (v0.9.38, same system): decode 23.5 / 22.3 / 20.4 / 17.3 / 13.3 / 9.0; prefill 612 / 731 / 742 / 686 / 523 / 343.
- SmolVLA (v1.0.0, Ryzen AI 9 370 Strix Point, 32 GB): 298 / 363 / 430 ms per inference with 1 / 2 / 3 camera images; max context 48 tokens.

## Release timeline (flm_story.md, README news)

- First public commit 2025-06-16; v0.1.0 2025-06-25; in AMD's Lemonade Server 2025-10-01; Linux support 2026-03-11; team joined AMD 2026-07-17; repo moved to the ROCm organization 2026-08-07; v1.0.0 2026-08-10 (README news gives 08/11/2026).
- The clone's tags run v0.1.0 to v0.1.8, v0.9.0 to v0.9.46, and v1.0.0 to v1.0.3.
- flm_story.md says FLM is built on IRON, AMD RAD's open NPU compiler stack.

## Doubtful

- README.md:108 claims FLM "Runs fully on AMD Ryzen AI NPU — no GPU or CPU load". F-0025 measured FLM 1.0.2 running prefill attention, gate, o_proj, residual and post-norm on the CPU.
- The hidden power table has no version or model; the visible page header says v0.9.30, but the commented decode table beside it shows older, lower numbers (LLaMA 3.2 1B 41.5 tok/s at 1k against 64.5 in the live table).
- flm_story's "built on IRON" sits beside the issue-690 speculation (AngryLoki) that FLM first depended on the Chess compiler; IRON can drive either compiler, so these need not conflict.
