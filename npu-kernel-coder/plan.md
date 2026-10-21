# npu-kernel-coder — plan (draft, in discussion)

Fine-tune a small open MoE to write AMD XDNA (Ryzen AI NPU) kernels: AIE C++ compute kernels plus the IRON/MLIR-AIE dataflow that feeds them.

Status: scoping. Nothing built yet.

## Target

**XDNA2 only (AIE-ML v2 / `aie2p`: Strix Point, Strix Halo, Krackan).** No XDNA1 (Phoenix, Hawk Point) data in training, eval, or RAG.

- Eval and verifier run on XDNA2.
- The filter is the verifier, not a label: an upstream example (mlir-aie, IRON, triton-xdna) gets in only if it builds and passes on `npu2`. Many mlir-aie designs take the device as a parameter, so this keeps the device-neutral ones and drops the XDNA1-only ones.
- Strip `npu1` branches and comments from kept examples so the model never sees XDNA1 names, array shapes, or intrinsics.
- Hardware docs: AM027 (AIE-ML v2) only. No AM020.

## Thesis

The model choice is not the hard part. The asset that decides whether this works is the **verifier**: compile → run on the NPU → compare against a reference → measure speed. With a verifier we can (a) measure the baseline honestly, (b) filter synthetic data, (c) do RL with execution reward. Without one, all we can do is imitate ~100k lines of kernels and hope it generalizes.

## Base model

Candidate: **K2-Horizon-MoVA-36B-A4B** (MBZUAI IFM, Apache-2.0, released 2026-09-03). 36B total / ~4B active, 512K context.

For:
- Fully open data and recipes (LLM360 line). That makes replay mixing (to limit forgetting) and decontamination checks possible, which closed-data models don't allow.
- We already run K2 on the NPU in OpenFlowLM (`openflowlm-k2attn`, `k2conv`, `k2pr`). The finished model could run on the same hardware it writes kernels for.
- 4B active is cheap to sample from, which matters a lot for rejection sampling and RL.

Risks:
- Mixture-of-Values attention is new. Check that HF transformers, PEFT, and our training stack (Unsloth? TRL?) actually support it before committing. If LoRA on MoVA is unsupported, that alone could cost weeks.
- MoE fine-tuning is fussier than dense (router drift, expert load balance). Default: freeze the router, LoRA on attention + shared experts first.
- Fallbacks if MoVA tooling is bad: Qwen3.6-35B-A3B (we know its quirks from the prune campaign), gpt-oss-20b.

## Data we have

| source | size (rough) | use |
|---|---|---|
| Kernel source, `open_kernels/` + per-model branches | ~1,550 files, ~105k lines C/C++ (openflowlm-next alone) | reference solutions; (task → kernel) pairs |
| Skills (`.claude/skills`, `.agents/skills`) | ~25 skills by family | distilled know-how; inference context and/or training text |
| Claude Code session transcripts | 359 sessions, 1.4 GB jsonl | multi-turn agent traces with compile errors and fixes — the most valuable and the dirtiest source |
| GitHub issues + PRs | TBD | natural (task description → diff) pairs, SWE-bench style |
| **Artlite** ([artlite-project/artlite](https://github.com/artlite-project/artlite), GPL-3.0, XDNA2-only) — kernel generators | 87 files in `src/artlite/operators` (gemm, prefill, fused, bfp16, act LUT, per-model qwen35 / gemma4 / llama / sd15); ~46k lines src + csrc | each operator renders AIE C++ + an IRON design from (shapes, formats, array placement). Render across many shapes → many verified (spec → kernel) pairs. Biggest data multiplier we have that isn't model-generated |
| Artlite — skills + docs | 8 skills (aie2p-coding, aie-kernel-perf-optimization + 8 reference files, iron-porting, measure-ii, npu-prefill, profiling, artlite-bench, update-deps); docs incl. design, methodology, npu-vs-igpu, models/{gemma4, llama3, qwen35, sd15}; ~7.5k lines | training text + RAG; compare against our skills and merge where theirs are better |
| Artlite — tests | 57 files in `tests/` (`tests/rtc/test_*`: per-operator and per-model NPU tests) | eval tasks with reference checks already written |
| Artlite — tools | `measure_ii.py`, `variant_search.py`, `aie2p_isa.py`, `profile_analysis.py` | verifier parts (see stage 0) |
| XDNA docs, see [sources.md](sources.md) | ~25 sources: AMD AM027/UG1603, mlir-aie/IRON, llvm-aie, triton-xdna, artlite skills, community roofline measurements | RAG corpus; maybe a small mix-in, not standalone continued pretraining (too small) |
| K2 pretraining data | huge | replay mix only |

Transcripts are in scope as training data. Anthropic's Commercial Terms give us ownership of outputs; the one restriction is training a *competing* AI model. A narrow XDNA-kernel specialist isn't that. Revisit only if this grows toward a general-purpose coding model.

Artlite is GPL-3.0; this repo is Apache-2.0. Call artlite's tools as an installed dependency; don't copy its code into our verifier. Training on GPL code is fine for internal use. Whether released weights trained on it carry GPL obligations is legally unsettled; decide before publishing weights. Reimplementing artlite techniques in OpenFlowLM (MIT): ideas yes, code no. One session reads artlite and writes a design note; a fresh session without artlite source in context implements from the note. Credit the idea in the commit.

## Stages

0. **Eval + verifier + baseline** (do first, decide later stages from the result)
   - Build task set from existing kernels: spec (op, shapes, dtype, tiling constraints) + reference output + perf target.
   - Hold out **whole kernel families** (e.g., all of deltanet, all of whisper), not random files. Random splits will measure memorization.
   - Wrap `open_kernels/harness` (`run_kernel.cpp`, `npu_host.py`, `bench.py`) into a pass/fail + speed scorer.
   - Two-tier scoring. **Compile-only tier** (CPU, parallel, no NPU): Peano compile + artlite `measure_ii`, which reads the compiler's optimization records and assembly to report each loop's initiation interval (cycles per iteration), pipelining, cycle count and register spills. Rejects non-compiling and badly scheduled attempts cheaply. **NPU tier**: correctness + wall-clock, only for attempts that pass the first tier. This is what makes stages 2–3 affordable.
   - Pin the NPU power mode during scoring (bandwidth swings 2.2× across modes) and report speed as a fraction of the roofline as well as against the reference kernel. Numbers in [sources.md](sources.md#digest-dram-bandwidth-on-strix-halo-angryloki-svg).
   - Baselines: base K2 cold; base K2 + skills in context + doc RAG; Claude + skills (ceiling). If K2 + skills + RAG is already close, fine-tuning may not be worth it.
1. **SFT**: kernels, PR pairs, cleaned trajectories. LoRA, router frozen.
   - Where OpenFlowLM and artlite both cover an op (qwen35 prefill, gemm, attention), score both on the verifier and use the faster one as the target. Artlite beats OFLM on some of these (see the OpenFlowLM-Next3 "fix issues and bugs found by ArtLite eval" issue), so "our code" can't be the default answer.
2. **Rejection sampling**: sample many attempts per task, keep the ones the verifier passes, retrain. Main data multiplier.
3. **RL with execution reward** (only if stage 2 plateaus and verifier throughput allows).

## Open questions

1. What's the goal: a local agent that runs on the NPU, a cheaper replacement for Claude on this work, or a research writeup? Changes how much the baseline comparison matters.
2. ~~Which target?~~ XDNA2 mainly (see Target). Still open: which layer does the model write — AIE C++ kernel only, IRON Python dataflow, or both?
3. Verifier throughput: how many NPU machines can run tests in parallel, and how long does one compile + run take? The compile-only tier moves most rejection off the NPU, but correctness still needs it.
4. Which GitHub repos hold the issues/PRs (Cyronius/OpenFlowLM-Next plus upstream)?
5. Is triton-xdna a candidate output language? Higher-level means fewer tokens per kernel and easier learning, but less control and a younger toolchain.
6. Artlite's kernels are rendered at run time from Jinja-templated C++. Train on the rendered C++ (what the model must write), the generator Python (how to parameterize), or both?
