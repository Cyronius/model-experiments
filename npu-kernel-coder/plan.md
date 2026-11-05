# npu-kernel-coder — plan (draft, in discussion)

Fine-tune a small open MoE so that, pointed at a model's GGUF metadata (architecture, shapes, quant types), it writes efficient XDNA2 kernels for that model: AIE C++ compute kernels plus the IRON dataflow that feeds them.

Status: scoping, revised 2026-11-03 toward direct emission (see [Revision 2026-11-03](#revision-2026-11-03-direct-emission)). Gates A and B have passed: [`npuasm/`](npuasm/) rebuilds all 1,863 local builds byte-identical from text, with no vendor toolchain in the loop. The two knowledge-injection pilots this depends on have first results (see [Starting from past findings](#starting-from-past-findings)).

## Revision 2026-11-03: direct emission

The proposal: GGUF metadata in, NPU programs out, with no Triton, IRON or mlir-aie needed at inference. The base model is Qwen3.8-27B. Findings go in through a trained projection to KV, read by attention with a runtime limit on how much it reads. The sections below keep the decisions this replaces, each marked *Replaced 2026-11-03*.

| Decision | Before | Now | Why |
|---|---|---|---|
| Output layer | AIE C++ and IRON; machine code only as an inner-loop experiment | Text assembly plus a declarative array configuration. A small deterministic assembler and packer turns it into the binaries ([`npuasm/`](npuasm/)). C++ and IRON stay as the source of training pairs and as the fallback. | hipfire's `pm-npu` (its own encoder from llvm-aie TableGen, raw ioctl, no XRT or Peano) shows that this backend can be built. Its hand-encoded int8 GEMM reached 19.6–33.4 TOPS against 13.09 for tuned mlir-aie (hearsay). Checking a candidate drops from minutes (qwen38-27b skill: `lx` ~3 min, `ax` ~7) to milliseconds, so RL waits on the NPU, not the compiler. |
| Encoding, alignment, addresses, checksums | Peano and bootgen | npuasm. Never the model. | hipfire hit these traps: 16 B branch targets, a hardware loop that ran once on silicon, lock spacing, delay slots, lane-index parity errors, and an AxQoS 15 wedge that needed a reboot. Rules like these belong in code. |
| Hang protection | aie-opt verifier passes on the MLIR | A rule gate built from findings (F-0006, BD and queue limits, F-0288, AxQoS…) runs before every submit. Every run is checked against a CPU oracle. | Below MLIR there is no compiler to catch mistakes. F-0044: a wrong instruction stream gives plausible wrong output. |
| Findings injection | Text first, LCLM-style encoded memory later | Text remains the stage-0 baseline. Then a trained claim→KV memory with a runtime top-k (see [Memory design](#memory-design)). | Context size isn't the reason: the curated notes are about 70k tokens, a few GB of KV. The reasons are choosing a different claim set per model without recomputing the prefix, prefill cost on the NPU, and keeping irrelevant or outdated claims out. |
| Base model | K2-Horizon-MoVA-36B-A4B | Target: Qwen3.8-27B (dense Qwen3.5 recipe, one layer in four full attention, 4 KV heads × 256). Build the memory and the output format on a 4B model first. | Dense is easier to modify (no router). But it samples about 6× slower than a 4B-active MoE, so rejection sampling and RL cost more. Training the projection backprops through the frozen 27B, which needs cloud GPUs and a bounded cost proposal first. |

### Memory design

- **Unit:** one claim-curation record becomes a few slots. Each slot group carries the claim ID, evidence level, stack and conditions, so hearsay and measured claims stay apart.
- **Injection:** projected K/V go into the existing full-attention layers with no rotary position (KBLaM-style), and only the projection trains. If that's too weak, add gated cross-attention blocks with the gate starting at 0, so the frozen model starts unchanged.
- **Runtime k:** first a hard filter on claim conditions against GGUF fields, then top-k by each layer's query-key score. Pick again as generation moves on. Log attention per claim ID so every kernel shows which findings it used.
- **Training:** context distillation. The teacher is the model with the claims as text, the student reads them from slots, and the loss is KL on kernel-writing tasks with held-out claims. Score the effect on generated code, not recall.
- **Second bank:** this machine's measurement log (op, shape, variant, latency, pass/fail) as slots. That is how output becomes specific to one machine without retraining.

### Gates

Each gate must pass by measurement before the next one depends on it. Stage 0 baselines (below) still come first on the model side.

- **A. Lossless round trip. Passed 2026-11-03.** npuasm lifts each build to text and rebuilds it. All 1,849 builds on this machine came back with every artifact identical: 2,267 distinct core programs (5,873 counting each build's distinct cores separately), three CDOs, the PDI and `insts.bin`. Of 11.8M bundles, 0.32% are kept as `.byte` because Peano's assembler rejects `vinsert.32`, `event` and `vinsert.16`. A median build is 127k tokens (CDO 74k, assembly 17k, `insts.txn` 21k; p90 490k). See [npuasm/README.md](npuasm/README.md).
- **B. Our own encoder. Passed 2026-11-03.** [`npuasm/rust`](npuasm/rust/) (`npuasm-core`) adds a text parser to hipfire's `peacemaker-npu` (Apache-2.0; encoder tables generated from llvm-aie TableGen). Every bundle of the 2,267 distinct programs (5.5M bundles) assembles from text to its exact bytes, with no ambiguous text. That includes `vinsert.32`, `event` and `vinsert.16`, which Peano can't assemble. hipfire's decoder needed one extension, for `vldb.fill`, whose FIFO and position registers have no field. The lifted text matches llvm-objdump on every bundle. The full round trip of 1,863 builds now runs only `npuasm-core` and Python, at a median of 0.061 s per build.
- **C. Compact form. Passed 2026-11-03 (lossless layer).** The CDO and transaction text use aie-rt register and field names, with tile context, lock runs, segment reuse, and folded loops (`repeat N i {…}` where numbers step by a constant). Core assembly has labels for branch and loop targets, and near-copy programs are patches on an earlier one. All 1,873 builds still rebuild byte-identically. Over 118 design families the median build fell from 127k to 58k Qwen tokens (p90 490k → 155k, max 2.0M → 282k); CDO −76%, assembly −73%, `insts.txn` −49% in total. Core assembly is now the bulk, at about 3.3 tokens per code byte, most of it real instructions (NOPs 7.7%); loop-folding the assembly would save only 5%. Smaller than this needs a higher layer that isn't lossless, for example a model that writes only the changed parts of a build.
- **D. Run path. Passed 2026-11-03.** `npuasm package` writes the xclbin itself. It parses and rewrites all 1,873 local xclbins exactly, and builds new ones around an npuasm PDI without xclbinutil's JSON copy, which XRT doesn't need. `npuasm rules` is the rule gate: program size, branch-target alignment, delay slots, lock spacing, zero-length valid descriptors, shim AxQoS 15. It was calibrated on the corpus, which passes with no violations. That calibration overturned two of hipfire's bans that it hadn't tested: hardware loops (Peano's form ran exactly on silicon, [npuasm/silicon/zol](npuasm/silicon/zol/)) and lock requests in delay slots (Peano does this in every build). On the NPU, a core edited as text went through rule gate → npuasm build → npuasm package → XRT and was correct (1,024/1,024 words). Still open: a raw-ioctl loader to drop XRT's userspace, routing and lock-protocol deadlock checks, and queue-depth checks.
- **Stage 0, started 2026-11-03.** [eval/](eval/) holds 77 shape-port tasks from 33 design families: port a build to a new shape, answered as a unified diff on npuasm text and scored against the compiler's real output (exact, gap closed, builds, rule gate, vendor bytes). The harness is oracle-checked on all 77. Pilot of 12 tasks through Codex on the ChatGPT subscription, medium reasoning effort: gpt-6.1-sol and gpt-6-astra each got 6/12 exact, byte-identical to the vendor build. That is every easy task, 2/4 medium and 0/4 hard. The hard tasks need the compiler's tiling and memory-layout choices, which build names don't carry. The rule gate refused misaligned branch targets in 5 hard answers. Full baseline (sol, medium effort, 77 tasks): 35 exact by text and 51 functionally correct. Functional correctness comes from a differential NPU check against the real build B, under `dibs run npu`; on hard ports it is 0 exact vs 11 correct. Six answers that passed the rule gate hung; every one timed out and the NPU recovered. Adding a worked example (another pair from the same design, as a diff) on 45 tasks: exact 25 → 29, with no task lost. After an alignment repair (pad branch targets to 16 bytes when that is the only rule-gate problem), functionally correct is 32 → 36, and 52/77 overall. Every equal verdict reproduced at a second input seed. Hangs are now most of what's left; they time out at about 2 s and the NPU recovers. hipfire's simulator can't pre-check them yet (no packet streams or float vector ops). A deterministic baseline with no model (rescale A's numbers from the design's other builds, plus 1/√P-style float constants) is functionally correct on 29/77 against sol's 52/77. Its correct set is nearly a subset of sol's, so models add value where code must change, not just numbers. Hard ports barely move (0 → 1). 91% of a hard port's changes are core assembly that Peano rescheduled, so exact match is the wrong bar there; hard tasks need NPU-functional scoring against a CPU reference. A null control (A unchanged as the answer) passed 16 tasks' NPU checks. On 6, shrinking ports wrote past the end of B's buffers, which guard regions now catch. On 10, random runtime parameters don't exercise the change, so NPU-only passes there no longer count: sol 52 → 51, baseline 29 → 28. Claude Opus 5.5 at `xhigh` effort, through Claude Code on the claude.ai subscription, got 37 exact (4 of them hard) and 61/77 functionally correct against sol's 51. On hard ports it is 23 against 12, at about 7× sol's output tokens. Its 4 losses are hangs, one a `mova`/`movx` swap the rule gate can't see. With effort matched (sol at `xhigh`), sol reaches 55/77, hard 16. Any of the three runs solves 66/77. Fifty correct answers are programs different from B, and on hard ports they edit a median 37 lines where Peano's port changes 209: smaller verified targets for an edit model. There are also 40 hardware-labelled correct-vs-wrong pairs. Qwen3.5-27B waits for the GPU (dibs hold by the manifold healing job). See [eval/results.md](eval/results.md).
- **E. Memory surgery on a 4B model.** Runs in parallel with B–D.
- **F. First direct emission.** The model writes one inner loop in the text form where `measure_ii` shows Peano above the ideal II. Compare II and wall-clock time; widen only if it wins.

## Goal

Input: GGUF metadata. Output: a kernel set that is correct and fast on the first attempt, or close to it.

Writing a kernel that works is the easier half. Optimizing it usually means rewriting it (restructuring loops, fusing ops, changing layouts and dataflow together), and that is the hardest part. So the model writes and rewrites kernel code itself; it does not pick parameters for fixed templates (see [Rejected: planner + fixed generator](#rejected-planner--fixed-generator)).

Two things get it to a good first draft instead of many iterations:
- **Past findings injected as context**: what worked, what failed, and why, selected for the model being ported.
- **Training on rewrites**: slow kernel → fast kernel pairs with the measured gain, not only finished kernels.

The verifier (compile → run on the NPU → compare → time) is a training tool: it filters generated data, scores attempts, and supplies the RL reward. It is not the product.

## Target

**XDNA2 only (AIE-ML v2 / `aie2p`: Strix Point, Strix Halo, Krackan).** No XDNA1 (Phoenix, Hawk Point) data in training, eval, or RAG.

- Eval and verifier run on XDNA2.
- The filter is the verifier, not a label: an upstream example (mlir-aie, IRON, triton-xdna) gets in only if it builds and passes on `npu2`. Many mlir-aie designs take the device as a parameter, so this keeps the device-neutral ones and drops the XDNA1-only ones.
- Strip `npu1` branches and comments from kept examples so the model never sees XDNA1 names, array shapes, or intrinsics.
- Hardware docs: AM027 (AIE-ML v2) only. No AM020.

## Output layer

*Replaced 2026-11-03: the model's target is now npuasm text (see [Revision 2026-11-03](#revision-2026-11-03-direct-emission)). C++ and IRON below remain the source of training pairs and the fallback.*

- **AIE C++ with intrinsics** for compute cores, compiled by Peano. Intrinsics map roughly one-to-one to vector instructions. Peano does register allocation, VLIW slot packing and software pipelining: exact combinatorial jobs a compiler does better than a language model.
- **IRON Python** for dataflow, not raw MLIR. The emitted aie-dialect MLIR is one instance per shape and much larger (`open_kernels/designs/dense/dx.py` is 39 KB and covers every shape; its MLIR for one K2 shape is 254 KB, 1.3 MB with addresses), and the dialect changes between mlir-aie releases.
- **MLIR is the check layer.** The verifier lowers each attempt and runs aie-opt passes, so placement, DMA, BD and lock errors fail in milliseconds without touching the NPU. The existing (IRON → MLIR) pairs (~1,090 emitted `.mlir` files under `open_kernels/designs/`, mostly untracked build output) can be training text that shows what the Python means at the BD/lock level.
- **Machine code: later experiment, inner loops only.** Emitting what actually runs is the preferred end state, but whole-xclbin emission loses compiler feedback (errors become NPU hangs), takes over routing and BD allocation, and costs many tokens. Test it where Peano is weakest: loops where artlite `measure_ii` reports an II above ideal. The model writes the assembly for one such loop; compare II and wall-clock against Peano; widen only if it wins. In its favor: the aie2p ISA is fixed for the generation, and every build yields (C++, ELF) pairs.

## Base model

*Replaced 2026-11-03: the target is Qwen3.8-27B, developed on a 4B model first (see [Revision 2026-11-03](#revision-2026-11-03-direct-emission)). K2 below is still the fallback if sampling cost decides it.*

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
| **Rewrite history**: perf PRs in OpenFlowLM-Next | 18 of the 65 commits touching `open_kernels` on `main` mention performance, most with the measured gain in the title (#22: 20.6× TTFT, 32.1× decode; #170: dense prefill ~3×; #172: q8 prefill ~10×). Intermediate attempts live on per-model branches and in transcripts | (slow → fast, why, measured gain): the core training data for the hardest part |
| **xdna2-notes** findings, recommendations, raw notes | 347 findings (`F-0001`…`F-0347`), 38 mentioning a speedup | injected context (see [Starting from past findings](#starting-from-past-findings)); claims extracted by `claim-curation` |
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

## Starting from past findings

Inject findings (hardware facts and the results of past experiments, successful and failed) so the model's first draft starts where earlier work ended instead of re-discovering it.

1. **Claim extraction** ([`claim-curation/`](../claim-curation/)): split notes into scoped, cited claims. Each record keeps `conditions` (dtype, sizes, toolchain version, workload). Pilot: NuExtract3 matched 30/33 target facts with 66/66 valid citations; stock Qwen3-4B labeled 10/10 claim pairs (contradicts / different scope / supports). A LoRA on the extractor did not help and was not promoted. **The whole-corpus run (590 documents, 937 passages) has not happened**: it kept yielding the GPU to other jobs on 2026-10-25/26 and produced 0 claims.
2. **Selection by the input**: match claim `conditions` against GGUF fields (head_dim, hidden size, expert count, tensor ggml types) first, then rank by similarity. Embedding similarity alone handles numbers badly.
3. **Conflict screening**: findings across toolchain versions disagree. Run the pair check on the selected claims and drop or label out-of-date ones. An out-of-date "success" is the worst thing to inject: the model starts confidently in the wrong direction.
4. **Injection**:
   - **Text in context** first. Works with any base model, no training. This is the stage-0 baseline.
   - **Encoded memory** later ([`context-memory/`](../context-memory/)), to cut prefill cost on the NPU. *2026-11-03: specified as a trained claim→KV memory, see [Memory design](#memory-design).* Pilot: the released LCLM 4x encoder + Qwen3-4B decoder recalled 13/14 facts from six notes (three of them findings: F-0009, F-0014, F-0038) vs 0/14 with zeroed or no memory, and followed swapped canaries. Limits: the whole bank was injected, so selection is untested; it only works with LCLM's own trained decoder, so K2 or Qwen3.6 needs a trained bridge (a separate project); it tests fact recall, not code generation.

Inject failures as well as successes: "tried X, it hung" saves as many iterations as "Y worked".

## Stages

0. **Eval + verifier + baseline** (do first, decide later stages from the result)
   - Build task set from existing ports: GGUF metadata (or an op spec cut from it: op, shapes, dtype) + reference output + perf target. Include rewrite tasks: a working slow kernel plus "make it faster", scored on the gain.
   - Hold out **whole kernel families** (e.g., all of deltanet, all of whisper), not random files. Random splits will measure memorization.
   - Wrap `open_kernels/harness` (`run_kernel.cpp`, `npu_host.py`, `bench.py`) into a pass/fail + speed scorer.
   - Two-tier scoring. **Compile-only tier** (CPU, parallel, no NPU): Peano compile + artlite `measure_ii`, which reads the compiler's optimization records and assembly to report each loop's initiation interval (cycles per iteration), pipelining, cycle count and register spills. Rejects non-compiling and badly scheduled attempts cheaply. **NPU tier**: correctness + wall-clock, only for attempts that pass the first tier. This is what makes stages 2–3 affordable.
   - Pin the NPU power mode during scoring (bandwidth swings 2.2× across modes) and report speed as a fraction of the roofline as well as against the reference kernel. Numbers in [sources.md](sources.md#digest-dram-bandwidth-on-strix-halo-angryloki-svg).
   - Baselines: base K2 cold; base K2 + skills + selected findings as text; Claude + skills + findings (ceiling). If K2 + findings is already close, fine-tuning may not be worth it.
1. **SFT**: kernels, rewrite pairs (slow → fast with the measured gain), PR pairs, cleaned trajectories, with the findings that applied in context so the model learns to use them. LoRA, router frozen.
   - Where OpenFlowLM and artlite both cover an op (qwen35 prefill, gemm, attention), score both on the verifier and use the faster one as the target. Artlite beats OFLM on some of these (see the OpenFlowLM-Next3 "fix issues and bugs found by ArtLite eval" issue), so "our code" can't be the default answer.
2. **Rejection sampling**: sample many attempts per task, keep the ones the verifier passes, retrain. Main data multiplier.
3. **RL with execution reward** (only if stage 2 plateaus and verifier throughput allows).

## Rejected: planner + fixed generator

`xdna2-notes/kernel_planner_agent_plan(1).txt` (scaffold in `xdna2-notes/kernel-planner/`) takes the same input, but the learned part only emits a structured plan (tiling, layout, partitioning, scheduling) and fixed code generates IRON/MLIR around existing compute kernels. That can produce working kernels, but optimizing them usually means rewriting them, and a generator only reaches what its templates already express. Its parameter search may still be useful as a tool inside the verifier.

## Open questions

1. ~~What's the goal?~~ GGUF metadata → efficient kernel code (see [Goal](#goal)).
2. ~~Which target? Which layer?~~ XDNA2 (see Target). Revised 2026-11-03: npuasm text, built by a deterministic assembler and packer (see [Revision 2026-11-03](#revision-2026-11-03-direct-emission)). Before that: AIE C++ + IRON, MLIR as the check layer, machine code as an inner-loop experiment.
3. Verifier throughput: how many NPU machines can run tests in parallel, and how long does one compile + run take? The compile-only tier moves most rejection off the NPU, but correctness still needs it. With npuasm output there is no compile step: a round trip takes a median 0.18 s per build, with Peano's assembler still in the loop.
9. ~~Is hipfire's `pm-npu` source available to read, and under what license?~~ Yes: [warpfront/hipfire](https://github.com/warpfront/hipfire), Apache-2.0 with some MIT files; `pm-npu` is now `peacemaker-npu`. Reused for gate B (path dependency on `~/code/hipfire` at `efa8fbb`; hipfire's NOTICE kept in `npuasm/rust/`).
11. ~~Do Peano's hardware loops (`movxm ls/le`, `add.nc lc`) work on silicon?~~ Yes, measured 2026-11-03 ([npuasm/silicon/zol](npuasm/silicon/zol/)): 1,024/1,024 words in every run, and an edited loop count stopped exactly there. Still unknown: what hipfire's encoder did differently.
10. Does the NPU firmware ignore the PDI's `metaHdrRevokeId` word? It holds leftover memory in about a third of vendor PDIs, which suggests it does. Test it at gate D by loading the same PDI with 0 and with a vendor value.
4. Which GitHub repos hold the issues/PRs (Cyronius/OpenFlowLM-Next plus upstream)?
5. Is triton-xdna a candidate output language? Higher-level means fewer tokens per kernel and easier learning, but less control and a younger toolchain.
6. Artlite's kernels are rendered at run time from Jinja-templated C++. Train on the rendered C++ (what the model must write), the generator Python (how to parameterize), or both?
7. Does the model write the whole kernel set for a model, or only what OFLM's recipes (`open_kernels/recipes/`) refuse: new shapes, ops and architectures?
8. After fine-tuning, do injected findings still help, or has training absorbed them? Score the stage-1 model with and without findings in context.
