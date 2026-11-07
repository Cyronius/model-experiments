# Stage 0 shape-port results

Measured 2026-11-03. Pilot: 12 fixed tasks (`run.py --tasks pilot`), 4 per difficulty. Codex CLI 0.164.0, signed in with ChatGPT, reasoning effort `medium`, prompt on stdin, empty read-only sandbox; neither model used a tool. Scoring and the harness check are described in [README.md](README.md).

| Model | Exact (= vendor bytes) | Easy | Medium | Hard | Builds | Rule gate | Gap closed, median | Seconds, median | Tokens in (cached) / out (reasoning) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gpt-6.1-sol-medium | 6/12 | 4/4 | 2/4 | 0/4 | 11/12 | 9/12 | 0.675 | 47.2 | 353,696 (145,024) / 32,041 (6,808) |
| gpt-6-astra-medium | 6/12 | 4/4 | 2/4 | 0/4 | 11/12 | 8/12 | 0.677 | 45.6 | 356,316 (133,376) / 25,934 (4,852) |

Every exact answer also rebuilt to B's vendor bytes.

Per task (gap closed; `x` = exact, `B` = does not build, `R` = fails the rule gate):

| Task | Difficulty | Design | A → B | Changed lines | sol | astra |
|---|---|---|---|---:|---|---|
| T0671b5da8d | easy | npu-dev | gemma4_ple_input_h2560_token → gemma4_ple_input_h1536_token | 1 | x | x |
| T0fce3ac302 | easy | gemma4_multimodal_ops_bundle_v2 | vision_online_attn_472x64 → vision_online_attn_1024x64 | 2 | x | x |
| T314bc26723 | easy | gemma4_e2b_text_kernel_bundle_v2 | mlp_gelu_12288 → mlp_gelu_6144 | 4 | x | x |
| Te5806288ba | easy | gemma4_e4b_text_kernel_bundle_v2 | attention_d512_r1024 → attention_d512_r512 | 2 | x | x |
| T17a0bce0c8 | medium | gemma4_multimodal_ops_bundle_v2 | audio_subnorm_32 → audio_subnorm_128 | 56 | B 0.2321 | B 0.2321 |
| T1f1b254ef9 | medium | llama32-1b-gguf-compact-g4-flashpair8k | ag_pv3840 → ag_pv3584 | 25 | x | x |
| T22b849941a | medium | llama32-1b-gguf-flash8k | ag_pv2304 → ag_pv2560 | 25 | x | x |
| T4c508e607b | medium | lm_head_q4 | build_100352_f32_c77b8bb1 → build_65536_f32_e97084fb | 77 | 0.1039 | 0.1039 |
| T0b5e0497b9 | hard | projections | n1024_k2560 → n2048_k2560 | 232 | -0.056 | R -0.0129 |
| T4394ba3d4b | hard | gemma4_e4b_text_kernel_bundle_v2 | head_norm_h2_d512 → head_norm_h8_d256 | 155 | 0.0903 | 0.0903 |
| T4469398b3c | hard | projections | bf16_n256_k2560 → bf16_n2560_k256 | 245 | R 0.3102 | R 0.302 |
| T65ca8bff72 | hard | projections | bf16_n256_k1536 → bf16_n1536_k256 | 217 | R 0.3502 | R 0.3548 |

## A worked example from the same design (45 tasks, gpt-6.1-sol, medium effort)

`add_examples.py` gives 45 of the 77 tasks a second pair of builds from the same design, shown as its unified diff. That pair never includes B; preferred is a pair where the same name positions change. The other 32 tasks come from designs with only two builds. The same 45 tasks were run with and without the example (`--variant ex1`); `compare.py` pairs them.

| difficulty | n | exact gpt-6.1-sol-medium | exact gpt-6.1-sol-medium-ex1 | gap median | gap median | gap better / same / worse |
|---|---:|---:|---:|---:|---:|---|
| easy | 5 | 5 | 5 | 1.000 | 1.000 | 0 / 5 / 0 |
| medium | 25 | 20 | 23 | 1.000 | 1.000 | 4 / 21 / 0 |
| hard | 15 | 0 | 1 | 0.090 | 0.152 | 5 / 6 / 4 |
| all | 45 | 25 | 29 | 1.000 | 1.000 | 9 / 32 / 4 |

- **Exact:** 25 → 29 of 45, and no task lost exactness. Answers that rebuild: 41 → 44; the rule gate passes 38 in both runs.
- **Medium tasks:** the example supplies what the names don't carry. In `lm_head_q4` 100352 → 65536 it shows how the core's buffer addresses and the chunk count move with the vocabulary, and the port becomes exact.
- **Hard tasks:** they barely move (0 → 1 exact; median gap closed 0.09 → 0.15). Of the 2,731 lines a hard port must change, 2,490 (91%) are core assembly. Even with the example, the answers leave 2,049 of those wrong. They leave 13 of 50 CDO lines and 80 of 191 `insts.txn` lines wrong.

When a shape changes enough, Peano reschedules the core: trip counts, unrolling, register allocation and addresses all move. Matching that output line for line is the wrong bar for hard ports, because a correct core scheduled differently should count. Scoring hard tasks needs an NPU run against a CPU reference.

## Functional check on the NPU (differential, against the real build B)

`npu_check.py` ran under `dibs run npu` on 2026-11-04 (Strix Halo, firmware 1.1.2.65, XRT 2.27.0). It checked every answer that builds and passes the rule gate.

- **Controls:** all 6 exact answers in the astra pilot came back `equal`, which confirms the harness and npuasm's xclbin packaging against the vendor builds.
- **No vacuous passes:** B wrote at least one buffer in every scored task (128 B to 1.8 MB), so no verdict was empty agreement.
- **Different code, same result:** 12 distinct inexact answers whose binaries differ from B gave B's output bit for bit. Their core code was 8–295 lines away from B's. They port the shape correctly with A's code structure where Peano generated new code.
- **Caveat:** these are equal on one seeded input per task, which is strong evidence but not proof.

**Functionally correct** means exact text, or byte-identical rebuild, or NPU `equal`/`close`:

| gpt-6.1-sol, medium effort | Tasks | Exact text | Functionally correct | Why the rest fail |
|---|---:|---:|---:|---|
| Easy | 12 | 12 | 12 | |
| Medium | 33 | 23 | 28 | 3 don't build, 2 compute wrong values |
| Hard | 32 | 0 | **11** | 9 wrong values, 7 rule gate, 3 don't build, 2 hang |
| All | 77 | 35 | **51** | |

On the 45 tasks with a worked example: 31 functionally correct without the example, 34 with it (medium 21 → 24, hard 5 → 5). With the example, wrong values fall from 5 to 1, but rule-gate rejections rise from 3 to 6. The model makes bolder core edits, and most rejections are branch targets knocked off 16-byte alignment, which Peano pads automatically.

**Hangs:** 6 answers that passed the rule gate hung, across 3 tasks and 3 runs. Every one ended at the 20 s timeout, and B ran correctly right after, so the NPU recovered each time. A typical cause is attention `r128 → r1`: the model fixed the host DMA length but not the core's data volume, so the core waits on a lock forever. A static dataflow-balance check would catch this; the timeout already makes it safe.

## Alignment repair and a second input seed (2026-11-04)

**Alignment repair.** When the rule gate rejects an answer *only* for branch-target alignment, `score.py` pads the referenced labels to 16 bytes (`core.align_targets`, the `.p2align 4` directive in npuasm-core, as Peano lays code out) and rechecks. That recovered every rule-gate rejection: sol 64 → 71 of 77, sol with example 38 → 44 of 45, astra 8 → 11 of 12. On the NPU the 16 repaired answers gave 4 `equal`, 3 `differs` and 9 hangs. So misalignment was usually a symptom of a broader wrong edit to the core.

**Second seed.** All 95 `equal`/`close` verdicts at the first seed reproduced at a second seed (7), including all 25 inexact answers.

**Functionally correct** (exact, byte-identical, or NPU `equal`/`close` after repair), gpt-6.1-sol, medium effort:

| Run | Easy | Medium | Hard | All | Failures |
|---|---:|---:|---:|---:|---|
| 77 tasks | 12/12 | 28/33 | 12/32 | **52/77** (51 after the null control below) | 14 wrong values, 6 don't build, 5 hang |
| 45 tasks, no example | 5/5 | 21/25 | 6/15 | 32/45 | 5 wrong values, 4 don't build, 4 hang |
| Same 45, with example | 5/5 | 24/25 | 7/15 | **36/45** | 7 hang, 1 wrong value, 1 doesn't build |

With the worked example, hangs are most of what's left. Hangs remain cheap and safe: across 25 hung answer runs (seed 1, including repaired answers), every one ended in `ERT_CMD_STATE_TIMEOUT` after about 2 s, and the vendor build ran correctly right after.

**Checking for deadlocks before the NPU is not done.** hipfire's CPU simulator (`peacemaker-npu sim`) loads exact PDI and TXN bytes, but it does not support packet-switched streams, memtile DMA, non-contiguous DMA or most bf16/float vector instructions. All 77 tasks use packet streams and float vector code, so it cannot run them yet. A pre-check would need either a "control skeleton" mode in that simulator (locks, DMA, scalar loop control, with vector math as no-ops, plus packet streams) or a configuration-level stream-volume check with route tracing, which cannot see the core's loop counts. Both are real projects; the runtime timeout covers the risk meanwhile.

## Deterministic baseline: rescale A's numbers (`baseline_scale.py`, no model)

For every number in A's text, predict its value at B's shape parameters (the numbers in the build name):
- **BL1, fitted (46 tasks):** the design's other builds (never B; up to 16 nearest in shape) are aligned with A line by line, with numbers masked. Each number position gets the simplest exact relation consistent with all of them: constant; a·P + b in one parameter; a·log₂P + b; a·P·Q + b or a·P/Q + b. Ambiguous or non-integer fits keep A's value.
- **BL0, guess (31 tasks with no other build):** a number equal to c·P or c·P ± 1 for a changed parameter P, with c round (2^k, 3, 5, 6, 12), is rescaled. Only size-like configuration fields and immediates of movs into general registers are touched. A first version without that restriction also rescaled stack offsets and base addresses that matched by coincidence.
- **Both:** a 32-bit immediate whose fp32 value is exactly f(P) for f ∈ {P, 1/P, √P, 1/√P} is re-evaluated at B. Example: `0x3ca1e89b` = 1/√2560 → `0x3cd105ec` = 1/√1536.
- **Labels** are never rewritten.

Results: exact text 24/77 (BL1 22: 5/5 easy, 17/25 medium, 0/16 hard; BL0 2). 72 answers build and pass the rule gate. On the NPU its 48 inexact answers gave 5 `equal`, 23 `differs` and 20 hangs. It rescales host DMA sizes but cannot restructure the core, which is exactly the volume mismatch that deadlocks. All 20 hangs again recovered.

**Functionally correct**, baseline vs gpt-6.1-sol (medium effort):

| Difficulty | Tasks | Baseline | sol | Both | Baseline only | sol only |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 12 | 9 | 12 | 9 | 0 | 3 |
| Medium | 33 | 17 | 28 | 17 | 0 | 11 |
| Hard | 32 | 3 | 12 | 2 | 1 | 10 |
| All | 77 | **29** | **52** | 28 | 1 | 24 |

After the null-answer control and guard regions (below), the baseline is 28/77 (9/17/2) and sol 51/77.

The baseline's correct set is nearly a subset of sol's (28 of 29). The model adds 23 tasks, mostly where code must change rather than numbers. That's the bar for any trained model, including an edit-diffusion one.

## Null-answer control and guard regions (2026-11-04)

**Control.** `run.py --backend null` answers every task with no edit, i.e. build A unchanged. If A itself passes the NPU check, that task's check can't tell a port from no port. At seed 1, A unchanged passed 16 of 77 tasks (10 `equal`, 6 `close`). There were two causes:

1. **Shrinking ports (6 tasks).** These are gelu 15360 → 12288 and 12288 → 6144, silu 4096 → 1024 (twice), gptoss gemv b90 → b1, and rope e4b → e2b. A does more work than B. Inside B's buffers it writes exactly what B writes, and the rest went past the end of each buffer, which was never compared. Fix: `--guard N` (`npu_exec.py`, `npu_check.py`) appends N bytes after every buffer, filled from a separate RNG. B never writes them, so an out-of-bounds write makes the verdict `differs`. With a 1 MB guard, all 6 tasks discriminate.
2. **Runtime parameters (10 tasks).** A unchanged still passes these at both seeds, even with guards. They are the attention capacity ports (`attention_d512_r128 → r512`, `r1024 → r512`, `r128 → r1`, `d256_r128 → r512`, `gemma4-attention-512-1024 → 512`, `vision_online_attn_472x64 → 1024x64`), `ple_input` h2560 → h1536 (token and context variants), `vision_pos_add` v2 → v3 and `gptoss-rope` v3 → v2. The 6 attention ports take a ui8 parameter buffer, and with random bytes in it the changed capacity is apparently never exercised. Scoring them needs real parameter values. For the other 4 the reason is not known: `pos_add` v2 → v3 may simply compute the same function.

**Scoring from here on:** an NPU-only pass on one of those 10 tasks counts as "unverifiable", not correct; exact text and byte-identical rebuilds still count (`functional.py`). Every `equal`/`close` verdict of every model run (Opus, sol, sol with example, astra) was rechecked with a 1 MB guard, and none flipped. One baseline answer did: it writes past the end of a buffer. Revised totals: sol 52 → 51, sol with example 36/45 unchanged, baseline 29 → 28.

## Claude Opus 5.5 at `xhigh` effort vs gpt-6.1-sol (2026-11-04)

**Setup.**
- Claude Code 2.1.318 `claude -p`, signed in with the claude.ai Max subscription: `--model claude-opus-5-5 --effort xhigh`.
- All tools off, no MCP servers or settings files, a one-line system prompt.
- The same prompts as sol, on stdin, in an empty temp directory, 4 tasks at a time.

Effort is not matched: sol ran at `medium`.

| 77 tasks | sol, medium | Opus 5.5, xhigh |
|---|---:|---:|
| Exact text (easy / medium / hard) | 35 (12 / 23 / 0) | 37 (10 / 23 / **4**) |
| Byte-identical rebuild | 38 | 39 |
| Builds | 71 | 74 |
| Rule gate (after alignment repair) | 64 (71) | 70 (74) |
| Gap closed, median: all / hard | 0.627 / 0.048 | 0.724 / 0.207 |
| Seconds per task, median: all / hard | 47 / 52 | 126 / 368 |
| Output tokens per task, median: all / hard | 2,017 / 2,334 | 15,046 / 42,548 |

**Functionally correct** (exact, byte-identical, or NPU `equal`/`close` with 1 MB guards; null control applied):

| Run | Easy | Medium | Hard | All | Failures |
|---|---:|---:|---:|---:|---|
| gpt-6.1-sol, medium | 12/12 | 27/33 | 12/32 | **51/77** | 14 wrong values, 6 don't build, 5 hang, 1 unverifiable |
| Claude Opus 5.5, xhigh | 10/12 | 28/33 | **23/32** | **61/77** | 6 hang, 6 wrong values, 3 don't build, 1 unverifiable |
| Baseline (no model) | 9/12 | 17/33 | 2/32 | 28/77 | 24 wrong values, 20 hang, 5 don't build |

| Paired | Tasks | sol | Opus | Both | sol only | Opus only |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 12 | 12 | 10 | 10 | 2 | 0 |
| Medium | 33 | 27 | 28 | 27 | 0 | 1 |
| Hard | 32 | 12 | 23 | 10 | 2 | 13 |
| All | 77 | **51** | **61** | 47 | 4 | 14 |

- **Opus wins on hard ports.** It gets 4 exact (sol none) and 23 functionally correct against sol's 12. On the 14 tasks only Opus solves, sol's answer computes wrong values on 7, hangs on 4 and doesn't build on 3. Opus's 25 inexact `equal` answers all reproduce at a second seed and with guards.
- **Opus's losses are all hangs.** Every task only sol solves is one where Opus's answer hangs. Two of them are easy ports:
  - **`vision_online_attn_472x64 → 1024x64`:** Opus rewrote `movx r1, #0x3ff` as `mova r1, #0x3ff`. The encoding is the same size and valid, and the value fits (`mova` takes a signed 11-bit immediate). The answer still hangs. The same swap in build A, at A's own value `0x1d7`, hangs too, so `mova` and `movx` are not interchangeable at that point in the schedule. LLVM's itineraries (hipfire `sched.rs`) give both a 1-cycle result, and the cause is not known. Neither the assembler nor the rule gate catches it.
  - **`1024x64 → 472x64`:** the DMA length came out as `0x3b00`, half of the right `0x7600`.
- **Cost.**
  - **Thinking:** Opus thinks a lot. Its median output is 15k tokens, about 10k of them thinking, against sol's 2k. One hard task used 127,975 output tokens of the 128k cap, and its answer was correct.
  - **Time:** the run took 4.95 hours of model time, about 1.4 hours of wall-clock time 4 at a time.
  - **Billing:** it was billed to the subscription, about $53 at API list prices.
- **Next:** an effort-matched comparison (sol at `xhigh`).

## gpt-6.1-sol at `xhigh`, and what the three runs give together (2026-11-05)

sol was run again at `xhigh` effort to match Opus's: Codex CLI on the ChatGPT subscription, 4 tasks at a time, with the NPU check using 1 MB guards and a second seed. All 19 inexact `equal` answers reproduced at seed 7.

| 77 tasks | sol, medium | sol, xhigh | Opus 5.5, xhigh |
|---|---:|---:|---:|
| Exact text (easy / medium / hard) | 35 (12 / 23 / 0) | 36 (12 / 24 / 0) | 37 (10 / 23 / 4) |
| Gap closed, median: all / hard | 0.627 / 0.048 | 0.681 / 0.106 | 0.724 / 0.207 |
| Seconds per task, median | 47 | 98 | 126 |
| Output tokens per task, median | 2,017 | 4,506 | 15,046 |
| **Functionally correct** (easy / medium / hard) | 51 (12 / 27 / 12) | 55 (12 / 27 / 16) | **61** (10 / 28 / 23) |
| Failures | 14 wrong values, 6 don't build, 5 hang, 1 unverifiable | 9 wrong values, 8 hang, 5 don't build | 6 hang, 6 wrong values, 3 don't build, 1 unverifiable |

**More effort helps sol a little, all of it on hard ports:** 12 → 16 correct. With bolder edits, wrong values fall (14 → 9) and hangs rise (5 → 8). At matched effort Opus still leads by 6: it solves 10 tasks sol `xhigh` doesn't (9 of them hard, where sol mostly hangs), and sol solves 4 Opus doesn't (the 2 easy Opus hangs and 2 hard).

**Together** (`ensemble.py`; selection by the NPU check, which needs B or a CPU reference):

| Difficulty | Tasks | sol medium | sol xhigh | Opus xhigh | Any of the three | All three |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 12 | 12 | 12 | 10 | 12 | 10 |
| Medium | 33 | 27 | 27 | 28 | 28 | 27 |
| Hard | 32 | 12 | 16 | 23 | 26 | 9 |
| All | 77 | 51 | 55 | 61 | **66** | 46 |

Opus with either sol run covers 65, and the third run adds 1. Eleven tasks nobody solves. Two of them sit among the 10 unverifiable tasks, where only exact text counts.

**Variety among correct answers.**
- On 16 of the 55 tasks that two or more runs solve, the correct answers differ: 72 distinct correct texts in all.
- 28 of them are programs whose bytes differ from B's: different code, same output on the NPU at two seeds. All but 4 are on hard ports.
- Across all three runs, 50 correct answers are programs different from B. They are small edits: on hard ports the median answer changes 37 lines of A, while Peano's A → B changes 209. The ratio is 0.25 on medium and hard alike.
- So for most ports a working edit about a quarter the size of the compiler's rescheduled output exists, and the models find it.

**Labelled failures.** The three runs also give 62 wrong answers with a hardware-verified failure (wrong values, hang, doesn't build, rule gate). 20 tasks have both a correct and a wrong answer, which makes 40 correct-vs-wrong pairs for the same prompt.

## Local base models, zero-shot (2026-11-06/07, in progress)

Which local model to fine-tune? `bakeoff.sh` serves each GGUF with llama.cpp build 12060 (Vulkan, a pinned copy of the binary) under a dibs GPU hold and runs the 12 pilot tasks: thinking off, temperature 0, 8,192 output tokens, 40,960 context. Model rows are in `bakeoff_models.tsv`; the tables come from `bakeoff_report.py`. Scores are text-only: model answers are not checked on the NPU (`runs/.npu_disabled`, after the 2026-11-05 freeze), so only exact text and byte-identical rebuilds count as correct.

| Model | Correct | Easy / medium / hard | Builds | Rule gate | Hunks applied | Gap closed, median | Hit 8k cap | Decode tok/s | Seconds, median |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-Coder-30B-A3B Q4_K_M | **0** | 0 / 0 / 0 | 8 | 8 | 51% | -0.024 | 5 | 18.7 | 327 |
| AREX-2 (Qwen3.8-27B) Q4_K_M | **0** | 0 / 0 / 0 | 11 | 11 | 90% | -0.034 | 4 | 5.0 | 1217 |
| Clover-35B-A3B Q4_K_M | **0** | 0 / 0 / 0 | 7 | 7 | 74% | -0.422 | 6 | 12.0 | 623 |
| *gpt-6.1-sol, medium* | 6 | 4 / 2 / 0 | 11 | 11 | 100% | 0.675 | 0 | – | 47 |
| *Claude Opus 5.5, xhigh* | 6 | 3 / 2 / 1 | 11 | 11 | 100% | 0.605 | 0 | – | 112 |
| *Rescale baseline (no model)* | 4 | 2 / 2 / 0 | 10 | 10 | 100% | 0.090 | 0 | – | 0 |

- **No local model gets a single task, not even the easy ones.** The easy ports change 1–4 lines, and the frontier models get all four.
- **The answers mostly make A worse.** Of 36 answers, 28 move further from B than A is (gap below 0). 4 change nothing that matters and only 4 close any of the gap. Qwen3-Coder emits hunks that replace a line with itself. AREX-2 and Clover rewrite lock and DMA tables that the port leaves alone.
- **The key edit is arithmetic, not pattern copying.** On `ple_input` h2560 → h1536 the whole port is one float constant in the core program: `movxm r2, #0x3ca1e89b` (1/√2560) → `#0x3cd105ec` (1/√1536). sol makes exactly that edit. No local model finds the line.
- **Answers run long.** 15 of 36 answers stop at the 8k output cap, partway through a diff (median output 6.5–8.2k tokens).
- **Runtime fixes made on the way.** Build 12060 with flash attention on degenerates (`GGGG…`, `3333…`) on prompts over about 16k tokens. This happened on qwen3moe, clover_moe and qwen35moe, so every row now runs `-fa off`; the flash-attention runs are in `runs/_superseded/`. Flint always reasons, and a 16k thinking budget ran out before the answer, so it uses `--reasoning-budget 0`.
- **Not run yet:** flint (the chain was waiting for the GPU when the machine went down on 2026-11-07 at about 02:05; `runs/bakeoff-flint/` is empty), Apex-35B, Qwen3.8-27B, Bonsai-2-27B Q2_0, Qwen3.6-35B-A3B (its flash-attention run was superseded), and the `r3` (toolchain-feedback retries) and `ex1` (one worked example) rows.
- **Caveat:** since 2026-11-05 some processes on this GPU compute wrong values without any error when several sessions share it. That was seen with ROCm. The bakeoff uses Vulkan and its server logs show no errors, but a faulty decode can't be ruled out completely.

So far, zero-shot score can't separate the local models. The choice will have to rest on how trainable each model is (tokenizer, architecture, licence, whether a training stack exists) and on decode speed, unless a model still to run scores above 0.

## What the pilot shows

- Both models get every easy port exact (1–4 changed lines) and half the medium ones. Neither gets a hard port, and the two models land on the same tasks.
- Shape ports need the compiler's tiling and memory-layout choices, which the build names don't carry. In `lm_head_q4` 100352 → 65536 both models scaled DMA lengths in proportion (`0x5be000 → 0x3c0000`). The real build retiles: `0x300000`, a chunk of `0x400` instead of `0x500`, and 112 changed lines in the core program as its data-memory addresses move. Neither model touched those core lines.
- The rule gate caught real defects: misaligned branch targets in 2 of sol's and 3 of astra's hard answers, which is the F-0288 hang condition. The assembler rejected a bundle no VLIW format can encode (`padda [p0], #0x200; mov dj0, r1`).
- Local base models zero-shot (see the bakeoff section above): 0 of 12 so far for Qwen3-Coder-30B-A3B, AREX-2 and Clover-35B-A3B, easy ports included.

## Next

- Finish the base bakeoff rows for flint, Apex-35B and Qwen3.8-27B. Run `r3`/`ex1` only if a model scores above 0. Then pick the model to fine-tune on trainability and decode speed.
- Re-enable NPU checks of model answers only after the user approves safeguards. Until then every local score is text-only.
- Real runtime parameters for the 6 attention tasks whose NPU check A unchanged passes, and a look at the other 4 (`ple_input`, `pos_add`, `rope`), so they can be scored functionally.
- Find out why `mova` can't replace `movx` before an `ltu` that reads the register, and add a check to the rule gate.
- A deadlock pre-check: a control-skeleton mode for hipfire's simulator, or a stream-volume check with route tracing (see above).
