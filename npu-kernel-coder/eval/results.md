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

## What the pilot shows

- Both models get every easy port exact (1–4 changed lines) and half the medium ones. Neither gets a hard port, and the two models land on the same tasks.
- Shape ports need the compiler's tiling and memory-layout choices, which the build names don't carry. In `lm_head_q4` 100352 → 65536 both models scaled DMA lengths in proportion (`0x5be000 → 0x3c0000`). The real build retiles: `0x300000`, a chunk of `0x400` instead of `0x500`, and 112 changed lines in the core program as its data-memory addresses move. Neither model touched those core lines.
- The rule gate caught real defects: misaligned branch targets in 2 of sol's and 3 of astra's hard answers, which is the F-0288 hang condition. The assembler rejected a bundle no VLIW format can encode (`padda [p0], #0x200; mov dj0, r1`).
- Local Qwen3.5-27B Q4_K_M has not run: the GPU is reserved in dibs by the manifold healing job until about 02:11. A first attempt was stopped after a few minutes when that hold was noticed.

## Next

- Let npuasm pad branch targets to 16 bytes (as Peano does), so alignment slips stop costing otherwise-correct ports; rerun the rule-gate failures.
- A second input seed for the functionally-equal answers.
- A dataflow-balance check in the rule gate (DMA volume vs. what the core consumes) for the hangs.
- Run Qwen3.5-27B under a dibs GPU hold once the GPU is free.
- Check answers that build and pass the rule gate on the NPU against a CPU reference, so correct non-identical ports also count.
