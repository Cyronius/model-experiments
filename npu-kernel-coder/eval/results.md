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

## What the pilot shows

- Both models get every easy port exact (1–4 changed lines) and half the medium ones. Neither gets a hard port, and the two models land on the same tasks.
- Shape ports need the compiler's tiling and memory-layout choices, which the build names don't carry. In `lm_head_q4` 100352 → 65536 both models scaled DMA lengths in proportion (`0x5be000 → 0x3c0000`). The real build retiles: `0x300000`, a chunk of `0x400` instead of `0x500`, and 112 changed lines in the core program as its data-memory addresses move. Neither model touched those core lines.
- The rule gate caught real defects: misaligned branch targets in 2 of sol's and 3 of astra's hard answers, which is the F-0288 hang condition. The assembler rejected a bundle no VLIW format can encode (`padda [p0], #0x200; mov dj0, r1`).
- Local Qwen3.5-27B Q4_K_M has not run: the GPU is reserved in dibs by the manifold healing job until about 02:11. A first attempt was stopped after a few minutes when that hold was noticed.

## Next

- Give the model a second pair from the same design in the prompt (a worked example of how the design's numbers move), and/or the selected findings. Measure the change on the same pilot, then on all 77 tasks.
- Run Qwen3.5-27B under a dibs GPU hold once the GPU is free.
- Check answers that build and pass the rule gate on the NPU against a CPU reference, so correct non-identical ports also count.
