# Klein denoiser: remove blocks without training, 2026-11-18

Plan: `.claude/plans/flux-denoiser-depth-prune.md`. Scripts: `block_ablate.py` (screen), `block_prune_render.py` (images).
Outputs: `artifacts/flux-conditioning/block-ablate/`, `artifacts/flux-conditioning/block-prune/` (ignored).

Setup: 10 probe prompts, 512 x 512, 4 steps, meanpad 128 text rows, CPU diffusers (bf16). Teacher = the stock model with the same
128 rows. Blocks are numbered 0-24: 0-4 are two-stream, 5-24 are one-stream. A removed block passes its input through unchanged.

## Screen: one block removed, teacher-forced (noise A)

Relative velocity error against the teacher, mean of 10 prompts. Each step starts from the teacher's latent.

| Region | Blocks | Mean error |
|---|---|---|
| Two-stream, first | 0 | 0.86 |
| Two-stream, middle | 1-3 | 0.43-0.47 |
| Two-stream, last | 4 | 0.78 |
| One-stream 0-7 | 5-12 | 0.33-0.47 (block 9: 0.47) |
| One-stream 8-17 | 13-22 | 0.24-0.30 |
| One-stream, last two | 23, 24 | 0.68, 0.53 |

Order, lowest error first: 18, 15, 19, 17, 20, 21, 14, 13, 16, 22, 6, 8, 7, 12, 10, 11, 1, 5, 2, 9, 3, 24, 23, 4, 0.
The order is stable across prompts: for 8 of 10 prompts the 8 lowest blocks all fall in 13-22. Most blocks do the most harm at
step 1 and the least at step 3.

## Images: top N of the order removed, full 4-step runs (noise A and B)

Sheets: `block-prune/sheet-noise-a.png`, `sheet-noise-b.png`.

| Blocks kept | Removed | Pixel MAD | Reading |
|---|---|---|---|
| 25 (teacher) | - | 0 | |
| 24 | 18 | 16.9 | near identical |
| 23 | 15, 18 | 24.8 | same content, small detail changes |
| 21 | 15, 17-19 | 34.1 | decent; drift: "OPEN" breaks on noise A, faces change, colors less saturated |
| 19 | 15, 17-21 | 46.9 | broken: haze, low contrast, blur, apples deformed |
| 17 | 13-15, 17-21 | 54.9 | broken |
| 15 | 13-22 | 53.7 | collapse: grey-brown fog, shapes only |

References: meanpad 128 against meanpad 512 is 31.3. A new seed is 50.8.

## Reading

- Without training, the limit is 21-23 blocks. The cliff is between 21 and 19.
- The failure mode is a wash toward grey: low contrast and fog, not noise or wrong layout at first. That looks like a systematic
  error in the size or mean of the velocity, which a cheap correction can partly fix. Not tested.
- The removal order is one-shot, not greedy. A greedy order can move the cliff a little. It is very unlikely to make 15 blocks usable without training.

## NPU: skip only, and skip with a linear bridge (2026-11-19)

Scripts: `npu_prune_render.py` (resident worker, open-kernels branch `exp/flux2-block-skip`), `bridge_fit.py`.
Setup: Q8_0 GGUF on the NPU, 10 probe prompts, stock 512 text rows, numpy seeds 1 and 2. Teacher = the same path with no skip.
Bridge: X_after = X_before + gate * (Zn W / gate), with Zn the one-stream modulated LN input at the first removed block. W is a
ridge fit per step and per gap, from 48 DiffusionDB prompts (40 fit, 8 check) on CPU bf16. It replaces the gap with one 3072x3072 GEMM.

| Blocks kept | Removed | Skip only MAD | Bridge MAD | Held-out bridge error (step 0-3) |
|---|---|---|---|---|
| 24 | 18 | 17.3 | - | |
| 23 | 15, 18 | 23.8 | - | |
| 21 | 15, 17-19 | 31.5 | 22.9 | 0.52-0.64, 0.63-0.70 |
| 19 | 15, 17-21 | 40.1 | 26.0 | 0.52-0.64, 0.61-0.70 |
| 17 | 13-15, 17-21 | 45.5 | 31.5 | 0.53-0.62, 0.61-0.70 |
| 15 | 13-22 | 47.5 | 34.0 | 0.65-0.75 |

Sheets: `npu-prune/sheet-s1.png` (skip only), `npu-bridge/sheet-s1.png` (teacher, then skip/bridge pairs for 21, 19, 17, 15).

- The NPU skip path agrees with the CPU path (24: 17.3 vs 16.9, 23: 23.8 vs 24.8).
- The bridge moves the cliff by about 4-6 blocks. 19 + bridge is better than 23 skip only. 15 + bridge is close to 21 skip only.
- Images: 19 + bridge is sharp, with small content drift. 17 + bridge is decent; faces and buildings change. 15 + bridge keeps
  layout and colors but is soft and low in contrast; it is not yet decent.
- A linear map misses 52-75% of the gap's change, but the images gain much more than that suggests. Possible cause: the
  bridge fixes the systematic part (the wash toward grey). Not tested.
- This supports a healing run that starts from the bridge, not from plain identity.

## Healing at 15 blocks (13-22 removed), RunPod RTX 3090, 2026-11-19

Script `heal.py`, pod files and results in `artifacts/flux-conditioning/heal/` (`run1/`, sheet `run1/sheet-s1.png`).
Trainable in every arm: bridge W per step (4 x 3072 x 3072) + LoRA r32 on blocks 12 and 23. Data: 1000 DiffusionDB prompts,
teacher-forced, stock 512 rows. 30 min per arm, batch 4, AdamW (bridge 2e-5, LoRA 2e-4), cosine. Held out: 40 prompts x 4 steps.
Images: 10 probe prompts x 2 seeds, diffusers bf16 on the GPU, MAD vs the 25-block teacher on the same path.

| Arm | Start | Loss | Steps | Velocity error | State-after-23 error | Image MAD |
|---|---|---|---|---|---|---|
| ridge | ridge | none | 0 | 0.562 | 0.271 | 36.1 |
| local | ridge | teacher state after block 23 | 1577 | 0.548 | 0.223 | 35.8 |
| global | ridge | final velocity | 3326 | 0.485 | 0.276 | 30.7 |
| global0 | zero bridge | final velocity | 3339 | 0.503 | 0.322 | 31.6 |

- Local healing fixed its own target (-18%) but barely moved the output (-2.5% velocity, images unchanged). Same as the user's
  LLM experience.
- Global healing is the useful one: 30 min on a 3090 gives sharp, colored, recognizable 15-block images (vs fog at no training).
  Content drifts (faces, building layout), so not yet "decent" in the teacher-match sense.
- The ridge start gives a small lasting gain over a zero start at equal time (0.485 vs 0.503, MAD 30.7 vs 31.6). Mainly a head start.
- The local loss costs more per step: it needs the teacher pass through the gap, so it ran about half the steps in the same time.

Cost: about $0.45 for this pod; about $4.30 lost earlier on six pods that a DNS failure left running. Total about $4.75 of the $10 limit.

### Long global heal (run2), RTX 4090 secure, 2026-11-19

Continued run1's global arm for 120 min more (24822 steps, same data, init check 0.4852 = run1). Results in `heal/run2/`, sheet `run2/sheet-s2.png`.

| | Velocity error | Image MAD (seed set of run2) |
|---|---|---|
| ridge | - | 35.0 |
| global, 30 min | 0.485 | 30.6 |
| global, 30 + 120 min | 0.468 | 29.3 |

- Diminishing returns: 4x more training gives -3.5% velocity error, -1.3 MAD. Training loss went 0.23 -> about 0.17 in the first hour, then flat.
- Images are cleaner and a little sharper; the content drift (faces, beards, layout) stays. The limit is probably the setup, not
  the training length: teacher-forced inputs, LoRA on only 2 blocks, a linear bridge.
- Cost about $2.40 (secure 4090, $0.89/h; 15 min of it was a slow 170 KB/s upload). Session total about $7.20 of the $10 limit.

## Healing v2 (run3): full fine-tune, half on the student's own path, A100 80 GB, 2026-11-20

Plan `.claude/plans/flux-denoiser-heal-v2.md`, script `heal2.py`, results `artifacts/flux-conditioning/heal/run3/`.
Trainable: all 15 kept blocks (2.45B, bf16 + fp32 master, bf16 Adam states, step in backward) + bridge W. Start: run2 merged.
Loss: velocity + 0.1 x (state after block 12 + state after block 24). Half the batches teacher-forced, half on the student's own
4-step path (teacher velocity at the student's latent). 4000 prompts, new noise every batch, batch 4, 1.3 s per iteration.

- lr 2e-5 made it worse (loss 0.24 -> 0.29 teacher-forced, 0.23 -> 0.34 own path, 600 iterations). Restarted at 5e-6 (bridge 1e-5).
- At 5e-6 the training loss fell slowly and steadily: teacher-forced 0.239 -> about 0.197, own path 0.26 -> about 0.21 (iteration 1500 -> 11000).

| Eval | Iterations | Velocity error, teacher-forced | Velocity error, own path | Final latent error | Image MAD |
|---|---|---|---|---|---|
| e0 (= run2 merged) | 0 | 0.458 | 0.474 | 0.700 | 28.8 |
| e1 | 5527 | 0.445 | 0.466 | 0.731 | 29.4 |

- e1 images look better than e0 (apple stems and texture back, portrait close to the teacher), but MAD and the final latent error
  are a little worse. The images drift less in quality and still drift in content.
- Lost: the e2 and final evals and the checkpoint. The local launcher stopped from about 08:20Z to 11:17Z (probably the PC slept),
  so nothing fetched them before the pod self-deleted at its cost cap (10:48Z). Training itself ran to the end.
- Cost $9.50 (410 min at $1.39/h, the cap). Session total about $16.70.

## Healing at 17 blocks (13-15 and 17-21 removed), run4, A100 80 GB, 2026-11-20

Same method as run3 (`heal2.py --gaps 13-15,17-21 --bridges bridge/b17.npz`, lr 5e-6, bridge 1e-5), started from the original
weights and the two ridge bridges. 2.70B trainable. Checkpoint fetched every hour (`heal/run4/ckpt.pt`, final, 5.7 GB).
Sheets: `heal/run4/work/full/` (full model, start, 2 h, final).

| Eval | Iterations | Velocity error, teacher-forced | Velocity error, own path | Final latent error | Image MAD |
|---|---|---|---|---|---|
| e0 | 0 | 0.503 | 0.515 | 0.795 | 32.2 |
| e1 | 2502 | 0.430 | 0.449 | 0.755 | 31.2 |
| e2 | 5025 | 0.413 | 0.429 | 0.718 | 29.0 |
| e3 | 7555 | 0.407 | 0.420 | 0.704 | 30.7 |
| e4 | 10063 | 0.399 | 0.412 | 0.692 | 27.8 |
| e5 | 12572 | 0.394 | 0.407 | 0.676 | 26.8 |
| final | 14459 | 0.392 | 0.406 | 0.675 | 26.8 |

- Best result of all runs: MAD 26.8 (15 blocks best 29.3). Training loss 0.27/0.25 -> 0.16/0.18 (teacher-forced / own path).
- Images: sharp, same layout and subject as the full model in most cases (3 apples in the same row, armor and pose match).
  Faces drift a little. Text is still broken ("OPEN" sign).
- Gains get smaller per hour but did not stop: -0.005 to -0.01 velocity error per hour in the last 3 hours.
- Cost: about $8.56 for this pod plus $0.31 for a first pod that failed on a script typo. Session total about $25.60.

### run4 on the NPU (2026-11-20)

`ckpt_to_gguf.py` wrote the 17 healed blocks into a Q8_0 copy (`C:\dev\klein-gguf\flux-2-klein-4b-heal17-Q8_0.gguf`).
Renders: `npu_prune_render.py --gguf ... --bridges heal/run4/bridge-v2.npz`, 10 prompts x 2 seeds, sheets `npu-heal17/full/`.

- MAD vs the 25-block NPU images: skip only 45.5, ridge bridge 31.5, healed 27.0 (GPU bf16 26.8, so Q8_0 adds almost nothing).
- Speed, `dibs run --measure` (CLEAN before and after), `npu_time.py`, 512 x 512, 4 steps, median of 6 after 1 warm-up:
  stock 25 blocks 3.836 s, healed 17 blocks 2.904 s: -24% time, 1.32x faster. Compute alone predicted -32%; the rest is cost
  that does not scale with blocks (embedders, modulation, the bridge GEMMs, per-step overhead).

### run4 with 128 text rows (CPU, noise A, 10 prompts, `artifacts/flux-conditioning/rows-heal17/`)

MAD: healed 512 rows vs full 512 = 28.2; healed 128 (mean padding) vs full 512 = 32.4, vs full 128 = 30.5, vs healed 512 = 24.4.
The 128-row change adds about the same drift to the healed model as to the full model (31.3). Not pursued on the NPU: dit_gemm
works in 512-row groups (M % 512), so 1152 rows cost the same as 1536; only attention keys would shrink (a few % per step).

## Step count (CPU, 2026-11-20, `artifacts/flux-conditioning/steps-test/`)

- Original model, 4 vs 8 steps, 4 prompts x 3 seeds: 8 steps adds clear detail (texture, sharper backgrounds) at 2x cost; layout
  stays. On seed 1, 6-8 steps changed content (2 apples, a beard); on seeds 2-3 it did not.
- 17-block run4 checkpoint (3 h) at 6 and 8 steps, nearest trained bridge per noise level: no break; more detail, mostly at 6 steps.
  17 blocks x 6 steps costs about the same as the original at 4 steps, so for the pruned model only 4 (and maybe 5) steps make sense.
