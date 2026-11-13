# Healing a pruned Qwen3.5-2B: which schedule heals fastest per FLOP?

When you prune a model you have to heal it, usually by distilling from the unpruned parent.
Healing can train the whole network at once, or a few layers at a time against the parent's
hidden states (block-wise or layer-wise reconstruction). This experiment races 15 such
schedules on a pruned Qwen3.5-2B, at equal compute on one local GPU. The aim was to find which
heals most efficiently, not to ship a model. One seed and small budgets throughout; see
[Caveats](#caveats).

Short version, for anyone setting up a healing run:

1. **Train the whole network first.** Every schedule that starts with whole-network distillation
   reached a quarter of the gap in about 0.5e15 FLOPs. Every block-wise schedule needed 2.25e15
   or more: four to seven times as much.
2. **Decay the learning rate.** At a constant learning rate, whole-network training improves fast,
   then stalls and gets worse. With a cosine decay to zero, the same training kept improving and
   finished first on loss, ARC-Easy and GSM8K.
3. **Get the learning rate right before anything else.** At batch size 1, a learning rate of 1e-4
   left the light student *worse* than it started. The learning rate mattered more than any
   schedule.
4. **Don't switch to small blocks after whole-network training.** It undid the gains and never
   recovered.
5. **Random windows, adaptive controllers and teacher-fed blocks didn't help.** None beat the
   simple options.

## Setup

**Teacher:** Qwen3.5-2B (1.88B unique text parameters).

**Students:** two prunes of the teacher, both made with activation-based (Minitron-style)
selection of residual coordinates and FFN neurons. Depth and attention are untouched.

| Student | Size | Cut | Validation CE at start | 100% rung | Gap |
|---|---:|---:|---:|---|---:|
| Light (`damaged-light-h1792-n1473244736`), hidden 1792, FFN 4800 | 1.47B | 25% of non-embedding parameters | 3.660 | the 2B teacher, 2.550 | 1.11 |
| Heavy (`damaged-v2-0.8b`), hidden 1024, FFN 3584 | 0.75B, exactly stock 0.8B's shape | 60% | 7.055 | stock Qwen3.5-0.8B, 2.780 | 4.28 |

The light student is the main line. The heavy student starts so far away that a practical
budget only covers the coarse first half of healing.

**What every schedule trains:** a window of contiguous decoder layers `[a, b)`. Each step runs
the student's own frozen prefix up to layer `a`, trains the window to match the teacher's hidden
state at layer `b` (relative MSE on the kept coordinates), and adds full-vocabulary KL plus
0.1 × next-token CE when the window reaches the top. All parameters are trained (no LoRA), in
FP32 with BF16 autocast, with AdamW and per-layer warmup. Adam state persists across window
changes.

**Budget and data:**
- 6e15 accounted training FLOPs per arm. That is about 0.3M tokens for whole-network training
  and about 1M for block-wise, which is cheaper per token. A run takes about 2 h on a Radeon 8060S
  (Strix Halo iGPU).
- Batch size 1 × 2,048 tokens.
- A nine-domain chat/reasoning/code/wiki mix.
- Validation and test use two held-out domains (biomed, argument) never trained on.

**Metric:** training FLOPs (plus controller FLOPs for adaptive arms) until validation CE closes
25%, then 40%, of the gap to the 100% rung. A rung counts at the second of two consecutive
evaluations at or below it. Downstream scores use a fixed custom harness:
- ARC-Easy, ARC-Challenge and 285 MMLU questions, zero-shot by likelihood.
- 128 GSM8K problems, greedy, thinking off, 1,024 new tokens.

Every GPU job checks the untouched teacher against a CPU reference at start and end (see
[Infrastructure lessons](#infrastructure-lessons)).

## The schedules

| Schedule | What trains |
|---|---|
| joint | All 24 layers every step |
| joint-cosine | joint, with the learning rate cosine-decayed to 0 over the budget |
| layerwise | One layer at a time, bottom to top, equal FLOP shares |
| fixed | Blocks of 4 layers, bottom to top, equal FLOP shares |
| hierarchical | Stages of widths 1, 2, 4, 8, 16, 24, each swept bottom to top |
| overlap | hierarchical with half-width strides, so windows cross earlier boundaries |
| coarse_to_fine | hierarchical reversed: widths 24, 16, 8, 4, 2, 1 |
| adaptive_error | 4-layer blocks; a controller picks the block that adds the most hidden-state error |
| adaptive_plateau | 4-layer blocks in order; move on when a block stops improving |
| global_fixed | Whole network for the first seventh of the budget, then fixed |
| alternating | fixed, with the last 20% of each of 24 slices spent on the whole network |
| fixed, teacher inputs | fixed, but each block reads the teacher's hidden states instead of the student's (no prefix compute) |
| fixed, ramp inputs | fixed, moving from teacher inputs to the student's own over each block's share |
| random | A new random window (any width, any start) every step |
| random_dwell | A random window, kept until its own training loss gains 5%, stalls for 15 steps, or reaches 60 steps |

## Results: light student

Learning rate 3e-5 (chosen by a sweep, below).

| Schedule | FLOPs to 25% | FLOPs to 40% | Final val CE | Gap closed | Tokens | ARC-E | ARC-C | MMLU | GSM8K (finished) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B teacher | | | 2.550 | 100% | | 0.644 | 0.428 | 0.396 | 0.711 (114) |
| Stock Qwen3.5-0.8B, for reference | | | 2.780 | | | 0.602 | 0.381 | 0.379 | 0.430 (112) |
| Light student, unhealed | | | 3.660 | 0% | | 0.425 | 0.278 | 0.309 | 0.008 (0) |
| random | 0.50e15 | never | 3.293 | 33.1% | 0.58M | 0.482 | 0.321 | 0.274 | 0.055 (60) |
| random_dwell | 0.51e15 | never | 3.282 | 34.0% | 0.60M | 0.456 | 0.314 | 0.316 | 0.016 (0) |
| **joint-cosine** | 0.53e15 | **0.76e15** | **3.142** | **46.7%** | 0.32M | **0.539** | 0.314 | 0.312 | **0.250 (82)** |
| coarse_to_fine | 0.53e15 | 0.76e15 | 3.299 | 32.5% | 0.85M | 0.451 | 0.278 | 0.288 | 0.062 (84) |
| global_fixed | 1.50e15 | 5.25e15 | 3.225 | 39.1% | 0.89M | 0.458 | 0.278 | 0.326 | 0.117 (95) |
| joint | 1.53e15 | never | 3.310 | 31.5% | 0.32M | | | | |
| alternating | 1.53e15 | 3.03e15 | 3.245 | 37.4% | 0.79M | 0.486 | 0.311 | 0.288 | 0.023 (26) |
| fixed | 2.25e15 | 5.99e15 | 3.189 | 42.4% | 0.99M | 0.468 | 0.301 | 0.298 | 0.203 (105) |
| fixed, ramp inputs | 2.25e15 | 5.99e15 | 3.192 | 42.2% | 1.06M | 0.484 | 0.314 | 0.298 | 0.125 (39) |
| fixed, teacher inputs | 2.25e15 | never | 3.221 | 39.5% | 1.13M | 0.449 | 0.288 | 0.312 | 0.172 (78) |
| adaptive_plateau | 2.29e15 | never | 3.274 | 34.8% | 0.93M | | | | |
| hierarchical | 3.01e15 | never | 3.240 | 37.8% | 0.85M | | | | |
| overlap | 3.02e15 | never | 3.247 | 37.2% | 0.84M | 0.493 | 0.324 | 0.312 | 0.047 (88) |
| layerwise | 3.75e15 | never | 3.270 | 35.1% | 1.36M | | | | |
| adaptive_error | 3.78e15 | never | 3.296 | 32.7% | 0.83M | | | | |

How to read the table:
- **Resolution.** The first rows that start with whole-network training all cross 25% at their
  first or second evaluation. The last four schedules were evaluated every 0.25e15 FLOPs, the
  others every 0.75e15, so joint's 1.53e15 against joint-cosine's 0.53e15 is mostly grid. The 40%
  rung separates the arms better.
- **GSM8K "finished"** counts replies that ended before the 1,024-token cap.
- **Blank cells** were not evaluated.
- **Noise.** ARC-C (n=299), MMLU (n=285) and GSM8K (n=128) move a few points on noise alone.

## What we learned

### 1. Whole-network training first is what makes a schedule fast

Every schedule whose first window is the whole network reached 25% by about 0.5e15 FLOPs, and
the only two that reached 40% early (joint-cosine and coarse_to_fine, 0.76e15) are both
whole-network at that point. Block-wise schedules need 2.25e15 to 3.8e15 for 25%. Block-wise
training is cheaper per token (fixed sees three times the tokens in the same FLOPs), but each
token does less: a block can match the teacher's hidden state at its own output while the
model's actual output barely improves.

### 2. Constant learning rate: improve, stall, get worse. Decay fixes it.

Whole-network training at a constant learning rate has the same curve every time we ran it:

| Run | Validation CE across the budget |
|---|---|
| joint, lr 3e-5, batch 1 | 3.66 → **3.19** at 0.76e15 → 3.29–3.36 → 3.31 |
| joint, lr 1e-4, batch 4 | 3.66 → **3.18** at 1.53e15 → rising steadily → 3.71 |
| joint, lr 1e-4, batch 1 | 3.66 → **3.40** at 0.76e15 → 4.0–4.5 → 4.23 (worse than unhealed) |
| joint-cosine, lr 3e-5 → 0 | 3.66 → **3.19** at 0.76e15 → wobble to 3.27 → steady decline → **3.14** |
| Heavy student: joint, lr 1e-4 | 7.05 → 5.38 → 4.88 at 2e15 → flat (4.87–4.95) → 4.80 |

The best constant-rate point and the cosine run's early point are the same (3.19). Decay
doesn't make the early gain faster. It stops the drift that follows, and lets later training
add to it. Continuing constant-rate joint to 1M tokens (18.6e15 FLOPs) made it worse, not
better (3.40).

**Confound:** only joint got cosine decay. The block-wise schedules all ran at a constant rate,
so part of joint-cosine's lead over fixed and global_fixed may be the decay, not the schedule.
That is the next experiment.

### 3. The learning rate matters more than the schedule

On the light student the three-run sweep above was the biggest effect in the study:
- lr 1e-4 at batch 1 made the model worse than it started.
- Batch 4 at 1e-4 helped early and then also drifted.
- lr 3e-5 at batch 1 was the best of the three and was used for every light-student arm.

A lightly damaged model needs a much smaller learning rate than a heavily damaged one; the heavy
student was fine at 1e-4.

### 4. Narrowing after whole-network training hurts

coarse_to_fine trains the whole network for its first sixth and matched joint-cosine exactly to
0.76e15 (3.187). When it switched to 16-layer windows, validation CE jumped to 3.331, and it
stayed at 3.28–3.34 through 1-, 2- and 4-layer stages to the end.

Per-depth logs show why. Local windows push intermediate hidden states closer to the teacher's
while the output gets worse. Hidden-state matching and output quality are not the same
objective, and once the whole network has been tuned for the output, local matching pulls
against it. global_fixed (one-seventh whole network, then 4-layer blocks) shows the same effect
more mildly: 3.187 at 0.76e15, still whole-network, then 3.216 and 3.248 once the blocks start.

### 5. Random windows: fast start, early plateau

A new random window every step, or one kept until its own loss improves or stalls, reached 25%
as fast as anything: random windows include the whole network often enough. Both then sat at
3.25–3.32 and never reached 40%.

random_dwell's triggers can't work at batch 1: training loss moves 10–30% from one step to the
next, more than a window improves over its whole dwell, so flips are mostly noise. Its final
model also didn't finish one GSM8K reply in 1,024 tokens. Loss-triggered controllers need a
smoothed or held-out signal, not per-step training loss.

### 6. Teacher-fed blocks don't heal more per FLOP

Feeding each block the teacher's hidden states makes blocks independent and skips the student
prefix's compute, so each FLOP buys more tokens (1.13M against fixed's 0.99M). It reached 25%
at the same point as fixed, never reached 40%, and finished worse (3.221 against 3.189). The
assembled student then reads its own, still imperfect, hidden states, which the blocks never
trained on. Ramping from teacher to student inputs inside each block (BERT-of-Theseus style) was
indistinguishable from plain fixed.

### 7. Adaptive controllers didn't earn their overhead

Both adaptive schedules spent about 0.75e15 FLOPs (an eighth of the budget) on controller
forward passes. adaptive_error spent its first third on layers 0–3 (until its cap of twice an
equal share), then mostly on the top block. adaptive_plateau walked the blocks in fixed's order
and wrapped back to the bottom. Neither beat fixed.

### 8. The heavy student ranks differently

On the 60% cut (6 evaluations over 6e15 FLOPs, lr 1e-4, constant):

| Schedule | Final val CE | Gap closed (to stock 0.8B) |
|---|---:|---:|
| adaptive_plateau | 4.445 | 61.0% |
| fixed | 4.454 | 60.8% |
| hierarchical | 4.582 | 57.8% |
| overlap | 4.597 | 57.5% |
| joint | 4.803 | 52.6% |
| layerwise | 4.866 | 51.2% |
| adaptive_error | 5.176 | 43.9% |

Joint was again fastest at the first evaluation (5.38 against 5.5–6.2 for the others), then
plateaued at the constant learning rate while fixed-block schedules kept going. All arms
crossed 25% between 1.8e15 and 3.0e15, which a 1e15 evaluation grid can't separate. The
light-student lessons (start global, decay the rate) were not rerun on the heavy student.

### 9. Healed versus stock: the 0.8B is very good

The best healed model is worse than stock Qwen3.5-0.8B on every benchmark, at twice its size:

| Model | Params | Val CE | ARC-E | ARC-C | MMLU | GSM8K |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 1.88B | 2.550 | 0.644 | 0.428 | 0.396 | 0.711 |
| Stock Qwen3.5-0.8B | 0.75B | 2.780 | 0.602 | 0.381 | 0.379 | 0.430 |
| Light student healed (joint-cosine) | 1.47B | 3.142 | 0.539 | 0.314 | 0.312 | 0.250 |
| Light student unhealed | 1.47B | 3.660 | 0.425 | 0.278 | 0.309 | 0.008 |

The more striking number is the 0.8B's: with 40% of the 2B's parameters it keeps 93% of its
ARC-Easy, 89% of ARC-Challenge, 96% of MMLU and 60% of GSM8K, and sits 0.23 nats from it on
held-out text.

What healing recovered (share of the drop from the 2B to the pruned start):

| Measure | Recovered | From → to (2B) |
|---|---:|---|
| Validation CE | 47% | |
| ARC-Easy | 52% | 0.425 → 0.539 (0.644) |
| GSM8K | 34% | 0.008 → 0.250 (0.711); finished replies 0 → 82 |
| ARC-Challenge | 24% | |
| MMLU | 3% | |

MMLU hardly fell from pruning and hardly moved with healing. Whatever knowledge the cut removed,
0.3M tokens of distillation don't bring back.

The 0.8B had its full pretraining budget, tens of trillions of tokens. These runs used 0.3–1M
tokens to rank methods. Matching the stock small model by prune-and-heal would need a budget
orders of magnitude larger, and nothing here says whether it would get there.

## Infrastructure lessons

- **Shared iGPUs can compute wrong answers silently.** On a Strix Halo box (gfx1151, ROCm)
  shared by several jobs, some processes produced wrong values with no error. Every job here
  scores the untouched teacher on four fixed sequences at start and end, against a CPU FP32
  reference (0.05 nats) and the first good GPU result (0.01 nats), and is rerun on failure.
  Every canary in the reported runs passed.
- **Memory pressure can hang the GPU outright.** With about 120 of 123 GiB of RAM mappable by
  the GPU (GTT) and four GPU processes resident, the firmware scheduler stopped responding
  (`MES failed to respond to msg=REMOVE_QUEUE`). All compute hung until a GPU reset. Running
  jobs don't fail; they just stop, so a watchdog should check for kernel GPU errors and for no
  progress since launch, not only for exit codes. We lowered the GTT cap to 112 GiB.
- **Batch-1 training loss is too noisy to steer by.** Use held-out evaluations, or a heavily
  smoothed loss, for any adaptive decision.

## Caveats

- **One seed** (17) for every arm.
- **Small budgets.** 6e15 FLOPs per arm (0.3–1.4M tokens). Rankings in the late, fine-tuning
  end of healing may differ.
- **Learning-rate confound.** Only joint-cosine had decay; every other arm used a constant rate.
- **Coarse evaluation grid.** Nine arms were evaluated every 0.75e15 FLOPs, so the 25% rung is
  resolution-limited.
- **Partial FLOP accounting.** FLOPs are PyTorch-registered matmul/attention forward and
  backward; elementwise, normalization and optimizer work are excluded.
- **Different 100% rungs.** The light student's 100% rung is the teacher's own loss (no stock
  model exists at that size). The heavy student's is stock 0.8B's.
- **Home-domain GSM8K.** The training mix includes GSM8K and MATH train problems, which gives
  the healed students an edge on GSM8K.

## What's next

1. Cosine decay for fixed, global_fixed and alternating, plus a second joint-cosine seed, on
   the 0.25e15 evaluation grid. This settles whether joint wins on its schedule or only on the
   decay.
2. More seeds for the top arms.
3. The same comparison on an FFN-only prune (`artifacts/damaged-ffn-h2048-n1542086464`), where
   the residual stream is untouched.
4. Branching from shared checkpoints, to test whether the best schedule changes by phase. The
   code is written: see the README.

Code, protocol, drivers and reproduction steps are in [README.md](README.md) and
[protocol.json](protocol.json). Per-run curves, step logs and evaluations are in `runs/`
(weights excluded), and `reports/` has the rung tables.
