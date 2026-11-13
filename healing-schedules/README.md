# Controlled Qwen3.5 pruning and healing

This experiment compares healing schedules, not finished model products: Qwen3.5-2B is
pruned to about 0.8B, healed under seven schedules, and judged against stock Qwen3.5-0.8B
(the bar) and the 2B teacher (the ceiling) on quality, healing tokens and wall time.
**The primary question is how the seven schedules compare**: which work, which heal best at
equal budget, and which heal most efficiently. Reaching stock 0.8B quality, the damage
operator and any finished model serve that comparison. `report.py` scores each arm by the
fraction of the damaged-to-target validation gap it closes, and by tokens, FLOPs and wall time
to confirmed 25/50/75/100% rungs of that gap (100% is stock 0.8B), so arms stay comparable
even when none reaches the target.
Since protocol version 3 the primary student is exactly stock 0.8B's size, **752,393,024 unique
text-model parameters**. All seven arms must reload the same `pruning.json`-verified damaged
checkpoint.

Current status (2026-11-13): the light-student comparison is done (15 schedules, one seed); the
write-up with results and lessons for healing runs is **[findings.md](findings.md)**. In short:
whole-network training first, with a decaying learning rate (joint-cosine), healed fastest and
best; block-wise schedules took four to seven times the FLOPs to close a quarter of the gap.
The heavy (0.8B-size, v3) student's seven-arm pilot is in `reports/pilot-v2` and in findings.md.
The earlier v1-damage pilot (below) showed the data-free damage operator wiped the model. The
harness passes 33 tests.

## Verified models and architecture

Checkpoint configurations and immutable revisions were verified on November 6, 2026.
See [metadata/parameter-audit.json](metadata/parameter-audit.json); `accounting.py`
reproduces the unique-parameter counts using meta-device model instances.

| Model | Unique text parameters | Residual width | FFN width | Layers |
|---|---:|---:|---:|---:|
| [Qwen3.5-2B teacher](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) | 1,881,825,088 | 2,048 | 6,144 | 24 |
| Primary damaged student (v3, `damaged-v2-0.8b`) | 752,393,024 | 1,024 | 3,584 | 24 |
| v1/v2 damaged student (`damaged-0.8b`, superseded) | 799,578,944 | 1,024 | 4,224 | 24 |
| [Qwen3.5-0.8B native reference](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) | 752,393,024 | 1,024 | 3,584 | 24 |
| Second, milder damage | 999,361,600 | 1,280 | 4,224 | 24 |

These are text-only models: vision and multi-token-prediction branches are excluded
from both teacher and reference. Shared embedding/output weights count once.
Hub stored-tensor totals are not the same quantity. The v1 student was 6.27% larger than
the native reference (all in the FFN); v3 matches it exactly. Its larger pretrained
initialization remains a confound.

FFN-only pruning cannot reach 0.8B for Qwen3.5-2B: even zero-width FFNs would leave
**975,855,424** parameters. Residual-width pruning is therefore an explicit part of
the fixed damage transformation, as selected by the user. Attention head counts,
head dimensions, recurrent-state dimensions, vocabulary, and depth are preserved.

`pruning.py` slices embeddings, both residual norms per layer, final norm, attention input
columns/output rows, and all three FFN matrices. Ties prefer lower original indices;
retained coordinates keep their original order. Two selection rules:

- **v1 `weights` (data-free):** residual coordinates by embedding-column L2 energy, FFN
  channels by the product of gate/up/down L2 norms. Result: validation CE 8.94, chance on
  ARC/MMLU, 0% GSM8K, i.e. close to a random model.
- **v3 `activation` (default, Minitron-style):** a teacher FP32 pass over the calibration split
  plus 24 seeded train rows (65,536 tokens). A residual coordinate scores by its mean squared
  output summed over all 49 hidden-size RMSNorms; an FFN neuron by its mean squared activation
  `silu(gate)*up` times the squared norm of its down-projection column on the kept coordinates.
  Each hidden-size RMSNorm gain is optionally multiplied by
  `sqrt(mean kept-coordinate input energy / mean all-coordinate input energy)`, so the pruned
  norm divides by about the teacher's RMS. The build probes start CE (student only, CPU FP32,
  16 controller and 16 validation rows) for `weights`, `activation` and
  `activation+norm_scale`, and keeps the rescale only if it lowers controller CE (validation is
  reported, not used). Statistics are saved in `activation-stats.safetensors`.

All selected indices, norm scales, probe results, source hashes, target configuration, and
artifact hashes are saved in `pruning.json`.

## Comparison protocol

`protocol.json` records the starting settings. The v3 pilot budget is 6e15
accounted training FLOPs per arm; the full-study budget is deliberately unset until
pilot feasibility is known. Use seeds 17, 29, and 43, paired across arms. Training
order is the same for a given seed; equal compute can expose different token prefixes.
Masks and initialization do not change between seeds or strategies.

Every arm updates the same eligible parameter set in FP32 with AdamW and BF16 autocast:
all decoder layers, and since v3 the tied embedding/output head and the final norm (v1/v2
froze those at their sliced values). Layers `[a, b)` train in a window; the tied
embedding/head trains when the window touches either end of the stack (`a == 0` reads the
embedding, `b == 24` uses the head) and the final norm when `b == 24`. "Joint" trains all of
it throughout. Optimizer state persists across window changes; inactive parameters have no
gradient and receive no optimizer update. No LoRA capacity difference is introduced.
Learning rate 1e-4 warms up linearly over each parameter group's first 32 updates (groups:
embedding, final norm, each layer), so a layer that first trains late in a schedule is
warmed up too. v1/v2 had no warmup; joint's training loss jumped to 18 in its first steps.

| Arm | Window allocation |
|---|---|
| Layerwise | One layer at a time, front to back, equal compute shares |
| Fixed | Contiguous groups of four, front to back, equal compute shares |
| Hierarchical | Widths 1, 2, 4, 8, 16, 24; equal compute per stage and per window within each stage |
| Overlap | Same widths/stage shares; half-width stride crosses prior boundaries |
| Joint | All 24 decoder layers throughout |
| Adaptive error | Fixed's width-4 blocks; at each decision, the block with the most discrepancy *added* inside it |
| Adaptive plateau | Fixed's width-4 blocks in order, wrapping; move on when the block's endpoint discrepancy improves <1% between decisions, or after holding twice its equal budget share |

The adaptive arms decide 24 times per budget from the per-depth discrepancy of the student's
own stream on 3 evenly spaced controller rows; those forward passes count against the arm's
training budget (`decision_flops`). v1/v2 controllers used one absolute discrepancy threshold
(0.15) and decided only at the 5 evaluation points; on a badly damaged student every layer
exceeded it and both stayed in layers 0-3 for the whole pilot.

Each step uses the **current student's prefix**, detached at the window boundary.
Teacher targets come from an independent full teacher forward pass, projected onto
the retained residual coordinates. The loss is relative endpoint MSE; windows ending
at the final layer additionally use full-vocabulary teacher||student KL plus 0.1 CE.
This is a comparison of reconstruction schedules with a conventional final-endpoint
language loss. It does **not** isolate window size under an identical scalar objective:
the endpoint and frequency of final-logit supervision change with schedule. If this
factor matters after the pilot, add an all-windows-final-logit objective control before
attributing improvements solely to window geometry.

### Data

`prepare_data.py` builds 2,048-token sequences that look like what a Qwen chat model is
for, with nine in-mix domains in equal shares (about 12M training tokens):

| Domain | Source | Format |
|---|---|---|
| chat | UltraChat (local parquet) | chat template |
| instruction | K2-Horizon SFT-Reasoning sample (local; only items with unambiguous Instruction/Reasoning/Response sections) | chat template, thinking |
| math | MATH train | chat template, thinking, boxed answer |
| math_word | GSM8K train | chat template, thinking |
| reasoning | OpenThoughts-114k | chat template, thinking |
| wiki | Wikipedia 20231101.en | plain |
| science | arXiv abstracts | plain |
| code | local Python standard library | plain |
| code_cpp | mlir-aie C++ (local) | plain |

Thinking turns carry non-empty `<think>` blocks that close into the answer and `<|im_end|>`.
On the Qwen3.5-9B MoE carve, a healing stream with only empty think blocks taught the
student to loop instead of stopping; templated thinking turns fixed it.

Two domains are never trained on: **biomed** (TREC-COVID corpus) and **argument**
(Touché-2020 corpus). Their documents form the `validation` split (stopping rule) and the
`test` split (read only after training). Items are split by a salted hash of their
identity: in-mix items go 3% to `controller` (what the adaptive controllers see), 3% to
`test_in_domain` (report only), 1% to `calibration` (FLOP profiling and the canary), and the
rest to `train`. An item never straddles splits. Items sharing a 13-word n-gram with a
GSM8K test, ARC validation/test or MMLU test question (or containing a whole 8-12-word
question) are dropped. Hub sources are pinned by revision; local sources are hashed per
file in the manifest. Known gaps: the argument documents come from one contiguous window
of the corpus, so they cover few debate topics; the GSM8K and MATH training problems give
the healed student a home-domain edge over stock 0.8B on GSM8K, which reports must note.

Downstream scoring uses a fixed custom harness (do not compare with lm-eval tables):
ARC-Easy and ARC-Challenge validation and a frozen 285-question MMLU subset by zero-shot
answer-token-normalized likelihood, and the first 128 GSM8K test questions (fixed hashed
order) by greedy generation with thinking off.

### Target

Before healing, freeze the target at the **native Qwen3.5-0.8B validation CE** on the
held-out domains (`baselines/mix-2048/target.json`). Require two consecutive validation
evaluations at or below it; report the first confirmed tokens, compute and wall time,
and the preceding evaluation interval. `report.py` also derives the native + 0.10 nats
threshold from the same curves.
The same capped trajectory supplies both quality-versus-compute and cost-to-quality
results. A strategy that does not reach the threshold is censored, never assigned an
invented cost.

### GPU canary

Since 2026-11-05 some processes on the shared local GPU have computed wrong values
without any error. Every GPU evaluation and healing run checks the untouched teacher's
per-sequence CE on four calibration sequences, at start and end, against a CPU FP32
reference (0.05 nats) and the first passing GPU result (0.01 nats). A failure exits with
code 3; the driver reruns the job in a fresh process, up to three attempts. Baseline test results must not alter this rule or controller settings.

## Version-2 pilot results (2026-11-07, v1 damage operator)

Baselines (`baselines/mix-2048/`; CE on held-out biomed+argument validation; GSM8K capped
at 384 new tokens, so 74 of 128 teacher and stock-0.8B replies were cut off):

| Model | Validation CE | ARC-Easy | ARC-Challenge | MMLU (285) | GSM8K (128) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B teacher | 2.550 | 0.644 | 0.428 | 0.396 | 0.406 |
| Stock Qwen3.5-0.8B (target) | 2.780 | 0.602 | 0.381 | 0.379 | 0.289 |
| v1 damaged student | 8.936 | 0.277 | 0.261 | 0.249 | 0.000 |

Seven arms, seed 17, 2e15 accounted training FLOPs each; none reached the target:

| Arm | Tokens | Final test CE | Validation gap closed |
|---|---:|---:|---:|
| overlap | 335,872 | 6.33 | 44.7% |
| hierarchical | 337,920 | 6.59 | 39.5% |
| fixed | 389,120 | 6.91 | 35.0% |
| layerwise | 514,048 | 7.51 | 25.3% |
| joint | 141,312 | 7.85 | 20.2% |
| adaptive_plateau | 587,776 | 8.12 | 16.0% |
| adaptive_error | 552,960 | 8.16 | 15.1% |

Joint full-depth training ran at about 9 s per 2048-token step (about 220 tokens/s, 0.8M
tokens/hour) on the local Radeon 8060S. Every canary passed (max CPU difference 0.0009 nats,
GPU repeat exact). One seed and a tiny budget: the ranking is not evidence. The finding is
that the v1 damage left too little to heal, which led to version 3.

## Follow-up experiment: does the best schedule depend on the phase?

Planned, and run only after the main seven-arm comparison. The question is whether one schedule
heals best early, another mid-way and another late. Comparing the independent arms phase by phase
cannot answer it cleanly. Each schedule spreads its plan over its whole budget, so what it does in
a phase depends on the budget. Its speed in a phase also mixes the schedule's quality with the
state it left the model in.

The design branches from shared checkpoints instead:

1. One trunk run saves a checkpoint the first time validation closes 25%, 50% and 75% of the gap.
2. From each checkpoint, all seven schedules continue with equal budgets and fresh schedules.
   Each branch inherits the trunk's weights, Adam state, per-group update counts and data
   position.
3. Branches are scored on the same ladder as the main arms, measured from the trunk's damaged
   start.
4. A composite (best early, then best mid, then best late) then races the best single schedule
   end to end, over several seeds.

The code is written and tested but not yet merged, because merging would change the source hashes
partway through the running pilot. It is in `followups/branching.patch` (run.py
`--checkpoint-gaps` and `--init-from`, report.py branch scoring, and a test showing that a trunk
checkpoint plus a branch reproduces an uninterrupted run bit for bit). Apply it with
`patch -p1 < followups/branching.patch` once the pilot's training arms are done. Still to choose:
the trunk schedule, the branch budget per phase, and the number of seeds.

## Follow-up experiment: a less damaged student

Planned, after the main comparison. The heavily damaged 0.8B student spends a practical budget in
the early, coarse part of the gap. A lighter prune (at most 25% of non-embedding parameters; candidates
hidden 1792 / FFN 4800, hidden 1792 / FFN 5376, hidden 1920 / FFN 5248) starts close enough to reach the fine-tuning end, so it tests whether the schedules rank the
same there. No stock model exists at those sizes, so the 100% rung is the 2B teacher's validation
loss (2.55). The cut will be chosen from CPU start-loss probes. This follow-up complements the
branching follow-up: branching studies the late phase of the same heavily damaged model, while a
light cut studies small, fresh pruning damage.

## Follow-up experiment: local/global combinations and teacher-fed windows

On the light student, joint closed 43% of the gap at an eighth of its budget and then drifted
back; fixed was slow but never drifted. Four arms at the light pilot's settings
(`drivers/followup-local-global.sh`, `protocol.json` `followup_local_global`):

- `global_fixed`: the whole network for the first of seven equal FLOP shares, then fixed blocks.
- `alternating`: the fixed sweep with the last 20% of each of 24 FLOP slices spent on the whole network.
- `fixed` with `--window-inputs teacher`: each block reads the teacher's stream instead of the
  student's own prefix, so blocks are independent and the prefix forward is not paid.
- `fixed` with `--window-inputs ramp`: teacher-fed early in each block's share, own prefix by its
  end (progressive replacement applied to the window's input).

Evaluation always runs the student's own full stream, so teacher-fed training is judged on the
assembled model.

## Follow-up experiment: random windows, cosine joint, coarse to fine

Four more arms on the light student (`drivers/followup-random.sh`, `protocol.json`
`followup_random`), at the same settings but with 24 evaluation points (every 0.25e15):

- `joint-cosine`: joint with the learning rate cosine-decayed to zero over the budget (`--lr-decay cosine`).
- `random`: a new window every step; width uniform over 1..24, start uniform.
- `random_dwell`: a random window kept until the decayed average of its own training loss is 5%
  under its first five steps (gain), stops improving for 15 steps (stall), or reaches 60 steps.
- `coarse_to_fine`: hierarchical reversed, six equal stages of widths 24, 16, 8, 4, 2, 1.

## Light-student results (2026-11-12)

Student `damaged-light-h1792-n1473244736` (25.1% non-embedding cut, validation CE 3.660, teacher
2.550, gap 1.11). Seed 17, 6e15 training FLOPs, lr 3e-5, batch 1. The headline is training FLOPs
(plus decision FLOPs for the adaptive arms) to the 25% rung (validation CE 3.382, second of two
consecutive evaluations at or below). The 40% rung (3.216) is shown because 25% is resolution-limited:
the first nine arms were evaluated every 0.75e15, the last four every 0.25e15, and every arm that
starts with whole-network training crosses 25% at the first or second evaluation.

| Arm | FLOPs to 25% | FLOPs to 40% | Final val CE | Gap closed | Tokens | ARC-E | ARC-C | MMLU | GSM8K (finished) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B teacher | | | 2.550 | | | 0.644 | 0.428 | 0.396 | 0.711 (114) |
| Damaged light student | | | 3.660 | 0% | | 0.425 | 0.278 | 0.309 | 0.008 (0) |
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

GSM8K is 128 problems, greedy, 1024 new tokens; "finished" counts replies that ended before the
cap. Blank downstream cells were not evaluated. Every canary passed. One seed, so differences of a
few points on ARC-C, MMLU and GSM8K are within noise (ARC-C n=299, MMLU n=285).

Findings:

- **joint-cosine is the best arm on every axis that discriminates**: first to 40% (with
  coarse_to_fine, which is joint for its first sixth), lowest final loss, best ARC-E and GSM8K.
  Plain joint's problem was its constant learning rate: it crossed 43% early and drifted back.
- **Whole-network training first is what makes an arm fast.** Every arm that reaches 25% by about
  0.5e15 starts global; every block-local arm needs 2.25e15 or more.
- **Narrowing to local windows after global training hurts.** coarse_to_fine matched joint-cosine
  to 0.76e15 (3.187), jumped to 3.331 when it switched to width-16 windows, and never recovered.
  Per-depth logs show local blocks reduce intermediate-layer mismatch while raising output loss.
- **Random windows are fast to 25% but plateau** (3.25 to 3.32) and never reach 40%. random_dwell
  dwelt 10 to 20 steps per window; its triggers ride batch-1 loss noise (10 to 30% per step).
  Its model never finished a GSM8K reply within 1024 tokens.
- Downstream scores track final validation loss only loosely: the fixed family is second on loss
  and GSM8K, overlap is second on ARC-E.

Open: a second seed; denser early evaluation (or the 40% rung) to separate the global-first arms;
cosine decay for the other arms; the FFN-only student (`artifacts/damaged-ffn-h2048-n1542086464`).

## Compute accounting and resource limits

`engine.py` profiles teacher forward, student prefix/window forward, window backward,
rotary context, and final vocabulary head costs. Every actual training step is counted
again and must exactly match the component prediction. A step that would exceed the
budget is not launched; report the remaining sub-step budget. Tiny budgets can skip
windows and are only engineering smoke tests. Inspect `steps.jsonl` for coverage.

The current ledger counts PyTorch's registered dense matmul, attention, and convolution
operations. **It is an accounted-FLOP estimate, not exact total training arithmetic**:
normalization, nonlinearities, optimizer arithmetic, and indexing are omitted. This
limitation matters for the recurrent/linear-attention implementation. Keep the eager
attention and unfused recurrence implementation fixed; the per-step assertion detects
cost drift. Profiling, controller, validation, and final-test FLOPs are reported separately
and in an all-in accounted total. Plot both training and all-in compute before making an
efficiency claim. Report tokens, steps, elapsed time, peak host RSS and device memory.

Full decoder AdamW needs roughly **13.0 GB decimal just for teacher, student,
gradients, and optimizer state**, before activations, vocabulary losses, allocator
overhead, or other users' memory. A 24 GB device is a sensible pilot target to verify,
not a measured guarantee. Shared GPU availability must be checked before each run.
No paid resources are provisioned by any script. Paid work requires a separate bounded
resource/rate/duration/budget approval under the user's AGENTS.md instructions.

Use Dibs for shared-resource work. Ordinary training logs are observational wall times;
use `dibs run gpu --measure ...` for isolated timing comparisons and retain both CLEAN
checks. A NOISY result cannot support a throughput ranking.

## Reproduction

From the repository root, the existing `target/manifold-rocm/env/bin/python` environment
has PyTorch 2.14.1+ROCm and Transformers 5.14.1. Use that executable below as `python`.
The scripts accept local models only and never launch remote compute.

```bash
dibs status
dibs run cpu --label "Qwen parameter audit" --eta 2m -- python healing-schedules/accounting.py
dibs run cpu --label "Qwen harness tests" --eta 3m -- python -m pytest healing-schedules/test_experiment.py -q

dibs run cpu --label "Freeze damaged Qwen" --eta 5m -- python healing-schedules/pruning.py \
  --source models/manifold-audit/Qwen3.5-2B --output healing-schedules/artifacts/damaged-0.8b

dibs run network --label "Native Qwen reference" --eta 10m -- python healing-schedules/fetch_native.py \
  --output healing-schedules/artifacts/native-0.8b

dibs run network --label "Healing corpus sources" --eta 20m -- python healing-schedules/prepare_data.py \
  --tokenizer models/manifold-audit/Qwen3.5-2B --output healing-schedules/data/mix-2048 \
  --raw healing-schedules/data/raw-mix-v1 --fetch-only
dibs run cpu --label "Healing corpus" --eta 10m -- python healing-schedules/prepare_data.py \
  --tokenizer models/manifold-audit/Qwen3.5-2B --output healing-schedules/data/mix-2048 \
  --raw healing-schedules/data/raw-mix-v1
```

The frozen corpus used here was fetched into `data/raw-mix-v1` (its `fetch.json` records
every revision and file hash) and built into `data/mix-2048`.

[drivers/pilot-mix-2048.sh](drivers/pilot-mix-2048.sh) ran the v2 baselines and pilot, each GPU
job through `dibs run gpu` with canary retries, skipping steps already done:

1. `canary.py` writes the CPU FP32 canary reference (`baselines/mix-2048/canary-cpu.json`).
2. `evaluate.py` scores the teacher, stock 0.8B and damaged student on all four corpus splits
   (per domain) and the downstream tasks (`baselines/mix-2048/{teacher,native,damaged}.json`).
3. The target is frozen from stock 0.8B validation CE (`baselines/mix-2048/target.json`).
4. The seven pilot arms run with seed 17 at 2e15 accounted training FLOPs each (about
   115 joint steps, roughly 0.24M tokens), 5 evaluation points of 16 sequences per split.

```bash
setsid nohup healing-schedules/drivers/pilot-mix-2048.sh > healing-schedules/logs/pilot-mix-2048.log 2>&1 < /dev/null & disown
```

[drivers/pilot-v2.sh](drivers/pilot-v2.sh) runs version 3, reusing the canary references and
frozen target: it builds `artifacts/damaged-v2-0.8b` on the CPU (activation selection and
start-quality probe), runs the seven arms with seed 17 at 6e15 accounted training FLOPs
(6 evaluation points of 16 sequences) into `runs/pilot-v2`, then scores the v3 student,
teacher and stock 0.8B with GSM8K at 1024 new tokens
(`baselines/mix-2048/{damaged-v2,teacher-gsm1024,native-gsm1024}.json`).

```bash
setsid nohup healing-schedules/drivers/pilot-v2.sh > healing-schedules/logs/pilot-v2.log 2>&1 < /dev/null & disown
```

A single arm by hand:

```bash
dibs run gpu --label "Qwen healing pilot" --eta 1h -- python healing-schedules/run.py \
  --teacher models/manifold-audit/Qwen3.5-2B \
  --damaged healing-schedules/artifacts/damaged-0.8b --data healing-schedules/data/mix-2048 \
  --output healing-schedules/runs/pilot-mix-2048/layerwise-seed17 --strategy layerwise --seed 17 \
  --budget-flops 2000000000000000 --target-ce TARGET_CE --eval-points 5 --eval-sequences 16 \
  --canary healing-schedules/baselines/mix-2048/canary-cpu.json \
  --canary-gpu healing-schedules/baselines/mix-2048/canary-gpu.json
```

The pilot budget is an engineering check: every arm is expected to stop short of the target.
Set the full-study budget from the pilot's measured seconds per step.

Repeat the same command with each of the seven strategy names and unique output paths.
Before a seven-arm suite, reserve disk with `dibs disk claim 25 --eta 1d --path healing-schedules/runs`
(increase the claim for multiple seeds), then release it when finished. Saved final
weights can exceed 10 GB in aggregate. Do not run the suite concurrently on this host.
`--max-seconds` defaults to 3600; hitting that or a step limit is recorded as censoring,
not an equal-budget result. Run final `evaluate.py` on each saved `final/` checkpoint
for downstream scores. Keep all data, code and initialization hashes in the report.
Only after the one-seed pilot works should the other seeds and second severity run.
`report.py --runs healing-schedules/runs --output healing-schedules/report-pilot`
writes a per-run CSV and seed summary, rejecting mixtures of different code, damage,
data, budgets, or target thresholds. It does not invent results for missing arms.

## Related work and reporting

[BRECQ](https://arxiv.org/abs/2102.05426) reconstructs quantized networks block by
block and analyzes cross-layer dependency. [CBQ](https://arxiv.org/abs/2312.07950)
addresses cross-block dependencies and accumulated quantization error in LLMs.
[AE-Qdrop](https://doi.org/10.3390/electronics13030644) describes blockwise reconstruction
followed by global fine-tuning for CNN quantization; the source supports that two-stage
description, not an established 1→2→4→8 overlapping LLM schedule. Novelty of the exact
schedule comparison remains unestablished; a broader literature review is still needed.

Additional relevant primary sources are [Sheared LLaMA](https://arxiv.org/abs/2310.06694),
which combines target-shape structured pruning with continued training, and the
[Minitron study](https://arxiv.org/abs/2407.14679) and
[practical follow-up](https://arxiv.org/abs/2408.11796), which compare pruning axes and
distillation recovery. They directly motivate the larger-initialization caveat.
[SliceGPT](https://arxiv.org/abs/2401.15024) reduces residual dimensions with dense
row/column deletion; this experiment's coordinate-magnitude selection does not
implement SliceGPT's computational-invariance construction.
[TesseraQ](https://arxiv.org/abs/2410.19103) uses progressive adaptive rounding within
block reconstruction; “progressive” there concerns rounding variables, which should
not be conflated with progressively enlarged optimization windows here.

Report CE/perplexity per domain, final-logit KL, ARC/MMLU/GSM8K accuracy, and discrepancy versus depth
against both training and all-in compute. Compare paired seeds, show every seed, and
report mean/spread rather than significance claims from three runs. Include failures
and regressions. Distinguish empirical findings from the six proposed hypotheses.
The native-size comparison is secondary; the controlled seven-arm schedule comparison
is the primary contribution. Transfer to dense-to-MoE remains a later experiment.
