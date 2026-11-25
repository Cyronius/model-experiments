# adaptive-loop: findings

See PLAN.md for the idea and setup.

## Circuit task, grid 1 (2026-11-18/19)

Random Boolean circuits of 48 nodes, trained on depth 1–12 (curriculum from depth 2), tested in distribution and on depth 13–24 with twice the step budget (40). 8k steps at batch 256, bf16, about 30–90 min per run on the local GPU. Accuracy is over gate outputs. "Readout" reads each token's answer at its own halting step (first step with halting probability > 0.5). "Freeze" stops updating a token's state once it halts, which is what saves compute. Training ran without freezing.

| model | params | acc | acc, freeze | mean steps | corr(steps, level) | depth 13–24 acc | depth 13–24, no halting (all 40 steps) | depth 13–24, freeze |
|---|---|---|---|---|---|---|---|---|
| dense (20 layers, early exit) | 2.66M | 99.96% | 92.8% | 7.9 | 0.74 | 97.5% | 92.4% | — |
| loop (one block) | 0.16M | 99.98% | 98.3% | 3.2 | 0.85 | 99.6% | 90.5% | ~89% |
| oploop K=8 | 0.62M | 99.98% | 99.7% | 3.7 | 0.71 | 99.7% | 99.6% | ~98% |

Steps per level (loop, in distribution): 1.0, 2.0, 2.9, 3.7, 4.5, 5.1, 5.7, 6.2, 6.6, 7.1, 7.5, 7.6 for levels 1–12. Fewer steps than levels at depth, probably because many gates are settled early (an AND with a 0 input, for example).

Results:

1. **The task is too easy to separate the models on accuracy.** A single looped block of 0.16M parameters solves it, adapts its depth to each gate's level, and extrapolates to deeper circuits. Adaptive depth works with or without an operation library.
2. **The op library is more stable.** With freezing, it loses 0.3 points, where loop loses 1.7 and dense 7. Run past its halting point to all 40 steps on deeper circuits, it keeps 99.6%, where loop drops to 90.5%. Its state seems to settle instead of drifting.
3. **Routing varies by token but its role is unclear.** All 8 experts are used (usage 10–14% each). The entropy of the choice given the step is 1.76 of a maximum 2.08, and knowing the gate type lowers it only to 1.68. So choices vary between tokens, but not much by gate type. Whether the choice matters needs an ablation that shuffles choices at evaluation time; grid 2 adds it.
4. **Freezing at inference needs freezing in training.** Without it, tokens learn to read neighbours' states that keep changing after those neighbours halt.

Next (grid 2): train with freezing on half the batches, the routing ablations (shuffle the choices across tokens within a step; random choices), and a narrow width (d=48), where one block might run out of capacity and a library could help. After that, a harder circuit variant with multi-bit words and mixed operations (ADD, MUX, shift, compare), then OCR cleanup.

## Circuit task, grid 2 (2026-11-19)

Same task. All runs train with halted tokens frozen on half the batches (`--freeze-frac 0.5`). New evaluation: **shuffle** permutes each step's expert choices across tokens (same mix of experts per step, no longer tied to the token); **random** picks experts uniformly.

| model | params | acc | acc, freeze | mean steps | corr(steps, level) | shuffle | random | depth 13–24 | 13–24, all 40 steps |
|---|---|---|---|---|---|---|---|---|---|
| loop, d=128 | 0.16M | 99.96% | 99.88% | 3.8 | 0.74 | — | — | 99.3% | 68.2% |
| oploop K=8, d=128 | 0.62M | 99.96% | 99.90% | 3.6 | 0.71 | 85.7% | 89.1% | 99.6% | 99.4% |
| loop, d=48 | 0.03M | 92.6% | 92.3% | 7.5 | 0.68 | — | — | 84.8% | 50.6% |
| oploop K=8, d=48 | 0.10M | 98.3% | 98.0% | 14.0 | 0.23 | 71.4% | 72.2% | 94.3% | 94.2% |

Results:

1. **Training with freezing closes the freeze gap.** Both d=128 models now lose under 0.1 points when halted tokens freeze.
2. **Routing matters, and matters more for harder tokens.** Shuffling choices drops the d=128 library from 99.96% to 85.7%. By level: 97% at level 1, falling to 67% at level 12. The deeper a gate, the more its result depends on getting its own sequence of operations.
3. **At d=48 the library beats the single block** (98.3% vs 92.6%; 94.3% vs 84.8% on deeper circuits). But it also has 3× the parameters and takes almost twice the steps (14 vs 7.5), and its steps track level less (corr 0.23: 11.8 steps at level 1, 16.8 at level 12). Grid 3 adds controls for both: a single block with the same parameter count, and one with a lower step cost.
4. **The single block breaks when run past its halting point; the library does not.** On deeper circuits with all 40 steps: loop 68% (d=128) and 51% (d=48), library 99.4% and 94.2%. This held in both grids.

## Circuit task, grid 3: controls for the d=48 result (2026-11-19)

| model (d=48, all trained with freezing) | params | MLP compute per step | acc | mean steps | corr(steps, level) | depth 13–24 | 13–24, all 40 steps |
|---|---|---|---|---|---|---|---|
| loop, hidden 96 | 0.03M | 1× | 92.6% | 7.5 | 0.68 | 84.8% | 50.6% |
| loop, hidden 96, step cost 0.003 | 0.03M | 1× | 93.3% | 16.6 | 0.33 | 83.0% | 81.2% |
| oploop K=8, hidden 96 | 0.10M | 1× | 98.3% | 14.0 | 0.23 | 94.3% | 94.2% |
| loop, hidden 768 (same params as oploop) | 0.09M | 8× | **99.7%** | 6.6 | 0.73 | **97.7%** | 97.2% |

Results:

1. **Extra steps alone do not help the single block.** With a lower step cost it takes 16.6 steps and gains only 0.7 points.
2. **A single block with the same parameter count beats the library.** It is more accurate (99.7% vs 98.3%; 97.7% vs 94.3% deeper), uses fewer steps (6.6 vs 14), and its steps track level better. It is also stable when run past halting, so the stability seen earlier is probably about capacity, not about routing. Its cost: 8× the MLP compute per step, about 3.8× the library's total MLP compute after counting steps.
3. **Both are adaptive looped models; the difference is the operation choice.** The library gives more capacity per unit of compute than a block of the same compute, and less than a block of the same size, while using about 1/4 of that block's total MLP compute. Routing is used and matters (grid 2's shuffle test). What this task can't show is a gain from choosing operations beyond capacity, probably because every gate needs the same operation. The fair comparison is accuracy against total compute (steps × active compute per step), not against parameter count.

Verdict on circuits: adaptive depth works; the op library is a capacity/compute trade, not yet a win from choosing strategies. The circuit task probably doesn't need distinct operations: every gate needs the same thing (fetch two inputs, apply a small truth table). A task where tokens need genuinely different operations is the next test: either the mixed-word circuit variant (ADD, MUX, shift, compare on 4-bit words) or OCR cleanup.

## Pedal task, grid 1 and diagnosis (2026-11-19/20)

Big Muff-family circuits (1–4 stages), node voltages at 16 phases of the last input period. 12k steps at batch 128 on 400k circuits. "w50": fraction of nodes with RMSE under 50 mV.

| model | params | w50 | output AC error (median) | mean steps | shuffle w50 | 5–6 stages w50 |
|---|---|---|---|---|---|---|
| dense (24 layers) | 3.19M | 47.4% | 53% | 24 | — | 0.1% |
| oploop K=8 | 0.62M | 45.2% | 60% | 3.1 | 8.2% | 33.0% |
| loop | 0.16M | 43.0% | 69% | 3.7 | — | 31.5% |
| loop, hidden 2048 | 0.62M | 42.1% | 72% | 4.0 | — | 29.5% |

Baselines on the test set: per-role mean waveform 17.6%; an oracle that knows each node's exact DC level and predicts a flat line 47.0%. So every model is at about the "right DC level, no signal" level. Collector nodes are 1% w50 (RMSE 1.2 V); passive nodes and the source are fine.

Diagnosis, on a loop memorising 1,024 training circuits for 6k steps with no step cost (all 24 steps):

| change | w50 | RMSE | output AC error |
|---|---|---|---|
| as in the grid (Huber 0.05) | 48.4% | 0.23 V | 70% |
| near-MSE loss (Huber 1.0) | 13.8% | 0.13 V | 21% |
| fp32 output head | 48.7% | 0.24 V | 87% |
| + Fourier features of component values | 47.9% | 0.26 V | 91% |
| + extra MSE term | 14.9% | 0.12 V | 18% |

Results:

1. **The models are all equal because none of them learns the task.** Capacity (8× MLP width, or 24 distinct layers) does not help, and a loop cannot fit even 1,024 circuits. The inputs carry every parameter the solver uses, so this is not a data bug.
2. **The loss sets the trade-off, not the result.** Huber 0.05 gets DC levels right and the AC wrong; MSE gets waveform shape closer (output error 18–21%) but almost no node within 50 mV. Neither fp32 output nor Fourier inputs changes this.
3. **Routing is used:** shuffling the op library's choices drops it from 45% to 8%. Dense breaks completely on longer circuits (0.1%); the looped models keep about 30%.

Verdict so far: as posed (full transient waveform of high-gain, clipping stages, 50 mV precision), the pedal task is too hard for these models at this budget, so it says nothing yet about the architecture. The bias point alone (the DC operating point, a nonlinear fixed point the loop should iterate toward) is the obvious smaller step.

## OCR task, grid 1 (2026-11-20)

wikitext-2 windows with OCR-style damage (about 3% of characters; heavy set 15%, evaluated with 1.5× the step budget). 12k steps at batch 128, trained with freezing on half the batches. CER = character error rate of the cleaned text (input: 3.22% normal, 15.21% heavy). "Damaged acc" is the fraction of damaged characters whose output slots are exactly right.

| model | params | CER | damaged acc | shuffle damaged acc | steps, clean / damaged chars | heavy CER | heavy damaged acc |
|---|---|---|---|---|---|---|---|
| dense (16 layers) | 2.16M | 1.06% | 72.9% | — | 16 / 16 | 23.8% | 41.2% |
| oploop K=8 | 0.65M | 1.08% | 72.4% | 53.3% | 16 / 16 (never halted) | 5.66% | 69.5% |
| loop, hidden 2048 | 0.65M | 1.18% | 68.3% | — | 4.5 / 7.0 | 5.95% | 66.5% |
| loop | 0.19M | 1.31% | 64.1% | — | 2.6 / 4.2 | 6.51% | 62.4% |

Clean characters are copied correctly by all models (99.8%).

Results:

1. **Adaptive depth works on OCR.** The loops spend about 1.6× more steps on damaged characters than on clean ones, without being told which are damaged.
2. **The op library never learned to halt.** Its expected steps stayed at exactly 16 (the cap) through training, after the step cost came in, so its halting head had saturated at "never halt" and got no gradient. The loops recovered (16 → 2.6 steps); the library did not. Its accuracy is therefore at full compute.
3. **Even so, it beats the same-size block on less compute.** Against the loop with hidden 2048 (same parameters, 8× MLP compute per step, about 37 step-units vs the library's 16): CER 1.08% vs 1.18%, damaged acc 72.4% vs 68.3%, heavy CER 5.66% vs 5.95%. Shuffling its choices drops damaged acc from 72% to 53%, so routing matters here too.
4. **Dense matches it in distribution and fails on heavier noise:** heavy CER 23.8%, worse than the uncorrected input (15.2%). The looped models all generalise (5.7–6.5%).

## OCR task, grid 2: halting prior (2026-11-20)

Op library and loop retrained with PonderNet's geometric-prior KL from step 0 (weight 0.01, prior 0.2). Both now learn to halt (expected steps in training about 3.4). The prior keeps each step's halting probability under 0.5, so the old evaluation rule (first step with lambda > 0.5) runs almost to the cap; these models are evaluated with the rule that matches training: halt at the first step where P(halted by t) >= 0.5 (`HALT_RULE=cdf`, also used for freezing).

| model | params | MLP compute per step | CER | damaged acc | shuffle damaged acc | steps, clean / damaged | heavy CER |
|---|---|---|---|---|---|---|---|
| oploop K=8, prior | 0.65M | 1× | **1.17%** | **69.5%** | 47.8% | 2.7 / 4.9 | **5.70%** |
| loop, prior | 0.19M | 1× | 1.27% | 65.4% | — | 2.7 / 4.2 | 6.35% |
| loop, hidden 2048 (grid 1) | 0.65M | 8× | 1.18% | 68.3% | — | 4.5 / 7.0 | 5.95% |

Results:

1. **First result for the idea at matched compute per step.** Against the loop with the same per-step compute, the library is better on every measure (CER 1.17% vs 1.27%, damaged acc 69.5% vs 65.4%, heavy CER 5.70% vs 6.35%), with the same steps on clean characters and more on damaged ones (4.9 vs 4.2).
2. **It matches the same-size block at a fraction of the compute.** Same accuracy as the hidden-2048 loop, with 1/8 of the MLP compute per step and fewer steps (about 1/12 of its total MLP compute).
3. **Adaptivity is clear:** 1.8× more steps on damaged characters than clean, without labels. Freezing halted tokens costs under 0.02 points CER.
4. **Routing matters:** shuffling choices drops damaged acc from 69.5% to 47.8%.

Seeds (2026-11-22): 3 seeds each, cdf rule. Every library seed beats every loop seed on every measure.

| model | CER | damaged acc | steps, clean / damaged | heavy CER |
|---|---|---|---|---|
| oploop K=8 (seeds 0/1/2) | 1.17 / 1.18 / 1.17% | 69.5 / 68.8 / 69.3% | 2.75 / 4.79 (mean) | 5.70 / 5.90 / 5.79% |
| loop (seeds 0/1/2) | 1.27 / 1.30 / 1.30% | 65.4 / 65.1 / 64.9% | 2.70 / 4.08 (mean) | 6.35 / 6.47 / 6.43% |

## Pedal task, DC bias point (2026-11-22)

Same circuits, but each node predicts only its DC operating point (signal source at 0 V). Labels fixed first: an unbiased diode pair left the clip node floating at DC, and the solver's 1e-9 S to ground dragged it toward 0 V; `GMIN_D` = 1e-6 S across each diode now holds it at the base voltage (|V| across any diode under 0.5 mV). 12k steps, halting prior from step 0, Huber 0.05 + MSE, evaluated with the cdf halting rule. Many nodes are trivially 0 V (ground, input, output after the coupling capacitor), so the role columns matter more than the total; guessing each role's mean voltage gets collectors 6%, tone low-pass 8%.

| model | params | MLP compute per step | w50 | collector w50 | tone low-pass w50 | mean steps | shuffle w50 | 5–6 stages w50 (collector) |
|---|---|---|---|---|---|---|---|---|
| oploop K=8 | 0.62M | 1× | **89.9%** | **84%** | **80%** | 3.5 | 30.0% | **73.9% (68%)** |
| loop | 0.16M | 1× | 86.7% | 78% | 68% | 4.9 | — | 66.5% (61%) |
| loop, hidden 2048 | 0.62M | 8× | 87.9% | 81% | 75% | 3.6 | — | 70.4% (65%) |

Results:

1. **The DC task is learnable**, unlike the full waveform: 87–90% of nodes within 50 mV, collectors 78–84%.
2. **The op library wins at the same compute per step, and also beats the same-size block** that has 8× its per-step compute: +3.2 points over the loop and +2.0 over the wide loop overall, +12 and +5 on tone low-pass nodes, and the same lead on longer circuits. It also takes fewer steps than the loop (3.5 vs 4.9).
3. **Steps follow dependency depth.** Base nodes (set locally by the transistor) halt at about 3 steps, collectors at 4, tone low-pass nodes, which depend on the previous stage's collector, at 5.3–6.
4. **Routing matters a lot:** shuffling choices drops the library from 89.9% to 30%.
5. **Halting helps accuracy as well as compute:** each model is more accurate read at its halting step than after all 24 steps (library 89.9% vs 88.4%).

## Standard-MoE reference runs (2026-11-23)

Distinct layers, each with its own 8 experts and router (`--kind moe`), trained like the op library (halting prior, cdf rule). Seed 0.

OCR:

| model | params | CER | damaged acc | steps, clean / damaged | heavy CER |
|---|---|---|---|---|---|
| op library K=8 | 0.65M | **1.17%** | **69.5%** | 2.75 / 4.89 | **5.70%** |
| MoE, 4 layers | 2.42M | 1.27% | 66.3% | 2.13 / 2.83 | 6.30% |
| MoE, 16 layers | 9.52M | 1.20% | 67.8% | 3.32 / 5.47 | 6.08% |
| loop | 0.19M | 1.27% | 65.4% | 2.71 / 4.16 | 6.35% |

Pedal DC:

| model | params | w50 | collector | tone low-pass | mean steps | 5–6 stages w50 |
|---|---|---|---|---|---|---|
| op library K=8 | 0.63M | 89.9% | 84% | 80% | 3.5 | **73.9%** |
| MoE, 4 layers | 2.40M | **91.3%** | 86% | 86% | 3.9 | 6.0% |
| MoE, 24 layers | 14.23M | 89.4% | 87% | 68% | 6.1 | 72.0% |
| loop | 0.17M | 86.7% | 78% | 68% | 4.9 | 66.5% |

Notes:

- On OCR the shared pool is ahead of both MoEs at 1/4 and 1/15 of their size.
- On pedal DC the 4-layer MoE is the most accurate on 1–4 stage circuits but has no way to go deeper: on 5–6 stages it collapses to 6% (its extra steps reuse its last layer). The op library is close in distribution at 1/4 the size and keeps working on longer circuits. The 24-layer MoE needs 23× the parameters for about the same result.

## Pedal task, one audio sample at a time (2026-11-23)

Each node gets its voltage at sample n and the input at n+1, and predicts its change over one 48 kHz sample (`gen_analog_step.py`: 50k circuits, 10 ms each from the DC point, 8 random samples per circuit = 400k examples). Closed-loop test (`eval_rollout.py`): start at the true DC point and run 480 samples on the model's own predictions. 12k steps, halting prior from step 0 (0.01), Huber 0.05 + MSE, cdf rule.

| model | params | one sample: within 50 mV / 5 mV / 1 mV | steps | closed loop, nodes within 50 mV | 5–6 stages |
|---|---|---|---|---|---|
| predict no change | — | 93.0 / 72.3 / 51.2% | — | 48.2% (hold DC) | 40.7% |
| oploop K=8 | 0.63M | 92.9 / 63.3 / 20.2% | 3.0 | 11.1% | 8.2% |
| loop | 0.17M | 92.9 / 64.5 / 22.1% | 3.0 | 10.5% | 9.0% |
| loop, hidden 2048 | 0.63M | 92.8 / 65.0 / 21.8% | 3.0 | 10.2% | 8.3% |

Results:

1. **Nothing was learned.** All three models match "predict no change" at 50 mV and are worse at finer thresholds; training loss did not fall; closed loop they drift (11% vs 48% for holding the DC value). Every model halts at exactly 3 steps (the halting prior, not adaptivity), and shuffling the library's routing changes nothing.
2. **Why: fuzz edges dominate.** About 5% of samples contain a node jumping more than 0.5 V in one sample (up to 8 V), and these carry about 90% of the squared voltage change. The real Muff layout is no milder (7% of samples, 86%). The output is close to a square wave, so the edges are the signal, and an MSE-type loss on volts is dominated by them. The solver itself is fine: with silent input every circuit stays exactly at its DC point.

### Edge-aware retry (2026-11-24)

`run_step2.sh`: target asinh(dv / 10 mV) with Huber 1 in that space (`--asinh 0.01 --huber 1`), 30% of each batch drawn from rows where some node moves > 100 mV (`--edge-frac 0.3`; 20% of rows), halting prior 0.001. "Edges" = nodes moving > 100 mV in that sample; predicting no change scores 0 there.

| model | one sample: within 50 mV / 1 mV | edges within 50 mV | edge error / jump (median) | steps | closed loop within 50 mV | 5–6 stages |
|---|---|---|---|---|---|---|
| predict no change | 93.0 / 51.2% | 0% | 1.00 | — | 48.2% | 40.7% |
| loop | 93.5 / 47.0% | 3.8% | 0.998 | 3.5 | 27.8% | 22.5% |
| oploop K=8 | 93.6 / 48.4% | 2.9% | 0.999 | 2.8 | 24.4% | 20.7% |
| loop, fp32 (5k steps) | 93.3 / 49.5% | 0.9% | 1.000 | 2.1 | — | — |

1. **Edges still not learned.** The models still predict about zero change where a node jumps. The compressed target stopped them from getting worse at small changes (1 mV close to the baseline), and closed-loop drift fell from about 11% to 25–28%, still below holding the DC value.
2. **Halting is now free of the prior:** 2.1–3.5 steps, correlation with depth about 0.24 (was exactly 3.0).
3. **Not numeric precision.** Training fully in fp32 gives the same loss curve as bf16 (0.553 / 0.520 at 1k / 5k steps vs 0.553 / 0.518).
4. Open question: whether one-sample edges are learnable from this state at all (the switching instant is a knife-edge function of the state), or the models just can't fit them. An overfit test on a few hundred edge rows would separate the two.

## OCR cleaner v1 (2026-11-24)

Op library K=8, d=128 (0.65M params, 2.6 MB fp32), trained 40k steps on 2M fineweb-edu windows with line breaks in the character set (`gen_ocr2.py`: wrapped lines, hyphenated wraps), `--freeze-z` so the sparse inference path is exact, halting prior, cdf rule. Test: held-out fineweb-edu (4.3% of characters damaged; heavy set 16.4%).

| | CER | damaged chars fixed | steps, clean / damaged | heavy CER |
|---|---|---|---|---|
| input | 4.28% | — | — | 16.38% |
| v1, readout | 0.79% | 84.2% | 2.9 / 5.0 | 4.36% |
| v1, freeze (what `infer.py` runs) | 0.98% | — | — | — |

- Shuffling the router's choices drops damaged-character accuracy from 84% to 47%.
- `infer.py ocr` (sparse: halted characters cost nothing): about 28,000 characters/s on 1 CPU thread. Model copied to `~/models/ocr-cleaner-v1/model.pt`.
- Known issues: it sometimes merges a real space ("era in" -> "erain") or leaves a stray one ("de layed"); windows do not overlap, so damage across a window edge is not fixed.
