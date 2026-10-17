# Mixture of Widths (MoW) — proof of concept

A small transformer testbed comparing **nested variable-width MLPs** (MoW) against
**mixture-of-experts** baselines at matched parameter and FLOP cost, on synthetic
tasks with controlled difficulty and type variation.

The question: does prioritized (nested) width with input-adaptive routing beat the
alternatives — or is it functionally equivalent to fine-grained adaptive-k MoE?
The second outcome is a null result, and the experiment is built to detect it
cleanly rather than to avoid it.

> **Status of the numbers.** The Results section below, and every table in this file, describe the
> **first grid**: 78 runs (26 conditions x 3 seeds) with learned positions and unpadded addition,
> preserved in `mow/runs_unpadded/`. After that grid the task was fixed (padded addition, RoPE,
> `recall_hard_m=(32,)`; see "Configuration the grid should now run" in `future-work.md`
>) and re-run with 10 seeds. Those runs are `mow/runs/`, and `figures/` and
> `figures/results_table.md` are rendered from them, so the figures do **not** match the tables in this
> README (for example addition is solved in the new grid). The Results text has not been rewritten
> for the second grid. A learned-position run with padding (`runs_pad_learned_BROKEN`, addition stuck
> at 0%) was deleted; its finding is the RoPE item in the configuration table.

## Pre-registered predictions

Written before any run. See `mow-experiment-plan.md`, which predates the code.

1. **MoW beats fixed widths on the frontier** (modest win). Routed MoW should sit
   below the fixed-tier interpolation line in figure 2.
2. **MoE wins the type axis; MoW wins the difficulty axis** (figure 4). MoE experts
   should specialize by task type, MoW tiers by difficulty.
3. **Fine-grained adaptive-k MoE without a balance loss develops an always-on
   expert core** — nesting emerging for free (figure 5).
4. **Open question the experiment decides:** does explicit nesting beat free-subset
   adaptive-k at this scale?

Falsification criterion for #1: if routed MoW does not sit below the fixed-tier
interpolation line, it bought nothing.

## Setup

- Plain PyTorch, nanoGPT-style layout. No framework dependencies beyond torch,
  numpy, matplotlib.
- Every run is fully described by an `ExperimentConfig` and seeded end to end.
- **Hardware:** this grid was run on CPU (12 physical cores). There is no usable
  GPU on the target box — no CUDA, and the integrated AMD iGPU has no practical
  PyTorch path on Windows. The model was sized so the full grid fits in hours
  rather than days; `config.PRESETS["full"]` holds the original larger trunk for
  rerunning the identical grid on a CUDA machine.

| | value |
|---|---|
| trunk | 4 layers, d_model 128, 4 heads, GELU, pre-LN, learned positions |
| reference MLP | hidden 64 (the 100%-FLOPs point) |
| MoW tiers | 8 / 16 / 32 / 64 — contiguous prefixes |
| MoE (matched) | 4 experts x 16 hidden |
| MoE (fine-grained) | 16 experts x 4 hidden |
| params | ~0.36M, matched within 0.9% across MoW / MoE / dense |
| vocab | 92 tokens, block size 32 |
| budget | 12,000 steps for every condition (MoW: 8,000 pretrain + 4,000 router) |
| seeds | 3 per condition, reported as mean with min–max range |

## Data — the 3x2 design

Format `<task><problem>=<answer><eos>`; loss covers answer tokens and the eos only.
Every example carries `{cell, task, difficulty}` metadata that never reaches the
model — it exists so evaluation can correlate router choices with difficulty.

| family | Easy | Hard | difficulty metric |
|---|---|---|---|
| **Arithmetic** | 1–2 digit addition | 5–8 digit addition | max operand digits (1–8) |
| **Brackets (Dyck)** | depth ≤ 2 | depth 5–8 | open-stack depth (1–8) |
| **Recall** | 256-entry table | 1024 / 4096-entry tables | log2(table size) (4–6) |

The recall family is an addition to the original plan, and the reason is the most
important empirical result in this repo.

### Why the plan's 2x2 could not test the hypothesis

MoW allocates MLP width **per token**. For that to buy anything, per-token MLP
capacity has to be the binding constraint. On the plan's arithmetic and bracket
tasks it is not — the bottleneck is depth and attention, and the MLP is idle
capacity at every width tested.

Measured on the original 4-cell task, dense models, identical recipe (12,000
steps, warmup + cosine, mixed data):

| MLP hidden | 16 | 32 | 64 | 128 | 512 |
|---|---|---|---|---|---|
| FLOPs fraction | 0.031 | 0.062 | 0.125 | 0.250 | 1.000 |
| answer-token loss | 0.3213 | 0.3661 | 0.2856 | 0.3560 | 0.3501 |
| exact-match | 0.750 | 0.750 | 0.750 | 0.750 | 0.746 |

**Hidden 16, at 3% of the FLOPs, matches hidden 512.** The ordering is not even
monotone; the 0.08-nat spread is noise larger than any width effect across a 32x
range. A compute-quality frontier measured on that task is flat by construction,
so no routing scheme — MoW, adaptive-k MoE, or otherwise — can be distinguished
from picking the cheapest tier. The comparison would have been vacuous.

The plan anticipated this failure mode and prescribed a remedy: *"if all conditions
saturate, increase difficulty spread (more digits, deeper nesting) before concluding
anything."* That remedy was tried and it does not work here, because in these tasks
difficulty scales the **sequence length**, not the per-token computation:

| variant | hidden 32 | hidden 512 | effect of 16x width |
|---|---|---|---|
| base-10, 5–8 digits | 1.8452 | 1.7569 | −0.088 |
| base-10, 10–14 digits | 2.0686 | 2.0988 | **+0.030 (wrong way)** |
| base-32, 5–8 digits | 2.8818 | 2.9523 | **+0.070 (wrong way)** |

Harder arithmetic makes the task harder for every width equally. It does not make
width matter.

### What does make width bind

Memorization. MLPs act as key-value memories, so storing N arbitrary associations
has a floor in MLP parameters that no amount of depth substitutes for. On a fixed
random table `T[a][b] -> (c, d)` with 4096 entries, width binds monotonically:

| MLP hidden | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| MLP params | 16,960 | 33,408 | 66,304 | 132,096 |
| loss | 0.1161 | 0.0072 | 0.0049 | 0.0033 |

This is why the recall family exists, and why the ladder was retargeted from the
plan's `[128, 256, 384, 512]` — which sits entirely in the saturated regime — down
to `[8, 16, 32, 64]`, which straddles the measured knee. Absolute widths carry no
scientific meaning here; what matters is that the tier ladder spans a region where
width actually trades off against loss.

Two smaller consequences. **Weight decay is 0**: it directly fights memorization,
and an earlier probe run at 0.1 inverted the width ordering (narrow beating wide),
which is an optimization artifact rather than a capacity result. **Addition operands
are reversed** alongside the answer, so the i-th least-significant digit of every
field sits at a fixed offset; this did not rescue long addition (5–8 digit addition
stays at 0% exact-match, and is reported as such) but it is the correct format.

Table sizes are fixed from a dedicated seed, so every condition and every run seed
memorizes the identical associations.

## Layout

```
mow/
  config.py    dataclass configs, presets, analytic FLOP accounting
  data.py      task generators, encoding, exact-match checker  (python -m mow.data)
  model.py     trunk, MoWMLP, MoEMLP, routers                  (python -m mow.model)
  train.py     training loop, aux losses, tier schedules
  eval.py      per-condition eval, FLOP accounting, router diagnostics
  plots.py     the five figures                                (python -m mow.plots)
  run_grid.py  sweep runner                                    (python -m mow.run_grid)
  run_layertest.py, run_kvswitch.py   width/KV-cache probes
  runs/        10-seed grid on the fixed task (config.json, metrics.csv, final.json;
               ckpt.pt is gitignored)
  runs_unpadded/   the first 78-run grid that the Results section describes
  runs_gpu/        partial GPU re-run (45 runs)
  runs_widthtest/ runs_ropetest/ runs_ropenopad/ runs_recalcal/ runs_fixsearch/
  runs_layertest/ runs_kvswitch/       small diagnostic runs behind the config choices
figures/       rendered figures + results_table.md (from mow/runs/)
logs/          transcripts for the diagnostic runs
gemma_phase1/  Gemma 3n / 4 nesting measurements (see writeup-nested-width-humanized.md)
router_demo/   serving demo for the router
```

Reproduce:

```
python -m mow.data                 # data sanity checks
python -m mow.model                # model self-tests + overfit gate
python -m mow.run_grid --seeds 0,1,2 --workers 8 --threads 2
python -m mow.plots
```

`run_grid` skips any run with a `done.json`, so an interrupted sweep resumes.

## Design notes worth knowing

**Soft mixing is free.** Both routed variants evaluate every choice at the cost of
one dense MLP forward. For MoW, tier *t*'s output is the running sum of per-block
projections, so computing all four blocks once costs exactly one full projection
and any tier mixture is a per-token reweight of those blocks:
`y = Σ_b p_b · (Σ_{t≥b} π_t · scale_t)`. For MoE, total expert hidden equals `d_ff`,
so a dense evaluation of all experts costs the same. This is what makes the
soft-to-hard schedule affordable without REINFORCE.

**FLOPs are analytic, never wall-clock.** Reported FLOPs come from the tier/expert
choices. Attention is identical across all conditions and reported separately.

**Matched optimization budget.** MoW trains in two phases (pretrain + router), so
every dense and MoE baseline gets that same total step count. Without this, figure 2
would be comparing training budgets rather than architectures.

**The mixed pretrain pass.** Nesting pretrain runs four uniform-tier passes *plus* one
pass with an independently sampled tier per layer. A router picks tiers per layer per
token, so a trunk trained only on uniform-tier forwards would meet ragged tier
combinations for the first time at routing time.

**Collapse is measured as marginal entropy**, i.e. the entropy of the choice
histogram across tokens. Evaluation runs with hard selection, so per-token entropy
is 0 by construction and would report collapse everywhere; what "collapse" actually
means is the whole token population funnelling into one choice.

**Two fixes to the original plan.** The fine-grained MoE uses k ∈ {1,2,4,8,16}: the
plan's k ∈ {1,2,3,4} over 16 experts spans only 6–25% of dense FLOPs, so that
condition could never reach the right half of figure 2 and the headline comparison
would have been unfair by construction. Per-tier learned output scales are kept, as
the plan requires.

## Figures

1. `fig1_nesting_tax.png` — loss at each fixed tier vs separately-trained dense models.
2. `fig2_frontier.png` — **headline**: loss vs mean per-token MLP FLOPs.
3. `fig3_router_interpretability.png` — MoW tier choice vs difficulty metadata.
4. `fig4_specialization_vs_prioritization.png` — MoE utilization vs MoW tier histogram, by cell.
5. `fig5_emergent_prioritization.png` — fine-grained MoE expert frequency, balance loss on vs off.

Categorical colors use a fixed, CVD-validated order; λ sweeps use a sequential ramp
because λ is a magnitude. Three palette slots fall below 3:1 contrast on white, so
every series is direct-labeled and `figures/results_table.md` carries the numbers.

## Results

78 runs, 3 seeds per condition, 0 failures. Every number below is a mean over 3
seeds. Ranges are min–max. Full table: `figures/results_table.md`.

**Read this first.** Five of the six data cells are solved at every width. So the
whole experiment lives in a narrow band: every condition scores between 0.62 and
0.71 loss. Differences are real but small, and several sit inside the seed range.

### Prediction 1 — MoW beats fixed widths on the frontier. REFUTED.

Routed MoW is worse than simply picking a fixed tier, at every compute level.

| compute (FLOPs frac) | fixed tier | routed MoW |
|---|---|---|
| ~0.13 | 0.6531 | 0.6967 |
| ~0.25 | 0.6418 | 0.6804 |
| ~0.85 | ~0.634 | 0.6724 |

The pre-registered falsification criterion was that routed MoW must sit below the
fixed-tier line. It sits above it everywhere. Adding a router to the nested model
costs more than adaptive width saves.

### Prediction 2 — MoE splits by task type, MoW by difficulty. REFUTED, both halves.

MoE experts were used almost equally by every cell. With 4 experts, uniform use is
0.25. Measured use ranged only from 0.19 to 0.32. No task-type specialization
appeared.

MoW did split clearly, but mostly by task family rather than by difficulty:

| cell | share of tokens at the narrowest tier (8) | at the widest tier (64) |
|---|---|---|
| dyck_easy | 0.88 | 0.00 |
| dyck_hard | 0.83 | 0.00 |
| recall_easy | 0.32 | 0.47 |
| arith_easy | 0.35 | 0.38 |

### Prediction 3 — a fine-grained MoE without a balance loss grows an always-on core. SUPPORTED, weakly.

Turning the balance loss off made expert use more uneven, as predicted. The effect
is small.

| | most-used expert | least-used expert |
|---|---|---|
| balance loss on | 0.118 | 0.019 |
| balance loss off | 0.135 | 0.008 |

About 8 of the 16 experts are used more than uniform when the balance loss is off.

### Prediction 4 — does explicit nesting beat free-subset adaptive-k? ANSWERED: no.

Neither beats a fixed width, but nesting is the weaker of the two.

The fine-grained MoE reached 0.063 FLOPs fraction at 0.660 loss. MoW cannot reach
that point at all — its narrowest tier costs 0.125. So the free-subset design covers
compute levels that nesting cannot.

One caution about the adaptive-k comparison. Every λ above 0.01 pinned it to k=1,
its cheapest setting. So its λ sweep produced one compute point with six noisy
repeats, not a compute-quality curve. Its best single result (0.6255) does beat the
fixed tier at the same compute (0.6418), but that spread is training variation.

### Unplanned finding 1 — nesting itself is free

One weight matrix, sliced four ways, matched four separately trained dense models.
Both trained for 12,000 steps.

| width | dense | nested | difference |
|---|---|---|---|
| 8 | 0.6792 | 0.6531 | −0.0261 |
| 16 | 0.6307 | 0.6418 | +0.0111 |
| 32 | 0.6446 | 0.6349 | −0.0097 |
| 64 | 0.6255 | 0.6292 | +0.0037 |

The difference flips sign and averages near zero. Exact-match was slightly higher for
the nested model at all four widths.

This matters for the negative result above. Nesting is not what fails. The router is.

**Step-matching was essential here.** An earlier comparison used an 8,000-step nested
model against 12,000-step dense models. That showed a cost of +0.065 to +0.129 at
every width, all positive. The entire effect was the step difference.

### Unplanned finding 2 — the router allocates by task family, not by difficulty

Figure 3 shows the opposite of the expected trend. Within each family, the router
spends *less* as difficulty rises, not more.

But difficulty is not the main driver. Task family is, and the gap is much larger:

| cell | solved (exact) | width helps (loss w8 − w64) | router spends |
|---|---|---|---|
| arith_easy | 0.953 | 0.0541 | 0.522 |
| arith_hard | 0.000 | 0.1342 | 0.415 |
| dyck_easy | 0.997 | 0.0038 | 0.143 |
| dyck_hard | 0.997 | −0.0038 | 0.154 |
| recall_easy | 1.000 | 0.0029 | 0.584 |
| recall_hard | 0.031 | 0.1313 | 0.395 |

Brackets get roughly a third of the compute the other families get.

**Two tempting explanations both fail on this table.**

*"The router learned it cannot solve those problems."* The bracket cells are solved at
99.7% and receive the least compute of anything. Going narrow is not a signal of
giving up.

*"The router allocates where width helps."* The two cells where width helps most
(arith_hard 0.134, recall_hard 0.131) get less compute than the two where it helps
least (arith_easy 0.054, recall_easy 0.003).

**A third reading fits all six cells, and is untested.** The bracket answer is fully
determined by the prompt — read the unclosed brackets and reverse them. That is
retrieval, which attention does; the MLP is barely involved. Recall is the opposite:
arbitrary stored associations, which is exactly what MLP width holds. Arithmetic is in
between. So the router may be giving MLP width to the tasks that use the MLP at all.

This is a reading of six data points. Across cells, the router's spend correlates
−0.17 with solvability and +0.31 with width-usefulness. At n=6 neither number is
meaningful. The family-level gap is large enough to trust; the explanation for it is
not tested. A direct test would ablate the MLP at fixed attention per family.

### Router collapse

Collapse was monitored as the marginal entropy of the choice histogram. MoW router
entropy fell steadily with the FLOP penalty:

| λ | 0 | 0.01 | 0.05 | 0.15 | 0.5 | 1.5 |
|---|---|---|---|---|---|---|
| entropy | 0.424 | 0.685 | 0.478 | 0.284 | 0.088 | 0.021 |

At λ = 0.5 and above, entropy falls below the 0.1 threshold. The router has collapsed
onto the narrowest tier. This is the failure mode the plan predicted, and it was
detected rather than hidden.

### What would change these conclusions

The width gradient in this task suite is small and comes from one cell. A stronger
gradient would be a fairer test of routing. The most direct route is a recall task
that is hard enough to need width but still learnable within the step budget — the
4096-entry table only reached 11% exact-match. See `future-work.md`.

