# Mixture of Widths (MoW) Proof-of-Concept — Implementation Plan

Plan for Claude Code. Build a tiny transformer testbed that compares nested variable-width MLPs (MoW)
against MoE baselines on synthetic tasks with controlled difficulty and type variation.

## Core Hypothesis

Prioritized (nested) width + input-adaptive routing beats alternatives at matched parameter/FLOP cost —
OR is functionally equivalent to fine-grained adaptive-k MoE (a null result we want to detect cleanly).

## Ground Rules

- Plain PyTorch, nanoGPT-style layout. Tight iteration loop > abstraction.
- Single repo, small files, no framework dependencies beyond torch + matplotlib + (optionally) wandb-style local CSV logging.
- Every run reproducible from a config dict. Seed everything.
- Target hardware: single consumer GPU or CPU-tolerable. Model ~5–10M params. Full training run should complete in minutes, not hours.
- Composability requirement: MoW MLP must be a module taking a width argument; MoE must wrap N MLP modules. MoE-of-MoW (phase 2) must fall out by composing the two classes.

## Repo Structure

```
mow/
  config.py          # dataclass configs for model / data / train / experiment grid
  data.py            # synthetic task generators (2x2 design)
  model.py           # transformer trunk, MoWMLP, MoEMLP, routers
  train.py           # training loop, aux losses, tier/expert sampling schedules
  eval.py            # per-condition eval, FLOP accounting, router diagnostics
  plots.py           # the five canonical figures
  run_grid.py        # sweep runner over experiment conditions
  runs/              # per-run output dirs (config.json, metrics.csv, checkpoints)
  README.md
```

## Phase 0 — Data (2x2 design)

Character-level synthetic tasks. Two axes: **task type** (for MoE specialization) x **difficulty** (for MoW adaptivity).

| | Easy | Hard |
|---|---|---|
| **Arithmetic** | 1–2 digit addition | 3–5 digit multiplication |
| **Brackets (Dyck)** | depth ≤ 2 | depth 5–8 |

- Format: `<task><problem>=<answer><eos>`, loss only on answer tokens (mask the prompt).
- Each example tagged with metadata: `{type, difficulty_level}` (digit count / nesting depth as integer). Metadata flows through to eval for router-interpretability plots. Never fed to the model.
- Mixed dataset = uniform over the 4 cells. Also generate per-cell eval sets.
- Deliverable: `data.py` with generators + a sanity script that prints samples and verifies exact-match answer checking.

## Phase 1 — Model

### Trunk
- 6 layers, d_model 256, 4 heads, MLP hidden 1024 (SwiGLU or GELU — pick GELU for simplicity), char vocab, rotary or learned positions (learned is fine).
- Pre-LN. Standard causal LM.

### MoWMLP module
- `forward(x, width_idx)` where tiers = [256, 512, 768, 1024] hidden neurons as **contiguous prefixes** of the weight matrices. Slice, don't mask, for the hard path; masking allowed for the soft-training path.
- **Per-tier learned output scale** (one scalar per tier) to fix residual-contribution statistics across widths. This is load-bearing — do not skip.

### MoEMLP module
- N experts, each a dense MLP. Parameter-matched headline config: 4 experts x 256 hidden (total = one 1024 MLP).
- Top-1 and top-2 modes. Switch-style load-balancing aux loss. Track expert utilization every eval.

### Routers
- One linear head per layer reading the pre-MLP residual (per-token), producing tier logits (MoW) or expert logits (MoE).
- MoW router training: **soft-to-hard**. Softmax-weighted mix of all tier outputs (compute full width, mask per tier — cheap at this scale), temperature annealed toward hard selection. No REINFORCE.
- **FLOP budget loss**: `λ * mean(expected_FLOPs_fraction)` added to total loss. λ is a sweep parameter. Expect router collapse to 100% without it; treat collapse as a detected failure (log tier histogram every N steps, alert if entropy < threshold).
- Adaptive-k MoE variant: same soft-to-hard trick over k ∈ {1,2,3,4}, with the same budget loss.

### Fine-grained MoE variant (phase 1.5, the equivalence test)
- 16 experts x 64 neurons, adaptive k, budget loss, **weakened or removed balance loss** (make balance-loss coefficient a config knob including 0).
- Log per-expert activation frequency as a function of chosen k — this feeds the "does prioritization emerge spontaneously?" figure.

## Phase 2 — Training Protocol

1. **Nesting pretrain (no router):** each step, either (a) sample one tier uniformly, or (b) compute loss at all 4 tiers and sum. Use (b) — it's affordable and lower-variance. Train to convergence on mixed data.
2. **Baselines:** 4 separately-trained dense models at widths 256/512/768/1024. Same steps, same data, same seeds x3.
3. **Router phase:** attach router, co-train (lower LR on trunk or freeze trunk — make it a config flag, try co-train first). Sweep λ over ~5 values to trace the compute-quality frontier.
4. **MoE conditions:** parameter-matched top-1 MoE, top-2 MoE, adaptive-k MoE, fine-grained adaptive-k MoE. Same schedule, same router architecture, same seeds.
5. 3 seeds per condition minimum. Report mean ± range.

## Phase 3 — Evaluation & The Five Figures

FLOP accounting: analytic per-token MLP FLOPs given tier/expert choices; report **average FLOPs fraction** per condition (attention FLOPs constant across conditions, report separately).

1. **Nesting tax:** loss at each fixed tier vs. the 4 dense baselines. (Bar/line, per tier.)
2. **The frontier (headline):** loss vs. avg per-token FLOPs. Curves/points: fixed tiers, routed MoW (λ sweep), top-1 MoE, top-2 MoE, adaptive-k MoE, fine-grained adaptive-k MoE. Routed MoW must sit below the fixed-tier interpolation line to claim a win.
3. **Router interpretability:** MoW tier choice vs. difficulty metadata (digit count / Dyck depth). Expect monotone relationship.
4. **Specialization vs. prioritization:** MoE expert-utilization heatmap and MoW tier histogram, split by the 2x2 data cells. MoE should split on type; MoW should split on difficulty.
5. **Emergent prioritization:** fine-grained MoE per-expert activation frequency vs. k, with balance loss on vs. off. Look for a stable always-on core.

Exact-match accuracy on answers reported alongside loss for all conditions.

## Pre-registered Predictions (write to README before running)

- MoW beats fixed widths on the frontier (modest win).
- MoE wins the type axis; MoW wins the difficulty axis (figure 4).
- Fine-grained adaptive-k MoE without balance loss develops an always-on expert core (emergent nesting).
- Open question the experiment decides: does explicit nesting beat free-subset adaptive-k at this scale?

## Phase 4 (deferred, design-for but don't build)

- MoE-of-MoW hybrid: compose MoEMLP over MoWMLP modules, two router heads (expert, width). Only justified if phase 1 routers train stably.
- Batch-1 wall-clock benchmark of hard-path prefix slicing vs. expert gather/scatter (deployment story).
- Speculative decoding with the 25% slice as draft model.

## Milestones / Task Order for Claude Code

1. `data.py` + sample/verify script. ✅ gate: printed samples look right, exact-match checker passes on known cases.
2. `model.py` trunk + MoWMLP, overfit a tiny batch at full width. ✅ gate: loss → ~0 on 64 examples.
3. Nesting pretrain + dense baselines. ✅ gate: figure 1 renders, nesting tax quantified.
4. MoW router + λ sweep. ✅ gate: no collapse at some λ; figure 3 shows difficulty correlation.
5. MoE conditions. ✅ gate: expert utilization spread (not collapsed); figure 4 renders.
6. Fine-grained adaptive-k condition. ✅ gate: figure 5 renders.
7. Full grid x3 seeds, figure 2 headline plot, README results section.

## Known Failure Modes to Watch

- Router collapse (tier or expert): monitor selection entropy, log histograms every eval.
- 25% tier under-contributing: verify per-tier output scales are learning (log their values).
- MoE instability at tiny scale: if top-1 won't train, fall back to top-2 for the headline and say so.
- Task too easy/uniform: if all conditions saturate, increase difficulty spread (more digits, deeper nesting) before concluding anything.

---

## Appendix — implementation decisions (added at build time)

Recorded here so the plan stays the document of intent and the divergences are
visible. Full rationale is in `README.md`.

**Environment.** No usable GPU on the target box: torch is a CPU-only build, there
is no CUDA, and the integrated AMD Radeon 890M has no practical PyTorch path on
Windows (DirectML would force a torch downgrade for an expected 1–2x, sometimes
negative, payoff). The trunk was therefore sized down to 4 layers / d_model 128 /
d_ff 512 (~0.80M params) so the full grid fits in hours. `config.PRESETS["full"]`
holds the plan's original 6-layer / d_model 256 trunk for rerunning the identical
grid on a CUDA machine.

**Data changes, both to protect the difficulty axis:**
- Hard arithmetic is 5–8 digit **addition**, not 3–5 digit multiplication, which
  small char transformers do not learn at this scale. A hard cell stuck at chance
  gives a flat loss across widths, and figures 3–4 would report unlearnability as
  an architecture result. Answers are emitted least-significant digit first.
- The bracket task predicts the **closing sequence**, not a valid/invalid bit. One
  bit saturates instantly and has a 50% exact-match floor; emitting closers makes
  answer length scale with depth.

**Fixed a FLOP-coverage bug.** Fine-grained MoE uses k ∈ {1,2,4,8,16}. The plan's
k ∈ {1,2,3,4} over 16 experts x 64 spans only 6–25% of dense FLOPs, so that
condition could never reach the right half of figure 2 — the headline comparison
would have been unfair by construction.

**Choices made where the plan was silent:**
- Nesting pretrain runs the four uniform-tier passes **plus one pass with an
  independently sampled tier per layer**, averaged. A router picks tiers per layer
  per token, so a trunk trained only on uniform-tier forwards would meet ragged
  combinations for the first time at routing time.
- Soft-to-hard anneals temperature over the first 70% of the router phase, then
  switches to a straight-through estimator. Nested prefixes have a worse soft-hard
  gap than MoE: mixing tiers reweights shared neuron blocks rather than selecting
  among disjoint experts.
- λ sweep is {0, 0.05, 0.15, 0.5, 1.5}. λ multiplies a FLOPs fraction in [0.25, 1],
  so the interesting scale is set by the loss gap between narrowest and widest tier.
- **Matched optimization budget:** MoW spends pretrain + router steps, so every
  dense and MoE baseline gets that same total. Otherwise figure 2 compares training
  budgets, not architectures.
- Collapse is measured as the **marginal** entropy of the choice histogram across
  tokens. Eval runs with hard selection, so per-token entropy is 0 everywhere and
  would report collapse universally.
