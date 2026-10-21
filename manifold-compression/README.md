# Manifold compression: can a model be projected into a smaller one?

The idea, in full, is in [`notes.md`](notes.md) §10. In short: treat each block of a model as a function on its activation manifold, not as a fixed set of weights. Then measure, before training anything, how much is lost when that manifold is projected to a smaller width, and which block boundaries a smaller model has to keep. Three pieces are combined:

- **SliceGPT**: a linear projection of the residual stream, which is the free part.
- **The function view**: how much of what the projection cuts can be recovered by a small nonlinear map. That recoverable part is what training would have to pay for.
- **The own-state test**: can a mixing layer's output be predicted from the token's own state, without context? Layers where it can are boundaries a smaller model can absorb.

All results to date are collected in [`findings.md`](findings.md); the idea backlog is in [`../.claude/plans/manifold-compression-ideas.md`](../.claude/plans/manifold-compression-ideas.md).

Everything is measured as KL divergence of the final next-token distribution from the untouched model, broken down by domain: prose, code, maths, chat, and a synthetic long-context copy task.

## Pre-registered predictions

Written on 2026-10-17, before any measurement.

1. **The free part is small.** Projecting every site to half width with PCA will cost a lot (mean KL > 0.5 nats). Single-site projections will be far cheaper than all-site ones, because errors build up.
2. **A real curved part exists, but it is modest.** The MLP decoder will recover a clear but minority share of the cut variance, roughly 10–40% R² at half width. That's enough to matter, but not enough to make projection free.
3. **The copy task is hurt most.** Exact recall of planted codes depends on low-variance directions, so its KL will be the worst of any domain at every width.
4. **Some boundaries are not real.** At least a third of the DeltaNet layers will be substitutable by an own-state predictor at small KL. Fewer attention layers will be, and on the copy task almost none.
5. **Middle layers attract, ends amplify.** Off-manifold perturbations in middle layers will shrink by the final layer, with gain < 1. Perturbations near the first and last layers will not shrink.

## Kill criteria

From the plan, in brief:

- **Dead:** at half width on Qwen3.5-9B, the decoder floor is worse than the real Qwen3.5-2B's gap to the 9B.
- **Reduces to SliceGPT:** the MLP decoder recovers ~0 of the cut variance.
- **Depth is fixed:** perturbations grow at most boundaries.

## Layout

| file | what |
|---|---|
| `common.py` | model loading, domain data, a layer-by-layer runner with edit hooks, the replace-and-continue KL routine |
| `collect.py` | runs the fit split and stores the residual state at every site |
| `m_width.py` | M1 + M2: PCA projection (per site, all sites, shared basis) and MLP decoders for the cut part |
| `m_boundary.py` | M3: own-state predictors per mixing layer, then greedy substitution |
| `m_perturb.py` | M4: on- and off-manifold perturbations, gain by depth |
| `m_idim.py` | M5: intrinsic dimension by neighbourhood size |
| `results/<model>/` | JSON results |

A site is one point in the residual stream. `L{i}.in` is the stream entering layer i. `L{i}.mid` is the stream after layer i's mixer and before its MLP. `final` is the stream before the output norm.

Phase 0 is Qwen3.5-0.8B on CPU, run to prove out the pipeline. Its numbers are small-sample and are not the result.

## Phase 0 results: Qwen3.5-0.8B (CPU, 2026-10-18)

Full tables are in [`results/Qwen3.5-0.8B/summary.md`](results/Qwen3.5-0.8B/summary.md).

**Read these with the caveats first:**
- It's the post-trained 0.8B, not base.
- The eval set is 8 sequences (2 each for prose, code, maths and copy; no chat).
- **The copy-task tail numbers rest on 16 answer tokens from 2 sequences**, so "worst 1%" is close to "worst token." Everything said below about the copy tail is a lead to confirm, not a finding.
- Position 0 is never edited. Applying that rule cut the all-sites quarter-width KL from 3.01 to 2.77, so the first token isn't a strong attention sink in this model.

**M1, the free part.** Projecting every site to width k with PCA:

| k / 1024 | variance kept | KL, per-site basis | KL, one shared basis |
|---|---|---|---|
| 256 | 0.77 | 2.71 | 5.32 |
| 512 | 0.90 | 1.34 | 2.24 |
| 768 | 0.97 | 0.41 | 0.47 |

Projecting a single site is far cheaper, and the cost is U-shaped over depth: 0.11–0.14 at k=256 for L6–L12, against 0.43–0.64 at L0, L21 and final. Most of the all-sites cost comes from errors building up across sites.

**M2, the curved part.** The first run (27K fit tokens, held out by domain) showed negative R². That was a data and split problem, not a flat manifold. With 306K tokens and an interleaved held-out split, an MLP recovers about **30% of the variance PCA cuts**, and that recovered part removes about **half the KL**:

| site | k | R² of the cut part | KL, PCA | KL, PCA + decoder | KL reduction |
|---|---|---|---|---|---|
| L6.in | 256 | 0.32 | 0.128 | 0.054 | 58% |
| L6.in | 512 | 0.28 | 0.027 | 0.015 | 46% |
| L15.in | 256 | 0.32 | 0.254 | 0.126 | 51% |
| L15.in | 512 | 0.27 | 0.045 | 0.026 | 42% |
| L20.in | 256 | 0.28 | 0.398 | 0.227 | 43% |
| L20.in | 512 | 0.25 | 0.084 | 0.054 | 36% |

The part of the cut that can be predicted from the kept coordinates is the part that matters most to the output. So the function view is worth something beyond SliceGPT: roughly halving the projection loss at a single site. It's smaller late in the model.

**M3, which boundaries are real.** Every mixing layer's update is 33–83% predictable from the token's own state. Substituting the prediction costs 2–9× less than deleting the update.
- Ranked by mean KL, the greedy pass replaces 13 of 24 mixing layers at 0.37 nats.
- Ranked by mean KL plus the copy-task tail, it replaces 10 at 0.28 nats, with 2.4× lower worst-case copy damage.
- Neither pass ever chose layers 0, 1, 6, 14, 15, 19 or 23. That set includes all three late attention layers. The early attention layers (3, 7, 11) were absorbed like DeltaNet layers.
- L23 is the averaging trap in action: R² 0.81 and a low mean cost (0.031), but the second-worst copy tail of any layer (0.31).

**M4, does the array compose.** A nudge of 10% of the state's size shrinks relative to the state by the final layer, from every site up to L20. Off-manifold nudges are pulled back about twice as hard as on-manifold ones in the middle of the model (ratio 0.25–0.27 vs 0.45–0.57). Only the last two layers amplify, and only slightly (1.07 at L22).

**M5, intrinsic dimension.** About 10–16, rising with neighbourhood size and never levelling off. The estimator is biased low at 3000 points, so this is a loose lower bound only.

**Against the predictions:**

| # | prediction | outcome |
|---|---|---|
| 1 | free part small; half width > 0.5 nats | confirmed (1.34) |
| 2 | curved part 10–40% R² | confirmed (25–32%), and worth ~40–60% of the KL |
| 3 | copy task hurt most by projection | **wrong**: at half and three-quarter width, copy is hurt less than average |
| 4 | ≥ a third of DeltaNet layers substitutable; fewer attention layers | confirmed, more strongly than predicted; the late attention layers are the ones kept |
| 5 | middle attracts, ends amplify | half right: the middle attracts and the last two layers amplify slightly, but the front attracts too |

**What Phase 1 has to fix:**
- Many more copy sequences in the eval set: at least 64, not 2.
- A chat domain in the eval set.
- Base checkpoints.
- Decoder KL at all sites at once, not one site at a time.
