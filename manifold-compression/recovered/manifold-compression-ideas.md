# Manifold compression: ideas backlog

Status as of 2026-10-18. Experiment folder: `manifold-compression/`. Results so far: `manifold-compression/README.md`. Main plan: `manifold-compression-big-to-small.md`. Measurement plan: `manifold-compression-measurement.md`.

Written for: the author and collaborators who will pick the next experiment.

## Core idea

Treat a model's residual-stream activations as a manifold. Approximate it with a smaller set of functions, then project the network onto a smaller architecture. Judge every projection by output damage (KL on domain data, plus task scores), not by variance kept.

## Ideas, in the order the author prefers

### 1. Fix the folds: choose projection and refit by output damage (preferred)

PCA keeps high-variance directions. Those are not always the directions the output depends on. Pick the kept set and refit by how much each candidate direction moves the output (KL, or a sensitivity-weighted error).

- Status: partly done. The MLP shrink already uses a sensitivity weighting (`build_pruned.py`, `--shape`). The 50% shaped build is pending a bench.
- Gate: at equal parameters, does the damage-chosen kept set beat variance-chosen on per-domain KL and chat tool use?
- Open: "fold" is still a guess. We have not measured where the folds are or how many points a better approximation needs.

### 2. Per-region projections with a router (atlas)

Split the manifold into regions. Give each region its own small projection and a router that picks one per token. This is the MoE version of the same idea.

- Status: not started. Placement results (`m_placement.py`) found that adaptive placement ties importance ranking, so a router needs a real gain to justify its cost.
- Gate: per-region projections beat one global projection at equal total parameters.

### 3. Width varying by layer

Give each layer its own width instead of one global ratio. Choose widths from per-layer damage, not parameter count.

- Status: not started. Per-layer KL curves exist (`m_width.py`, `m_decoder_kl.py`).
- Gate: at equal total parameters, per-layer widths beat uniform width on KL and tool use.

### 4. Calibration and sensitivity weighting (in progress)

Calibrate the MLP shrink on the model's own chat answers, and weight the importance by per-token sensitivity (`--self-chat`, `--shape`).

- Results (50% build): self-chat calibration tools 0.425, GSM8K 0.467, MMLU 0.233, loops 0.212, ppl 18.20 (plain: 0.225, 0.40, 0.533, 0.224, 17.14). Adding sensitivity weighting at 0.5 was worse: tools 0.175, GSM8K 0.067, loops 0.424, ppl 19.12. Full table in `findings.md`.
- Gate: tool-use rate and loop rate on the bench, not build-time KL. Next: calibration-mix sweep.

### 5. Domain-carved MoE: keep the experts a domain uses

For a MoE model, keep the experts a target domain routes to, and drop the rest. Smaller total size.

- Gate (cheap, no training): count routing mass per domain on the existing MoE. If a domain concentrates on 10–20% of experts (about 80% of routing mass), carving is worth testing. If routing spreads evenly, it is not.
- Required parts: remove dropped experts from the router logits and renormalize, or retrain the router. Without this, the carved model fails on its own domain.
- Required out-of-domain check: per-domain KL and task scores on the other domains.

### 6. Expert width cut for speed (shared-memory target)

Shrink the width of each expert in a MoE. This cuts the bytes read per token, which is what sets decode speed on shared memory. Our MLP shrink applies directly, per expert.

- Gate: decode tok/s and quality on the target hardware (below), at 50% and 25% widths.
- Caveat: dropping whole experts shrinks total size but not active bytes per token, so it helps fit, not speed.

### 7. Shared-memory target: Strix Point

Target: a 35B-A3B-class model on Strix Point (iGPU, LPDDR5X, roughly 100 GB/s). Estimates only, not measured:

- Stock 35B-A3B at Q4 (~20 GB): about 20–30 tok/s.
- With a 50% per-expert width cut (~1.5B active): roughly 1.6–1.9× faster.
- Fit: tight on 32 GB, comfortable on 64 GB.

- Status: no Strix Point machine has been tested. The pod is an NVIDIA A40.
- Needed: stock tok/s and memory headroom on the actual hardware, and which backend (Vulkan or ROCm) people use.

### 8. Dense-to-MoE conversion, with nested sizes

Split a dense FFN (e.g., Qwen 27B) into E experts, route to k, and order experts by importance so smaller models use a prefix of them. One model then ships at several sizes.

- Speed: active FFN work per token drops to about k/E of the original. Total FFN parameters stay the same unless you also shrink.
- Risk: a dense FFN is not naturally partitioned. If every token uses most neurons, a partition loses quality until a router is trained (MoEfication setting).
- Cheap gate (one forward pass, on the 2B since it is the same family): measure how concentrated each token's FFN output is across neurons. If the top quarter of neurons carries most of it, partition plus router is promising. If the mass is spread evenly, conversion needs heavy training.
- Cost: the 27B in bf16 (~54 GB) does not fit an A40 (48 GB). Router distillation needs an 80 GB card or several GPUs. This is well above the few-dollars budget used for the 2B.
- Related: `mow-nested-width/`. Its open question (does explicit nesting beat free adaptive routing?) must be answered before the nested-size claim is trusted.

### 9. Flat spots as free capacity (unexplored)

Directions in the activation manifold that carry little variance may be unused. Idea: reuse them for capacity, for example by storing a second function there.

- Status: raised in discussion, not measured. Needs the idle-direction measurement first (how much of each site's variance and output mass sits in the low-variance tail).

### 10. Shared residual basis for carved experts

When experts from a big model move into a smaller model, they read and write a shared residual basis. That basis is the projection P, fixed across experts and the router.

- Status: part of idea 1 and idea 5. Not built.

### 11. Deform the manifold, then project (nonlinear warp)

Learn a smooth warp that unrolls the manifold's curved directions, then project the warped coordinates to fewer dimensions. In practice: a nonlinear encoder and decoder (autoencoder) per site, instead of PCA plus a decoder.

- What it can and cannot do: empty regions cost no dimensions, because a linear projection already ignores space with no data. What costs dimensions is curvature and folds. A warp helps only by flattening those, so it is the "fix the folds" idea (1) done with a learned map.
- Evidence so far: the decoder from PCA coordinates (M2) recovers about 30% of the cut variance on the 0.8B and 15% at half width on the 2B, with KL payoff 12–58%. That is the floor for a nonlinear map: a fully nonlinear encoder could do better, but we have not measured it.
- Cost problem: the layers consume the original 2048-wide vectors through linear maps, and only linear warps fold into those weights. A nonlinear warp needs an encoder and decoder at each site, which adds compute and can cancel the saving. The alternative is one warp at the embedding and one unwarp at the output, with every block refit in the small space. That is distillation (the student-training step in the main plan), not a post-hoc projection.
- Evidence against one shared warp: a single shared linear basis costs about 2× more KL than per-site bases (0.8B, k = 256: 5.32 vs 2.71 nats), so the manifold changes enough across depth that one global map is hard.
- Evidence for tolerance: off-manifold nudges are pulled back about 2× harder than on-manifold ones in mid-model, so small warp errors off the data should wash out.
- Risk: rare tokens sit in sparse regions, and deforming to close gaps may distort exactly those (the copy-task tail). Report tail statistics.
- Gate result (0.8B, single site, k = 64 and 256, three sites): the learned encoder beat PCA + decoder by 1.7-2.5x KL and plain PCA by 3-5x. Gate passed. See `findings.md`.
- 2B with an equal-parameter arm (decoder with twice the hidden width): the warp is still 1.4-2.5x lower KL than the equal-parameter decoder at every site and width tested.
- All sites at once (2B): warp 1.776 vs equal-parameter decoder 2.140 at width 512 (1.2x), 3.438 vs 3.550 at width 128 (1.03x). Compounding removes most of the single-site gain; width 512 all-sites roughly equals PCA at width 1024. Not usable as a post-hoc projection. Copy recall at width 512 is much better (0.455 vs 2.322 for PCA).
- Next gates: encoder compute cost; training the model to run in the small space (distillation) so errors do not compound; then a version where the whole network runs in the small space (distillation).

### 12. Mix the stock 0.8B and our projected 0.8B through a learned layer (exploratory, not a focus)

Added 2026-10-19 as a follow-up to hold. Freeze both 0.8B models and train only a small layer that mixes them. This is a known method family, model composition: CALM (Google, 2024) uses learned cross-attention between two frozen models. That reference was not checked.

- Question it answers: does our projected model hold 2B knowledge that the stock 0.8B lacks and can use? The stream test (`findings.md`, "The stock 0.8B against the 2B") suggests that our basis holds more of the 2B's state than the stock model carries.
- Cheap version: a per-token gate between the two output distributions, trained on KL to the 2B, with both models frozen. Take per-token hidden states from llama.cpp, so our GGUF needs no conversion (check that the tool returns the last-layer state for each token). Use the 2B's top-k probabilities as targets to keep the files small, because disk C: is nearly full. About 3–4 hours of work, no pod.
- Measures: KL to the 2B and wikitext perplexity, against the stock 0.8B alone (KL 0.30) and ours alone (KL 0.75). Then the 40-prompt smoke test with the gate on.
- Expectation: the gate trusts the stock model for chat and uses ours on prose, where our wikitext perplexity is better (13.85 against 15.48). Generation stays near stock quality.
- Why it is not a focus: both models run on every token (about 1.5B parameters of compute), and a 4-bit 2B is smaller and better. At 27B → 14B no stock partner exists, so the result cannot help the compression goal. It is a research probe for the 0.8B only.
- Do not merge weights instead: only the head parameters line up directly (`findings.md`, "Direct weight comparison").

## What has been ruled out

- Learned own-state predictor with an unscaled input: diverged. Standardizing inputs and using the MLP only when it beats linear fixed it. (`m_boundary.py`)
- Edits at position 0 (attention sink): not a strong sink in these models, but keep_first stays on by default.
- Pure variance-ranked projection as the final method: not ruled out, but Phase 1 showed the ranking must be checked against output damage.

## Next experiment

Run the cheap gates first, in this order:

1. Calibration-mix sweep at 50% (self-chat alone: tools 0.425, MMLU 0.233; sensitivity weighting at 0.5 made it worse).
2. FFN activation concentration on the 2B (idea 8 gate, one forward pass).
3. Routing count on the existing MoE (idea 5 gate, one forward pass).

Nothing in ideas 2, 3, 6, 9, or 10 is worth building until one of these gates passes.
