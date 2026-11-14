# Manifold compression: can a model be projected into a smaller one?

**2026-10-26 capacity versus projection:** A local matched stock/projected ×
rank-16/rank-64 study identifies a training-path bottleneck. With 32,975 answer
tokens per fresh arm, the projected model scores **5/39 at rank 16, 10/39 at
rank 64, and 11/39 after compressing the trained rank-64 update back to rank 16**
without further training. The full-rank reconstruction control matches all 47
answers exactly. All projected variants still fail the four program tasks;
truncation leaves one science loop. Stock itself regresses **19/39 → 14/39 or
17/39**, so the healing curriculum also needs better skill retention. This does
not prove the projection is lossless or justify replacing the earlier selected
checkpoint. Nine tests, cache gates and independent reload checks pass; **$0
cloud spend**. See the [controlled comparison and diagnostic](results/capacity-projection-20261026/README.md).

**2026-10-26 expanded-curriculum continuation:** Local correctness healing raises
original closed accuracy **43/56 → 45/56** and new closed accuracy **12/38 → 16/38**,
but new arithmetic only moves **3/24 → 4/24**, and programs regress **1/4 → 0/4**
(stock 4/4). Original repetition flags disappear while both science explanations
remain false. Development selects total update 24; total update 32 ties on dev
and loses previously sound science diagnostic answers. This is a mixed experimental
checkpoint, not a general upgrade. Five tests and independent reconstruction pass;
**$0 cloud spend**. See the [complete results and checkpoints](results/curriculum-healing-20261026/README.md).

**2026-10-25 broad-projection healing:** A local 16-update correctness-only
LoRA repair of the 64k broad Qwen2B projection improves the original closed
score **24/56 → 43/56** (stock 51/56), and a fresh set **18/42 → 32/42**,
with no lost previously correct answers in either set. Development selects
update 16. Fresh loop flags fall 3 → 0, but both original science answers
remain wrong and repetitive. No program meets all requirements; the one
passing original I/O check still uses linear search. Arithmetic remains weak
(1/8 original, 0/6 fresh). This supports further correctness healing with a
richer verified curriculum, without claiming general reasoning or stock parity.
Three tests, cache/adapter gates, and independent reconstruction of six answers
including both long explanations pass. All work is local, **$0 cloud spend**.
See the [results, adapter, and limitations](results/broad-healing-20261025/README.md).

**2026-10-25 broader + more follow-up:** Extending the successful broad
calibration mixture from 64k to 262,691 tokens changes Qwen2B's closed-answer
score from **24/56 to 25/56**: two gains and one loss, at identical model size.
Arithmetic, Python execution and JSON construction remain at zero. No program
meets all requirements, and both science explanations remain wrong and
repetitive; token-limit hits increase from one to four. More data improves
energy retention and chat KL, but adds little task benefit here. The local
run is complete with $0 cloud spend and no whole-model healing. See the
[combined comparison and artifacts](results/broad-more-20261025/README.md).

**2026-10-25 local projection ablations:** On Qwen3.5-2B at roughly 75%
parameter retention, a broader 64k-token calibration mixture improves closed
answer correctness from **8/56 to 24/56**, versus stock 51/56 (content score;
strict type/format scores are separate). Increasing chat-only calibration to
256k reaches 14/56. Per-layer bases score 1/56 both within the storage budget
and with extra space for their bridges. Token-dependent RMS correction scores
8/56; a fitted linear decoder scores 7/56 despite better reconstruction and KL.
The small nonlinear decoder probe gives little additional held-out recovery.
The broad arm still fails arithmetic, Python execution and full programming
requirements, and both science explanations remain wrong. This supports
broader activation calibration followed by correctness healing, without yet
claiming transfer to 27B. Full-width/cache/reload checks and nine tests pass.
All compute is local, with **$0 cloud spend** and no whole-model healing.
See the [comparison, raw outputs and limitations](results/local-projection-20261025/README.md).

**2026-10-25 correctness healing result:** The strongest 75% student
(Qwen3.5-27B → 20.10B parameters) improved from 92/128 to 103/128 at update 16
and **117/128 at update 32**, after one pass over verified Horizon targets with
no teacher KL. Stock scored 121/128. Long-form loop flags fell from 3/6 to 1/6;
photosynthesis still repeats, and interval-merging code remains invalid Python.
Full-width projection lost zero stock-correct answers in this screen.
Both H200 jobs are complete and their pods deleted, with **$11.93 estimated
combined cost** under the approved $15 cap. The selected update-32 adapter and
optimizer are saved and verified. This supports continued repair of the current
projection; broader correctness and long-form evaluation are still needed.
See the
[results, limitations and frozen protocol](results/correctness-healing-20261025/README.md).

**4,096-token follow-up:** The selected update-32 adapter reran eight previously
flagged prompts with their original batch companions. Binary search finished at
1,285 tokens, but its code remained invalid. Photosynthesis repeated through the
4,096-token cutoff. Same-pod shorter-limit controls matched all 20 output
prefixes; historical long outputs differed, so full-model reconstruction
reproducibility needs a separate audit. The inference job is complete, the pod
is deleted, and estimated cost is $2.24 under its $3.50 cap. Details are in the
[healing report](results/correctness-healing-20261025/README.md).

**2026-10-24 larger-model results:** The Qwen3.5-27B, Mistral Small 24B and
Qwen2B projection screen is complete. Scale helped at 75% parameter retention,
but neither large architecture reached stock parity; all half-size arms failed
badly. Full-width numerical controls passed. Strict task/format checks were
62→40→2 for Qwen27 and 47→23→2 for Mistral (stock/75%/50%); these are not general
accuracy scores. An exact Mistral readout improved formatting but left content
errors. The H200 pod is deleted, with estimated cost $3.58. See the
[full comparison, limitations and next experiment](results/scale-probe-20261024/summary.md).

**2026-10-24 healing pilot:** Four matched short repair runs are complete.
Corrected SmolLM2 improved; both Qwen runs regressed. The legacy SmolLM2 arm
improved KL while becoming more repetitive and less compliant, showing why KL
alone cannot be the success criterion. The recovered 500M-token Qwen student
also has better KL than the fresh Qwen initializations but much worse task
checks. Keep the output correction and stabilize healing before scaling it.
See the [full pilot comparison](results/healing-pilot-20261024/summary.md),
including raw generations, saved checkpoints, and the $0 local compute result.

**2026-10-24 audit:** Full-width PCA rotation preserves Qwen3.5-2B and
SmolLM2-1.7B predictions when the final normalization is handled exactly. The
builder's diagonal approximation introduces measurable error even without
compression. Both checks ran locally in under 30 seconds each. See the
[rotation audit](results/rotation-audit-20261024/summary.md). Later experiment
history, including the completed 500M-token repair run, was recovered from the
Windows machine in [recovered/findings-windows.md](recovered/findings-windows.md).

**Output correction:** The builder now keeps the full projected final-norm
matrix and multiplies by the measured RMS correction. The matrix stays small
and preserves tied embeddings. See the paired Qwen/Llama width-cut experiment,
checkpoint locations, limitations, and next gates in the
[output-fix report](results/output-fix-20261024/summary.md).

`build_projected.py --rotation-test` exercises a full-width PCA rotation without
shrinkage or healing. `--output-norm legacy-diagonal` reproduces the old final
norm for ablations; the default is `matrix`. The existing full builder and
layer-by-layer runner are Qwen-specific; `probe_output_fix.py` handles both
Qwen3.5 and Llama for the width-only experiment.

Corrected HF checkpoints include `projection_models.py`. In-repo scripts use
`projection_models.load_causal_model(path)`; external Hugging Face callers use
`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` after
reviewing the bundled module. The tested environment is pinned in
`results/output-fix-20261024/requirements-lock.txt` (Transformers 5.14.1).
For stock-model export, `export_projected_stock.py --model PATH --out NEW_PATH`
folds the adapter into a separate output head. That duplicates the vocabulary
weights and adds storage; it does not imply verified GGUF/runtime parity.

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
