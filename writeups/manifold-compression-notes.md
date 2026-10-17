# Can a huge model be projected into a tiny one? Notes on the manifold view

*Working notes, 2026-10-17. Thinking-out-loud session on whether a massive model (the trigger was Muse-Spark-Open-816B-A42B) can be turned into a much smaller one by treating it as a continuous object and down-projecting, rather than by pruning and retraining. Conclusion up front: the idea is sound, most of it is already in the literature under other names, and the part that is not is a specific, cheap set of experiments listed at the end.*

---

## 1. The question

Shouldn't it be possible to take a trained 800B-parameter model and produce a 3B or 8B model that is at least a strong head start, by projecting it into a smaller shape, rather than training the small one from scratch?

Three versions of the idea came up, each stronger than the last:

1. Treat each weight matrix as samples of a continuous function and resample it at lower resolution.
2. Treat the **activations** as the manifold. Push data through, fit the manifold, and project the model into its coordinates.
3. String all the layers end to end and treat the whole model as one continuous flow. Then re-discretize it into any architecture, including a different depth, width, and mix of attention and MLP.

## 2. The hard limit that no reparameterization moves

Before the three versions, the bound that caps all of them.

A model's knowledge is stored in its weights, and a tiny model has fewer bits of storage. An 800B model is not 100x redundant. Any method, continuous or not, can only remove what is redundant. Past that, the small model must give something up.

Three reasons the big model's weights stop being useful as a head start at large ratios:

- **Features are in superposition.** The model stores more features than it has dimensions, spread across all of them at once. Any projection to lower width keeps the high-variance directions and cuts the long tail. The long tail is where rare facts live.
- **Kept weights were tuned to work with cut weights.** At 2x, survivors still mostly make sense. At 10x, each kept channel has lost most of its partners, and after a few billion tokens of healing the init is barely better than random.
- **Weight spectra are flat.** The singular value spectrum of an LLM weight matrix has no small set of dominant directions. Effective rank is close to full rank. This is the direct measurement of "how low-dimensional is this layer," and the answer is "not very."

Rough map, from published prune-and-heal results (Minitron, Sheared LLaMA):

| size ratio | does weight surgery help? | what is needed |
|---|---|---|
| 1.5x to 2x | yes, a lot | layer / expert drop, light LoRA heal |
| 2x to 4x | yes | width cut, full fine-tune, billions of distillation tokens |
| 4x to 10x | a little | mostly a distillation run |
| 10x and up | nearly nothing | train a small model with the big one as teacher |

So "turn a massive model into a tiny one" is already how tiny models are made. It happens through the big model's **outputs** (distillation), not its weights. The question is whether the manifold view can make that cheaper.

## 3. Version 1: the weight matrix as a manifold

Treating a weight matrix as discrete samples of a continuous function, then down-projecting, is already rigorous math. A weight matrix is a linear map between vector spaces; the numbers are its coordinates in one basis. "Best lower-dimensional approximation of this map" is, by the Eckart-Young theorem, truncated SVD. So this version reduces to the methods SliceGPT, SVD-LLM, and ASVD already use. They get 20 to 30 percent off before quality falls hard, because of the flat spectrum above.

The "neuron index as a spatial coordinate" variant (resample the field at lower resolution) fails for a different reason: neuron order carries no information. Permuting neurons changes nothing. There is no neighbor structure to exploit. Sorting neurons by similarity and merging neighbors is just neuron merging, which works only where neurons are redundant.

Where the continuous view **does** hold, because there is real geometry:

- **Positions.** RoPE interpolation is exactly "fit a continuous function, resample it." It works because position has an order.
- **Depth.** A residual network is roughly a discretized ODE. Dropping layers and rescaling residual branches is "taking bigger time steps." This is why layer pruning survives at all. Adjacent layers are similar in **function**, not in weights, so averaging their weights fails.

## 4. Version 2: the activations as the manifold

This is a real improvement and has a paper: it is SliceGPT, almost exactly.

Push calibration data through. At each layer, collect the residual stream, run PCA, rotate the layer into those coordinates, and drop the low-variance directions. The rotation folds into the neighboring weights, so there are no new parameters. The activation spectrum is much steeper than the weight spectrum, because the residual stream is strongly anisotropic. This fixes the flat-spectrum problem. SliceGPT removes about 25 to 30 percent of the residual width on LLaMA-2 70B with a modest perplexity hit, mostly recovered by a light fine-tune.

Three things cap it short of "tiny":

1. **Loud is not important.** PCA keeps high-variance directions. Rare tokens and specific facts live in quiet directions. Cutting them costs little variance and a lot of knowledge. The fix is to weight directions by effect on loss (gradient or Fisher information), but then it is no longer a manifold fit.
2. **The manifold is curved and a weight projection must be linear.** The measured intrinsic dimension of LLM activations is low, often tens to a few hundred in a 4096-wide stream. But the manifold curls through the whole ambient space. A linear projection must cover its **span**, not its intrinsic dimension. Using the low intrinsic dimension requires a nonlinear encoder and decoder at each layer boundary, which are new parameters that need training.
3. **The residual stream forces one coordinate system.** Every layer reads from and writes to the same stream, so you cannot pick a different ideal subspace per layer without adapters between them.

The fully nonlinear version (autoencoder per layer, then small layers trained to map code at layer i to code at layer i+1) is progressive layer-wise distillation. It is the most sample-efficient known way to build a small model from a big one, because the student gets a supervised target at every layer, not just at the logits. It is still a training run.

## 5. Version 3: the whole model as a continuous flow

Think of a token's residual vector as a particle and depth as time. Each layer is one step of a vector field: it reads the particle's state and nudges it. The whole model is a time-varying field **F(x, t)**. Fit the field as continuous in t and you can re-discretize it with fewer steps. That is fewer layers.

### Where attention goes

The field has two terms:

- The **MLP** is the "self" term: how a particle moves based on its own state.
- **Attention** is the "coupling" term: how a particle moves based on all the other particles in the sequence.

In continuous form the model is a system of interacting particles, and there is real math under that exact name. Two things fall out:

- **Placement is measurable.** In the teacher, record the norm (or better, the loss-ablation effect) of the attention update and the MLP update at every layer. The curve of coupling strength over t says where the student needs attention and where cheap self-terms suffice.
- **"Any architecture" means "any coupling form."** Full attention, linear attention, Gated DeltaNet (GDN), and state-space models are different function classes for the coupling. Swapping one for another is a lossy projection of the coupling term. This has been done (transformer-to-Mamba distillation) and partly works, with training. The Qwen3.5 family's 3:1 GDN-to-attention interleave lives in this world already.

### What cannot be done

The coupling cannot be collapsed into a per-token manifold. A single token's trajectory is not a function of its own state; it depends on the whole sequence. The manifold that would be needed is over **sequences**, and that manifold is the model. The coupling has to stay a coupling.

### What the manifold view actually buys

One big thing. The teacher's field is defined on all of space, but it only matters on the thin manifold the data visits. The student only has to reproduce the field **there**. That is the real source of compression, and it is why distillation beats training from scratch.

But the smaller field is found by fitting it, which is training. The projection is a fit. What the framing changes is the supervision: every (x, t) on the manifold is a target. It turns "learn a model" into "trace a known field," which is a much easier optimization problem.

## 6. The recipe the framing suggests

1. **Width.** Pick the student width from PCA of the teacher's residual stream, at a chosen energy fraction. Fix one projection P shared across depth.
2. **Depth and layout.** Pick the student depth. Place attention (and choose attention vs. GDN per slot) using the measured coupling curve from the teacher.
3. **Trajectory matching.** Train each student layer to match the teacher's field at its time step, with teacher-forced inputs. For a student layer at non-integer teacher depth t, interpolate between the adjacent teacher layers' states.
4. **End to end.** Finish with logit distillation to fix compounding error.

Steps 1 and 2 are free and are the manifold part. Step 3 should converge much faster than end-to-end training. Step 4 is standard.

## 7. What is already well trod

| piece | prior work | status |
|---|---|---|
| Structured prune + distill, 2x to 4x | Minitron (NVIDIA), Sheared LLaMA | solid, measured |
| Weight SVD compression | SVD-LLM, ASVD | ~20 to 30% ceiling |
| Activation-PCA width cut | SliceGPT | ~25 to 30%, light heal |
| MoE to dense + distill | Kim et al. 2026 (arXiv 2605.28207) | beat dense-to-dense pruning by ~6 pts at ~4B tokens on Qwen3-30B-A3B; **the state of the art for the original MoE question** |
| Expert pruning with distillation | MoE-Pruner (2024) | keeps MoE form |
| Changing coupling type (attention to SSM) | Mamba-in-Llama (20B tokens, keeps 1/4 attention), Llamba via MOHAWK (~12B tokens, pure Mamba) | works, staged alignment |
| Staged alignment: matrices, then per-block hidden states, then end to end | MOHAWK | same width as teacher |
| Matching feature deltas, not just features | "Beyond Logits" (ACL 2025) | discrete form of the field view |
| Transformers as continuous flows | Flowing Through Layers (2025), Continuous-Depth Transformers (2026), adjoint-sensitivity analysis (2026) | theory and control, not compression |
| Which attention layers can be dropped | ShortGPT, "not all attention is needed" line of work | per-layer importance, not layout design |

## 8. What is not done, as far as a search could find

In order of confidence (and a missing search hit is not proof nobody has done it):

1. **Fractional-depth matching.** Every layer-wise method found maps student layer k to some teacher layer j, or aligns only the last few layers. None matches the student to the teacher's field at non-integer depth by interpolating between teacher layers.
2. **Deriving the student layout from measured coupling.** Prior work drops the least important attention layers or keeps every nth one. None measures the teacher's coupling strength over depth and uses it to design a fresh student, including choosing attention vs. GDN per slot. Attention span and entropy per layer would say which slots can take a cheap local coupling.
3. **Different width and depth at once, with one shared projection and per-layer targets.** MOHAWK keeps the teacher's width. Minitron changes width but trains on logits. The combination of a PCA-chosen width, a resampled depth, and dense per-layer supervision is the untested stack.

## 9. Experiment plan, with kill gates

Cheap gates first. Each one can end the idea in an afternoon, which is the point.

**Gate A: spectrum comparison (hours).** On one Qwen3.6 layer, plot cumulative energy of (a) the weight matrix's singular values and (b) PCA of the layer's input activations on a calibration set. Expect (b) far steeper than (a). The gap between them is the headroom the activation view buys. If (b) is also flat, stop.

**Gate B: interpolation test (hours).** Take teacher states at layers i and i+1, form the midpoint, feed it into layer i+1 (or i+2) and measure perplexity. If interpolated states behave like real states, the field is smooth enough to resample in depth. If perplexity blows up, fractional-depth matching dies here.

**Gate C: coupling curve (hours).** Per teacher layer, record attention-update norm vs. MLP-update norm, plus loss change when each is ablated. Also record attention entropy / effective span. This produces the student layout for free and is interesting on its own.

**Main experiment: the scaling curve nobody has published.** Teacher: Qwen3.6-35B-A3B. Student: ~3B dense, roughly 10x smaller in total params. Build it three ways under the same token budget:

1. from scratch,
2. logit distillation,
3. trajectory matching (recipe in section 6).

Measure quality vs. tokens at two or three budgets. The claim under test: dense supervision from the manifold lets a 10x smaller student reach a given quality with far fewer tokens than logit distillation. If (3) wins by a wide margin at 10x, that is the result the earlier papers stop short of. If it only wins at 2x, the manifold view is a nicer story for a known result.

Cost note: student training at ~10B tokens for a 3B model is on the order of 6 * 3e9 * 1e10 = 1.8e20 FLOPs, roughly a hundred H100-hours at reasonable utilization. Teacher activations can be cached once and reused across all three arms.

## 10. Sources

Compression and distillation:

- Pruning and Distilling Mixture-of-Experts into Dense Language Models (Kim et al., 2026): https://arxiv.org/pdf/2605.28207
- MoE-Pruner: https://arxiv.org/html/2410.12013v1
- The Mamba in the Llama (Wang, Paliotta, et al., NeurIPS 2024): https://arxiv.org/html/2408.15237v3
- Llamba: Scaling Distilled Recurrent Models (Bick et al., 2025): https://www.alphaxiv.org/abs/2502.14458.md
- Thinking Slow, Fast: Scaling Inference Compute with Distilled Reasoners (2025): https://arxiv.org/pdf/2502.20339
- Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation (ACL 2025): https://aclanthology.org/2025.acl-long.1125.pdf
- Layer-wise Curriculum Learning for Efficient LLM Compression (2026): https://arxiv.org/pdf/2609.19213
- A Survey on Knowledge Distillation of Large Language Models: https://arxiv.org/pdf/2402.13116

Continuous-depth and dynamical-systems views:

- Flowing Through Layers: A Continuous Dynamical Systems Perspective on Transformers (2025): https://arxiv.org/pdf/2502.05656
- Continuous-Depth Transformers with Learned Control Dynamics (2026): https://arxiv.org/abs/2601.10007
- An Adjoint-Sensitivity Framework for Lost-in-the-Middle Phenomena in Causal Residual Transformers (2026): https://arxiv.org/pdf/2607.17696
- Towards Understanding the Universality of Transformers for Next-Token Prediction (2024): https://arxiv.org/pdf/2410.03011

Named from memory, not re-fetched in this session (verify before citing):

- SliceGPT (Ashkboos et al., ICLR 2024), SVD-LLM, ASVD: activation-aware and weight SVD compression.
- Minitron / Compact Language Models via Pruning and Knowledge Distillation (NVIDIA, 2024).
- Sheared LLaMA (Xia et al., ICLR 2024).
- MOHAWK (Bick et al., 2024): the staged alignment procedure behind Llamba.
- ShortGPT and the "not all attention is needed" line on dropping attention layers.
- Geshkovski, Letrouit, Polyanskiy, Rigollet: transformers as interacting particle systems.
- NeRN and implicit weight fields: continuous representations of weights, 2x to 5x on CNNs.

In-repo context: `writeups/qwen36-27b-a2.8b.md` (layer prune + heal survived), `writeups/qwen38-flash-next-131b-a6b.md` (layer prune fatal, n-gram table prune free), `aggressive-multiple-layer-prune.md` (lm_head floor, GDN/attention split).
