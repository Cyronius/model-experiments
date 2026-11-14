# Manifold projection roadmap

*Agreed 2026-10-27 in the Manifold pair session (paird 245–262). This replaces the allocation and healing line (P0/P1) as the main direction; G1 finishes as a baseline for the fixed linear fold. Not a protocol: each stage freezes its own.*

*Update 2026-10-27: the pair session has ended and Claude continues alone. H0b ran under a frozen protocol (`h0_rules.json`, `results/h0-native-mlp-20261027/`): the gate FAILED at all three sites. Native functions fitted on the activations beat restricted weights above the chart floor, but at k = 768 the PCA chart's own floor costs more than deleting the MLP. The follow-up `h0_floor_probe.py` found the floor came mostly from sharing one basis across layers, but even with per-layer bases and exact blocks, squeezing the whole stack costs 0.56 nats/token at k = 1536 and 2.2 at 768. Before healing, the narrow residual is the obstacle, not the block functions, so the programme's real test is H1 (see findings.md). G1 finished with single-reviewer science (see findings.md).*

*Update 2026-10-29: **branch closed.** The user ended the projection line after repeated negative results. M1 finished (`results/moe-carve-20261028/README.md`); dense→MoE conversion continues in a separate session. H1, the M1 follow-ups below and the Flash-Next moonshot were not started; no GPU cap was granted.*

*Update 2026-10-28: the user set two concrete lines; they replace H1 as the next steps.*

- **M1, dense → MoE by domain (questions 1 and 8).** Target: Qwen3.8-27B dense. Measure MLP neuron contributions across domains; neurons used by every domain form a shared expert, the rest form domain experts. Tested first on Qwen3.5-9B, which has the same `qwen3_5_text` architecture (Qwen3.8 has no 9B). The user no longer trusts 2B models as stand-ins for large ones. Prior work to beat: CMoE (shared expert by activation rate plus co-activation clusters) and LLaMA-MoE. First run: `moe_carve.py`, rules `moe_carve_rules.json`, results `results/moe-carve-20261028/`: training-free, oracle routers, whole-stack KL at 50/25/12.5% of MLP neurons active. **Result:** co-activation experts beat domain experts at every budget (25% active: KL 0.95 vs 1.59); a perfect document-level domain router recovers only a quarter of what a prune loses; each token's own top 25% of neurons gives KL 0.10, so the MLP is sparse per token but not per domain. Next: learned routers, then healing (see findings.md).
  - *Update 2026-10-29, dense→MoE session.* The user's plan for the 9B, then progressively larger models (Qwen3.8-27B next, same architecture and tokenizer): keep English, symbols and code only; per-token co-firing experts with a learned router, sizes possibly varied by depth; a 64K vocabulary; one output head whose recently used rows are scored first (the drafter scores the same rows) with a miss detector falling back to the rest; DeltaNet state stored in 16 bit; a cut to the DeltaNet weights; then healing. Benchmark v2 (`moe_carve_rules_v2.json`): ten English and code domains. Each step was first measured on its own, unhealed, in one RunPod A40 batch (`slim_batch.py`, `slim_rules.json`, `run_slim_pod.py`; results `results/moe-carve-v2-20261029/`, summary in its README).
- **Moonshot, ~128B → ~9B.** Teacher: Qwen3.8-Flash-Next (125B with 6B active, plus a 51B n-gram table and 4B MTP; 48 layers, residual 2560, 512 experts of width 640, 10 routed + 1 shared). About 121B of it is routed experts, so a 9B version keeps the residual, attention and n-gram table and shrinks the experts about 20× (about 25 per layer), measured and merged by domain and co-activation and refitted on the teacher's activations. References: stock Qwen3.5-9B and Qwen3.8-27B (same tokenizer). Needs a newer transformers (`qwen4_exp`), about 350 GB of weights, and GPU time; waiting on the user's dollar cap.

## The goal

Take a large model, measure its activation manifold block by block, fit compact functions on it and re-express them in a smaller architecture. The long-range target is a 128B model projected to about 9B. Healing (training after the projection) is an accepted repair, not a failure; a model that works without it would be a bonus.

Compression is reported two ways, side by side: on all parameters (total) and on non-embedding parameters, which leave out both vocabulary tables (the input embedding and the output head). Each is given for stored weights and for weights active per token; an embedding lookup reads one row, so it counts as zero active. *(Changed 2026-10-29 at the user's request; until then only non-embedding parameters counted and the vocabulary tables were outside the test.)*

## What the user wants to learn

1. Can manifold functions turn a dense model into an MoE?
2. Can the technique beat distillation on training budget for a smaller model?
3. Can it produce a model smarter than comparably sized models?
4. Does it at least give a head start that cuts pretraining cost against other methods?
5. Can a small model be projected upward into a larger one? (Less practical.)
6. Can model weights be combined with projections?
7. Can pieces of a large model be projected and bolted onto a smaller one?
8. Can domain-specific measurements give a small domain expert, discarding the rest?

## Stages

| Stage | Questions | What it tests | Compute |
|---|---|---|---|
| **H0** (now) | 6, groundwork for all | Strict k-wide bottleneck function fits on SmolLM2-1.7B (Llama architecture). Native MLP blocks in a PCA or learned chart against the restricted-weight baseline (the current line). Selects initializers for H1; not a kill gate for the programme. | Local CPU, $0 |
| **H0c** | groundwork | Chains of adjacent native blocks, including attention with full sequence and KV conditioning. | Local CPU |
| **H1** | 2, 4 | Primary mechanism test: same native architecture and initialization; logit-only training against block targets plus common logit healing. Initialization ablation separately, or a small 2×2. Quality against all counted compute, including teacher collection, chart and block fits, and healing. Pre- and post-healing curves. | GPU; needs explicit user approval and a cap |
| **H1 evaluation** | 3 | Best H1 student against public models of similar size on a fresh frozen suite, built after any H0-based initializer selection. Public models are quality references, not training-efficiency controls. | After H1 |
| **Domain branch** | 8 | Charts and fits calibrated on one domain, against a same-size domain-distilled or domain-fine-tuned baseline with matched in-domain token exposure and total compute. Out-of-domain degradation is reported as a tradeoff, not required. A lower task-relevant dimension for one domain is a hypothesis to measure, and in-domain gain alone does not show that the projection discarded information selectively. H0 keeps balanced held-out domain groups as a first look; the branch freezes separately. | Local first |
| **MoE branch** | 1 | Region-partitioned compact experts fitted on the MLP-input manifold. Reports total stored scalars, active operations, router overhead and rare-domain quality. | Local first |
| **Stitch branch** | 7 | Project blocks of a larger model into a smaller host from the same family. Same host, a matched-size adapter baseline and host regression protection, all after equal healing. | Local first |
| **Upward** | 5 | Later. | — |

Qwen3.5 (hybrid attention and DeltaNet) is a later replication. The architecture swap cannot show that the hybrid design caused earlier failures, because tokenizer and training also differ.

## Rules carried forward

- Every learned chart reconstructs only from its declared k-wide code. The earlier warp arms read the exact PCA coordinates as well, so they were not k-wide (see the correction in `findings.md`).
- Inference carries compact states directly. A wide decoder or teacher continuation is a labelled diagnostic.
- Hold out whole documents. Report tails (p99, copy task) next to means.
- Count all work: teacher collection, chart and block fits, healing, and setup.

## Prior work to beat or cite (verify before citing)

MoEfication and sparse upcycling (dense to MoE); Minitron and Sheared-LLaMA (pruning with distillation); SliceGPT (the linear fold); model stitching (bolt-on); Net2Net and LiGO (growth); D-Pruner (domain pruning).
