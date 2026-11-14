# Notes for a Hugging Face model card

*Started 2026-10-29 at the user's request, for the slimmed Qwen3.5-9B (dense → MoE) if it is ever uploaded. Not a card yet: collect here what the card should say, with sources.*

## Relation to CMoE

The expert layout follows CMoE (Pei et al., 2025). The router, the scoring of neuron activity and the healing differ, and the model also gets a smaller vocabulary and a low-rank DeltaNet cut.

**CMoE in brief.** CMoE turns a dense model's MLPs into experts without training:

1. On a little calibration text, it marks each MLP neuron as active for a token when it is among the 10 largest by magnitude.
2. The neurons active most often become an always-on shared expert.
3. The rest are grouped by balanced k-means on their activation patterns into equal-size routed experts.
4. The router is not trained. Each routed expert has one representative neuron, the member nearest its cluster's centre, and the router scores an expert with that neuron's own gate and up weights.
5. An optional short LoRA fine-tune recovers quality.

CMoE was tested on Llama-2-7B and Llama-3-8B. Its 25% setting is 2 shared experts plus 2 of 14 routed experts.

| | CMoE | This model |
|---|---|---|
| Shared expert | the neurons most often active | same idea: 1,536 of 12,288 neurons per layer, half of the active budget |
| Routed experts | balanced k-means on activation patterns (L1 distance, one assignment pass) | same idea, finer: 64 experts of 168 neurons, 9 run per token. Balanced k-means on co-activation patterns, cosine similarity, up to 20 passes. |
| When a neuron counts as active | among the 10 largest activations of a token | among the top 25% of a token's neurons by contribution: activation size × the norm of the neuron's output weights |
| Router | built from each expert's representative neuron, no training: top-k of \|silu(x·g) · (x·u)\| | a linear router per layer, trained to predict each token's share of contribution in each expert, and trained further during healing |
| Expert weighting | on/off; fine-tuning learns a scale (1 + softmax · u) and a load-balancing bias | on/off, no scale, no load balancing |
| Healing | LoRA rank 8, alpha 32, on 2,048 WikiText-2 samples | LoRA rank 64 on every body projection plus the routers, on 8.2M tokens from 8 English and code domains. Loss: KL to the dense model's next-token distribution plus half-weight cross-entropy on the text. |
| Model and other cuts | Llama (all attention) | Qwen3.5-9B (DeltaNet and attention hybrid). Also cut: a 64K vocabulary, a low-rank DeltaNet, an output head that scores recently used rows first. |

**Router comparison, before healing** (25% of MLP neurons active, the same experts, the v2 benchmark's KL from the dense model; `cmoe_router.py`, results in `results/moe-carve-v2-20261029/cmoe/`):

| Router | KL | experts found |
|---|---:|---:|
| perfect (reads the real activations) | 0.821 | 100% |
| ours, linear, trained | 1.288 | 68% |
| CMoE's, from representative neurons | 2.475 | 26% |
| *for scale: one fixed set of neurons, no routing* | *2.21* | — |

On our experts, CMoE's router is far worse than the trained one: +1.19 nats (95% interval +1.15 to +1.23). It is even worse than keeping one fixed set of neurons for every token. It finds 26% of the right experts, where picking at random would find 14% (9 of 64). It does best in the last layer (48%).

This tests CMoE's router on our experts, not CMoE as a whole. CMoE builds its router for its own clusters: fewer and larger experts (2 of 14 routed at 25%), grouped by its own activation marker, and its fine-tuning then learns a scale and a load-balancing bias. With coarser experts, picking the right ones is easier. Run on one A40 for $0.07 (`run_cmoe_pod.py`).

## Results after the healing pilot

The pilot was trained on 8.2M tokens: 3.2 hours on one RTX 6000 Ada, about $3.70. Details are in `results/moe-carve-v2-20261029/README.md`, section 8.

| | dense | slimmed, before healing | slimmed, healed |
|---|---:|---:|---:|
| KL from dense (v2 benchmark, 10 domains) | 0 | 1.490 | 0.612 |
| bits per byte, 64K vocabulary | 0.656 | 1.061 | 0.734 |
| greedy agreement with dense | 100% | 55.2% | 71.1% |
| GSM8K, 16 prompts, greedy, 2,048 tokens | 9 | — | 6 |
| MMLU, 1,135 questions, zero-shot (section 19) | 67.5% | — | 42.6% |
| ARC-Challenge, 600 / ARC-Easy, 400, zero-shot | 53.0% / 73.3% | — | 32.7% / 54.2% |
| HellaSwag, 600, zero-shot | 69.3% | — | 48.0% |
| parameters stored, total / non-embedding | 8.95B / 6.92B | 6.93B / 6.39B | 6.97B / 6.43B |
| parameters active per token, total / non-embedding | 7.94B / 6.92B | 2.95B / 2.76B | 2.95B / 2.80B |

**Held-out domains.** The two benchmark domains kept out of the healing mix, biomedical and argument, improved by the same share as the domains the model trained on.

**Multiple-choice tests** (section 19: plain text, zero-shot, the same 64K-token text for every 9B row). The healed model keeps 0.45 of the dense model's accuracy above chance (95% interval 0.41–0.50). Scored the same way, Qwen3.5-2B gets 53.4, 39.5, 60.5 and 59.7, above the healed model on all four tests. With every neuron running, the same healed model scores above the dense model (MMLU 71.5), so the loss comes from running 25% of the MLP neurons. This setup scores Qwen3.5-2B 2 to 6.5 points below its fp32 scores without a leading <|endoftext|>, and probably the other models too, so a card should report these numbers only beside the dense model and the 2B scored the same way.

**What is left after healing** (section 9 of the results README). Of the remaining 0.612:
- the learned router accounts for 0.095 (a perfect router gives 0.517);
- running only 25% of the MLP neurons accounts for 0.275 (every neuron gives 0.242);
- the rest of the model accounts for 0.242.

So after healing, the router is no longer the main cost, unlike before healing (see the router comparison above).

With every neuron running, the healing adapter itself adds 0.087: it was trained to make up for missing neurons and overshoots when they all run (every neuron without the adapter: 0.155). So the rest of the model's share is about 0.155 for the DeltaNet cut and the vocabulary, plus this overshoot (section 10).

**Known problem, not ready to ship.** In 7 to 10 of 16 greedy GSM8K generations, the healed model got stuck repeating itself; the dense model never did. It loops as often with fp32 DeltaNet state and with a perfect router, so neither causes it. The cause is running only 25% of the MLP neurons: with every neuron running, the same healed model never loops, finishes 14 of 16 and gets 10 right (dense: 9). Without the healing, the other cuts alone (low-rank DeltaNet, 64K vocabulary) never finish an answer, so the healing is needed. Switching experts on and off makes the 25% model 18 times as sensitive to tiny numerical changes as the dense model, but fading experts in and out (a soft edge on the router) removes 65% of that and the loops stay (7 of 16, section 11). The skipped neurons carry 42% of the MLP output on average, and the neurons that run can predict only about a quarter of that. Folding it into their output weights after healing (no training) made the model worse: KL 0.743, 10 of 16 looping (section 12). Subtracting that fold instead is far worse (KL 1.97, 13 of 16 looping), but a quarter of it lowers KL to 0.597 and looped in only 4 of 16, too few prompts to be sure (section 13). On 48 new GSM8K prompts the healed model loops in 18 (section 14). Better-fitted folds lower KL to as little as 0.581 (fitted layer by layer, aiming at the every-neuron model's layer outputs) and still loop in 15 or 16, no better than chance; which prompts loop changes with any small change. With Qwen's recommended sampling (temperature 1, top-p 0.95, top-k 20, presence penalty 1.5) nothing repeats, but 10 of 48 answers still go round in circles until the 2,048-token cap without repeating, and it gets 14 right against 22 greedy. On-policy distillation from the dense model (7.2 hours, 2,496 prompts, section 15) made it worse: 20 of 48 looping, 6 finished, 7 right, and KL 0.644. The student took on the dense model's long, self-checking answers and could not finish them at 25%. On the same prompts, the healed model with every neuron finishes 45 and gets 40 right (dense: 23 finished within the cap, 28 right). Distilling from that every-neuron model instead, trained only on the answers it finishes (section 16), improved every count: greedy 13 looping, 33 finished, 26 right; sampled 0 looping, 27 right (dense, sampled: 29). The loop reduction is not yet significant (p = 0.19), and KL rose to 0.651, spread over all domains. Section 17 carried that run on for 34 more rounds, with healing text mixed back in, and checked it on 96 prompts. Greedy, it loops in 17 against the healed model's 36 (p = 0.002) and gets 54 right against 42. Its KL is 0.602, below the healed model's 0.612. Sampled, it never repeats and gets 47 right (healed: 46), but it finishes only 61 (healed: 81); the other 35 run to the 2,048-token cap. With a 4,096-token cap (section 18), 19 of those 35 finish: sampled, it finishes 80 and gets 52 right. The dense model finishes 78 and gets 73 right; the every-neuron model finishes 95 and gets 64 right. Greedy, the longer cap turns more of the distilled model's long answers into loops (27 of 96), and the dense model also loops in 7 of 96, so the dense model is loop-free only within 2,048 tokens. A card must report the loops unless later training fixes them, and should report both greedy and sampled results.

## Sources

- CMoE: Pei et al., *CMoE: Fast Carving of Mixture-of-Experts for Efficient LLM Inference*, [arXiv 2502.04416](https://arxiv.org/abs/2502.04416); code at [github.com/JarvisPei/CMoE](https://github.com/JarvisPei/CMoE). The router and representative-neuron details above were read from the code: `construct_moe` and `construct_experts_k_means` in `CMoE_utils.py`, `Router` in `CMoE_model.py`.
- Our side: `moe_carve.py` (carving, learned router), `slim_batch.py` (the other cuts), `slim_heal.py` (healing), results in `results/moe-carve-v2-20261029/README.md`.
- Other prior work to cite on the card: LLaMA-MoE (dense to MoE with continued pretraining) and MoEfication.
