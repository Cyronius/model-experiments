# M1: Can a dense model be split into experts by domain?

*Experiment run 2026-10-28 on Qwen3.5-9B, local CPU, $0. Code: `moe_carve.py`; rules and predictions: `moe_carve_rules.json`; data: this folder.*

## In short

- **Domain is the wrong way to split the MLPs into experts.** Experts built from neurons that fire together beat domain experts at every size we tried. At 25% of MLP neurons active, the output drifts 0.95 nats from the full model with co-firing experts and 1.59 with domain experts.
- **A domain router barely beats a plain prune.** Even when it is told the true domain of each document, a domain expert recovers only about a quarter of what a plain prune loses.
- **The useful sparsity is per token.** If every token could keep its own most useful 25% of neurons, the drift would be only 0.10. Each token needs a small, specific set of neurons, but the set changes token to token, not domain to domain.
- **Your shared-expert idea is partly right.** In the middle layers about half of each domain's important neurons are shared by all eight domains. In layers 20-26 only 8-22% are, so those layers are where the domains really split.
- **Nothing here works without healing.** The best carve at 25% more than doubles perplexity. That matches the published methods (CMoE, LLaMA-MoE), which all train after carving.

**Recommendation:** build experts from co-firing neurons, not domains, and test the two things this run assumed away: a router that has to predict its choice, then healing. Details under [Recommendations](#recommendations).

## The question

Your thesis was:

1. measure activations across several domains;
2. use those measurements to split a dense model into smaller experts;
3. put neurons that fire in every domain into a shared expert.

The real target is Qwen3.8-27B as an MoE with fewer experts per layer than Qwen3.6-35B. We tested it first on Qwen3.5-9B. It has the same architecture as Qwen3.8-27B, and you no longer trust 2B models as stand-ins for big ones.

## What "carving" means here

Each layer of the 9B has an MLP that is 12,288 neurons wide. For every token, each neuron produces one number and writes it back into the model's main signal. The MLPs hold about 4.8B of the model's weights.

An MoE runs only some of those neurons for each token. To measure how much a neuron matters for one token, we used its **contribution**: the size of its activation times the size of what it writes back (`|a_i| x ||W_down[:, i]||`).

This experiment changed no weights. Every version keeps the original 9B and only chooses which neurons run for each token, in all 32 layers at once. That isolates one question: which way of choosing neurons does least damage? It says nothing about how much training could repair; that comes later.

## How the experiment worked

**Data.** Eight domains:

- chat (UltraChat);
- code (Python standard library);
- math (MATH problems);
- Wikipedia in English, French and Chinese, kept as three separate domains;
- biomedical papers (TREC-COVID);
- argument essays (Touché).

All text was cut into documents of 512 tokens:

- **Calibration:** 40 documents per domain, used to measure the neurons.
- **Evaluation:** 10 held-out documents per domain.
- **Copy test:** 16 extra documents that repeat a passage. These check that the model can still copy text it has already seen, which is where pruned models often break first.

**Measuring.** Running the 9B over the calibration documents recorded three things for every neuron in every layer:

- its average contribution in each domain;
- how often it fires;
- which neurons fire together on the same token.

**Five ways to choose neurons.** At a given size every method runs exactly the same number of neurons per token, so the comparison is fair.

| Method | How it picks the neurons | What it stands for |
|---|---|---|
| **Plain prune** | One fixed set for every token: the neurons with the highest average contribution | The simplest baseline. No experts. |
| **Domain expert, true label** | The set that matters most for the document's domain, chosen by its true label | Your thesis with a perfect domain router |
| **Best domain expert per token** | For each token, whichever of the eight domain sets holds most of that token's contribution | Your thesis with a perfect router that can switch mid-document |
| **Co-firing clusters** | A shared expert (the neurons that fire most often, half the budget) plus 64 equal clusters of neurons that tend to fire together; each token gets the clusters with most of its contribution | The CMoE recipe, the main published competitor |
| **Each token's own top neurons** | Every token keeps its own highest-contribution neurons | The best any router could do with no fixed experts |

At 25% active, the co-firing version has a 1,536-neuron shared expert plus 64 experts of 168 neurons, and each token uses 9 of the 64.

**Sizes tested.** 50%, 25% and 12.5% of each MLP active.

**Score.** For every token we compared the carved model's prediction of the next token with the full model's prediction. KL divergence measures that gap in nats:

- **0** means identical.
- **About 0.1** is barely noticeable.
- **1 or more** is heavy damage.

We also report the rise in loss on the true text (NLL) and the worst 1% of tokens on the copy test.

**Fairness rules.** I wrote six predictions and the pass/fail criteria into the rules file before any measurement. Evaluation documents were never used to build the experts.

## Results

### The main comparison

Drift from the full model (KL, lower is better); the worst 1% of the copy test is in brackets:

| Neurons active | Plain prune | Domain, true label | Domain per token | **Co-firing clusters** | Each token's own top |
|---|---|---|---|---|---|
| 50% | 1.12 (1.53) | 0.75 (1.73) | 0.57 (0.58) | **0.34** (0.41) | 0.016 (0.01) |
| 25% | 2.50 (4.08) | 1.87 (3.71) | 1.59 (2.52) | **0.95** (1.40) | 0.10 (0.06) |
| 12.5% | 3.64 (8.61) | 3.01 (7.47) | 2.75 (6.54) | **2.13** (3.87) | 0.31 (0.40) |

Rise in loss on real text at 25%:

| | Plain prune | Domain, true label | Domain per token | Co-firing clusters | Each token's own top |
|---|---|---|---|---|---|
| NLL (nats) | +2.36 | +1.71 | +1.44 | +0.81 | +0.08 |
| Perplexity | x10.6 | x5.5 | x4.2 | x2.2 | x1.08 |

What the comparisons show:

1. **Co-firing clusters beat domain experts.** This was the main test, set in advance. At 25% the gap is 0.64 nats, with a 95% interval of 0.59-0.69, so it is not noise. The clusters also have the better copy tail, and they win in all eight domains at all three sizes.
2. **The domain router fails its own test.** I set the bar at "recovers at least half of what a plain prune loses". With the true label it recovers 25%, and switching domain per token raises that to 36%.
3. **The ceiling is far away.** Each token's own top 25% of neurons gives 0.10 nats, nine times better than the best fixed experts. The 9B's MLPs really are sparse for each token. The loss comes from forcing neurons into fixed groups, and grouping by domain loses the most.

### By domain (25% active)

| Domain | Plain prune | Domain, true label | Co-firing clusters | Each token's own top | Gain of domain expert over prune |
|---|---|---|---|---|---|
| chat | 3.00 | 2.58 | 1.31 | 0.11 | 14% |
| code | 2.24 | 1.48 | 0.92 | 0.17 | 34% |
| math | 1.31 | 0.82 | 0.50 | 0.05 | 37% |
| Wikipedia | 2.45 | 2.20 | 0.91 | 0.10 | 10% |
| biomedical | 2.55 | 1.84 | 0.72 | 0.05 | 28% |
| argument | 2.63 | 2.11 | 0.86 | 0.08 | 20% |
| French | 2.75 | 1.98 | 1.11 | 0.10 | 28% |
| Chinese | 3.09 | 1.92 | 1.24 | 0.16 | 38% |

Domain experts help most where the text is unusual: Chinese, math and code. They help least for Wikipedia and chat, which look like the model's ordinary text. Even at their best, domain experts stay well behind co-firing clusters.

### Where the domains really split

For each layer we took each domain's top 25% of neurons and asked how many of them every domain shares (the "shared core").

| Layers | Shared by all 8 domains | Average overlap between two domains |
|---|---|---|
| 0-3 | 23% | 0.41 |
| 4-9 | 27% | 0.41 |
| **10-18** | **47%** | 0.55 |
| **19-27** | **17%** | 0.31 |
| 28-31 | 35% | 0.48 |

- **Middle layers are mostly shared.** This is where a large shared expert would fit, as your thesis predicted.
- **Layers 20-26 split by domain.** Only 8-22% of neurons are shared there.
- **The split is real.** Two halves of the same domain overlap 0.72; two different domains overlap 0.45. In layers 20-26 the figures are 0.66-0.71 against 0.27-0.39.

Which domains use similar neurons (overlap between pairs, 25% active, averaged over layers):

| Pair | Overlap |
|---|---|
| Wikipedia and French | 0.58 |
| chat and argument | 0.57 |
| Wikipedia and Chinese | 0.54 |
| Wikipedia and chat | 0.50 |
| French and Chinese | 0.49 |
| code and math | 0.46 |
| code and anything else | 0.35-0.38 |

The model groups text by **kind of writing** more than by language. Encyclopedia prose in three languages shares more neurons than chat shares with English Wikipedia. Code and math sit apart from everything else.

**One odd result.** Co-firing clusters keep only slightly more of each layer's total contribution than the plain prune (37% against 34%), yet their drift is 2.6 times lower. How much contribution a method keeps is a poor guide to how well it works. What matters is that each token keeps the particular neurons it needs.

### How my predictions did

| Prediction (written before the run) | Outcome |
|---|---|
| Plain prune drifts more than 1 nat at 25% | Held (2.50) |
| Each token's own top neurons drift less than 0.3 at 25% | Held (0.10) |
| Domain expert beats the plain prune, but by less than half | Held (25% better) |
| Choosing among domain experts per token adds little | **Failed:** 15% better, clearly not zero |
| Co-firing clusters beat both domain methods at every size | Held |
| More than half of each domain's neurons are shared by all eight | **Failed:** 30% overall (47% in the middle layers) |

## What this means for the thesis

- **Measuring by domain is useful; splitting by domain is not.** The measurements found real structure: a large shared core in the middle layers and domain-specific neurons in layers 20-26. But a token's needs track its neighbours and its role in the sentence more than the topic of the document. Neurons that fire together are a much better expert boundary.
- **The shared expert is worth keeping.** The winning recipe already uses one, made of the most often active neurons. A refinement would size it by layer, large in layers 10-18 and small in 20-26.
- **Domains could still be a coarse layer on top.** In layers 20-26, co-firing experts could be built separately within each domain group. That is a hypothesis to test, not a result.
- **For a small domain expert (question 8),** keeping one domain's neurons is a better starting point than a generic prune. Math at 25% drifts 0.82 instead of 1.31; Chinese 1.92 instead of 3.09. It still needs healing.

## What this experiment did not test

- **No real router.** The per-token methods chose neurons by looking at the activations they were about to keep, which a real MoE cannot do: it has to predict the choice from the layer's input before running the experts. So these methods save no compute yet, and their numbers are upper bounds. The true-label domain expert is the only realistic router here, and it is the weakest.
- **No healing or refitting.** Every number is "damage before repair".
- **The 9B, not the 27B.** The architecture matches, but the 27B result is still to confirm.
- **Smaller storage was not the goal.** An MoE keeps all of its weights and only runs fewer per token; it cuts compute, not size. Shrinking the experts themselves is a separate step.
- **Small evaluation set.** 10 documents per domain of 512 tokens each. The intervals are tight because the differences are large, but rare tasks and long contexts were not tested.
- **KL and loss, not task accuracy or generation quality.**

## Recommendations

**1. Learned routers (next; local, $0, about a day).**
For each layer, train a small router that predicts from the MLP's input which co-firing experts to run. It costs about 0.26M weights per layer, which is tiny. Compare it with CMoE's training-free router at 50% and 25% active.
- *Pass:* the learned router keeps the 25% drift within about 1.3 times the oracle's 0.95.
- *Add:* a hybrid arm where layers 20-26 use co-firing experts built within domain groups. This is the one place your domain idea could still add something.

**2. Healing pilot on the best carve (GPU; needs a cap from you).**
Carve, add the learned routers, then train briefly to match the full 9B's outputs. Measure the curve of quality against training tokens, and compare it with simply fine-tuning a plain prune of the same active size. This tells us whether the MoE carve is a head start (your questions 2 and 4). A first pass of about 100M tokens on one rented GPU should cost tens of dollars. I'd scope it and give you a number before asking.

**3. Confirm on Qwen3.8-27B (local, $0).**
Repeat the training-free measurements and the router test on the real target. Its 55.6 GB of weights fit in RAM. This shows whether the 9B's pattern holds at 27B before we spend GPU money on the 27B.

**4. The Flash-Next moonshot (still waiting on your cap).**
Flash-Next is already an MoE with 512 small experts per layer. Two lessons carry over:
- Merge its experts by which ones fire together, not by domain.
- Expect the unhealed result to be very poor. Shrinking it to 9B means cutting the experts about 20 times. Here, an 8-times cut of a dense MLP (the 12.5% row) already costs 2.1 nats even with perfect routing, so the healing curve, not the unhealed score, is the real measurement.

There are two ways to run it:
- **GPU:** I suggested a $50 cap, stopping after the unhealed check for your go-ahead on healing.
- **Local:** about a day at $0, plus a 350 GB download.

## Files

| File | What it holds |
|---|---|
| `../../moe_carve.py` | All stages: fetch, prepare, collect, build, evaluate, report |
| `../../moe_carve_rules.json` | Settings, predictions and pass/fail criteria, frozen before the run |
| `../../test_moe_carve.py` | 7 tests (expert sizes, equal budgets, exactness at full width, batching) |
| `corpus.json` | The 416 documents and their calibration/evaluation split |
| `build.json` | Expert structure and domain overlap by layer |
| `evaluation.jsonl` | Per-document results for all 16 versions |
| `report.json` | Totals, per-domain results, bootstrap intervals and verdicts |
| `../../findings.md` | The short technical entry ("M1") |

Run time on the shared CPU: 42 minutes measuring, 5 minutes building experts, 2 hours 38 minutes evaluating.
