# Plan: whole-network distillation of the 2B-to-0.8B build, tokens against quality

Status: proposed, not started (2026-10-21).
Experiment: `manifold-compression/`. Background: `findings.md` ("Damage isolation", "9B to 4B shape").

## Question

Every cut without training breaks chat, and more fitting data does not help (damage isolation). Does whole-network distillation repair it? How many tokens does it take? This is the first run with a real repair budget. Earlier: 2M tokens did not fix it.

## Why the 0.8B shape

- Student is 0.75B, teacher is 2B. Cost per token is about 5 times lower than 9B to 4B.
- A stock 0.8B exists as a baseline (smoke coherence 3.75), and the stock 2B is the teacher (4.20).

## Method

1. **Rebuild the student** on the pod: `build_projected.py` (2B to 0.8B shape, layer-wise healing, tail polish). About 35 minutes. The old checkpoint was lost with the deleted pods.
2. **Distill the whole network** with forward KL to the 2B's output distribution on fixed corpus text. No generation during training, so it runs fast.
   - Data: UltraChat in the chat template (most of it), plus wiki, code and math rows. About 100M tokens of text; rows are shuffled and not repeated.
   - Full fine-tuning of the student, bf16, AdamW, lr 2e-5 with cosine decay, about 32K tokens per step, KL computed in chunks over the full vocabulary.
   - Teacher in bf16, forward pass only.
3. **Checkpoints** at 5M, 20M, 50M and 100M tokens, each saved in bf16 (1.5 GB).
4. **Measure** each checkpoint:
   - KL to the 2B on held-out rows, logged on the pod.
   - Smoke test (40 prompts, greedy, thinking off, 400 tokens) on the local GPU after download: convert to GGUF, generate with llama.cpp, judge blind with Haiku. Stock 0.8B and stock 2B go into the same judge round as anchors.

## Cost

| Item | Estimate |
|---|---|
| Rebuild | about $0.25 |
| 100M tokens of distillation | about 4 hours on an A40 at $0.4 per hour, about $1.60; about 2 hours on an A100 at $1.79 per hour, about $3.60 |
| Evaluation | $0 (local) |
| Total | about $2 to $5 |

These are estimates from FLOP counts (student forward and backward plus teacher forward, about 8.5 GFLOP per token) at 40% hardware use. The first 1M tokens give the real rate. If the rate is worse than half the estimate, I stop and report before going on.

## Decision rules

Judge noise is about 0.1; a gap below 0.5 does not count.

| Result | Meaning |
|---|---|
| Coherence rises with tokens and reaches within 0.5 of stock 0.8B (3.75) | the cut is repairable; next scale it up (9B to 4B, then larger) |
| Coherence rises but is still below 3 at 100M | repair works but needs more tokens; fit the curve and estimate the budget |
| Coherence stays flat (under 2.2) from 5M to 100M | the projected start does not help; stop this line |

## Risks

- The 2B and student share a tokenizer and chat template, so corpus text needs no conversion.
- UltraChat answers come from another model, not from the 2B. The teacher's probabilities still supply the targets. If chat style looks off, a second run can use the 2B's own answers.
- GPU memory: the KL over a 248K vocabulary must run in chunks of about 4K tokens.
- Pod rules from the last run: set `terminate-after` as an ISO time, but do not rely on it (it did not stop the last pod). I will delete the pod myself as soon as the checkpoints are downloaded, and I will not wait on dibs while the pod runs.

## Not in scope

Larger models, the 100M-token run at 4B shape, on-policy distillation.
