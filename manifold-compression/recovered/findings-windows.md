# Manifold compression: what we have learned so far

Status as of 2026-10-18. This collects every result, build number and lesson from the experiment so far. The first-pass tables for the 0.8B are in [README.md](README.md); the idea backlog is in [`../.claude/plans/manifold-compression-ideas.md`](../.claude/plans/manifold-compression-ideas.md); the background reasoning is in [notes.md](notes.md).

Written for: someone picking up the experiment cold, who needs the numbers and the reasons behind each next step.

## The question and the win condition

Can a big model's activation manifold be treated as a set of approximating functions and projected onto a smaller architecture, giving a usable small model for a few dollars instead of thousands? "Usable" means it keeps chat, tool use and maths well enough to ship, not just a low perplexity.

## What the activations look like

- **Intrinsic dimension is low.** About 10–16 on both models, rising with neighbourhood size and never levelling off. The estimator is biased low at 3,000 points, so this is a loose lower bound.
- **PCA captures most of the variance in few directions.** The 2B keeps 0.76 / 0.89 / 0.96 of variance at k = 512 / 1024 / 1536 of 2048 (median site).
- **The manifold is curved, not flat.** An MLP predicting the PCA-cut part from the kept coordinates recovers about 30% of the cut variance on the 0.8B. On the 2B it recovers 15% at k = 512 and less above that. The earlier negative R² was a data-starvation and split problem: 27K tokens, held out by domain. A learning curve confirmed this, and 306K tokens with an interleaved held-out split fixed it.
- **Projection cost is U-shaped over depth.** Mid-model sites are cheap; the first site and the late sites are expensive. On the 2B at k = 512, single-site KL runs 0.13–0.19 for L2–L12, against 0.62 at L0 and 0.73 at the final site.
- **Errors build up across sites.** All-sites projection costs far more than any single site (2B, k = 512: 3.18 nats against 0.13–0.73 per site).

## How much the function view buys over plain projection

PCA plus a decoder removes part of the projection damage, and the gain shrinks with scale and with depth.

| model | k | site | KL, PCA | KL, PCA + decoder | reduction |
|---|---|---|---|---|---|
| 0.8B | 256 | L6 | 0.128 | 0.054 | 58% |
| 0.8B | 256 | L20 | 0.398 | 0.227 | 43% |
| 2B | 512 | L6 | 0.161 | 0.089 | 45% |
| 2B | 512 | L15 (attn) | 0.408 | 0.358 | 12% |
| 2B | 512 | L20 | 0.381 | 0.279 | 27% |

All-sites 2B at k = 512: PCA 3.18 nats, PCA + MLP 2.40. The decoder helps, but not enough to turn a hard cut into a cheap one.

## Which layers are real boundaries

Replacing a mixing layer's update with a prediction from the token's own state (no context) costs 2–9× less than deleting the update.

- **0.8B:** greedy substitution replaces 13 of 24 mixing layers at 0.37 nats by mean KL. Ranking by mean KL plus the copy-task worst 1% replaces 10 at 0.28 nats with 2.4× lower worst-case copy damage.
- **2B:** per-layer own-state R² is 0.35–0.86 (linear) and up to 0.85 (MLP). Greedy by mean KL replaces 15 layers at 0.54 nats; by mean + copy tail it replaces 8 at 0.18 nats.
- **Never replaced on the 0.8B:** layers 0, 1, 6, 14, 15, 19, 23. That set includes all three late attention layers. Early attention layers (3, 7, 11) were absorbed like DeltaNet layers.
- **L23 on the 0.8B is an averaging trap:** R² 0.81 and low mean cost (0.031), but the second-worst copy tail of any layer (0.31). The mean hid the tail.
- **Layer 0 on the 2B** has high R² (0.86) but dropping its update costs 8.1 nats, so it is essential.

## Whether the array of blocks composes

A nudge of 10% of the state's size shrinks relative to the state by the final layer, from every site up to L20. Off-manifold nudges are pulled back about twice as hard as on-manifold ones in the middle of the model (2B: 0.23–0.31 vs 0.31–0.48 relative deviation ratio at L0–L13). Only the last few layers go to about 1 (2B L22: 0.86 off, 1.01 on). Errors injected mid-model are mostly corrected before they reach the output, which favours cutting in the middle.

## Does smart placement of kept neurons help?

Placement test (0.8B, MLP rebuild at width m, then least-squares refit of the output):

- **Importance ranking and OMP selection tie.** Adaptive placement (OMP, mixed greedy) did not beat ranking neurons by importance. Both are far ahead of random: validation R² 0.41 vs 0.20 at L6 and 0.52 vs 0.20 at L15.
- **So a router or smarter placement needs a real gain to justify its cost.** This is why the atlas idea (per-region projections) is not a sure win.

## The MLP shrink build, and what it did to chat behaviour

The build keeps the top neurons by importance and refits the down projection with ridge least squares, layer by layer on pruned inputs. Chat calibration data comes from the real Qwen3.5 chat template.

Bench results on Qwen3.5-2B (llama.cpp, GGUF bf16):

| model | size (GB) | tools | GSM8K | MMLU | loop rate | wikitext2 ppl |
|---|---|---|---|---|---|---|
| stock 2B | 3.90 | 0.95 | 0.80 | 0.567 | 0.00 | 10.65 |
| MLP 75% | 3.32 | 0.90 | 0.60 | 0.533 | 0.047 | 12.49 |
| MLP 50% | 2.87 | 0.225 | 0.40 | 0.533 | 0.224 | 17.14 |
| stock 0.8B | 1.56 | 0.875 | 0.60 | 0.467 | 0.00 | 15.48 |

- **75% keeps the model usable.** Tools 0.90 against 0.95, MMLU 0.533 against 0.567, perplexity 17% worse. Against the stock 0.8B (1.56 GB, less than half its size) it is better on tools (0.90 vs 0.875), MMLU (0.533 vs 0.467) and perplexity (12.49 vs 15.48), ties on GSM8K (0.60), and loops more (0.047 vs 0).
- **50% breaks chat behaviour while keeping knowledge.** MMLU stays at 0.533, but tool use collapses to 0.225 and 22% of outputs loop. Perplexity worsens 1.6×.
- **Comparison with the author's own Qwen3.6 expert-width cut:** ours at 50% has perplexity damage of 1.6× against their 3.7×, and keeps maths better. Tools and loop rate are similar (about 17–24% tools, about 22% loops). Caveat: different base models and different benchmark setups, so this is suggestive, not a clean head-to-head.
- **Why chat takes the hit (hypothesis, not tested):** the knowledge survives (MMLU) but format-following may be brittle, and calibration data that does not cover the chat template may leave the pruned MLP unprotected where the format lives. The experiments below test this.

### Calibration and sensitivity experiments (2B, 50%)

Hypothesis: the chat collapse comes from calibration coverage, and sensitivity weighting ("protect the directions the output can feel") reduces it.

| 50% build | tools | GSM8K | MMLU | loop rate | wikitext2 ppl | build KL (all) | chat KL |
|---|---|---|---|---|---|---|---|
| plain calibration | 0.225 | 0.400 | 0.533 | 0.224 | 17.14 | n/a | n/a |
| self-chat calibration | 0.425 | 0.467 | 0.233 | 0.212 | 18.20 | 0.454 | 0.693 |
| self-chat + sensitivity (`--shape 0.5`) | 0.175 | 0.067 | 0.267 | 0.424 | 19.12 | 0.487 | 0.756 |

- **Own-chat calibration helps tool use and maths** (tools 0.225 to 0.425, GSM8K 0.40 to 0.47) but MMLU fell to 0.233, near chance for four options. The suites are small (scores move in steps of about 0.033), and the format the grader expects may be a cause. The drop is unverified: rerun MMLU with more questions and read raw answers before treating it as lost knowledge.
- **Sensitivity weighting made things worse** at `--shape 0.5`: higher build KL on every domain, tools 0.175, GSM8K 0.067, loops 0.42. One setting only; other shapes were not tried.
- **Loops did not improve with calibration alone** (0.212 vs 0.224).
- **Build-time prose ppl (29.17, 30.48) is not the bench perplexity (18.20, 19.12).** They use different data; do not compare them.
- Raw files: `results/bench-pod-20261018/` (`mc-pod-20261018_result_mc-2b-mlp50-*.json`, `build_report_*.json`).
- Infrastructure bugs found on the way: forward hooks returning a tuple replaced the layer outputs; a batch of 4 sequences with full-vocabulary logits ran the GPU out of memory in the sensitivity pass. Fixed at `build_pruned.py:93-99` and `build_pruned.py:96`.

## Deforming the manifold with a learned encoder (0.8B, single site)

Script: `m_autoencoder_kl.py`; log `logs/autoencoder_kl.log`; results `results/Qwen3.5-0.8B/autoencoder_kl.json`. Replace-and-continue KL at one site, same eval set, 1,500 training steps, hidden width 1,024. "Warp" = PCA coordinates plus a learned nonlinear correction computed from the full vector, then the decoder.

| site | k | KL, PCA | KL, PCA + decoder | KL, warp | warp vs PCA + decoder |
|---|---|---|---|---|---|
| L6.in | 64 | 1.501 | 0.703 | 0.280 | 2.5x lower |
| L6.in | 256 | 0.132 | 0.085 | 0.037 | 2.3x lower |
| L15.in (attn) | 64 | 1.995 | 1.090 | 0.494 | 2.2x lower |
| L15.in (attn) | 256 | 0.255 | 0.179 | 0.076 | 2.3x lower |
| L20.in | 64 | 1.950 | 1.264 | 0.736 | 1.7x lower |
| L20.in | 256 | 0.409 | 0.310 | 0.123 | 2.5x lower |

- The learned encoder beat PCA + decoder at every site and width, by 1.7-2.5x, and plain PCA by 3-5x. The gate for idea 11 passes at the single-site level.
- **Caveats before this means a cheaper model:** (1) the warp arm has more trainable parameters than the decoder-only arm, and the encoder compute (a 1,024-wide MLP over the full vector) is not counted; an equal-parameter comparison is still needed. (2) It is one site at a time; errors compound across sites, and the all-sites version has not been run. (3) The 2B has not been tested; the decoder payoff shrank at 2B, so the warp's gain may too. (4) The layers still read the original wide vector, so a per-site warp adds compute unless the whole network is refit in the small space.

### Same test on the 2B, with an equal-parameter arm

Qwen3.5-2B-Base, single site, 3,000 steps, hidden width 2,048. `decoder2x` is a decoder with twice the hidden width, which matches the warp's parameter count. Results: `results/Qwen3.5-2B-Base/autoencoder_kl.json`, log `logs/autoencoder_2b.log`.

| site | k | PCA | PCA + decoder | decoder 2x | warp | warp vs decoder 2x |
|---|---|---|---|---|---|---|
| L6.in | 128 | 1.400 | 0.647 | 0.516 | 0.254 | 2.0x lower |
| L6.in | 512 | 0.161 | 0.088 | 0.067 | 0.031 | 2.2x lower |
| L15.in (attn) | 128 | 1.895 | 1.078 | 0.925 | 0.609 | 1.5x lower |
| L15.in (attn) | 512 | 0.408 | 0.353 | 0.326 | 0.176 | 1.9x lower |
| L20.in | 128 | 2.044 | 1.369 | 1.220 | 0.854 | 1.4x lower |
| L20.in | 512 | 0.381 | 0.275 | 0.234 | 0.093 | 2.5x lower |

- The warp beats the equal-parameter decoder at every site and width, by 1.4-2.5x. Extra parameters alone do not explain the gain; the learned encoder does.
- The gain is smallest at the narrow late sites (L20, k = 128: 1.4x) and largest at k = 512.
- Still open: all sites at once, compute cost of the encoder, and refitting the whole network in the small space (distillation).

### All sites compressed at once (2B)

Every one of the 49 sites replaced together, so errors compound. Script `m_autoencoder_all.py`; results `results/Qwen3.5-2B-Base/autoencoder_all.json`; log `logs/autoencoder_all_2b.log`. KL against the untouched model, mean over all domains (copy-task answers in brackets).

| code width | PCA | PCA + decoder | decoder 2x | warp |
|---|---|---|---|---|
| 512 (quarter) | 3.190 (2.322) | 2.477 (1.473) | 2.140 (0.959) | 1.776 (0.455) |
| 128 | 5.305 (3.823) | 3.743 (2.636) | 3.550 (2.457) | 3.438 (2.297) |

- **Compounding takes most of the single-site gain.** At one site the warp scored 0.03-0.18 at width 512; across all sites it scores 1.78. It is nowhere near usable: the 50% MLP cut that broke tool use was at about 0.45.
- **The warp's lead over the equal-parameter decoder shrank** from 1.4-2.5x at one site to 1.2x (width 512) and 1.03x (width 128).
- **Copy-task recall is the exception at width 512:** 2.32 for PCA down to 0.455 for the warp.
- **For scale**, plain PCA at half width (1024) scored 1.69 all-sites in the earlier 2B run. The warp at a quarter width roughly matches it. That is about a 2x narrower code for the same error, with the warp's extra compute not counted.
- At width 128 all four arms are far above usable, so there is nothing to build there.

## 2B -> 0.8B: the first real down-projection (2026-10-19)

Scripts: `build_projected.py` (shared PCA basis folded into the weights, MLP 6144 -> 3584 by importance + least squares, layer-wise healing, final-norm fit), `polish_tail.py` (last 4 layers + final norm trained end to end on KL to the 2B, 600 steps, about 5 minutes). Pod logs and results: `results/pod-proj-20261019/`. The output is a standard Qwen3.5-0.8B checkpoint (0.752B parameters, GGUF 1.52 GB, same as stock). Unit test: folding with an identity basis reproduces the model exactly (KL 0.000000).

Build: 485 calibration rows (fit mix + 192 of the 2B's own chat answers); the 1024-wide basis keeps 86% of the pooled energy; RMS fixes 1.22-1.36. KL with the 2B finishing from the student's stream: 0.29 after layer 3, 0.41 after 11, 0.67 after 19, 1.15 after 23.

| | KL vs 2B | tools | GSM8K | MMLU | IFEval prompt / instruction | answers hitting the cap | wikitext ppl |
|---|---|---|---|---|---|---|---|
| stock 2B | 0 | 0.95 | 0.80 | 0.567 | 0.66 / 0.76 | 13% | 10.65 |
| stock 0.8B | 0.30 | 0.875 | 0.60 | 0.467 | 0.53 / 0.64 | 24% | 15.48 |
| ours, layer-wise healing | 1.60 | 0.075 | 0.00 | 0.133 | 0.15 / 0.26 | 99% | 42.2 |
| ours + tail polish | 0.75 | 0.075 | 0.27 | 0.067 | 0.19 / 0.34 | 82% | **13.85** |

- **Teacher-forced quality is good; generation is not.** The polished student's wikitext perplexity (13.85) beats the stock 0.8B (15.48), and its KL to the 2B is 0.75 (stock 0.8B: 0.30). But under greedy decoding 82% of its open-ended answers run to the length cap, and most collapse into repeated characters or numbers after a fluent first sentence or two. MMLU below chance comes from never stating a letter.
- **Not a conversion bug.** llama.cpp and Hugging Face produce the same greedy text for the same prompt, and both are coherent for the first ~120 tokens of a simple request.
- **The likely cause is exposure bias.** Every training signal so far was teacher-forced (the student always saw real text). It was never trained on its own outputs, so small errors compound in generation until it locks into a loop. The fix that targets this is on-policy distillation: the student generates, the 2B scores the student's tokens, and the student is trained toward the 2B's distribution on its own text.
- The tail polish more than halved KL (1.60 -> 0.75) for minutes of GPU time, so the cheap end-to-end steps work; they just need to run on the student's own generations.
- Bench caveat: `degenerate_rate` flags 32% of stock 0.8B and 13% of stock 2B IFEval answers, because some IFEval prompts ask for long or repetitive text; on IFEval, the cap-hit rate is the cleaner loop signal.

## Distillation from the 2B to repair generation (0.8B student, 2026-10-19)

Script: `onpolicy_distill.py`. Results and logs: `results/pod-onpolicy-20261019/`. Health check = greedy answers to 64 held-out UltraChat prompts, 400-token cap.

| run | tokens | answers at the cap | loops | KL vs 2B |
|---|---|---|---|---|
| start (polished student) | 0 | 84% | 53% | 0.750 |
| A: pure on-policy, reverse KL | 0.52M | 100% | 98% | 0.869 |
| B: half teacher answers (cut at 256 tokens), half student answers scored on first 96 tokens | 1.02M | 88% | 72% | 0.716 |
| C: 75% full-length teacher answers (768 tokens, thinking off), 25% student answers | 0.55M / 1.03M / 2.05M | 75% / 83% / 88% | 70% / 78% / 72% | 0.728 / 0.713 / 0.699 |

Bench of run C at 2M tokens: tools 0.025, GSM8K 0.13, MMLU 0.23, IFEval 0.17 / 0.31, cap-hit 82%, wikitext ppl 14.8.

- **Pure on-policy distillation made the loops worse.** The student's own answers already looped. The 2B, given a looping prefix, predicts more of the loop. Reverse KL then sharpened the loop.
- **Teacher-written answers must be long enough to end.** In run B every teacher answer hit the 256-token cut, so the student never saw an answer stop.
- **2M tokens is not enough.** KL fell about 0.025 per million tokens. Generation did not improve. The stock 0.8B sits at KL 0.30. If the rate stays linear, the gap needs roughly 15-20M more tokens; this is not verified.
- **The stop token is not lost in the projection.** `<|im_end|>` keeps 87% of its embedding norm after projection (vocabulary median 78%). The model can produce the stop signal.
- Infrastructure: two out-of-memory crashes (full-vocabulary scores on whole sequences; long teacher answers at 4 per batch); fixed by `logits_to_keep` and a smaller micro-batch. The pod host rebooted once; files on the container disk survived and the bench resumed from its work files.

## Smoke test: 40 chat prompts, judged blind by Haiku (2B: 75% and 50% MLP builds vs stock)

Files: `smoke/` (`prompts.jsonl`, `answers_*.jsonl`, `divergence.json`, `judge/first/tally.json`). Greedy decoding (temperature 0), thinking off, 400-token cap, llama.cpp b10673 on CPU. Coherence was judged by Haiku with the two models' answers shuffled and unlabelled; agreement compared each 75% answer with the stock answer. The 50% build was judged in a second round (`judge/second/`) with the stock answers re-judged alongside it; the stock score moved by 0.08, so the judge is consistent to about 0.1.

| | stock 2B | 75% MLP build | 50% MLP build (own-chat calibration) |
|---|---|---|---|
| answers identical to stock | — | 2 of 40 (5%) | 1 of 40 |
| median characters before the answers diverge | — | 5 | 2 |
| coherence, mean 1-5 | 4.17 (4.25 on re-judging) | 2.75 | 2.25 |
| coherence 4 or 5 | 70% | 33% | 20% |
| repetition loops | 0% | 50% | 50% |
| checkable answers correct | 14/20 | 10/21 | 4/22 |
| followed format instructions | 13/13 | 9/13 | 8/13 |
| same substance as stock (yes / partly / no) | — | 12 / 12 / 16 | 7 / 8 / 25 |
| better answer (stock / build / tie) | — | 30 / 5 / 5 | 36 / 2 / 2 |

- **The 75% build loops badly under greedy decoding.** Every one of its 20 loops ran to the 400-token cap, and spot checks confirm real repetition (a sentence or code line repeated until the cap), often after a correct start. The stock 2B never looped.
- **The bench's loop rate does not catch this.** `bench/run_bench.py` counts a loop only when a tools answer hits the length cap with empty content (thinking that never ends). It reported 4.7% for the 75% build; repetition inside visible answers is invisible to it. The tools score (0.90) therefore overstates how usable the 75% build is for open-ended chat.
- **The 50% build is clearly worse:** 4 of 22 checkable answers right (maths 2.2 vs 4.2 for stock), 25 of 40 answers with different substance from stock. Own-chat calibration lifted its tools score to 0.425 but did not make its chat usable.
- **By category** (mean coherence, 75% vs stock): maths 4.2 vs 4.2 and rewriting 4.0 vs 4.75 held up; facts 1.6 vs 3.8, explanations 1.8 vs 3.8, advice 1.7 vs 4.7 and code 2.0 vs 4.2 did not.
- **Judge reliability:** spot checks of checkable answers (17×23, 221 prime, list aliasing, yes/no) matched the judge's correctness calls. Its coherence scores are noisier (it gave a correct, clear stock multiplication a 3).

### Sampling check: is the 0.8B looping a greedy-decoding effect? (2026-10-19)

Same 40 prompts, 400-token cap, thinking off, llama.cpp on the iGPU. "Sampled" = the Qwen3.5 model card's non-thinking text settings (temperature 1.0, top_p 1.0, top_k 20, presence_penalty 2.0), seed = prompt id. Stock 0.8B is the unsloth Q8_0 GGUF; ours is `mc-proj-0.8b-tail.gguf` (bf16, polish only, no distillation). All six sets judged together, blind. Files: `smoke/answers_{stock08,proj08,*-sampled}.jsonl`, `smoke/judge/sampling/tally.json`.

| model | decoding | coherence | 4 or 5 | loops | correct | follows format |
|---|---|---|---|---|---|---|
| stock 2B | greedy | 4.15 | 78% | 0% | 18/22 | 12/12 |
| stock 2B | sampled | 3.38 | 40% | 3% | 16/22 | 10/12 |
| stock 0.8B | greedy | 3.65 | 55% | 3% | 14/20 | 12/13 |
| stock 0.8B | sampled | 3.02 | 25% | 0% | 10/23 | 10/11 |
| ours | greedy | 1.68 | 3% | 72% | 2/23 | 1/16 |
| ours | sampled | 1.77 | 3% | 3% | 1/22 | 3/15 |

- **The gap is real, not a test effect.** Under greedy decoding the stock 0.8B is usable (3.65, 3% loops); ours is not (1.68).
- **The presence penalty removes the loops but not the damage.** Our loop rate fell from 72% to 3%, but coherence and correctness did not move. The answers turn into fluent-looking nonsense ("dewth", "downting") instead of repetition. Looping is a symptom; the model cannot keep a line of thought past the first sentences.
- **These sampling settings hurt the stock models** (2B 4.15 to 3.38, 0.8B 3.65 to 3.02), so greedy is the fairer test here.
- **Wikitext perplexity misleads.** Ours beats the stock 0.8B on it (13.85 vs 15.48) and loses on every chat measure.
- Judge consistency: the stock 2B greedy answers scored 4.15 here against 4.17 and 4.25 in the earlier rounds.

## The stock 0.8B against the 2B and against our basis (2026-10-19)

Scripts: `m_embed_relation.py`, `m_stock_relation.py`. Results: `results/stock-relation-20261019/`. CPU, fp32, Hugging Face. 64 rows of 512 tokens (wikitext, UltraChat with thinking off, MBPP), position 0 dropped, 32,704 tokens per model. Ridge maps are fit on 3 of every 4 rows and scored on the 4th. "Our basis" = the 2B's stream times P, the 1024-wide shared basis of the polished build. P comes back exactly from our embeddings, which are the 2B embeddings times P (orthonormality error 2e-4).

| layer | R², 2B → stock 0.8B | R², our basis → stock 0.8B | R², stock 0.8B → 2B | 2B energy in our basis |
|---|---|---|---|---|
| 0 (embeddings) | 0.93 | 0.85 | 0.82 | 0.77 |
| 1 | 0.79 | 0.75 | 0.71 | 0.86 |
| 6 | 0.75 | 0.73 | 0.63 | 0.86 |
| 12 | 0.75 | 0.73 | 0.63 | 0.87 |
| 18 | 0.78 | 0.75 | 0.65 | 0.85 |
| 23 | 0.73 | 0.70 | 0.61 | 0.83 |
| 24 (after the final norm) | 0.71 | 0.68 | 0.62 | 0.81 |

- **The stock 0.8B was not made by keeping a subset of the 2B's embedding channels.** No 0.8B embedding dimension copies a 2B dimension: the median best single-column correlation is 0.25, and none is above 0.9. A rotation-style width cut (like ours) also leaves no column copies, so this does not rule out a derivation; see the weight comparison below.
- **The two models compute in step.** For every even stock 0.8B layer from 0 to 24, the 2B layer that predicts it best (searched within ±6) has the same index. About three quarters of the stock 0.8B's stream is a linear readout of the 2B's stream at the same depth.
- **Our basis keeps almost all of that readout.** R² falls by only 0.02–0.05 after layer 0, and by 0.08 at the embeddings. The projection keeps the information that a working 0.8B shares with the 2B. So the damage in our model is in how its weights compute, not in what the subspace holds.
- **A working 0.8B copies less of the 2B than our healing asks for.** The stock 0.8B explains linearly about 0.61–0.66 of the 2B's stream after layer 1. Our healing asks the student to reproduce the 2B's state inside our basis, about 0.85 of its energy. The two numbers use different measures (R² against energy), so the gap shows a direction, not a size. About a quarter of the stock 0.8B's stream is not linear in the 2B's stream: features of its own. Hypothesis, not tested: healing toward the full projected 2B state asks too much of a 0.8B.
- Over the whole vocabulary with no frequency weights, the embedding relation is weaker (R² 0.51; rare tokens fit worse). The principal-angle overlap between the stock readout and our basis is near random (0.54–0.58 against 0.50), but it does not weight by energy, so the R² columns are the useful measure.
- **No control pair.** No unrelated model with the same tokenizer was compared, so we do not know if 0.75 is high for two models of one family.
- **Our model's own stream was not measured.** Its Hugging Face checkpoint is lost; only the GGUF exists. The "our basis" column is a ceiling for our model, which matches the 2B times P exactly only at layer 0.

### Direct weight comparison: the 2B and the stock 0.8B share their heads

Script: `m_weight_relation.py`. Results: `results/stock-relation-20261019/weight_relation.json`. The 0.8B and the 2B have the same layer count and the same heads (8 full-attention heads with 2 KV heads, 16 linear-attention heads, the same head sizes). Only the residual width and the MLP width differ. So every parameter indexed by head or channel has the same shape in both models and can be compared entry by entry, with no alignment step. Pearson correlation, same layer against other layers and against a random head or channel order:

| parameter (per head or channel) | same layer, mean (min) | other layers | shuffled |
|---|---|---|---|
| linear attention `A_log` (decay, 16 heads) | 0.97 (0.93) | 0.04 | −0.02 |
| linear attention `dt_bias` (16 heads) | 0.99 (0.98) | 0.01 | −0.07 |
| `conv1d` kernels, q / k / v channels (2048 each, taps centered) | 0.95 / 0.95 / 0.95 (≥ 0.86) | 0.00 | 0.00 |
| gated output norm (128) | 0.84 (0.72) | −0.01 | 0.03 |
| row norms of `in_proj_qkv` (6144) / `in_proj_z` (2048), column norms of `out_proj` (2048) | 0.92 / 0.89 / 0.85 | ≤ 0.04 | ≤ 0.01 |
| full attention `q_norm` / `k_norm` (256) | 0.88 / 0.91 | 0.14 / 0.16 | ≤ 0.02 |
| full attention row norms q / k / v, column norms o | 0.79 / 0.89 / 0.84 / 0.89 | 0.17 / 0.59 / 0.07 / 0.02 | ≤ 0.01 |

- **The heads line up one to one, in the same order, with nearly the same values.** Independent training from random weights gives an arbitrary head order, so this cannot happen by chance.
- **Only the head parameters line up directly.** The width-dependent weights (embeddings, residual-side projections, MLPs) line up only partly, and only through the measured map between the residual streams (see the units section below).
- **These values are far from any random start.** `dt_bias` runs from −7.3 to +9.5, and the `conv1d` kernels are sparse (mean |w| 0.03 against 0.25 for the default start), with the same sparse pattern per channel in both models. So both models carry the same *trained* head parameters. One was built from the other, or both from one parent, and then trained further. The data does not show which way.
- **The stock 0.8B therefore has the same design as our model:** the 2B's heads, with a narrower residual stream and a narrower MLP. It shows that this design works after enough training (KL 0.30, smoke coherence 3.65). Qwen's training budget for it is unknown.
- Our model also keeps the 2B's heads (healing changed them a little), so our model, the stock 0.8B and the 2B share heads. They differ in the residual basis, the MLPs and the embeddings.

### Units: heads and MLP neurons, by activations

Script: `m_unit_relation.py` (`UNIT_SIZE=medium`: 24 rows of 512 tokens, 12,264 tokens, even layers only; a full-size run did not fit in free memory beside other jobs). Results: `results/stock-relation-20261019/unit_relation_medium.json`. Best matches are chosen on 3 of every 4 rows and scored on the 4th.

- **Full-attention heads match by index.** In all 6 full-attention layers, the best one-to-one assignment of 0.8B heads to 2B heads is the identity. Attention-pattern correlation 0.81–0.88; text-shifted baseline 0.17–0.37.
- **Linear-attention heads match by index too.** In all 12 layers measured, the best assignment is the identity. Held-out R² of a 0.8B head output from the same-index 2B head output: 0.26–0.39.
- **MLP neurons match only partly.** Per 0.8B neuron, the best single 2B neuron has median held-out correlation 0.20–0.42 (text-shifted baseline 0.07–0.13). 7–35% of 0.8B neurons have a match above 0.5, and at most 1.6% above 0.9. So the MLPs did not keep a one-to-one neuron identity the way the heads did.
- **Matched neurons have aligned weights.** Through the measured map between the residual streams, matched pairs write in similar directions (down-projection cosine 0.37–0.53) and read similar directions (gate cosine 0.13–0.38). Random pairs give 0.02–0.03, and the same comparison with no basis map gives 0.025.
- **Our importance score keeps the neurons that the stock 0.8B has.** 90–99.8% of the matched 2B neurons are inside the 3584 that our MLP shrink keeps (chance 58%).
- The R² from all 6144 2B neurons to each 0.8B neuron came out negative: 6144 inputs overfit 9,200 training tokens. That measure needs the full-size run.

## Layer-by-layer healing (0.8B, width 256 of 1024, all sites)

Plan: `../.claude/plans/manifold-compression-layerwise-healing.md`. Script `heal_layerwise.py`; results `results/Qwen3.5-0.8B/heal_*.json`; logs `logs/heal_*.log`. Each half-layer is trained, in order, on the compressed stream it really receives, to match the original model's output for that half-layer, then frozen. 320 fit sequences of 512 tokens, batch 8 sequences.

| arm | lr, steps per half-layer | KL before | KL after | prose | code | math | chat | copy |
|---|---|---|---|---|---|---|---|---|
| PCA, blocks healed | 2e-5, 300 | 3.45 | 2.86 | 1.80 | 3.80 | 1.89 | 4.21 | 0.91 |
| PCA, blocks healed | 2e-4, 1000 | 3.45 | **1.93** | 1.25 | 2.76 | 1.43 | 2.50 | 0.37 |
| PCA, blocks healed | 6e-4, 2000 | 3.45 | 2.43 | 1.65 | 3.32 | 1.79 | 3.18 | 0.54 |
| warp + blocks healed | 2e-4, 1000 | 1.65 | **0.97** | 0.56 | 1.65 | 0.64 | 1.13 | 0.21 |

- Healing works and the warp stacks with it. The best arm reaches 0.97, inside the plan's "mixed" band (0.5-1.0); the pass line is 0.5.
- Learning rate matters: 2e-4 beats both 2e-5 (undertrained) and 6e-4 (training loss rose at several layers).
- The KL curve through depth climbs steadily (warp arm: 0.18 after layer 0, 0.41 after layer 12, 0.97 after layer 23), with no blow-up, but the late layers add the most.
- Code and chat are the weakest domains; the copy task is the best.
- **Every model here still runs at full width.** The k-wide code is decoded back to the full hidden size before each block, and the blocks keep their full widths. This measures how much information the k-wide code can carry, not a model that is cheaper to run. A real k-wide model needs the PCA rotation folded into the weights (possible for the PCA arm, as in SliceGPT) or blocks that read the code directly (the warp arm; not built).

## What the visualisation showed

The 3D viewer (published privately; source in `viz/`) shows the 0.8B residual manifold at every site. Two things stood out:

- **The cloud is not a smooth surface.** A kernel-smoothed sheet fit (bandwidth 0.35, leave-one-out R²) explains only 0.34–0.41 of variance mid-model and 0.15 at the output. This is why "fix the folds" matters.
- **Flat spots** (directions with little variance) look unused. That is an idea, not a measurement; see idea 9 in the backlog.

## Process and infrastructure lessons

- **The Runner matches HF exactly** (verified), so replace-and-continue KL is trustworthy.
- **Eval size matters.** The first eval set had 8 sequences and the copy-tail numbers rested on 16 answer tokens. Anything about the copy tail from that set is a lead, not a finding. Phase 1 used 64 copy sequences and added chat.
- **Averages hide tails.** L23 and the mean-vs-tail greedy comparison both show it. Always report a tail statistic next to the mean.
- **Position 0.** Editing it hurt; protecting it cut all-sites KL from 3.01 to 2.77, so it is not a strong sink here but `keep_first` stays on.
- **Pod mechanics (RunPod A40, 48 GB):** a 4090 community pod was out of stock; SSH needs `runpodctl pod update <id> --ports 22/tcp`; the container RAM cap is about 50 GB, and compiling llama.cpp CUDA on 96 cores while a build runs gets OOM-killed. Fix: pause the compiler processes (SIGSTOP), finish the build, then SIGCONT. Scripts copied from Windows carry CRLF line endings and break bash; strip with `sed -i 's/\r$//'`. `pkill -f <name>` over SSH kills the SSH shell itself if the name is in the command line.
- **GPU memory in the sensitivity pass:** logits over the 248,320-token vocabulary at 4 sequences of 512 tokens do not fit alongside a 2B model's activations and gradients. Use 1 sequence per step.
- **Peer coordination:** another job requested a pause. Stopping jobs by matching command lines is risky; check process IDs first.
- **Viewer:** Plotly 2.35.2 is not on cdnjs; use 2.33.0. Embedded token text with U+FFFD blocked publishing; escape the JSON to ASCII.

## What we still do not know

- Whether a balanced calibration mix (chat plus prose, code, maths and MMLU-style questions) recovers both tools and MMLU at 50%. Self-chat alone gave tools 0.425 but MMLU 0.233; sensitivity weighting at 0.5 made it worse.
- Where the folds are and how many points a better approximation needs. The 5–15 point improvement guessed earlier is a guess.
- Whether per-region projections beat one global projection at equal parameters.
- Whether per-layer widths beat uniform width.
- How these results carry to a 27B model or a MoE, and to Strix Point hardware (none tested).
- Whether the build-time ppl and bench ppl are comparable.

## Part swap: stock 0.8B parts inside our projected 0.8B (2026-10-20)

Method: `m_part_swap.py`, GGUF through llama.cpp Vulkan, greedy, thinking off, 40 smoke prompts, 400 tokens, blind Haiku judges (24 batches of 20). Each swap has a matched control: the stock 0.8B with the same sites sent through our coordinates and back (lossy round trip, no our weights).

| Model | Coherence (1-5) | Loop | Correct |
|---|---|---|---|
| stock 2B | 4.20 | 0.00 | 18/22 |
| stock 0.8B (Q8) | 3.75 | 0.03 | 15/20 |
| ps-stock (rebuilt stock 0.8B) | 3.73 | 0.05 | 15/22 |
| proj08 (ours) | 1.75 | 0.70 | 2/27 |
| ps-ours (ours, rebuilt) | 2.02 | 0.68 | 1/28 |
| ps-mlp, ps-attn | 1.00 | 1.00 | 0 |
| ps-io | 1.20 | 0.88 | 2/25 |
| ctrl-mlp, ctrl-attn, ctrl-io, ctrl-all | 1.00-1.02 | 1.00 | 0 |

- The pipeline is sound: ps-stock matches the stock 0.8B, and ps-ours matches our earlier build.
- Every single-part swap collapses to gibberish, and so does every control. The swaps and the controls give the same result.
- A lossy round trip through our coordinates (held-out R2 0.60-0.82) is enough to destroy the stock model at any one site. The test cannot say which part of ours breaks chat, because the basis mismatch alone breaks every mixed model.
- Our all-ours build (2.02) is better than any mix. Mixed parts do worse than parts that were fitted together.
- Conclusion: the part-swap does not localize the damage. A swap needs a near-exact basis map. Our maps are not.

## 9B to 4B shape (2026-10-20)

Method: `build_projected.py` on a rented A100 80 GB, Qwen3.5-9B into the Qwen3.5-4B shape (hidden 4096 to 2560, MLP 12,288 to 9216, 32 layers, untied head, 4.84B parameters). Layer-wise healing at 400 steps per half-layer (the 0.8B build used 1000), then tail polish of the last 4 layers. KL to the 9B: 1.04 after the build, 0.79 after polish (the 0.8B build: 0.75). The fp32 identity test did not fit in 80 GB and was not run. Smoke test: 40 prompts, greedy, thinking off, 400 tokens, Q8 GGUF, llama.cpp Vulkan, blind Haiku judges.

| Model | Coherence (1-5) | 4 or more | Loop | Correct | Hit the 400-token cap |
|---|---|---|---|---|---|
| stock 4B | 4.92 | 1.00 | 0.00 | 21/21 | 12 |
| stock 9B | 4.72 | 0.97 | 0.00 | 21/21 | 13 |
| ours (9B to 4B, polished) | 2.23 | 0.10 | 0.35 | 5/23 | 25 |

- The decision rule in the plan said: far below the stock 4B means stop this line. Ours is 2.7 below the stock 4B. At 0.8B the gap was 2.0 (1.75 against 3.75).
- Scale did not close the gap. Loops fell (0.35 against 0.70 at 0.8B) and correctness rose (5/23 against 2/27), but chat is still broken for most prompts.
- This cut is milder than the 0.8B cut (width 0.625x against 0.50x), and the teacher is bigger. Neither fixed it.
- This is a tiny healing budget, not a failed method. Healing was local only: about 250K calibration tokens, 400 steps per half-layer, and 600 tail-polish steps. Nothing trained the whole network on the teacher. The stock 4B had trillions of training tokens. The result says only that a cut with this little repair fails.
- Not tested: whole-network distillation after the cut at a realistic token budget (the next test is KL and coherence against tokens), more healing steps, and the sample-count change.
- Cost: about 6.5 hours of A100 at $1.79 per hour, about $12. Three crashes from GPU memory: gradients of healed layers stayed allocated, and the least-squares step made fp32 copies of the whole stored state. Both fixed in `build_projected.py`. The 4-hour pod limit did not stop the pod.

## Damage isolation: data against structure (2026-10-21)

Method: `scale_gate.py`, 2B cut to the 0.8B shape (width 1024 of 2048, MLP 3584 of 6144 neurons), no training anywhere. Each cut is applied alone and together, with the data behind each fit varied. Activation tokens set the shared basis and neuron importance (32K, 128K, 512K). Refit tokens set the MLP least-squares refit (8K, 32K, 128K). The oracle fits on the 15K eval tokens themselves. KL is on held-out rows. Smoke test is 40 prompts, greedy, thinking off, blind Haiku judges. The fit stream is mostly prose and chat (code and math are small); the oracle is partly overfit (15K tokens for 3,584 weights per output).

| Arm | KL to stock 2B | Smoke coherence (1-5) | Loop | Correct |
|---|---|---|---|---|
| stock 2B | 0 | 4.20 | 0.00 | 16/20 |
| width fold, 32K / 128K / 512K activation tokens | 2.57 / 2.47 / 2.50 | not run / not run / 1.02 | 0.40 | 0/20 |
| width fold, oracle | 1.91 | 1.00 | 0.90 | 0/19 |
| MLP shrink, refit 8K | 0.81 | 1.18 | 0.93 | 1/21 |
| MLP shrink, refit 32K | 0.46 | not run | | |
| MLP shrink, refit 128K | 0.40 | 1.73 | 0.82 | 3/21 |
| MLP shrink, oracle | 0.11 | 1.75 | 0.85 | 1/23 |
| both (128K activation, 32K refit) | 2.53 | 1.02 | 0.88 | 0/21 |
| both, oracle | 1.97 | not run | | |

- **Neither cut alone gives working chat without training.** The width fold alone gives coherence 1.0 at every data size. The MLP shrink alone (58% of neurons kept) gives 1.2 to 1.75.
- **Activation data is not the cause.** 16 times more activation tokens moved the width-fold KL from 2.57 to 2.50, and the MLP KL from 0.50 to 0.45.
- **Refit data matters for the MLP shrink, then saturates.** Coherence rose from 1.18 (8K tokens) to 1.73 (128K), and the oracle gives 1.75. More refit data past about 128K tokens would not fix chat.
- **The oracle does not fix either cut.** Fitting on the evaluation tokens themselves gives KL 1.91 for the width fold and coherence 1.00. The damage is structural: the cut removes capacity that fitting on more data does not restore.
- **KL does not predict chat.** The MLP shrink at KL 0.40 still breaks chat (coherence 1.73). This repeats the 0.8B finding.
- Conclusion: the repair has to be training, not better fits. The next test is whole-network distillation after the cut, with checkpoints against tokens.

## Whole-network distillation of the 0.8B student (2026-10-21)

Method: `distill_whole.py`, 2B teacher, 0.8B-shaped student (the polished build, KL 0.75). Forward KL to the teacher's full distribution on fixed corpus text (UltraChat in the chat template, plus wiki, code and math). No generation during training, so it runs at 3,525 tokens per second on an A40. Checkpoints at 5M, 20M, 50M, 100M, 200M, 350M and 500M tokens.

Held-out KL: 1.0036 at the start, 0.8298 at 2.5M, 0.4739 at 50M, 0.3548 at 135M, 0.2942 at 267.5M (the last point before the pod was stopped).

Smoke test at each checkpoint (40 prompts, greedy, thinking off, 400 tokens, blind Haiku judges, stock 2B and stock 0.8B in the same round as anchors). Each checkpoint was its own judge round, so the anchors show the round-to-round noise: the stock 2B scored 4.22, 4.35 and 4.22, and the stock 0.8B scored 3.75, 3.67 and 3.80. Judge noise is about 0.1 to 0.15. Files: `smoke/judge/d20/tally.json`, `smoke/judge/d50/tally.json`, `smoke/judge/d100/tally.json`.

| Checkpoint | Coherence | 4 or 5 | Loop | Correct | Follows format | Gap to the stock 0.8B |
|---|---|---|---|---|---|---|
| polished build, 0 tokens | 1.75 | - | 0.70 | 2/27 | - | 2.00 |
| 20M tokens | 1.73 | 5% | 0.72 | 2/33 | 3/14 | 2.07 |
| 50M tokens | 2.48 | 20% | 0.53 | 1/28 | 7/13 | 1.19 |
| 100M tokens | 2.25 | 10% | 0.55 | 3/30 | 6/17 | 1.50 |
| 200M tokens (Codex review, different judge) | 2.55 | 30% | 0.35 | 5/31 | 6/14 | 1.10 against the same Codex review |
| stock 0.8B | 3.65 to 3.80 | 55 to 60% | 0.05 | 20 to 22 | 12 to 15 | - |
| stock 2B | 4.22 to 4.35 | 75 to 85% | 0.00 | 23 to 24 | 14 to 16 | - |

The 200M smoke evaluation was added on 2026-10-22. The original Haiku judge was unavailable because its weekly usage limit had been reached, so the primary Codex assistant reviewed all 40 answers and the same 40 cached stock 0.8B answers directly, without blinding. In that review stock 0.8B scored coherence 3.65, loops 0.05, correctness 13/31 and format 11/14. The stricter correctness assessment gives a different stock score from Haiku; do not treat the 200M row as a clean trend comparison with earlier checkpoints. It remains 1.10 coherence points below stock under the same review. At 200M, held-out KL was 0.3203 and 18/40 answers hit the 400-token cap. Full ratings, reasons and methods: `smoke/judge/d200/README.md`, `codex_ratings.json` and `tally.json`. Agreement with the stock 2B was not judged for this round.

Against the stock 2B answer for each prompt, the same round:

| Candidate | Same substance (yes / partly / no) | Better answer (reference / candidate / tie) |
|---|---|---|
| stock 0.8B, 50M round | 17 / 20 / 3 | 24 / 4 / 12 |
| stock 0.8B, 100M round | 18 / 19 / 3 | 26 / 4 / 10 |
| ours, 50M tokens | 2 / 13 / 25 | 38 / 1 / 1 |
| ours, 100M tokens | 3 / 14 / 23 | 39 / 0 / 1 |

- **The curve rises, then flattens by 50M.** 1.75 at 0 tokens, 1.73 at 20M, 2.48 at 50M, 2.25 at 100M. The 50M and 100M points are within judge noise of each other. The gap to the stock 0.8B goes 2.00, 2.07, 1.19, 1.50. More tokens past 50M bought nothing measurable.
- **The same-substance metric tells the same story.** 2 to 3 of 40 answers carry the substance of the stock 2B, against 17 to 18 for the stock 0.8B. The judge preferred the distilled answer once in 80 comparisons.
- **Distillation helps, but far less than the KL curve suggests.** Coherence rose from 1.75 to about 2.4, while held-out KL fell 3x (1.00 to 0.34). The plan's stop line was "flat under 2.2"; the model is barely above it, and far below the 3.25 repair line.
- **The failure mode is unchanged: repetition.** The loop rate stays above 50% at every checkpoint (0.70, 0.72, 0.53, 0.55). The loops are verbatim paragraph repeats. The judge also caught answers that state the right answer and then contradict it, and confident wrong arithmetic.
- **Correctness does not move at all.** The distilled model answers 1 to 3 of 28 to 33 checkable questions correctly, against 20 to 22 for the stock 0.8B. It is not less accurate; it is broken.
- **Only the categories that need no sustained reasoning survive.** Coherence by category at 50M, ours against the stock 0.8B: rewrite 4.00/5.00, creative 3.33/4.00, howto 3.25/3.00, advice 2.33/3.67, code 2.00/3.60, math 2.20/3.60, explain 1.80/3.60, fact 1.80/2.80, format 2.67/5.00.
- **KL does not predict chat, again.** This is the third time. Forward KL on corpus text measures next-token agreement under teacher forcing; the damage is exposure bias at generation time, and corpus-text training cannot touch it.
- **What this points at:** the repair needs training on the student's own generations, not more teacher-forced corpus text. The earlier on-policy attempt failed for its own reasons (teacher answers cut at 256 tokens, so the student never saw an answer end; the teacher then reinforced the student's loops). A run with the teacher's own complete answers as the on-policy source is the next test.
- Cost: $0.59 per hour on the A40. The run was stopped at 267.5M tokens, 22 hours, about $13.00. The 200M checkpoint was taken; the 350M and 500M checkpoints were not.
- **The pod was stopped on purpose.** The coherence curve had flattened, so the remaining 230M tokens (about $7) were not worth spending. The KL curve kept falling (0.2942 at 267.5M) but KL is not the target.

## Continuation to 500M (2026-10-22)

The user requested completion of the 500M-token experiment. The previous run's 267.5M point was a logged evaluation, not a scheduled model checkpoint. The latest confirmed saved weights are `ckpt_200M`, taken at **200,015,872 tokens**. The old pod `r81k4ypxoiqdmd` cannot currently restart because its host has no free GPU; the CPU-only recovery request also failed. No 267.5M checkpoint was available locally.

The continuation uses a new A40 pod, `381y3y7lnnzt9x` (`mc-distill-08b-500m`), at $0.59/hour, with a 100GB persistent volume. The uploaded 200M safetensors match the local SHA-256 (`83a9d3ed9a08e7ff9fdae51a84239c5a88ee9840961715a7de974b343f4541be`). This is a **weights-only continuation with a fresh AdamW optimizer**, not an exact restoration of the interrupted process: the old trainer saved neither optimizer state nor a shutdown checkpoint. The global token count, corpus offset and learning-rate schedule continue from the 200M point, using the same corpus-building procedure. The corpus cache is rebuilt on the new pod. Batched tokenization preserves conversation order and was checked against individual encoding on all 40 templated smoke prompts.

The target is 500M total training tokens (500,006,912 after rounding up to a complete batch), with model exports at 250M, 300M, 350M, 400M, 450M and 500M. There is no early-stop rule based on KL. Remaining training is about 300M tokens, approximately 24 hours / $14 at the previous measured throughput; setup and checkpoint transfer add overhead.

The rebuilt cache contains 976,626 rows of 512 tokens. Its UltraChat source yields 385,783 unique full chunks (197,520,896 tokens), repeated to fill the 90% chat share of the original 500M training stream. The token budget counts training exposure, not unique text. This repeats the original corpus-building policy.

`distill_whole.py` now saves full fp32 model state, AdamW state, RNG state and progress atomically as `resume_latest.pt`, after the first update, every 10M tokens in this run, at model-export checkpoints and on a handled SIGTERM/SIGINT. The export no longer rounds live fp32 parameters to bf16 in place. A tiny-model CPU integration test verified that interruption and full-state resume produce exactly the same final weights and training losses as uninterrupted training; it also checks the weights-only continuation's token offset and learning-rate position.

Remote outputs: `/workspace/distill-500m`; training log: `/workspace/logs/distill-500m.log`. `watch_distill_500m.py` runs locally, downloads and size-checks the exported models into `models/manifold-compression/distill-08b/continuation-500m/`, and stops only the new pod after completion or failure. It requires the local computer to remain running. The remote training command has a 32-hour graceful timeout; the local watcher has a 36-hour pod-stop watchdog. Full resume state remains on the persistent volume after stopping. Connection and run metadata: `logs/continuation-500m-pod.json`; watcher output: `logs/continuation-500m-watch.log`.

The 500M run is complete; its final sampled smoke evaluation is recorded below. Success is judged by generated-answer coherence, loops and correctness against stock baselines, rather than closeness to either model's outputs.

**After a local reboot:** the existing remote training wrapper was launched with `nohup` and disconnected input, so the RunPod training job continues when this computer or Codex session closes. Reopen this workspace and reconnect to pod `381y3y7lnnzt9x` using `logs/continuation-500m-pod.json`; check the live training log before considering any trainer launch. Restart the local `watch_distill_500m.py` monitor in a hidden process, preserve its previous logs, and update `logs/continuation-500m-watch.pid` with its new PID. The monitor skips size-verified checkpoint downloads and fetches missing ones. Local downloading and automatic pod shutdown are unavailable while that monitor is offline; completion of the training process alone does not stop GPU billing. A local reboot does not require restarting the trainer or creating another pod.

Startup verified: the new pod reproduced held-out KL **0.3203** from the saved 200M student. Its first optimizer update completed at **200,048,640 tokens** (step 6105, lr 1.38e-5, batch KL 0.3052), and the **9.4 GiB full resume state** was successfully written to the persistent volume. Training is active beyond that point.

### 250M correctness check (2026-10-23)

Generated all 40 smoke answers from the continuation's `ckpt_250M` (250,019,840 actual training tokens), using the same b10673 Vulkan server, Q8_0, greedy decoding, thinking disabled, one slot and 400-token limit. Tokenizer, tokenizer config and chat template match the 200M checkpoint byte-for-byte. The new export stores the tied embedding and output head separately; their values were verified identical and both quantized to Q8_0. The larger exported file therefore does not represent a wider student.

Correctness was reviewed by the primary Codex assistant without blinding, with the same rubric and exactly the same 31 checkable prompt IDs as the 200M Codex review. The nine subjective prompts remain excluded; format failures are separate. Existing 200M and stock ratings were reused unchanged.

| Model | Correct | Accuracy |
|---|---:|---:|
| 200M | 5/31 | 16.1% |
| 250M | 3/31 | 9.7% |
| Stock 0.8B, earlier Codex review | 13/31 | 41.9% |

The three passing answers are the inflation explanation, faithful council-summary content (though it uses two sentences), and the shopping list. No previously incorrect answer became correct. The train-speed question regressed from 72 km/h to 120 km/h, with wrong time conversion and arithmetic. The other regression is borderline: 17 times 23 is answered as "391 (or 391.001)"; rejecting the contradictory alternative follows the existing strict rule. Accepting the first number raises the 250M score to **4/31 (12.9%)**. The palindrome example was executed and contradicts its claimed output; SQLite rejects the SQL example because `order_amount` is not a supplied column.

These 31 cases do not show a correctness improvement at 250M and are too few to establish an overall trend. No change was made to the requested 500M training target. Coherence was not scored. Twenty of 40 generations hit the token limit; that is not a judged loop rate.

A subsequent Codex review applied the same repetition-loop criterion to all 40 saved 250M answers: **13/40 (32.5%)** were flagged, against **14/40 (35%)** in the unchanged 200M ratings. Three prompts gained loops (1, 33, 36), and four lost them (5, 6, 7, 20). Several judgments are borderline, so a one-answer difference is not persuasive evidence of reduced looping. Token-cap hits rose from 18/40 to 20/40, a separate mechanical measure. The correctness difference is only one or two questions depending on the multiplication judgment; it does not by itself establish forgetting. Full loop reasons and paired summary: `smoke/loops_d250M_ratings.json` and `smoke/loops_d250M.json`.

Artifacts: `smoke/answers_d250M.jsonl`, `smoke/correctness_d250M_ratings.json` (all per-answer reasons), and `smoke/correctness_d250M.json` (validated paired summary). All 40 answer IDs and all 31 checkable IDs were verified complete.

### Evaluation convention and model lineage (2026-10-23)

The user selected Qwen's recommended non-thinking text preset as the primary decoding convention for future checkpoint comparisons, applied equally to our models and stock references. Greedy runs remain separately labeled diagnostics. `smoke/gen.py` now uses the recommended preset for bare model names and reserves greedy for explicit `--greedy`; sampled output filenames retain the `-sampled` suffix so historical greedy answers are not mixed with them. This preference does not alter any existing scores. Qwen notes that higher presence penalties can reduce performance, but its recommendation is not a guarantee of any particular accuracy tradeoff on our small test.

The current student comes from **the post-trained chat checkpoint `Qwen/Qwen3.5-2B`**: `pod_distill_08b.sh` builds it into the 0.8B shape, polishes its tail against that same model, then launches whole-network distillation. `distill_whole.py` defaults to `Qwen/Qwen3.5-2B` as teacher, including the 200M-to-500M continuation. The stock reference is **the post-trained `Qwen/Qwen3.5-0.8B`**, quantized to Q8_0 by Unsloth. This is a comparison of chat-derived models with different training histories, not an untuned-base-versus-chat mismatch. The earlier `Qwen3.5-2B-Base` manifold measurements in this document are separate experiments. Using “base” informally for the unchanged stock reference should not be confused with the actual `-Base` pretrained checkpoint.

### Qwen recommended sampling pilot (2026-10-23)

Tested the checkpoints available at launch (20M, 50M, 100M, 200M, 250M and 300M), plus stock 0.8B, with the Qwen3.5-0.8B model card's **non-thinking text best-practice preset**: temperature 1.0, top_p 1.0, top_k 20, min_p 0.0, presence_penalty 2.0, repetition_penalty 1.0 (`repeat_penalty` in llama.cpp). Source: https://huggingface.co/Qwen/Qwen3.5-0.8B#best-practices. This is distinct from the lower-temperature configuration reported for the model card's benchmark results.

All seven models generated the same 40 smoke prompts, with thinking disabled, Q8_0, b10673 Vulkan, a 400-token output cap and one sample per prompt with seed equal to prompt ID. This is a controlled short-output pilot, not the model card's generally recommended 32,768-token output allowance. The server's effective settings were inspected, including its default **64-token penalty window** (`repeat_last_n=64`); framework penalty implementations can differ. 20M ran with one slot; the remaining models used four independent slots with 4,096 context tokens per slot. Parallel batch execution can affect numerical reproducibility even with fixed seeds. A one-answer serial 50M partial was archived before its fresh full parallel pass.

The primary Codex assistant reviewed all 280 answers directly, without blinding, using the existing coherence and repetition rubric and the same fixed 31 checkable questions. Correctness evaluates the whole answer, including supporting claims and arithmetic; subjective prompts remain excluded and format failures are separate. The greedy ratings below were reused unchanged from the previous Codex reviews. Earlier Haiku scores are not a matched-judge comparison.

| Model | Coherence / 5 | Judged loops | Correct / 31 | Output-cap hits / 40 |
|---|---:|---:|---:|---:|
| 20M | 2.050 | 1/40 (2.5%) | 2 | 21 |
| 50M | 2.275 | 0/40 (0%) | 1 | 13 |
| 100M | 2.125 | 1/40 (2.5%) | 0 | 20 |
| 200M | 2.350 | 1/40 (2.5%) | 1 | 13 |
| 250M | 2.400 | 0/40 (0%) | 1 | 15 |
| 300M | 2.350 | 1/40 (2.5%) | 3 | 18 |
| Stock 0.8B | 3.075 | 0/40 (0%) | 8 | 15 |

| Model | Greedy loops -> sampled loops | Greedy correct -> sampled correct | Greedy coherence -> sampled coherence |
|---|---|---|---|
| 200M | 14/40 -> 1/40 | 5/31 -> 1/31 | 2.55 -> 2.35 |
| 250M | 13/40 -> 0/40 | 3/31 -> 1/31 | Not previously scored -> 2.40 |
| Stock 0.8B | 2/40 -> 0/40 | 13/31 -> 8/31 | 3.65 -> 3.075 |

The preset greatly reduced judged repetition in this pass, but did not improve overall answer quality in the matched comparisons. Many answers replaced repeated paragraphs with varied invented facts or incorrect reasoning. At 200M, the sampled answers lost four greedy passes (inflation, train speed, multiplication and council summary) and gained none. At 250M, inflation and the summary were lost; the shopping list was the only strict pass. At 300M, the product 391, JavaScript deduplication and shopping list passed; the deduplication function passed simple executed checks for duplicates, empty input and strings. No greedy 300M review was performed in this pilot.

These are small, single-seed results with subjective, unblinded ratings. Borderline calls can move counts by one: 50M inflation was accepted despite confused wording; 100M's shopping list was rejected for its whole-or-half-loaf alternative; 300M's loose summary was rejected and its repetitive adoption workflow was flagged as a borderline loop; stock's nearby-station French was accepted despite omitting a literal nearest superlative. The changes do not isolate temperature from presence penalty or establish a training trend. The requested 500M training run was left running.

One 200M response (prompt 18) caused the server's peg-native chat parser to return HTTP 500. Its actual 229-token raw output was recovered from the preserved server log and scored, rather than replaced. The remaining 200M prompts and subsequent models used `--skip-chat-parsing` to preserve malformed output; generation parameters and the thinking-off prompt template were unchanged. Resumed metadata retains the earlier run details.

Artifacts: `smoke/answers_<model>-sampled_qwen-20261023.jsonl`, matching `metadata_*.json` and server logs; `smoke/runtime_sampling_qwen-20261023.json` and `runtime_sampling_parallel_qwen-20261023.json`; all per-answer scores and reasons in `smoke/qwen_sampling_20261023_ratings.json`; validated aggregates and paired changes in `smoke/qwen_sampling_20261023_summary.json`. Reproduce aggregation with `python manifold-compression/smoke/summarize_qwen_sampling.py`.

### Final 500M checkpoint evaluation (2026-10-23)

Training reached **500,006,912 tokens**, with final held-out KL **0.2688**. The watcher downloaded and size-verified `ckpt_500M`, copied training history and logs, then stopped pod `381y3y7lnnzt9x` at **20:40:11 UTC (3:40 PM Central)**. Full training resume state remains on the persistent pod volume; it has not been downloaded locally. This evaluation used local inference and did not restart the pod.

Generated all 40 smoke prompts from the final checkpoint using the same Q8_0 quantization, b10673 Vulkan server, thinking-off template, 400-token cap, four slots with 4,096 context tokens each, seed equal to prompt ID, and Qwen sampling preset as the preceding 300M comparison. The tokenizer files and chat template match 300M byte-for-byte. Captured runtime settings confirm the sampling parameters and 64-token penalty window. Used `--skip-chat-parsing` as in the 300M run.

The primary Codex assistant directly reviewed all 40 outputs, unblinded, with the unchanged coherence/repetition rubric and fixed 31 checkable IDs. Previous checkpoint and stock ratings were reused unchanged.

| Model | Coherence / 5 | Judged loops | Correct / 31 | Output-cap hits / 40 |
|---|---:|---:|---:|---:|
| 200M | 2.350 | 1/40 | 1 | 13 |
| 250M | 2.400 | 0/40 | 1 | 15 |
| 300M | 2.350 | 1/40 | 3 | 18 |
| **500M** | **2.350** | **1/40** | **1** | **20** |
| Stock 0.8B | 3.075 | 0/40 | 8 | 15 |

The shopping list is the only strict correctness pass at 500M. The two passes at 300M that fail here are multiplication and JavaScript deduplication. The 500M answer starts with the correct product 391 but follows it with false supporting arithmetic (`270 * 5 = 1750`, actually 1350), so it fails the same whole-answer rule used earlier. Similarly, the shirt answer starts with the correct $30 but contradicts it with half-price equations and invented unit costs. Counting only those opening numbers would raise 500M to 3/31; that is a different metric. The council summary invents October approval rather than preserving the scheduled next-month vote; accepting its broad gist would add another borderline pass.

Executed checks confirm that the palindrome function returns `None` for a long palindrome and its own long example, while the accompanying examples have invalid Python syntax. SQLite rejects both supplied SQL queries, and Node's syntax checker rejects the JavaScript at an unexpected `else`. The logic answer repeatedly circles through invented lavender/rose transformations; its loop flag is borderline, so the judged loop count is 0-1 depending on that call. Twenty token-limit hits are a separate mechanical measure.

Four- or five-point coherence answers increased from 6/40 at 300M to 8/40 at 500M, concentrated in rewriting and creative tasks, while the overall mean stayed unchanged. This small, single-seed, unblinded test shows **no clear answer-quality improvement from 300M to 500M** and leaves the student well below the matched stock reference. Lower teacher KL has not translated into improved quality on this test. It does not establish a general regression or predict the outcome of a changed training objective, corpus or schedule. These results do not provide a positive reason to spend another 500M tokens on the unchanged setup.

Artifacts: `smoke/answers_d500M-sampled_qwen-20261023.jsonl`, matching metadata/server logs, and `smoke/runtime_sampling_d500M_qwen-20261023.json`; all 40 judgments and format checks in `smoke/qwen_sampling_500m_20261023_ratings.json`; executed checks in `smoke/qwen_sampling_500m_20261023_code_checks.json`; validated comparison in `smoke/qwen_sampling_500m_20261023_summary.json`. Reproduce generation with `python manifold-compression/smoke/gen.py d500M --tag qwen-20261023 --slots 4 --skip-chat-parsing` (existing answer IDs are skipped), and aggregation with `python manifold-compression/smoke/summarize_500m.py`.

### Full IFEval: stock, 300M and 500M (completed 2026-10-23; audited 2026-10-24)

Ran all **541 prompts / 834 instruction checks** for each model, locally, using Q8_0, the same b10673 Vulkan server, four slots of 4,096 context tokens each, thinking disabled, a 1,280-token response cap, and the user's selected Qwen sampling preset. Seeds are IFEval document keys, identical across models. These are one-sample local comparisons under this decoding convention, not reproductions of published model-card scores. All 1,623 responses completed without request or scoring exceptions. Generation completed at 8:02 PM Central on October 23; the RunPod remained stopped.

The independent audit found an installed lm_eval 0.4.13 checker issue: `keywords:letter_frequency` substitutes a random alphabetic letter when the dataset asks for `#` (key 1122) or `!` (key 1129). These are the only two non-alphabetic letter-frequency cases in the dataset. The local correction preserves the requested character; installed packages, generated responses and native result files remain unchanged. This behavior is also documented in the upstream report https://github.com/google-research/google-research/issues/3427. Every corrected answer was rescored twice; all results repeated exactly, and all score differences from the original run were confined to these two keys. The table below uses the disclosed local correction.

| Model | Prompt strict | Instruction strict | Prompt loose | Instruction loose | Output-cap hits |
|---|---:|---:|---:|---:|---:|
| Stock 0.8B | **249/541 (46.03%)** | **473/834 (56.71%)** | 285/541 (52.68%) | 526/834 (63.07%) | 24/541 (4.44%) |
| 300M | 85/541 (15.71%) | 228/834 (27.34%) | 95/541 (17.56%) | 250/834 (29.98%) | 14/541 (2.59%) |
| 500M | 77/541 (14.23%) | 221/834 (26.50%) | 91/541 (16.82%) | 246/834 (29.50%) | 11/541 (2.03%) |

Prompt strict requires all instruction checks in a prompt to pass. Instruction strict counts checks separately. Loose checking applies the evaluator's standard response-cleanup variants. IFEval checks instruction compliance, not general factual correctness or coherence.

500M is **1.48 percentage points below 300M** on strict prompt accuracy: 50 prompts pass in both, 35 pass only at 300M, 27 only at 500M, and 429 fail in both. This single-sample difference does not establish a general regression, but provides no evidence of improved instruction following at 500M. Both students remain far below stock. Cap-hit rates are low for the students and do not explain the large stock gap by themselves; some requested long outputs can still be truncated.

Native uncorrected scores have the same strict prompt counts. Native instruction strict for 500M was 220/834 (26.38%), while the literal-character correction credits one additional check. Native loose scores were stock 286/541 and 527/834, 300M 95/541 and 251/834, and 500M 91/541 and 247/834. Original and corrected results are both retained.

Artifacts: `results/ifeval-stock08-300M-500M-qwen-20261023/summary-corrected.md`, `comparison-corrected.json`, `scoring-audit-corrected.json` (exact changed fields), `instruction-breakdown-corrected.json`, and all per-prompt responses, runtime settings, manifest and logs. Original scores are in `summary.md` and `comparison.json`. The background run's initial audit failure is preserved in `finish.err`; the completed correction and repeat audit are recorded in `rescore.log` and the corrected artifacts. Reproduce the audit with `python bench/summarize_manifold_ifeval.py --runid ifeval-stock08-300M-500M-qwen-20261023 --literal-characters`.

## Whole-network distillation: what is on disk

`models/manifold-compression/distill-08b/`: checkpoints at 5M, 20M, 50M, 100M and 200M tokens (bf16 safetensors, 1.5 GB each), the GGUF conversions at 5M, 20M, 50M, 100M and 200M (Q8_0, 811 MB each), and `history.json` with the full KL curve (108 eval points). The 200M checkpoint was smoke-tested and reviewed by Codex on 2026-10-22; see the different-judge caveat above.

The judge rounds for 20M, 50M and 100M are in `smoke/judge/d20`, `smoke/judge/d50` and `smoke/judge/d100`, each with a `tally.json`.

## Open ideas

See [`../.claude/plans/manifold-compression-ideas.md`](../.claude/plans/manifold-compression-ideas.md). The cheap gates, in order: the shaped and self-chat bench, FFN activation concentration on the 2B, and the routing count on an existing MoE.
