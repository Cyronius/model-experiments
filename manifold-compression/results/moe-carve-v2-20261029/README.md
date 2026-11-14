# Slimming Qwen3.5-9B one step at a time (benchmark v2)

*Run 2026-10-29 on one RunPod A40: 2.2 hours, about $1.32 of the $8 cap. Nothing was trained except the routers and the miss detector; every number is before healing. Code: `moe_carve.py` and `slim_batch.py`. Rules and predictions written before the run: `moe_carve_rules_v2.json` and `slim_rules.json`. Pod: `run_slim_pod.py` and `slim_worker.py`. Section 8, the healing pilot, ran afterwards on one RTX 6000 Ada for about $3.70, section 9's checks on the healed model on one A40 for $0.79, section 10's on one A40 for $0.76, section 11's on one A40 for $0.51, section 12's on one A40 for $0.68 (two attempts), section 13's on one A40 for $0.96, section 14's on six A40s side by side for $5.03, section 15's on one A40 for $4.75, section 16's on one A40 for $5.03, section 17's on an A40 and then an RTX A6000 for $7.11, and section 18's on one RTX A6000 for $1.14. Section 19's benchmark ran on the local GPU at no cost.*

## In short

- **Most of each step's cost is known now, and the costs add up.** All the steps together drift 1.49 nats from the dense model, close to the 1.45 their separate costs sum to. Each step can be judged and tuned on its own.
- **The learned router is the biggest single cost.** A linear router per layer finds 68% of the right experts. That raises the drift at 25% active from 0.82 with a perfect router to 1.29.
- **Sizing experts by depth did not help without training.** Both depth rules were slightly worse than equal sizes (+1.6% and +5.7%).
- **16-bit DeltaNet state is free.** bf16, fp16, fp16 with stochastic rounding and int16 with row scales all match fp32 within the noise of a 1-bit fp32 perturbation, on 512-token text and in 2,000-token generations. int8 is clearly worse.
- **The 64K vocabulary costs 5% in bits per byte.** Almost all of that comes from words the model now sees split in unfamiliar ways. Text whose tokens did not change lost nothing. Training on the new tokenization should recover most of it.
- **The cache of recently used head rows works, but its miss detector is weak.** With 16K rows, the model's own top pick is outside the cache 4.6% of the time, and 99.7% of greedy picks come out unchanged. To catch the misses, though, the detector sends a third of all positions to the full head, so it saves only half of the 64K head's work.
- **Low rank beats dropping DeltaNet blocks at the bigger cut.** Removing 539M DeltaNet parameters costs 0.156 nats with activation-aware low rank, against 0.274 for dropping the 8 least important blocks. At 270M the two tie (0.09).
- **Unhealed, the whole thing is not usable yet.** Greedy picks agree with the dense model 55% of the time, and bits per byte rise 75%.
- **Healing on 8.2M tokens cut the drift by 59%, from 1.49 to 0.61.** Bits per byte with the 64K vocabulary fell from 1.06 to 0.73 (dense: 0.66), and greedy agreement rose from 55% to 71%. Most of the gain came in the first million tokens. The held-out domains improved as much as the trained ones. But in long generations the healed model often gets stuck repeating itself, which the dense model never did within 2,048 tokens.
- **After healing, the router is no longer the main cost.** Of the 0.61 left, the router accounts for 0.10. Running only 25% of the MLP neurons costs 0.27, and the rest of the model 0.24. Neither int16 state nor the router causes the loops: they happen just as often with fp32 state and with a perfect router.
- **The loops come from running only 25% of the MLP neurons, not from the healing.** With every neuron running, the healed model never loops, finishes 14 of 16 GSM8K answers and gets 10 right (dense: 9). Without the healing, the other cuts alone (low-rank DeltaNet, 64K vocabulary) never finish an answer, so healing repairs them.
- **Switching experts on and off makes the 25% model jumpy, but that is not what makes it loop.** With every neuron running, a tiny numerical change moves the healed model as little as it moves the dense one; with on/off experts it moves it 18 times as much as the dense one. Fading experts in and out instead (a soft edge) removes two thirds of that, at almost no cost in drift, yet the model loops just as often (7 of 16).
- **Folding the skipped neurons into the running ones, after healing, makes the model worse.** Fitted layer by layer, the neurons that run can stand in for 26% of what the skipped ones add; the other three quarters are not predictable from them. Applied to the whole healed model, the fold raises the drift from 0.61 to 0.74 and the loops from 8 to 10 of 16. Turning the shared expert up recovers under 1% of what is lost.
- **Subtracting the fold is far worse; a quarter of it helps a little.** Subtracted, the fold sends the drift to 1.97 and 13 of 16 generations loop. At a quarter of its strength it lowers the drift from 0.612 to 0.597, in every domain, and gives the best 25% generations so far: 4 loops, 11 finished, 7 right. Sixteen prompts cannot prove the loops dropped. The fold points the right way but overshoots at full strength.
- **No change made after healing stops the loops, and sampling only hides them.** On 48 new prompts the healed model loops in 18. Three better-fitted folds lower the drift by up to 0.032, to 0.581, the lowest yet without training, and still loop in 15 or 16: no better than chance. So section 13's 4 of 16 was luck. Which prompts loop changes from run to run. With Qwen's recommended sampling nothing repeats, but 10 of 48 answers still go round in circles until the cap, and fewer are right (14 against 22 greedy).
- **On-policy distillation from the dense model made the 25% model worse.** After 7.2 hours, greedy, it loops in 20 of 48 (before: 18), finishes 6 (before: 30) and gets 7 right (before: 22); its drift rose by 0.031. It took on the dense model's long, self-checking way of answering, and at 25% it cannot bring that to an end. The healed model with every neuron, on the same prompts, never loops, finishes 45 and gets 40 right, more than the dense model's 28.
- **Distilling from the healed model with every neuron instead helped on every count, though the loops are not yet significantly fewer.** Trained only on the answers that teacher finishes, the 25% model loops in 13 of 48 greedy (18 before, p = 0.19), finishes 33 and gets 26 right (22 before). Sampled, it never repeats and gets 27 right (13 before, p = 0.003), about as many as the dense model (29). The drift rose by 0.038, over the guard, so a longer run needs the original healing text back in the mix.
- **With the healing text mixed back in, a longer run brought the loops significantly down and removed the drift, but the gain has levelled out.** Section 16's student was trained for 34 more rounds (9.5 hours, every remaining training prompt) and checked on 96 prompts. Greedy, it loops in 17 against the healed model's 36 (p = 0.002) and gets 54 right against 42. Its drift fell to 0.602, below the healed model's 0.612. But section 16's student already looped in only 21 of these 96. The loop share on the training answers stayed level, and with sampling a third of the answers run to the cap. More LoRA distillation at 25% is not the fix on its own.
- **With twice the room (4,096 tokens), the distilled model's sampled answers turn out mostly long, while its greedy loops get worse.** Sampled, 19 of the 35 answers the 2,048-token cap had cut finish, and it finishes 80 of 96, about as often as the dense model (78). But it gets 52 right against the dense model's 73. Greedy, more of the long answers end in loops: 27 of 96. The dense model also loops in 7 of 96 greedy answers at this length, so "the dense model never loops" held only within 2,048 tokens.
- **On multiple-choice tests, the 25% model keeps less than half of the dense model's ability and scores below the dense Qwen3.5-2B.** On MMLU, ARC and HellaSwag it keeps 0.45 of the dense model's accuracy above chance (MMLU: 42.6% against 67.5%). It is 8.9 points below Qwen3.5-2B, which runs 1.9B parameters per token against its 2.95B. With every neuron running, the same healed model scores above the dense model, so all of the loss comes from running 25% of the MLP neurons. The distilled model is no better than the healed one.

- **More experts per token help steeply, even without retraining.** With the routers picking 16 experts instead of 9 (34% of the MLP, about 3.4B parameters per token), the benchmark KL falls from 0.61 to 0.49 and the multiple-choice share of the dense model rises from 0.45 to 0.65, level with Qwen3.5-2B. At 32 experts it is 0.90, above the 2B. The GSM8K loop check at 16 is still running.

## The benchmark

Ten English and code domains: chat, Python, C++, MATH, GSM8K, Wikipedia, biomedical, argument, reasoning traces and arXiv abstracts. Each has 40 calibration and 10 held-out documents of 512 tokens, and there are 16 copy documents. The score is the KL of the next-token distribution from the stock dense model, averaged per domain and then across domains. Everything was evaluated on the held-out documents; all choices were made on calibration documents.

## 1. Experts with a perfect router

At 25% of MLP neurons active per token:

| Carving | KL | copy p99 |
|---|---:|---:|
| plain prune | 2.21 | 2.91 |
| domain experts, true document domain | 1.67 | 5.90 |
| domain experts, best domain per token | 1.33 | 1.93 |
| **co-firing experts (shared + 64 routed)** | **0.82** | 0.86 |
| each token's own top neurons (a bound) | 0.08 | 0.09 |

The first run (with French and Chinese) gave 0.95 for co-firing experts. At 50% active they give 0.29, at 12.5% 1.90.

**Sizing by depth** used the same total of active neurons:

- **Shared expert sized by depth:** the shared expert grows where the domains share more neurons (640 to 2,496 neurons). Result: 0.834, +1.6% (95% interval +0.004 to +0.023 nats).
- **Active neurons moved between layers:** more go to layers 7–12, fewer to the last layers. Result: 0.867, +5.7%.

Neither helped without training.

## 2. Learned router

A linear router per layer, trained for four passes over the calibration documents:

- **KL:** 1.29 against 0.82 for the perfect router (difference +0.47, interval +0.43 to +0.50).
- **Experts found:** it picks 68% of the perfect router's experts on average, 54–82% depending on the layer.
- **Copy p99:** 1.15.
- **By domain:** chat is worst (2.21), GSM8K best (0.61).
- **CMoE's training-free router on the same experts** (added 2026-10-29, `cmoe_router.py`, results in `cmoe/`): KL 2.475, worse than the trained router by 1.19 (interval 1.15 to 1.23) and worse than a fixed prune (2.21). It finds 26% of the right experts; random picks would find 14%. CMoE's router reuses one representative neuron per expert, the member nearest the cluster centre: it ranks experts by \|silu(x·g)·(x·u)\| from that neuron's normalized gate and up weights. Our experts are smaller and more numerous than CMoE's, which makes this harder for its router. CMoE's own clustering and fine-tuning were not reproduced.

## 3. 64K vocabulary

The kept tokens are:

- the 26 special tokens and the 256 bytes;
- every token seen in calibration;
- then English, symbol and code tokens in merge order;
- plus the merge ancestors every kept token needs.

Token counts grow 0.3% (MATH) to 2.6% (Wikipedia). Bits per byte go from 0.624 to 0.656 (+5.0%), by domain from +1.7% (C++) to +8.1% (Wikipedia). The 9 documents whose tokenization did not change came out 0.2% *better*, because the smaller output drops probability on removed tokens. So the cost comes from the model meeting unfamiliar splits.

## 4. Cache of recently used head rows

Every token is admitted as it is seen and the least recently used row is evicted. The cache is warmed on the calibration stream. A logistic miss detector reads the cached logits plus a predicted full log-normaliser, with its threshold set to catch 95% of misses on calibration positions.

| Rows | next token in cache | model's top pick in cache | mean probability outside | falls back to full head | misses caught | greedy pick unchanged | rows scored per token (of 64K) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 83.4% | 87.5% | 16.5% | 36.0% | 94.4% | 99.37% | 26.2K |
| 16,384 | 93.7% | 95.4% | 6.5% | 33.0% | 92.2% | 99.68% | 32.6K |

Math and reasoning fall back least (15–19%), and argument, Wikipedia and chat most (42–49%). In the combined model, the detector trained on the dense model falls back 64% of the time, so it has to be retrained on whatever model ships.

## 5. DeltaNet state precision

The state is stored after every token; the math stays in fp32. Each format below is measured as KL from the stock chunked kernels.

| Format | KL | Long generations: mean length | finished | correct | median tokens before the text differs from fp32 |
|---|---:|---:|---:|---:|---:|
| fp32, token by token | 6.2e-4 | 1,508 | 9 | 9 | — |
| fp32 + 1-bit jitter (chaos control) | 6.3e-4 | 1,475 | 10 | 9 | 173 |
| bf16 | 7.7e-4 | 1,491 | 11 | 12 | 186 |
| fp16 | 5.9e-4 | 1,501 | 11 | 11 | 141 |
| fp16, stochastic rounding | 5.8e-4 | 1,395 | 13 | 12 | 154 |
| int16, scale per row | 8.6e-4 | 1,562 | 8 | 8 | 173 |
| int8, scale per row | 8.3e-2 | 1,865 | 4 | 6 | 14 |

The generations used 16 GSM8K test questions with greedy thinking, up to 2,048 tokens. Many runs hit the cap, so "correct" mostly tracks whether the reasoning finished in time. Greedy text drifts from fp32 after about 170 tokens even with a 1-bit perturbation, and every 16-bit format behaves like that control. Nothing overflowed. This sample is too small to rule out the slow verbosity drift Nemotron reported at scale.

## 6. Fewer DeltaNet parameters

**Block importance.** Each DeltaNet block's mixer was removed alone on calibration documents. Block 0 is essential: removing it alone costs 6.1 nats. The least important blocks are 29, 26, 28 and 25, then 8, 9, 22 and 10.

| Cut | parameters saved | KL | copy p99 |
|---|---:|---:|---:|
| drop 4 blocks | 270M | 0.089 | 0.044 |
| low rank, same saving | 270M | 0.090 | 0.044 |
| drop 8 blocks | 539M | 0.274 | 0.257 |
| **low rank, same saving** | 539M | **0.156** | 0.079 |

The low-rank cut is SVD-LLM style: in_proj_qkv, in_proj_z and out_proj are truncated after whitening by their input covariance, keeping 67% of their parameters. It won on separate calibration documents, so the combined run used it.

## 7. Everything together

The combined model has:

- co-firing experts at 25% with the learned router;
- int16 state;
- the low-rank DeltaNet cut;
- the 16K cache;
- the 64K vocabulary.

Results:

- **Drift (KL):** 1.49, against 1.45 summed from the separate steps. By domain it runs from 0.67 (GSM8K) to 2.49 (chat).
- **Copy p99:** 1.35.
- **Greedy agreement with dense:** 55.2%, the same with and without the cache.
- **Bits per byte with the 64K vocabulary:** 0.624 → 1.061 (+75%).

## 8. Healing pilot

*Added 2026-10-29. One RTX 6000 Ada, 3.2 hours of training, $3.70 of a $5 cap (ledger `runpod_budget_heal_20261029.json`). Code: `slim_heal.py`. Rules and predictions written before the run: `heal_rules.json`. Pod: `run_heal_pod.py` and `heal_worker.py`. Outputs: `heal/`.*

The combined model of section 7 was distilled from the dense model:

- **What trains:** LoRA rank 64 on every body projection (160M parameters), with the routers (8.4M) trained alongside.
- **Loss:** KL from the dense model's next-token distribution over the 64K kept tokens, plus cross-entropy on the text at half weight.
- **Data:** 2,048-token sequences from 8 of the 10 domains, drawn from items outside the benchmark. Biomedical and argument are held out to test transfer. 16 sequences sharing text with benchmark documents were dropped.
- **Length:** planned for 10M tokens. The run stopped on time at 8.24M (503 of 608 steps), when the learning rate had already decayed to 17% of its peak.

| Tokens | KL | trained domains | held-out domains | 64K bits per byte | greedy agreement | router recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.490 | 1.393 | 1.878 | 1.061 | 55.2% | 68.0% |
| 1.0M | 0.766 | 0.735 | 0.891 | 0.799 | 67.5% | 65.5% |
| 3.0M | 0.676 | 0.643 | 0.809 | 0.756 | 69.5% | 69.1% |
| 8.2M | **0.612** | 0.574 | 0.763 | **0.734** | 71.1% | 72.5% |

Dense: 0.624 bits per byte with the full vocabulary, 0.656 with the 64K one.

- **Before training, the student reproduced section 7:** KL 1.490 against 1.491.
- **The gain comes early.** 82% of it came in the first million tokens and 93% by 3M. From 3M to 8.2M, KL fell by only 0.064, about 0.045 nats per doubling of tokens.
- **Healing generalizes.** The held-out domains lost the same share of their drift as the trained ones (59%). Healing repairs the model in general, not only on the text it sees.
- **The vocabulary cost mostly closed.** 81% of the gap to the dense model's 64K bits per byte closed. On GSM8K the student now beats it (0.547 against 0.612), probably because the healing mix contains GSM8K-style text.
- **The router improved little.** Its recall first fell (65.5% at 1M) as the experts changed under it, then rose to 72.5%.
- **Copy p99:** 1.25 → 0.75.

| Domain | KL before | KL after | 64K bits per byte after | dense, 64K |
|---|---:|---:|---:|---:|
| chat | 2.49 | 0.94 | 0.625 | 0.535 |
| Python | 1.12 | 0.61 | 0.614 | 0.445 |
| C++ | 0.94 | 0.41 | 0.677 | 0.654 |
| MATH | 0.73 | 0.30 | 0.552 | 0.521 |
| GSM8K | 0.67 | 0.27 | 0.547 | 0.612 |
| Wikipedia | 2.00 | 0.84 | 0.952 | 0.796 |
| reasoning traces | 1.55 | 0.60 | 0.479 | 0.358 |
| arXiv abstracts | 1.64 | 0.63 | 0.903 | 0.850 |
| biomedical (held out) | 2.03 | 0.81 | 0.868 | 0.762 |
| argument (held out) | 1.73 | 0.72 | 1.067 | 0.994 |

**DeltaNet state.** The state check used 2 documents per domain:

- With int16 state, the healed student's KL from dense is 0.681, against 0.679 with fp32.
- But the two versions differ from each other by 0.014 nats, 17 times the 8.6e-4 the dense model showed in section 5.
- This run had no fp32 token-by-token control, so it cannot tell how much of that comes from int16 and how much from running token by token. Section 9 checks this: it is the healed model's sensitivity to any small change, not int16.

**Cache detector, retrained on the healed student:**

| Rows | next token in cache | model's top pick in cache | falls back to full head | misses caught | greedy pick unchanged | rows scored per token (of 64K) |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 83.7% | 89.8% | 39.9% | 94.3% | 99.49% | 28.6K |
| 16,384 | 94.3% | 96.7% | 41.9% | 93.8% | 99.81% | 37.0K |

The fallback is down from 64% with the dense model's detector, but above the 33% the dense model gets with its own. As in section 4, 4K rows are cheaper per token and 16K rows keep more greedy picks.

**Long generations.** The same 16 GSM8K prompts as section 5, greedy, up to 2,048 tokens:

| | correct | finished | mean length | stuck repeating |
|---|---:|---:|---:|---:|
| dense, fp32 state | 9 | 9 | 1,508 | 0 |
| healed, int16 state | 6 | 8 | 1,145 | 7 |

- **Loops are the main failure.** 7 of the 8 runs that hit the cap were stuck in a loop: at least 45% of the 20-token spans in their last 1,000 tokens repeat an earlier span. One of them repeated "#### 20" after the right answer, which is why it counts as correct.
- **When it finishes, the student is brief.** It used 145 to 473 tokens, where the dense model used 647 to 2,048. Five of its 8 finished answers are right, including 2 prompts on which the dense model ran out of room.
- **The benchmark cannot see this.** It scores KL on 512-token documents with the true text as input, so it never shows the model running on its own output.

| Prediction (written before the run) | Result |
|---|---|
| Student reproduces section 7 before training | yes, KL −0.001, bits per byte −0.03% |
| KL below 0.6 | **no**, 0.612 (stopped on time at 8.2M of 10M tokens) |
| 64K bits per byte below 0.75 | yes, 0.734 |
| At least 70% of the gain by 3M tokens | yes, 93% |
| Held-out domains lose at least half the trained domains' share | yes, the same share (59%) |
| Router recall up at least 5 points | **no**, +4.6 |
| int16 state changes KL by under 0.01 | yes, +0.002 |
| Retrained detector falls back under 40% (16K rows) | **no**, 41.9% |
| At least 4 of 16 GSM8K prompts right | yes, 6 (dense 9), but 7 runs loop |

The rule set before the run for a longer healing run was KL below 0.6 and still falling by at least 10% between the last two checkpoints. Both parts **failed narrowly**: KL is 0.612, and it fell 9.5%.

## 9. Checks on the healed model

*Added 2026-10-30. One A40, 1.3 hours, $0.79 of a $1.50 cap (ledger `runpod_budget_probe_20261030.json`). Code: `heal_probe.py`. Rules and predictions written before the run: `heal_probe_rules.json`. Pod: `run_probe_pod.py` and `probe_worker.py`. Outputs: `probe/`.*

The healing pod did not keep the low-rank DeltaNet factors, so they were rebuilt from the calibration documents and combined with the saved adapter. On the A40 the rebuilt factors are not bit for bit those of the healing run, but the student reproduces it: KL +0.0002, 64K bits per byte +0.05%.

**Where the remaining drift sits.** The healed model with three ways of choosing its experts:

| Experts | KL | trained domains | held-out domains | 64K bits per byte | greedy agreement |
|---|---:|---:|---:|---:|---:|
| learned router | 0.612 | 0.575 | 0.763 | 0.734 | 71.3% |
| perfect router | 0.517 | 0.496 | 0.601 | 0.700 | 73.5% |
| every neuron runs | 0.242 | 0.242 | 0.244 | 0.616 | 81.0% |

- **The router costs 0.095** (95% interval 0.086 to 0.104), 16% of what is left.
- **Running only 25% of the neurons costs 0.275** (0.259 to 0.290), 45%. This is the perfect router against every neuron.
- **The rest costs 0.242**, 40%: the low-rank DeltaNet, the vocabulary and whatever else healing changed.
- **With every neuron running, the held-out domains drift no more than the trained ones** (0.244 against 0.242). Bits per byte (0.616) are below the dense model's with the 64K vocabulary (0.656), and even with the full one (0.624).
- **Caveat:** healing trained the LoRA with 25% of the neurons running. The every-neuron model was never trained that way; it only shows what the sparse MLP costs.

| Domain | learned router | perfect router | every neuron |
|---|---:|---:|---:|
| chat | 0.93 | 0.84 | 0.34 |
| Python | 0.61 | 0.55 | 0.30 |
| C++ | 0.41 | 0.37 | 0.21 |
| MATH | 0.30 | 0.27 | 0.17 |
| GSM8K | 0.27 | 0.23 | 0.15 |
| Wikipedia | 0.84 | 0.67 | 0.29 |
| reasoning traces | 0.60 | 0.52 | 0.24 |
| arXiv abstracts | 0.63 | 0.53 | 0.24 |
| biomedical (held out) | 0.81 | 0.62 | 0.24 |
| argument (held out) | 0.72 | 0.59 | 0.24 |

The router matters most on Wikipedia and the held-out domains, the sparse MLP most on chat.

**DeltaNet state.** On 5 documents per domain:

| State | KL from dense |
|---|---:|
| fp32, chunked kernel | 0.621 |
| fp32, token by token | 0.620 |
| int16, token by token | 0.619 |

- **int16 costs nothing:** it is as close to the dense model as fp32.
- **The healed model is sensitive to any small numerical change.** Switching the fp32 kernel from chunked to token by token moves its output by 0.011 nats. The same switch moved the dense model by 6.2e-4 (section 5), 18 times less. int16 moves it by about the same amount: 0.013 from either fp32 run. So the 0.014 seen in section 8 is this sensitivity, not an int16 error.
- **A likely cause, not tested:** the hard choice of the top K experts. A tiny change can flip an expert that is nearly tied with another.

**Generations.** The same 16 GSM8K prompts, greedy, up to 2,048 tokens:

| | correct | finished | mean length | stuck in a loop at the cap |
|---|---:|---:|---:|---:|
| dense, fp32 state | 9 | 9 | 1,508 | 0 |
| healed, learned router, int16 state | 6 | 8 | 1,145 | 7 |
| healed, learned router, fp32 state | 6 | 6 | 1,354 | 8 |
| healed, perfect router, fp32 state | 6 | 5 | 1,519 | 10 |

- **Neither the state format nor the router causes the loops.** fp32 state loops as often as int16, and the perfect router loops more.
- **The healed model either answers briefly or never finishes.** In all three runs, every finished answer took 108 to 531 tokens; the dense model's took 647 to 1,588.
- **Which prompts fail changes from run to run**, as the sensitivity above would predict. Each run got 6 right, but different ones, and only 3 prompts looped in all three runs.

| Prediction (written before the run) | Result |
|---|---|
| The rebuilt student reproduces the healed one | yes, KL +0.0002, bits per byte +0.05% |
| A perfect router lowers KL by at least 0.1 | **no**, narrowly: 0.095 (0.086 to 0.104) |
| Every neuron gives a lower KL than the perfect router | yes, 0.242 against 0.517 |
| int16 itself is under half of the 0.014 | **no**: 0.013 from fp32 token by token, but the fp32 kernel switch alone moves it 0.011 |
| fp32 state: at least 5 loops | yes, 8 |
| Perfect router: at least 4 loops | yes, 10 |

## 10. Every neuron, with and without healing

*Added 2026-10-30. One A40, 1.3 hours, $0.76 of a $1.20 cap (ledger `runpod_budget_probe_all_20261030.json`). Code: `heal_probe.py`. Rules, predictions and how to read the result, written before the run: `heal_probe_all_rules.json`. Pod: `run_probe_all_pod.py` and `probe_all_worker.py`. Outputs: `probe_all/`.*

The question: do the loops come from running only 25% of the MLP neurons, or from the healing? Both versions here run every neuron. The unhealed one drops the healing adapter, so its only cuts are the low-rank DeltaNet and the 64K vocabulary. The rebuilt student again reproduced the healing run (KL +0.0002, 64K bits per byte +0.05%), and the healed version reproduced section 9's 0.242.

| Every neuron runs | KL | trained domains | held-out domains | 64K bits per byte | greedy agreement |
|---|---:|---:|---:|---:|---:|
| healed | 0.242 | 0.242 | 0.244 | 0.616 | 81.0% |
| unhealed | 0.155 | 0.153 | 0.166 | 0.618 | 85.5% |

- **With every neuron running, the adapter adds 0.087 of drift** (95% interval 0.082 to 0.093). It was trained to make up for missing neurons, and overshoots when they all run. So of section 9's 0.242 "rest", about 0.155 is the DeltaNet cut and the vocabulary, and the rest is this overshoot.
- **Bits per byte stay the same**, because the adapter also learned the 64K tokenization.

**Generations.** The same 16 GSM8K prompts, greedy, up to 2,048 tokens:

| | correct | finished | mean length | stuck in a loop at the cap |
|---|---:|---:|---:|---:|
| dense, fp32 state | 9 | 9 | 1,508 | 0 |
| healed, learned router, int16 state | 6 | 8 | 1,145 | 7 |
| healed, learned router, fp32 state | 6 | 6 | 1,354 | 8 |
| healed, perfect router, fp32 state | 6 | 5 | 1,519 | 10 |
| **healed, every neuron, fp32 state** | **10** | **14** | 1,334 | **0** |
| unhealed, every neuron, fp32 state (15 prompts) | 2 | 0 | 2,048 | 4 |

- **The loops come from the 25% sparsity, not from the healing.** The same adapter with every neuron running never loops, finishes 14 of 16 and gets 10 right. Its finished answers take 445 to 2,014 tokens (median 1,148), like the dense model's. So the short answers at 25% came from the sparsity too.
- **Healing repairs what the other cuts break.** Without the adapter, the model is closer to dense on text (0.155) but never finishes an answer. All 15 ran to the cap: 4 stuck in a loop, the rest going round in circles ("Wait, let's check the grammar again"). Only 2 of 15 closed their thinking, and both then repeated `</think>` until the cap. Which of the two cuts does this was not tested.
- **The unhealed run stopped after 15 of the 16 prompts**, at its 0.6-hour limit, because every answer ran to the cap. The last prompt cannot change the reading: none of the 15 finished.
- **Against the rule written before the run:** the healed set is clean (0 loops; clean meant at most 2). The unhealed set is "unclear" (4 in 15; looping meant at least 5). The rule did not foresee that combination. Read plainly, the healing is cleared and the sparsity causes the loops.

| Prediction (written before the run) | Result |
|---|---|
| The rebuilt student reproduces the healed one | yes, KL +0.0002, bits per byte +0.05% |
| Every neuron reproduces section 9's 0.242 | yes, 0.2425 |
| Without the adapter, every neuron drifts less than with it | yes, 0.155 against 0.242 |
| Without the adapter, at most 2 loops | **no**, 4 of 15, and none finished |
| With the adapter and every neuron, at least 3 loops: healing is part of the cause | **no**, 0 |
| The healed model's finished answers stay short (median under 600); the unhealed one's do not | **no**: median 1,148, and the unhealed model finished none |

## 11. A soft edge on the router

*Added 2026-10-30. One A40, 0.9 hours, $0.51 of a $1.00 cap (ledger `runpod_budget_probe_soft_20261030.json`). Code: `heal_probe.py`. Rules, predictions and how to read the result, written before the run: `heal_probe_soft_rules.json`. Pod: `run_probe_soft_pod.py` and `probe_soft_worker.py`. Outputs: `probe_soft/`.*

The question: do the loops come from switching whole experts on and off? The learned router picks 9 of 64 experts, so a tiny change can swap the 9th for the 10th. The soft edge keeps the same 9 experts and the same 25% budget, but weights each chosen expert by min(1, (its score − the best score left out) / width). An expert swapping places with the first one left out weighs about 0 on both sides of the swap. Each layer's width is its median gap between the 9th and 10th router score on 20 calibration documents (0.008 to 0.046). Nothing was retrained.

| Healed model, 25% of neurons | KL | trained domains | held-out domains | 64K bits per byte | greedy agreement |
|---|---:|---:|---:|---:|---:|
| learned router | 0.612 | 0.575 | 0.763 | 0.734 | 71.3% |
| learned router with a soft edge | 0.620 | 0.582 | 0.773 | 0.736 | 71.0% |

The soft edge costs 0.008 of drift (95% interval 0.006 to 0.009).

**Sensitivity:** how far the chunked fp32 kernel and fp32 token by token, the same math in a different order, move the output apart. On 5 documents per domain:

| | KL between the two |
|---|---:|
| dense model (section 5) | 6.2e-4 |
| healed, every neuron | 4.7e-4 |
| healed, learned router | 1.1e-2 |
| healed, learned router with a soft edge | 3.9e-3 |

- **The switching causes all of the extra sensitivity.** With every neuron running, the healed model is as stable as the dense one.
- **The soft edge removes 65% of it.** The rest comes from experts that are not near the edge but still change rank.

**Generations.** The same 16 GSM8K prompts, greedy, up to 2,048 tokens:

| | correct | finished | mean length | stuck in a loop at the cap |
|---|---:|---:|---:|---:|
| dense, fp32 state | 9 | 9 | 1,508 | 0 |
| healed, learned router, fp32 state | 6 | 6 | 1,354 | 8 |
| healed, perfect router, fp32 state | 6 | 5 | 1,519 | 10 |
| healed, every neuron, fp32 state | 10 | 14 | 1,334 | 0 |
| **healed, learned router with a soft edge, fp32 state** | 7 | 8 | 1,182 | **7** |

- **The soft edge does not stop the loops.** They stay at 7 of 16, and finished answers stay short (median 239 tokens; dense 984).
- **So the jumpiness is not what drives the loops.** What does is what each token loses by skipping 75% of the MLP, however the experts are switched. By the rule written before the run, the next things to try are the shared expert (larger at the same budget, or weighted more) and merging more neurons into each expert.

| Prediction (written before the run) | Result |
|---|---|
| The rebuilt student reproduces the healed one | yes, KL +0.0002, bits per byte +0.05% |
| The soft edge raises KL by under 0.05 | yes, +0.008 |
| Every neuron brings sensitivity within 3 times the dense model's | yes, 4.7e-4 against 6.2e-4 |
| The soft edge at least halves the learned router's sensitivity | yes, by 65% |
| With the soft edge, at least 5 still loop | yes, 7 |

## 12. Folding the skipped neurons into the ones that run

*Added 2026-10-30. One A40, two attempts: the first ran out of GPU memory after fitting 16 of 32 layers ($0.13); the second ran in 0.9 hours ($0.54). Total $0.68 of a $1.20 cap (ledger `runpod_budget_probe_fold_20261030.json`). Code: `heal_probe.py` (`fit_layer`). Rules, predictions and how to read the result, written before the first attempt: `heal_probe_fold_rules.json`; its `rerun` note says what changed for the second. Pod: `run_probe_fold_pod.py` and `probe_fold_worker.py`. Outputs: `probe_fold/`.*

The idea, from the user: fold all the experts into each expert by some kind of averaging, but give each expert's own neurons most of the weight. Tested here on the healed model, at the same 25% budget and with no training:
- The output weights of the neurons that run change so that, on calibration text, they also add what the skipped neurons would have added, as far as the running neurons can predict it.
- This is one ridge regression per layer, toward the original weights. Its penalty is the preference for each neuron's own weights; held-out calibration text sets the penalty per layer.
- The target is the healed model with every neuron running, which never looped.

Two weaker versions fit the same loss with fewer numbers: one gain per expert, or a single gain on the shared expert (the "weigh it more" idea).

Each layer is fitted alone, on the inputs of the unfolded model: 300 calibration documents to fit, and 100 more to choose the penalty.

**Layer by layer, the fold does what it was fitted to do.** On held-out calibration text, averaged over the 32 layers:

| | share of the skipped neurons' output recovered |
|---|---:|
| the fold | 26% (19% to 68% by layer; 23% at the median) |
| one gain per expert | 1.3% |
| one gain on the shared expert | 0.7% |

- **What the skipped neurons carry:** 42% of the every-neuron MLP output on average. That is up to 58% in the middle layers and 4% in the last one.
- **The shared-expert gain:** the fit picks 1.06 on average (1.00 to 1.18), but turning the shared expert up that much recovers under 1% of the loss.
- **How much the fold moves the weights:** to recover its 26%, the fold changes each running neuron's output weights by 37% of their size at the median. The held-out text chose a light penalty.

**For the whole model, it makes things worse.**

| Healed model, 25% of neurons | KL | trained domains | held-out domains | 64K bits per byte | greedy agreement |
|---|---:|---:|---:|---:|---:|
| learned router | 0.612 | 0.575 | 0.763 | 0.734 | 71.3% |
| + one gain on the shared expert | 0.618 | 0.580 | 0.773 | 0.736 | 71.0% |
| + one gain per expert | 0.620 | 0.582 | 0.773 | 0.737 | 70.7% |
| + the fold | 0.743 | 0.703 | 0.903 | 0.788 | 67.5% |

The shared-expert gain costs 0.006 (95% interval 0.004 to 0.008). The fold costs 0.123 more than the expert gains (0.111 to 0.133), and it is worse in all 10 domains.

**Generations**, the same 16 GSM8K prompts as in section 11:

| | correct | finished | mean length | stuck in a loop at the cap |
|---|---:|---:|---:|---:|
| healed, learned router, fp32 state | 6 | 6 | 1,354 | 8 |
| healed, every neuron, fp32 state | 10 | 14 | 1,334 | 0 |
| **healed, learned router with the fold, fp32 state** | 2 | 5 | 1,462 | **10** |

- **The fold does not stop the loops; it adds two.** Its finished answers are short (median 154 tokens; dense 984).
- **Why it helps each layer but hurts the whole model.** There are two likely reasons, not yet told apart:
  - Each layer was fitted alone, on the unfolded model's inputs. With all 32 layers folded, every layer's inputs change, and the 37% weight changes add up.
  - The healing tuned the whole model, its adapter included, to the 25% outputs. Pushing those outputs toward the every-neuron ones moves the model away from what healing found. Even the expert gains, which change the outputs very little, make it slightly worse.
- **What is left of the idea:**
  - About three quarters of what the skipped neurons add cannot be predicted linearly from the neurons that run. A fold can still be fitted before healing, as healing's starting point, or learned by healing itself, which would then train every neuron's output weights instead of a rank-64 update. Neither can recover more than the running neurons carry.
  - A fixed gain on the shared expert is ruled out. CMoE's gain, trained and different for each token, is untested.
- **The rules' reading does not apply cleanly.** The fold recovered 26%, between the one-tenth and one-third thresholds, and the rules did not foresee the drift rising.

| Prediction (written before the run) | Result |
|---|---|
| The rebuilt student reproduces the healed one | yes, KL +0.0002, bits per byte +0.05% |
| The fold recovers at least 20% of the lost output (mean over layers) | yes, 26% (the first attempt's 16 layers were seen before the rerun) |
| One gain per expert recovers under a quarter of the fold's share, and the shared gain less still | yes, 1.3% and 0.7% |
| Neither gain lowers KL by more than 0.02 | yes; both raise it slightly |
| The fold lowers KL below 0.56 | **no**, it rises to 0.743 |
| With the fold, at least 5 still loop | yes, 10 |

## 13. Subtracting the fold instead

*Added 2026-10-30. One A40, 1.6 hours, $0.96 of a $1.50 cap (ledger `runpod_budget_probe_minus_20261030.json`). Rules, predictions and how to read the result, written before the run: `heal_probe_minus_rules.json`. Pod: `run_probe_minus_pod.py` and `probe_minus_worker.py`. Outputs: `probe_minus/`.*

The user's question after section 12: adding the fold made the loops worse, so what happens if the other neurons are subtracted instead?
- Subtracting the skipped neurons' real output would need every neuron to run. So this subtracts the fold, the part of that output the running neurons predict, at no extra cost.
- The fold was refitted exactly as in section 12: the same penalty and the same recovery in every layer, and the same KL at full strength.
- It was then applied at strengths from −1 to +1 on the healed model, with no training.
- The two gains were mirrored too: each gain's distance from 1 was reversed, so the shared expert is weighed about 0.94 instead of 1.06.

**Drift along the fold's direction:**

| Fold strength | KL | against no fold (95% interval) | 64K bits per byte | greedy agreement |
|---:|---:|---:|---:|---:|
| −1 (subtracted) | 1.972 | +1.360 (+1.320 to +1.400) | 1.225 | 47.9% |
| −½ | 0.878 | +0.266 (+0.255 to +0.276) | 0.827 | 65.6% |
| −¼ | 0.689 | +0.077 (+0.072 to +0.081) | 0.760 | 69.4% |
| 0 (learned router, no fold) | 0.612 | — | 0.734 | 71.3% |
| **+¼** | **0.597** | **−0.015 (−0.019 to −0.012)** | 0.730 | 71.3% |
| +½ | 0.621 | +0.009 (+0.003 to +0.014) | 0.741 | 70.6% |
| +1 (section 12's fold) | 0.743 | +0.131 (+0.120 to +0.141) | 0.788 | 67.5% |

The mirrored gains cost a little too:
- expert gains: 0.623, +0.011 (+0.009 to +0.012);
- the shared expert at about 0.94: 0.615, +0.002 (0.000 to +0.005).

- **Subtracting is far worse than adding.** With the whole fold subtracted, the drift is 1.97, more than the unhealed carve with every cut (1.49). The fold carries real signal, and taking it out of the running neurons breaks the model.
- **A quarter of the fold helps a little.** It lowers KL by 0.015 in all 10 domains, held-out ones included (0.763 → 0.740). A parabola through the three middle points bottoms out near +0.2, so +¼ is about as good as this line gets.
- **The fold points the right way, but at full strength it overshoots.**
- **The healed model sits at the bottom along both gains.** Turning the shared expert up or down both cost.

**Generations**, the same 16 GSM8K prompts:

| | correct | finished | stuck in a loop at the cap | finished median length |
|---|---:|---:|---:|---:|
| healed, learned router, fp32 state | 6 | 6 | 8 | 186 |
| + the fold (section 12) | 2 | 5 | 10 | 154 |
| + the fold subtracted | 1 | 2 | 13 | 477 |
| **+ a quarter of the fold** | **7** | **11** | **4** | 254 |
| healed, every neuron, fp32 state | 10 | 14 | 0 | 1,149 |
| dense, fp32 state | 9 | 9 | 0 | 984 |

- **Subtracted, the fold loops in 13 of 16.**
- **A quarter of the fold gives the best 25% generations so far.** It looped in 4, where every earlier 25% run looped in 7 to 10, and it finished 11.
  - The rules call 4 "unclear" (3 or 4).
  - With 16 prompts, and runs that already vary from one to the next (section 9), this is a hint, not proof.
  - Finished answers stay short: median 254 tokens, against the dense model's 984.
- **Reading, as the rules set it out:** a small positive strength lowers KL, so the fold points the right way and overshoots at full strength.
  - The likely cause is that each layer was fitted alone, on the unfolded model's inputs, so with all 32 layers folded the changes compound.
  - The next step the rules name is fitting the layers in order, each on the folded model's own inputs.

| Prediction (written before the run) | Result |
|---|---|
| The rebuilt student reproduces the healed one | yes, KL +0.0002, bits per byte +0.05% |
| The refitted fold reproduces section 12's KL within 0.01 | yes, 0.743, with the same penalty and recovery in every layer |
| Subtracting the whole fold raises KL to at least 0.66 | yes, 1.97 |
| No strength lowers KL by more than 0.01 | **no**: a quarter lowers it by 0.015 |
| The mirrored gains raise KL by more than 0.002 each | yes for the expert gains (+0.011); only just for the shared gain (+0.0023, its interval reaching 0) |
| With the fold subtracted, at least 5 still loop | yes, 13 |

## 14. Six checks side by side: can a fold or sampling stop the loops?

*Added 2026-10-31. Six A40 pods side by side, $5.03 in all of $10.30 in caps (one ledger per pod, `runpod_budget_probe_loops_<name>_20261030.json`). Rules, predictions and how to read the result, written before the run: `heal_probe_loops_rules.json`, and one `heal_probe_loops_<name>_rules.json` per pod. Pod: `run_probe_loops_pod.py` and `probe_loops_worker.py`. Outputs: `probe_loops_<name>/`, with the combined report in `probe_loops/`.*

The user, after section 13: keep pursuing this line, see if the loops can be eliminated, and run several experiments in parallel.
- Every pod used the healed model with its learned routers at 25%, and nothing was trained.
- Generations used 48 GSM8K test prompts the model had never been run on (rows 16–63; healing used the train split), three times as many as before.

| Pod | What it changes |
|---|---|
| control | nothing |
| fold | section 13's fold at strengths 0.1 to 0.3; generates at the strength with the lowest KL |
| order | the fold fitted layer by layer in order, each layer on the inputs of the model already folded before it; strengths ⅛ to 1, the same choice |
| track | also fitted in order, but each layer's fold aims at the every-neuron model's output for that whole layer, so it also makes up for drift from earlier layers; strengths ⅛ to 1, the same choice |
| sample | nothing, but generates with the sampling the model card recommends for thinking mode: temperature 1, top-p 0.95, top-k 20, presence penalty 1.5 (Qwen recommends it because greedy decoding can repeat endlessly) |
| sample_fold | a quarter of the fold, with that sampling |

**Drift.** Every pod reproduced the healed model (KL +0.0002, bits per byte +0.05%), and both pods with section 13's fold reproduced its quarter strength (0.597).

| Fold strength | 0.1 | ⅛ | 0.15 | 0.2 | ¼ | 0.3 | ½ | ¾ | 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fitted together (sections 12–13) | 0.601 | | **0.596** | 0.598 | 0.597 | 0.599 | *0.621* | | *0.743* |
| fitted in order | | 0.595 | | | **0.590** | | 0.598 | 0.633 | 0.697 |
| tracking every neuron | | 0.588 | | | **0.581** | | 0.590 | 0.649 | 0.747 |

KL without a fold: 0.612. Values in italics are section 13's.

| Best of each fold | KL | against no fold (95% interval) | held-out KL | 64K bits per byte | greedy agreement |
|---|---:|---:|---:|---:|---:|
| none | 0.612 | — | 0.763 | 0.734 | 71.3% |
| fitted together, × 0.15 | 0.596 | −0.017 (−0.021 to −0.014) | 0.740 | 0.729 | 71.6% |
| fitted in order, × ¼ | 0.590 | −0.022 (−0.027 to −0.019) | 0.732 | 0.728 | 71.5% |
| tracking every neuron, × ¼ | **0.581** | **−0.032 (−0.037 to −0.027)** | 0.713 | 0.726 | 72.0% |

- **Each fold lowers the drift in all 10 domains** at its best strength. The fitted-together fold is flat from 0.15 to 0.3 (within 0.004).
- **Fitting in order helps a little but still overshoots.** Each layer's fold recovers more of what the layer loses (33% against 26%), and its best KL is 0.006 lower. But the best strength stays at a quarter, and at full strength KL is 0.697. So the overshoot does not come only from layers fitted alone.
- **Tracking the every-neuron model gives the lowest drift yet without training: 0.581.** The held-out domains gain most (0.763 → 0.713).
- **Tracking also shows how far the 25% model drifts.** The gap between each layer's output and the every-neuron model's grows with depth.
  - At layer 0 it carries 0.23 times the energy of the every-neuron MLP output.
  - From layer 4 to 30 it carries more than that whole output: 1.2 to 4.2 times, 2.4 on average.
  - The neurons that run predict only 6% of it there.

**Generations**, the 48 prompts, up to 2,048 tokens:

| | correct | finished | at the cap, stuck in a loop | at the cap, not repeating | finished median length |
|---|---:|---:|---:|---:|---:|
| healed, learned router (control) | 22 | 27 | 18 | 3 | 236 |
| + the fold × 0.15 | 23 | 33 | 15 | 0 | 244 |
| + the fold fitted in order × ¼ | 22 | 32 | 16 | 0 | 245 |
| + the tracking fold × ¼ | 22 | 31 | 16 | 1 | 242 |
| sampled | 14 | 38 | 0 | 10 | 216 |
| + the fold × ¼, sampled | 20 | 32 | 0 | 16 | 259 |

All use fp32 state. Greedy unless marked sampled.

- **No fold reduces the loops.** They loop in 15 and 16 of 48, against 18. The one-sided Fisher p values are 0.33, 0.42 and 0.42; the rules asked for p < 0.05. Section 13's 4 of 16 was luck. The best fold also finishes 6 more answers (33 against 27), but that is not significant either (p = 0.15).
- **Lower drift does not bring fewer loops.** The tracking fold lowers KL by 5% and loops as often as the others.
  - KL is scored with the real text as input, and loops happen when the model runs on its own output (section 8: the benchmark cannot see this).
  - The every-neuron model, at 0.24, never loops (section 10). The 25% models sit at 0.58 to 0.61 and loop in about a third, so 0.03 is far too small a step.
- **Which prompts loop is mostly chance.**
  - Only 3 prompts looped in all four greedy runs, and 14 in none. The other 31 looped in one, two or three.
  - Of the control's 18 looping prompts, each fold finished 10.
  - So any small change reshuffles which prompts loop, and the rate stays near a third. Section 9's 18-fold sensitivity predicts this.
- **Sampling ends the exact repetition but not the failure to finish.**
  - With the recommended sampling, no answer repeated itself; the presence penalty prevents that.
  - Still, 10 of 48 ran to the cap, against 21 greedy (one-sided p = 0.014).
  - None of those 10 repeated a 20-token span. Instead they went round in circles in new words ("But wait, maybe the problem is trying to set up a different scenario…"), as the unhealed every-neuron model did in section 10.
  - Sampling also costs accuracy: 14 right against 22 greedy (p = 0.07).
- **With sampling, the fold finished fewer but got more right.** It finished 32 against 38 (p = 0.13) and got 20 right against 14 (p = 0.14). Neither difference is significant.
- **Not measured here:** the dense and every-neuron models on these 48 prompts, greedy or sampled. So it is unknown how often the dense model runs to the cap with this sampling, or how many of these it gets right.

**Reading, as the rules set it out:**
- No fold reduces the loops. So the fold line ends here, and the next options are a bigger shared expert or on-policy distillation.
- The sampled control is clean by the rules: 0 loops, where at most 3 counted as clean. So the loops are mostly an effect of greedy decoding, and later work should be judged on sampled generations as well.
- It does not count as eliminated, which needed 40 of 48 finished; it finished 38.
- Read plainly: the exact repetition is how greedy decoding shows the problem, but the drift beneath it stays. With sampling, a fifth of the answers still never finish.

| Prediction (written before the run) | Result |
|---|---|
| Every pod reproduces the healed model; a quarter of the fold gives 0.597 within 0.005 | yes: +0.0002, and 0.597 in both pods |
| The control loops in at least 15 of 48 | yes, 18 |
| The fold's best strength is 0.15 to 0.25, with KL within 0.005 of 0.597 | yes: 0.15, at 0.596 |
| The best fold loops in more than half as many as the control | yes, 15 against 18 |
| Fitted in order, the best strength is at least ½ and KL beats the best fold fitted together | **no** for the strength (¼); yes for the KL (0.590 against 0.596) |
| The tracking fold's best KL is below 0.58 and the lowest of the three | **no**, 0.5806, by 0.0006; it is the lowest |
| No greedy fold arm is clean | yes: 15, 16 and 16 |
| Sampled, the control loops in at most 5 but finishes fewer than 40 | yes: 0 loops, 38 finished |
| Sampled, a quarter of the fold finishes at least as many as the control | **no**, 32 against 38 |

## 15. On-policy distillation from the dense model

*Added 2026-10-31. One A40, about $4.75 of the $10 cap (ledger `runpod_budget_opd_20261031.json`). Rules, predictions and how to read the result, written before the run: `opd_rules.json`. Code: `slim_opd.py`. Pod: `run_opd_pod.py` and `opd_worker.py`. Outputs: `opd/`. Partway through, Claude Code stopped the local launcher and its watchdog because the computer was low on memory. `tend_pod.py`, run every 5 minutes by a systemd timer the user approved, then copied the outputs, deleted the pod when the worker finished and settled the ledger.*

After section 14 the user approved a $5–10 on-policy distillation run. The student is the healed model with its learned routers at 25%, and only its LoRA updates train.
- Each round has 96 prompts: GSM8K training questions and UltraChat first turns, alternating. None are among the 48 check prompts.
- The dense model answers 48 of them, sampled. The student answers the other 48, half greedy and half sampled. Each student answer is cut at its first long repeat.
- Each round then takes 6 optimizer steps on both sets: forward KL from the dense model on the dense model's answers, and reverse KL on the student's answers, over the 64K rows.
- In all: 26 rounds, 156 steps, 2,496 prompts and 3.8M positions (2.2M on the dense model's answers, 1.6M on the student's), in 7.2 hours.

**Before training.** The rebuilt student reproduced the healed one (KL +0.0002). Left padding moved its outputs by 0.016 nats and the dense model's by 0.0006, both well under the limits.

**Generations**, the 48 check prompts of section 14, batched, up to 2,048 tokens, fp32 state:

| | correct | finished | at the cap, stuck in a loop | at the cap, not repeating | finished median length |
|---|---:|---:|---:|---:|---:|
| dense | 28 | 23 | 0 | 25 | 1,082 |
| dense, sampled | 29 | 23 | 0 | 25 | 1,009 |
| healed, every neuron | **40** | **45** | 0 | 3 | 893 |
| healed, every neuron, sampled | 30 | 43 | 0 | 5 | 837 |
| healed, 25% (control) | 22 | 30 | 18 | 0 | 194 |
| healed, 25%, sampled | 13 | 32 | 0 | 16 | 251 |
| distilled, halfway (round 14) | 18 | 9 | 19 | 20 | 970 |
| distilled | 7 | 6 | 20 | 22 | 991 |
| distilled, sampled | 16 | 12 | 0 | 36 | 1,147 |

Greedy unless marked sampled. Sampling is the model card's, as in section 14. An answer counts as correct when its last number is right, so an answer cut at the cap can still count.

- **The references were run on these prompts for the first time.**
  - The dense model never loops, but it runs to the cap in 25 of 48: it thinks long, and 2,048 tokens is often not enough.
  - The healed model with every neuron is the best of all: 45 finished and 40 right. It thinks almost as long, but ends.
- **The batched control matches section 14**, which ran one prompt at a time: 18 loops either way, and 30 finished against 27.
- **Distillation made every count worse.**
  - Greedy: 20 loops against 18 (one-sided p = 0.73 for fewer), 6 finished against 30, 7 right against 22.
  - Sampled: 12 finished against 32, and 16 right against 13 (p = 0.33).
- **The student took on the dense model's way of answering.**
  - Before: the healed student answered directly ("Here's the step-by-step solution:"), in about 200 tokens.
  - After: 40 of its 48 greedy answers open as the dense model's do ("Here's a thinking process that leads to the solution:"), and they check themselves over and over.
  - At 25%, that checking does not end. A typical answer, at the cap: "Let me re-read the prompt again… Wait, is it possible the prompt implies she works 50 weeks *total*? No… Let me re-read the prompt again…".
  - The halfway check had already changed the same way.
- **Training did not lower the loops on the student's own answers either.**
  - The share of greedy training answers cut for a loop was 0.54 in round 0. From round 1 on it was 0.75 to 0.96 in all but two rounds (0.50 and 0.58).
  - By quarter of the run: 0.74, 0.85, 0.84, 0.79. Sampled: 0.38, 0.39, 0.40, 0.38.
  - The share of greedy answers that finished fell from 0.46 in round 0 to 0.17 or less in every later round but one (0.38). Their mean length rose from 410 tokens to about 1,100.
- **The losses fell all the same.** The forward KL on the dense model's answers fell from 0.78 to 0.58, and the reverse KL on the student's own answers from 0.48 to 0.37. So the student matched the dense model token by token better, while its generations got worse.
- **The dense model's answers made a poor target.** Only about 35% of them finished within 2,048 tokens (mean length 1,770), and the unfinished ones were trained on too.

**Drift**:

| | KL | held-out KL | 64K bits per byte | greedy agreement |
|---|---:|---:|---:|---:|
| healed | 0.612 | 0.763 | 0.734 | 71.3% |
| distilled | 0.644 | 0.773 | 0.747 | 70.6% |

- The difference is +0.031 (95% interval +0.023 to +0.041).
- Most of the rise is in chat (+0.158) and reasoning (+0.062). The other 8 domains each rose by 0.008 to 0.018.

**Reading, as the rules set it out:**
- The loops were not reduced.
- The loop share on the student's own answers stayed flat over the rounds, so a longer run is not the answer.
- The drift is just over the guard of 0.03, so before anything longer, teacher-forced healing text would go back into the mix.
- The rules' next options are structural: a bigger shared expert, or training every MLP weight.

Read plainly: the user asked whether distilling a dense model into an MoE is a problem, and here it was. The dense teacher pulls the student toward long, self-checking answers that it cannot finish at 25%.

The healed model with every neuron answers in much the same style (all 48 of its greedy answers open with "Here's a thinking process…"), and finishes. Section 16 uses it as the teacher, trained only on the answers it finishes.

| Prediction (written before the run) | Result |
|---|---|
| The rebuilt student reproduces the healed one: KL within 0.01 of 0.612 | yes, +0.0002 |
| Left padding moves outputs under 0.05 (student) and 0.005 (dense) | yes: 0.016 and 0.0006 |
| Batched, the control loops in 12 to 24 greedy and at most 3 sampled | yes: 18 and 0 |
| Dense and every neuron loop in at most 3 each; dense finishes fewer than 40 | yes: 0 and 0; 23 |
| The distilled student loops less than the control (p < 0.05), but not clean | **no**: 20 against 18 (p = 0.73) |
| Its finished greedy answers have a median above 400 tokens | yes, 991 (of only 6) |
| Its KL is within 0.03 of the healed student's, either way | **no**, +0.031 |
| The on-policy greedy loop share is lower in the last quarter of rounds than the first | **no**: 0.79 against 0.74 |
| It gets at least as many greedy answers right as the control | **no**: 7 against 22 |
| Sampled, it finishes at least as many as the control | **no**: 12 against 32 |

## 16. On-policy distillation from the healed model with every neuron

*Added 2026-10-31. One A40, about $5.03 of the $10 cap (ledger `runpod_budget_opd_self_20261031.json`). Rules, predictions and how to read the result, written before the run: `opd_self_rules.json`. Code: `slim_opd.py` (the `Teacher` class). Pod: `run_opd_self_pod.py` and `opd_self_worker.py`. Outputs: `opd_self/`.*

The user doubted that a dense model is the right teacher for an MoE, and suggested a better MoE, such as Qwen3.6-35B-A3B.
- We chose the healed student itself with every neuron running instead.
  - It differs from the student only in the skipped neurons.
  - It is the best model on the check prompts (section 15).
  - It costs nothing extra to run.
- A logits-only teacher passes on nothing specific to MoE, and the 35B-A3B would be further from the student than the dense 9B is.
- Everything else is as in section 15, except:
  - **The teacher** runs the healing run's adapter, frozen, with every neuron, while the student's copy of the adapter trains.
  - **Only the teacher answers that end within 2,048 tokens are trained on**, so the student never learns thinking that does not end.
- In all: 16 rounds, 96 steps, 1,536 prompts and 1.3M positions (0.5M on the teacher's answers, 0.8M on the student's), in 7.4 hours.
  - That is a third of section 15's positions. Rounds took twice as long, because the teacher runs every neuron through the student's expert code: 184 tokens/s against the dense model's 298 in the checks before training.
  - The teacher finished 67% of its answers (the dense teacher: 34%), and 514 of 768 were trained on.

**Before training**, the same checks as section 15 gave the same numbers: KL +0.0002, padding 0.016 and 0.0006, and identical reference generations.

**Generations**, the same 48 check prompts:

| | correct | finished | at the cap, stuck in a loop | at the cap, not repeating | finished median length |
|---|---:|---:|---:|---:|---:|
| dense | 28 | 23 | 0 | 25 | 1,082 |
| dense, sampled | 29 | 23 | 0 | 25 | 1,009 |
| healed, every neuron (the teacher) | 40 | 45 | 0 | 3 | 893 |
| healed, every neuron, sampled | 30 | 43 | 0 | 5 | 837 |
| healed, 25% (control) | 22 | 30 | 18 | 0 | 194 |
| healed, 25%, sampled | 13 | 32 | 0 | 16 | 251 |
| distilled, halfway (round 10) | 22 | 30 | 14 | 4 | 915 |
| **distilled** | **26** | **33** | **13** | 2 | 835 |
| **distilled, sampled** | **27** | 32 | 0 | 16 | 749 |
| section 15's distilled (dense teacher) | 7 | 6 | 20 | 22 | 991 |
| section 15's distilled, sampled | 16 | 12 | 0 | 36 | 1,147 |

- **Every count improved, but the loops are not significantly fewer.**
  - Greedy: 13 loops against 18 (one-sided p = 0.19), 33 finished against 30, 26 right against 22 (p = 0.27).
  - Sampled: 27 right against 13 (p = 0.003), with the same 32 finished.
- **The distilled 25% model is now about as accurate as the dense model on these prompts:** 26 and 27 right, against the dense model's 28 and 29.
  - Of its 33 finished greedy answers, 26 are right. Its 13 loops get none right.
  - The every-neuron teacher, at 40, is still well ahead.
- **It answers like its teacher.**
  - All 48 greedy answers now open with "Here's a thinking process…".
  - Its finished answers run to a median of 835 tokens (teacher: 893).
- **The remaining loops are the same self-check cycle as section 15's**, though fewer: "So the answer is 6 mph. Wait, let me check the problem again. "If she wants…" So the answer is 6 mph. Wait, let me check…".
  - Which prompts loop still moves around: 6 of the control's 18 still loop; 7 new ones do.
- **The teacher matters a great deal.**
  - On the same prompts, against section 15's dense-teacher student, this student loops in 13 against 20, finishes 33 against 6, and gets 26 right against 7.
  - And that with a third of the training positions.

**On the student's own training answers, the loop share fell.**
- Greedy, by quarter of the run: 0.62, 0.62, 0.47, 0.49. The dense teacher's run held at 0.74 to 0.85.
- Sampled: 0.20, 0.18, 0.09, 0.12.
- The share of greedy answers that finished rose from about 0.35 in rounds 1–9 to about 0.49 in rounds 10–15.
- Each round has only 24 greedy answers, so a single round's share moves by ±0.1 by chance. The share fell in the middle of the run, and the last two quarters are level.

**Drift**:

| | KL | held-out KL | 64K bits per byte | greedy agreement |
|---|---:|---:|---:|---:|
| healed | 0.612 | 0.763 | 0.734 | 71.3% |
| distilled, dense teacher (section 15) | 0.644 | 0.773 | 0.747 | 70.6% |
| distilled, every-neuron teacher | 0.651 | 0.805 | 0.749 | 70.4% |

- The difference is +0.038 (95% interval +0.035 to +0.042), over the 0.03 guard.
- Unlike section 15, the rise is spread over every domain: chat +0.071, reasoning +0.061, the other 8 domains +0.023 to +0.044, with the two held-out domains (biomedical, argument) at +0.039 and +0.044.
- So training only on GSM8K and chat answers, with no text from the original mix, costs some general fit. This is so even though the teacher itself sits much closer to the dense model (0.24, section 10).

**Reading, as the rules set it out:**
- Not reduced (p = 0.19 against the p < 0.05 needed), so at this length even this teacher does not fix the loops.
- The loop share on the student's own answers fell over the first half and then levelled. So whether a longer run helps is the open question, but not a clear yes.
- The drift is above the guard. So before anything longer, the original teacher-forced healing text goes back into the mix.

Read plainly:
- This is the first change at 25% that improves the generations on every count, and it is clearly better than the dense teacher.
- Sampled, the 25% model never repeats and is as accurate as the dense model.
- Greedy, it still loops in about a quarter of the prompts.
- What remains looks like the same limit as before: a quarter of the neurons cannot always end a self-check.

| Prediction (written before the run) | Result |
|---|---|
| The rebuilt student reproduces the healed one: KL within 0.01 of 0.612 | yes, +0.0002 |
| Left padding moves outputs under 0.05 (student) and 0.005 (dense) | yes: 0.016 and 0.0006 |
| Batched, the control loops in 12 to 24 greedy and at most 3 sampled | yes: 18 and 0 |
| Every neuron loops in at most 3 and finishes at least 40; dense loops in at most 3 | yes: 0 and 45; 0 |
| The teacher finishes at least half its training answers | yes, 67% |
| The on-policy greedy loop share is lower in the last quarter than the first | yes: 0.49 against 0.62 |
| The distilled student loops less than the control (p < 0.05), but not clean | **no**: 13 against 18 (p = 0.19) |
| It finishes at least as many greedy answers as the control | yes, 33 against 30 |
| Its KL is within the guard and lower than the control's | **no**, +0.038 |
| It gets at least as many greedy answers right as the control | yes, 26 against 22 |
| Sampled, it finishes at least as many as the control | yes, 32 against 32 |
| It loops less and finishes more than section 15's distilled student | yes: 13 against 20, 33 against 6 |

## 17. The same distillation, longer, with the healing text back in

*Added 2026-11-01. Two pods, about $7.11 of the $10 cap (ledger `runpod_budget_opd_long_20261031.json`): an A40 that ran out of memory after 22 rounds ($4.53), then an RTX A6000 that carried on from its checkpoint ($2.58). Rules, predictions and how to read the result, written before the run: `opd_long_rules.json`. Code: `slim_opd.py` (`train --resume`). Pods: `run_opd_long_pod.py` with `opd_long_worker.py`, then `run_opd_long_resume_pod.py` with `opd_long_resume_worker.py`. Outputs: `opd_long/`, and each pod's own folder in `results/opd-long-pod-20261031/`.*

Section 16 left two questions: whether a longer run keeps lowering the loops, and whether healing text in the mix holds the drift. So:
- **Start**: section 16's distilled student, and its next training prompts (from the 1,537th).
- **The teacher and the rounds** are as in section 16: the healed model with every neuron, frozen, trained on only the answers it finishes; 48 teacher and 48 student answers and 6 steps per round.
- **New in each round**: 24 of the healing run's 2,048-token sequences, on which the student learns toward the dense model, as in healing.
- **Twice the check prompts**: 96 GSM8K test questions (rows 16–111). The first 48 are sections 14–16's.
- **The halfway check** is a greedy run on the 96.

**What happened to the run.**
- **The first pod ran out of GPU memory at the halfway check**, after 22 rounds (about 6.5 hours).
  - The cause: it generated all 96 check answers at once while the optimizer state was still on the GPU.
  - The fallback halves the batch after a failure. But Python keeps the failed call's memory alive inside the `except` clause, and the halving ran inside it, so every smaller batch failed too, down to a single prompt.
- **The fix in `slim_opd.py`**:
  - the batch is halved only after leaving the `except` clause and freeing the failed attempt;
  - the halfway check generates 48 answers at a time;
  - the allocator uses expandable segments.
  
  A test that fails on the old code covers the freed memory.
- **The second pod continued from the checkpoint saved after 21 rounds** (`train --resume`):
  - it kept those rounds' prompts and text and found its place on the learning-rate cosine;
  - it redid round 21;
  - since the optimizer state was not saved, it warmed up again over 6 steps.
  
  Peak memory stayed at 30 GB.
- **The run ended when the training prompts ran out, not at 12 hours.**
  - Section 15 built 4,800 prompts (2,400 GSM8K, 2,400 UltraChat). After section 16's 1,536, they were enough for 34 rounds, about 9.5 hours. I did not check this when I proposed 12 hours.
  - The learning rate ended at 8.8e-6, 81% of the way along its cosine (floor 5e-6).
- **In all**: 34 rounds and 204 steps on 3,264 new prompts, 4.6M positions:
  - 1.1M on the teacher's answers (1,110 of 1,632 trained; the teacher finished 68%);
  - 1.9M on the student's answers;
  - 1.7M on 816 text sequences.

**Before training**, everything reproduced:
- the healed model: KL +0.0002;
- padding: 0.016 and 0.0006;
- the start: KL 0.651, as in section 16.

**Generations**, the 96 check prompts:

| | correct | finished | at the cap, stuck in a loop | at the cap, not repeating | finished median length |
|---|---:|---:|---:|---:|---:|
| healed, 25% (control) | 42 | 59 | 36 | 1 | 214 |
| healed, 25%, sampled | 46 | 81 | 0 | 15 | 303 |
| start (section 16's distilled) | 52 | 65 | 21 | 10 | 848 |
| start, sampled | 46 | 68 | 0 | 28 | 801 |
| halfway (round 22) | 56 | 63 | 23 | 10 | 933 |
| **distilled** | **54** | **65** | **17** | 14 | 898 |
| **distilled, sampled** | 47 | 61 | 0 | 35 | 831 |

- **The loops are now significantly fewer than the control's: 17 against 36 of 96** (one-sided p = 0.002).
  - It finishes 65 against 59 (p = 0.23) and gets 54 right against 42 (p = 0.056).
  - Of its 65 finished answers, 49 are right.
- **Most of that was already there at the start.**
  - On 96 prompts, section 16's student loops in 21 against the control's 36 (p = 0.013). The 48 prompts in section 16 could not show this (p = 0.19).
  - This run took it from 21 to 17 (p = 0.29 for fewer), with 23 at the halfway point.
  - On the first 48 prompts alone, the start loops in 10 and the control in 19; section 16 measured 13 and 18.
- **Which prompts loop still moves around.**
  - Of the distilled student's 17 loops, 5 also loop for the start, 7 for the control, and 3 for both. 8 loop for neither.
  - The loops are the same self-check cycle as before: "*Final Answer:* 230 miles. *Wait, let me check the question again.* … The answer is 230 miles. *Final Answer:* 230 miles. …".
- **Sampled, it never repeats, but it runs to the cap more often.**
  - It reaches the 2,048-token cap without repeating in 35 of 96, against the control's 15 (p = 0.0008 for fewer finished), and still gets as many right: 47 against 46.
  - The start was already at 28 (p = 0.019). Section 16 put the two level on its 48 prompts (32 finished each).
  - These students think the way their teacher does, at a median of 800–900 tokens against the control's 214–303, and with the model card's sampling more of that thinking does not end within 2,048 tokens.

**On the student's own training answers, the loop share stayed level.**
- Greedy, by quarter (rounds 0–8, 9–17, 18–25, 26–33): 0.50, 0.48, 0.58, 0.53. Section 16 ended at 0.49.
- Sampled: 0.13, 0.10, 0.07, 0.10.
- Greedy finished: 0.47, 0.47, 0.40, 0.46.
- So the gain from this teacher levelled out during section 16's second half and did not resume.

**Drift**: the healing text did more than hold it.

| | KL | held-out KL | 64K bits per byte | greedy agreement |
|---|---:|---:|---:|---:|
| healed | 0.612 | 0.763 | 0.734 | 71.3% |
| start (section 16's distilled) | 0.651 | 0.805 | 0.749 | 70.4% |
| **distilled, with healing text** | **0.602** | **0.757** | 0.743 | 71.4% |

- Against the healed model: −0.011 (95% interval −0.016 to −0.007). Against the start: −0.049 (−0.057 to −0.043).
- It is below the healed model in 9 of 10 domains, most of all in chat (−0.052). The exception is reasoning, at +0.011.
- The held-out domains improved too: biomedical −0.006, argument −0.007.
- Bits per byte with the 64K vocabulary are still a little above the healed model's: 0.743 against 0.734.
- The student's KL on the healing text itself fell over the rounds: 0.515, 0.516, 0.493, 0.474 by quarter.

**Reading, as the rules set it out:**
- Reduced: p = 0.002.
- Not still improving: p = 0.29 against the start.
- Within the guard: 0.602, under the healed model itself.
- The loop share on the training answers was not falling at the end.

So the every-neuron teacher's gain is real, but it has levelled out for LoRA updates at 25% hard routing. The next options are structural: a bigger shared expert at the same budget, or training every MLP weight. Each would be followed by this distillation.

Read plainly:
- This is the first result at 25% with significantly fewer loops than the healed model. It is also more accurate (54 against 42 right) and drifts less.
- Healing text in the mix removes the drift cost of distillation, so this recipe can follow any structural change.
- It is still not clean: 17 of 96 greedy answers loop.
- Sampled, a third of the answers go round until the cap.
- Most of the gain came in section 16's first 16 rounds; 34 more rounds added little.

| Prediction (written before the run) | Result |
|---|---|
| The healed student reproduces the healing run: KL within 0.01 of 0.612 | yes, +0.0002 |
| Left padding moves outputs under 0.05 (student) and 0.005 (dense) | yes: 0.016 and 0.0006 |
| The start reproduces section 16's student: KL within 0.01 of 0.651 | yes, 0.651 |
| Batched, the control loops in 24 to 48 greedy and at most 6 sampled | yes: 36 and 0 |
| The start loops in 16 to 36 greedy and at most 6 sampled | yes: 21 and 0 |
| The teacher finishes at least half its training answers | yes, 68% |
| The on-policy greedy loop share is lower in the last quarter than the first | **no**: 0.53 against 0.50 |
| The text KL is lower in the last quarter than the first | yes: 0.474 against 0.515 |
| The distilled student loops less than the control (p < 0.05), but not clean | yes: 17 against 36 (p = 0.002) |
| It loops less than the start (point estimate) | yes: 17 against 21 (p = 0.29) |
| It finishes at least as many greedy answers as the start | yes, 65 against 65 |
| It gets at least as many right as the start, greedy and sampled | yes: 54 against 52, 47 against 46 |
| Sampled, it loops in at most 6 and finishes at least as many as the start | **no**: 0 loops, but 61 finished against 68 |
| Its KL is within the guard (at most 0.642) and below the start's | yes: 0.602 against 0.651 |

## 18. The same checks with a 4,096-token cap

*Added 2026-11-01. One RTX A6000, about $1.14 of the $3 cap (ledger `runpod_budget_opd_cap_20261101.json`). Rules, predictions and how to read the result, written before the run: `opd_cap_rules.json`. Code: `slim_opd.py` (the `check` stage, and `against_shorter` in the report). Pod: `run_opd_cap_pod.py` and `opd_cap_worker.py`. Outputs: `opd_cap/`.*

Section 17's distilled student never repeats with the model card's sampling. But 35 of its 96 sampled answers ran to the 2,048-token cap unfinished, against the healed control's 15. So the question: are those answers long, or do they go round in circles?
- **The prompts and seeds are section 17's**; only the cap changed, to 4,096 tokens.
  - The repeated arms therefore begin with exactly section 17's answers: 96 of 96 in each.
  - So each answer the old cap cut can be followed to its end.
- **References on all 96 prompts**: the dense model and the every-neuron model (the teacher). Section 16 had run them only on the first 48, at 2,048.
- **The healed student reproduced** the healing run first (KL +0.0002).

| at 4,096 tokens | correct | finished | at the cap, stuck in a loop | at the cap, not repeating | finished median length |
|---|---:|---:|---:|---:|---:|
| dense | 60 | 61 | 7 | 28 | 1,140 |
| dense, sampled | 73 | 78 | 0 | 18 | 1,590 |
| healed, every neuron, sampled (the teacher) | 64 | 95 | 0 | 1 | 769 |
| healed, 25%, sampled (control) | 48 | 88 | 0 | 8 | 343 |
| **distilled** | 54 | 68 | **27** | 1 | 898 |
| **distilled, sampled** | **52** | **80** | 0 | 16 | 918 |

**Sampled, about half of the cut answers were only long.**
- Of the distilled student's 35, 19 finish by 4,096 tokens and 12 of those are right.
  - In all, it now finishes 80 (was 61) and gets 52 right (was 47).
- The other 16 are still going at 4,096 and almost never repeat a 20-token span.
  - They keep re-checking: "So my first calculation was wrong ($97.111...)$. Let's check. Wait, $18 \times 9 = 162$."
- The healed control: of its 15 cut answers, 7 finish. In all it finishes 88 and gets 48 right.
- Distilled against control, 16 against 8 unfinished: one-sided p = 0.063.
- **The dense model is no quicker.**
  - Sampled, it leaves 18 of 96 unfinished at 4,096, with a finished median of 1,590 tokens.
  - On the 25 of the first 48 prompts where section 16's dense answers hit the 2,048 cap, 16 of its new answers finish. These are new samples, not the same answers continued.
  - So long thinking is this model's normal way of answering, and 2,048 tokens were too few to judge it.
- **But the dense model is far more accurate**: 73 right, against the teacher's 64 and the distilled student's 52 (p = 0.001).
  - The teacher finishes 95 of 96. Against its 64 right, the distilled student's 52 gives p = 0.052.

**Greedy, the longer cap makes the loops worse, not better.**
- Of the 31 distilled greedy answers the old cap cut (17 of them looping), only 3 finish, and none of those 3 is right.
- 10 of the 14 that were still varying at 2,048 settle into a loop. That makes 27 loops of 96 (was 17).
- Their last 1,000 tokens are 73–99% repeats.
- **The dense model loops too, given room**: 7 of 96 greedy, where at 2,048 it never had.
  - One of its loops: "*Let's consider the "Two Trains" as a trick for "They are actually the same train".* No. *Let's consider…*".
  - The distilled student loops on that prompt too, and on one more of the dense model's 7.
  - So "the dense model never loops" held only within 2,048 tokens.
  - The distilled student still loops far more: 27 against 7 (p = 0.0001).
- The healed control was not run greedy at 4,096.

**Reading, as the rules set it out:**
- Long, not circling:
  - unfinished not significantly more often than the control: p = 0.063, against the p < 0.05 that would mean circling;
  - at least half of the 35 finish: 19.
  
  So the 2,048-token cap, not the student, explains most of section 17's sampled shortfall, and later checks use 4,096.
- The dense model leaves as many unfinished as the distilled student (18 against 16). So long thinking is normal for this model, and the bar is the every-neuron model's: 95 finished and 64 right, sampled.

Read plainly:
- With sampling, the distilled 25% model finishes about as often as the dense model. It is less accurate: 52 right against 64 for the every-neuron model and 73 for the dense one.
- Greedy is where it fails: given more room, more answers end in loops (27 of 96).
- The dense model loops in 7 of 96, so some greedy repetition belongs to this model family at length. The 25% model's is about four times as common.
- The sampled verdict is narrow: p = 0.063 against the 0.05 line, and 19 of 35 finished against the 18 needed. "Mostly long" is fairer than a clean "long".

| Prediction (written before the run) | Result |
|---|---|
| The healed student reproduces the healing run: KL within 0.01 of 0.612 | yes, +0.0002 |
| At least 90 of 96 answers in each repeated arm begin with section 17's | yes: 96 and 96 |
| Of the distilled student's 35 cut sampled answers, 4 to 17 finish | **no**: 19 |
| Of the control's 15 cut sampled answers, at most 5 finish | **no**: 7 |
| At 4,096, the distilled student is still unfinished more often than the control (p < 0.05) | **no**: 16 against 8 (p = 0.063) |
| The dense model, sampled, finishes 50 to 85 | yes, 78 |
| The every-neuron model, sampled, finishes at least 85 | yes, 95 |
| The distilled student, greedy, loops in 17 to 31 and finishes at most 72 | yes: 27 and 68 |

## 19. A multiple-choice benchmark: how much does the 25% model know?

*Added 2026-11-01. Local GPU (Radeon 8060S), 1.9 hours, no pod. Rules, predictions and how to read the result, written before the run: `slim_bench_rules.json`. Code: `slim_bench.py`. Outputs: `bench/`.*

So far the 25% model has been judged on drift and on GSM8K answers. This run asks how much of the dense model's knowledge and reasoning it keeps, on four public tests.
- **The items** are the sample `tf_metrics.py` scored Qwen3.5-2B on: MMLU 1,140 (20 per subject), ARC-Challenge 600, ARC-Easy 400 and HellaSwag 600.
  - 5 MMLU math and statistics questions share a 13-token window with the healing data (its MATH domain) and were dropped, leaving 1,135.
- **Zero-shot plain text**, no chat template and no thinking. Each choice is scored by its log-probability:
  - MMLU lists the four choices and scores the letters;
  - ARC scores each answer after "Question: … Answer:", and HellaSwag each ending after the text, both per character.
- **The four 9B arms read the same 64K-token text**, so they differ only in the model body. The dense model was also scored as shipped (full vocabulary), and so was Qwen3.5-2B, a smaller model of the same family with the same tokenizer (1.9B parameters per token).
- The healed model reproduced the healing run first (KL −0.0006).

| accuracy, % | MMLU | ARC-Challenge | ARC-Easy | HellaSwag | share of dense above chance, mean of the four |
|---|---:|---:|---:|---:|---:|
| chance | 25.0 | 25.0 | 25.1 | 25.0 | 0 |
| dense | 67.5 | 53.0 | 73.3 | 69.3 | 1 |
| dense as shipped (full vocabulary) | 67.1 | 54.3 | 76.2 | 70.5 | 1.03 |
| healed, every neuron | 71.5 | 55.2 | 77.7 | 72.7 | 1.08 (1.04–1.13) |
| **healed, 25%** | **42.6** | **32.7** | **54.2** | **48.0** | **0.45 (0.41–0.50)** |
| distilled (section 17), 25% | 41.3 | 32.2 | 51.0 | 46.7 | 0.42 (0.37–0.46) |
| Qwen3.5-2B as shipped | 53.4 | 39.5 | 60.5 | 59.7 | 0.68 |

**The 25% model keeps less than half of what the dense model can do here.**
- Averaged over the four tests, it keeps 0.45 of the dense model's accuracy above chance (95% interval 0.41–0.50). It is far below the dense model on every test (each p < 10⁻¹¹).
- Knowledge suffers more than continuing a text: MMLU keeps 0.41, HellaSwag 0.52. ARC-Challenge, the hardest reasoning set, keeps the least: 0.27 (0.15–0.40).
- **It is below Qwen3.5-2B on every test**, by 8.9 points on average (p from 2·10⁻⁸ on MMLU to 0.017 on ARC-Easy). The 2B runs 1.9B parameters per token against the 25% model's 2.95B (1.4B against 2.8B without the output head).

**All of the loss comes from the 25% routing.**
- With every neuron running, the same healed model scores above the dense model on all four tests, by 3.5 points on average (significant on MMLU, ARC-Easy and HellaSwag).
  - So the low-rank DeltaNet, the 64K vocabulary and the healing adapter cost nothing on these tests.
  - It probably scores above the dense model because healing on plain text suits this plain-text format, while the dense model is tuned for chat. This was not tested.
- Running 25% of the MLP neurons, with the same adapter, takes MMLU from 71.5 to 42.6.

**Distillation did not help here.** The distilled model is 1.6 points below the healed one on average. Only ARC-Easy's 3.2 is significant (p = 0.015). What distillation bought was finished answers (sections 16–18), not knowledge.

**The 64K vocabulary costs 1.3 points on average.** On ARC-Easy it costs 3.0 (p = 0.036); the other three are within noise.

**The absolute scores in this setup run low, so compare models only within this table.**
- Qwen3.5-2B scores 2 to 6.5 points less here than it did with `tf_metrics.py` on the same items: MMLU 53.4 against 58.7, ARC-Easy 60.5 against 67.0, HellaSwag 59.7 against 66.0.
- The setups differ in three ways: bf16 here instead of fp32, the GPU instead of the CPU, and the <|endoftext|> token that starts each sequence here. Which of them matters was not tested.
- Every model here was scored the same way, so the comparisons stand. The absolute numbers are not comparable with published ones, which also use few-shot prompts or thinking.

**Reading, as the rules set it out:**
- The share is 0.45, below 0.7: the 25% budget costs a large part of what the dense model can do, so later changes are judged on this benchmark as well as on GSM8K.
- The model is below Qwen3.5-2B: as carved and healed so far, it is worse than a smaller dense model that is cheaper to run.

Read plainly:
- The loops were the visible problem. The bigger one is that the 25% model knows and reasons much less than the dense model.
- Nothing outside the routing is to blame: with every neuron, the same healed model is at least as good as the dense one.
- So the next change at 25% has to give each token more MLP capacity than LoRA healing has. A carve worth keeping should at least beat Qwen3.5-2B on this benchmark.

| Prediction (written before the run) | Result |
|---|---|
| The healed model reproduces the healing run: KL within 0.01 of 0.612 | yes, −0.0006 |
| The dense model scores 68–82 on MMLU, 48–62 on ARC-Challenge, 72–85 on ARC-Easy, 72–82 on HellaSwag | **no**: ARC in range, but MMLU (67.5) and HellaSwag (69.3) just below |
| The healed model keeps 0.5–0.85 of the dense model's accuracy above chance, and is significantly below it on every test | **no**: 0.45; significantly below on every test, yes |
| MMLU loses a larger share than HellaSwag | yes: keeps 0.41 against 0.52 |
| The distilled model is within 3 points of the healed one on every test, significantly different on at most one | **no**, narrowly: ARC-Easy −3.2, the only significant one |
| The every-neuron model keeps at least 0.85 and beats the healed model on every test | yes: 1.08, every test |
| The dense model and the dense model as shipped are within 1.5 points on every test | **no**: ARC-Easy 3.0 |
| The healed model scores above Qwen3.5-2B on average | **no**: 8.9 points below |
| Qwen3.5-2B is within 3 points of its `tf_metrics.py` scores on every test | **no**: 2.0 to 6.5 points lower |

## 20. More experts per token, without retraining: 16 and 32 instead of 9

*Added 2026-11-01. Local GPU, no pod. Rules, predictions and how to read the result, written before the run: `slim_bench_k_rules.json` (benchmark) and `opd_k16_rules.json` (GSM8K generations). Code: `slim_bench.py` (arms `healed_k16`, `healed_k32`) and `slim_opd.py` (`check` with `eval.experts`). Outputs: `bench_k/` and `opd_k16/`.*

The user asked: should we just switch from 9 experts to 16 and see if that fixes a bunch? This changes only how many routed experts the learned routers pick. The adapter and routers are the healing run's, trained at 9, so every number below is a lower bound for a model healed at that count.
- 16 experts run 4,224 of 12,288 MLP neurons per layer (34%, about 3.4B parameters per token against 2.95B); 32 run 6,912 (56%, about 4.5B).
- The healed model reproduced the healing run first (KL 0.611).

**Drift** (benchmark KL from the dense model): 9 experts 0.612, **16 experts 0.490** (in-mix 0.465, held-out 0.588), bits per byte 0.690 (9: 0.734; dense 0.656), greedy agreement 74.0% (9: 71.2%).

**Multiple-choice benchmark** (section 19's items and scoring):

| accuracy, % | MMLU | ARC-C | ARC-E | HellaSwag | share of dense above chance |
|---|---:|---:|---:|---:|---:|
| dense | 67.5 | 53.0 | 73.3 | 69.3 | 1 |
| healed, 9 experts (section 19) | 42.6 | 32.7 | 54.2 | 48.0 | 0.45 |
| healed, 16 experts | 47.7 | 41.2 | 63.5 | 55.2 | **0.65** (0.60–0.70) |
| healed, 32 experts | 60.9 | 50.2 | 69.0 | 66.2 | **0.90** (0.85–0.94) |
| Qwen3.5-2B | 53.4 | 39.5 | 60.5 | 59.7 | 0.68 |

- **Each step helps a lot, on every test, with no training.** 16 beats 9 by 7.5 points on average, and 32 beats 16 by 9.7 (every difference significant, p < 0.005).
- **16 experts match Qwen3.5-2B** (1.4 points below on average; ahead on ARC, behind on MMLU and HellaSwag). **32 are 8.3 points above it** and within 4.2 points of the dense model, significantly below only on MMLU and HellaSwag.
- Knowledge (MMLU) recovers last: 47.7 at 16, 60.9 at 32, against 67.5 for the dense model.
- Each step costs neurons: 32 experts is 56% of the MLP per token, well outside the 25% budget the user set on 2026-10-30 ("more experts can't be the final answer"). This measures how steeply capacity buys back ability, not a new target.

**GSM8K generations at 16 experts** (96 prompts, 4,096-token cap, greedy and sampled, also counted at 2,048): running when this was written. Results will be in `opd_k16/check.json` and the `gen_healed_*.jsonl` files beside it. Predictions are in `opd_k16_rules.json`; compare with the healed model at 9 (section 17: greedy 36 loops of 96 at 2,048) and the dense model (section 18: 7 loops at 4,096).

| Prediction (written before the run) | Result |
|---|---|
| The healed model reproduces the healing run (KL within 0.01 of 0.612) | yes, 0.611 |
| 16 experts keep 0.55–0.80 of the dense model's accuracy above chance | yes, 0.65 |
| 32 experts keep 0.75–0.95 | yes, 0.90 |
| On MMLU, 16 beats 9 and 32 beats 16, each with the interval above 0 | yes: +5.1 and +13.2 points |
| 16 are within 5 points of Qwen3.5-2B; 32 are above it | yes: −1.4 and +8.3 |
| At 16 experts the benchmark KL is 0.40–0.52 | yes, 0.490 |
| Greedy loops, sampled finishes and right answers at 16 | pending |

**Reading so far.** Capacity, not training recipe, is what limits the 25% model on these tests: the same adapter that leaves 0.45 at 9 experts leaves 0.65 at 16 and 0.90 at 32. A model healed at 16 would do at least as well as these numbers. Whether it also stops the loops is the pending part. The choice between 16 experts (about the size of a dense 4B in active parameters, matching the 2B) and the 25% budget is the user's.

## Parameters

These are text-model parameters. "Active" means per token: the embedding lookup counts as zero, and the cache row uses the measured fallback rate.

| Step | stored, total | stored, non-embedding | active, total | active, non-embedding |
|---|---:|---:|---:|---:|
| dense | 8.95B | 6.92B | 7.94B | 6.92B |
| + experts, 25% active (routers included) | 8.96B | 6.93B | 4.31B | 3.29B |
| + 64K vocabulary | 7.46B | 6.93B | 3.56B | 3.29B |
| + 16K cached rows (64% fallback as measured) | 7.46B | 6.93B | 3.49B | 3.29B |
| + DeltaNet low rank | 6.93B | 6.39B | 2.95B | 2.76B |
| + healing (section 8; detector retrained, 42% fallback) | 6.97B | 6.43B | 2.95B | 2.80B |

Healing adds 44M parameters: the LoRA updates of the low-rank DeltaNet projections add to their rank, while every other LoRA update merges into its weight. The active total does not change, because the retrained detector saves about as many head rows per token.

## Predictions

| Prediction (written before the run) | Result |
|---|---|
| Plain prune above 1 nat at 25% | yes, 2.21 |
| Co-firing beats both domain carvings at every budget | yes |
| A document-level domain router recovers less than half of the prune's loss | yes, a quarter |
| Shared core above v1's 0.30 | **no**, 0.28 |
| Shared expert sized by depth within 10% of uniform | yes, +1.6% |
| Neurons moved between layers at least 10% better | **no**, 5.7% worse |
| Learned router 1.2–2.5× the perfect router, picks 50–75% | yes, 1.57×, 68% |
| 64K vocabulary costs under 2% bits per byte; science and biomed most | **no**, 5.0%; Wikipedia and Python most |
| 16K cache: top pick outside under 5% | yes, 4.6% |
| Detector falls back under 25% | **no**, 33% |
| 4K misses the top pick at least twice as often | yes, 12.5% vs 4.6% |
| int16 and stochastic fp16 within 3× of the jitter control | yes |
| bf16 at least 10× the jitter control | **no**, 1.2× |
| int8 worse than bf16 | yes |
| int16 and stochastic fp16 lengths within 10% of fp32; bf16 differs more | first part yes; **bf16 no** |
| Drop 4 blocks under 0.3; 8 more than twice that | yes, 0.089 and 0.274 |
| Low rank worse than dropping blocks | **no**: tie at 4, better at 8 |
| Combined within 30% of the summed parts | yes, +3% |
| Cache lowers greedy agreement by under 2 points | yes, no change |

## What this means for the plan

1. **Fix the loops at the same 25% budget.** Running more experts is ruled out as the answer (decided 2026-10-30). Healing works and is not the cause: with every neuron, the healed model never loops and gets 10 of 16 GSM8K right (section 10). The router is only 0.10 of the drift (section 9). Switching experts on and off makes the model jumpy, but removing most of that does not stop the loops (section 11). So the loops come from what each token loses with 75% of the MLP skipped: 42% of the MLP's output on average, three quarters of it not predictable from the neurons that run (section 12). At full strength, changes fitted to the healed model after the fact make it worse (section 12). Subtracting the fold is far worse still. A quarter of it lowers the drift a little (section 13). On 48 prompts, no fold reduces the loops, however it is fitted: the best, tracking the every-neuron model, lowers the drift to 0.581 and loops as often as the plain model (section 14). Which prompts loop changes from run to run. So the fold line ends here. Judge generations both greedy and with the recommended sampling from now on: sampling stops the exact repetition, but a fifth of the answers still go round in circles until the cap, and fewer are right. The bar to reach, on the same 48 prompts (section 15): the healed model with every neuron finishes 45 and gets 40 right greedy; the dense model gets 28 right but runs to the 2,048-token cap in 25. Judge with a 4,096-token cap from now on (section 18): at 2,048 the dense model's answers are cut as often as the student's. On section 17's 96 prompts at 4,096, sampled, the dense model gets 73 right and finishes 78, the every-neuron model 64 and 95, and the distilled 25% model 52 and 80; greedy, the dense model loops in 7 and the distilled model in 27. Generation-only checks like these can now run on the local GPU (`LOCAL_GPU.md`). Judge every change on the multiple-choice benchmark too (section 19, `slim_bench.py`, two hours on the local GPU). There the healed 25% model keeps 0.45 of the dense model's accuracy above chance and is 8.9 points below Qwen3.5-2B, while with every neuron it beats the dense model. A carve worth keeping should at least beat the 2B.

   What is left, at the same budget, all needs training:
   - a larger shared expert, with fewer routed neurons per token (untested);
   - training every MLP weight during healing instead of rank-64 updates, so the experts can take over more of what is skipped, optionally starting from a fold fitted before healing;
   - on-policy distillation on the student's own generations, which trains directly on the drift that ends in loops. With the dense 9B as the teacher it made things worse (section 15): the student took on the dense model's long self-checking and could not finish it at 25%. With the healed model running every neuron as the teacher, trained only on answers that finish, every count improved (section 16): greedy, 13 loops, 33 finished and 26 right against 18, 30 and 22; sampled, 27 right against 13. But the loops were not yet significantly fewer, and the drift rose by 0.038. Carried on for 34 more rounds with the healing text mixed back in (section 17), it loops in 17 of 96 greedy against the healed model's 36 (p = 0.002), gets 54 right against 42, and drifts less than the healed model (0.602). But most of that gain came in section 16's 16 rounds, and the loop share on the training answers has stayed level since. So this recipe, with the healing text in the mix, is the step to run after any structural change below, not a fix on its own;
   - a fixed gain on the shared expert is ruled out; CMoE's trained, per-token gain is untested.
   Each needs healing before its generations mean anything (unhealed, even the every-neuron model never finishes an answer), so compare them on short healing runs against the current carve healed the same way.
2. **Keep expert sizes equal for now.** Revisit sizing by depth after healing.
3. **Choose the DeltaNet state format by what is cheapest on the NPU.** bf16 is native; int16 needs scales. Stochastic rounding is not needed.
4. **Use low rank for the DeltaNet cut, and keep block 0.**
5. **The row cache needs a better miss detector.** Even retrained on the healed model, it falls back on 42% of positions; try hidden-state features next. With the retrained detector, 4K rows are cheaper per token than 16K (28.6K vs 37.0K rows scored), while 16K keeps more greedy picks (99.8% vs 99.5%).
