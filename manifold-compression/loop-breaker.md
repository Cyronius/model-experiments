# Four hand-written neurons that stop a carved model's thinking loops

*Project log, November 27–28, 2026. One session on a shared iGPU, no cloud spend, no gradient descent. The neurons were designed and written by an LLM coding agent (Claude) on request; every weight is a closed-form function of activation statistics.*

- **Code:** [`loop_breaker.py`](loop_breaker.py) (first pair), [`loop_breaker2.py`](loop_breaker2.py) (second pair and the blind tests).
- **Rules written before each run, with outcomes:** [`loop_breaker_rules.json`](loop_breaker_rules.json), [`loop_breaker2_rules.json`](loop_breaker2_rules.json).
- **Answers and summaries:** [`results/moe-carve-v2-20261029/heal_pieces/`](results/moe-carve-v2-20261029/heal_pieces/) (`loop_breaker/`, `loop_breaker2/`).

## TL;DR

A Qwen3.5-9B that I carved into a mixture of experts loops when it thinks under greedy decoding: on GSM8K with a 2,048-token cap, more than a quarter of its answers end as a verbatim cycle inside `<think>`. Four extra neurons beside the last MLP, with weights written by hand, fix that:

| 96 fresh GSM8K prompts (blind), greedy | right | right and finished | finished | loops |
|---|---:|---:|---:|---:|
| unedited | 46 | 43 | 58 | 36 |
| two neurons (`</think>`) | 55 | 54 | 80 | 12 |
| four neurons (`</think>`, then `<\|im_end\|>`) | 56 | 56 | 93 | 1 |

Against the unedited model the four neurons fix 35 loops and cause none (paired sign test 3e-11), and gain 10 right answers while losing none (0.001). Of the 58 answers the unedited model finished by itself, 57 are token-for-token unchanged. A control with the same detector and a random write of the same size changes no token in any of 96 answers.

The neurons do one thing. They end loops; they do not make the model think better, and at the model card's sampling settings, where this model does not loop, they stay silent.

## The model and the failure

The model is an in-progress experiment in this folder that is not written up here yet: every MLP of Qwen3.5-9B (32 layers, 12,288 neurons each) is cut into 96 experts of 128 neurons, a router picks 24 of them per token, attention is untouched, and the result is healed against the original in four-layer pieces. Its KL to the original is 0.578 nats per token on held-out documents, and it is far behind the original on knowledge benchmarks (MMLU 41% against 67%). So this write-up is about repairing one failure of a damaged model, not about a finished model.

The failure: with thinking on and greedy decoding, the model reaches an answer inside `<think>`, writes "Wait, let's check the question again", re-derives the same lines word for word, and repeats until the token cap. I call an answer a loop when it hits the cap unfinished and at least 45% of the 20-token windows in its last 1,000 tokens occurred earlier in the same answer. By that rule 28 of the 96 check prompts loop, and 36 of the 96 fresh ones.

Sampled at the model card's settings (temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5) the same model does not loop at all: 0 of 96. The loop is the greedy form of a deeper defect, thinking that does not close, which sampling shows as 37 of 96 answers still wandering inside `<think>` at the cap.

## What had already failed

Twice before, on an earlier version of this carve, I removed a "loop direction" from the model: contrast MLP outputs inside the loop with those before it, and subtract the difference from MLP down-projection columns in the middle layers. The second attempt followed the recipe of the weight-space programming post ([Takehi et al.](https://rikiyat.github.io/blog/weight-space-programming)), 64 columns a layer. Both times a random direction of the same norm did as well.

The reason is worth more than the failure. This model's greedy answers are extremely path dependent: any small change to the weights changes the outcome of about 20 of 96 prompts, in both directions. I measured the extreme case by accident during this work. The same weights on two PyTorch builds (2.14 and a 2.16 nightly) share no token-identical answer out of 96, first differ at token 3 in the median, and score 50 against 41 right. A loop count that moves after an edit is therefore not evidence that the edit did anything.

That set three requirements for this attempt:

1. The edit must be exactly silent outside loops, so that most answers stay token-identical and the rest can be attributed.
2. For every answer that changes, the neuron's own activation must show that it fired at the token where the answer left its unedited text.
3. Every comparison is paired by prompt, on one build, in one session, with a random-write control.

## The program

```
beside the last MLP, at every position:
    if the text is repeating itself  and  the current token ends a line:
        if the thinking is still open:   raise the logit of </think>
        else:                            raise the logit of <|im_end|>
```

All three conditions are linear reads of `x`, the input of the last layer's MLP. Each is fit in closed form on the model's own greedy answers to 160 GSM8K training questions, teacher-forced through the model (46 of those answers loop with a clean cycle, 106 finish). The closed read also uses the answers the first pair gave to the same questions:

| read | how it is fit | how well it separates (held-out answers) |
|---|---|---|
| `w·x` "repeating" | Fisher discriminant: `(S_w + λI)⁻¹(μ₊ − μ₋)`. Positives are positions at least 20 tokens into the second copy of a cycle; negatives are every position of finished answers and the part of looping answers before the loop | AUROC 0.9988. The threshold `b = 4.40` is exceeded by 0.01% of the negatives |
| `n·x` "this token ends a line" | ridge regression onto 0/1 | 0.76 to 1.04 at line ends, at most 0.25 elsewhere |
| `m·x` "a `</think>` came earlier" | ridge regression onto 0/1 | where the detector fires: median 0.80 after a close, 0.06 before. The tails overlap (lowest closed 0.39, highest open 0.50) |
| `c·x` "one" | the least-variance direction with mean 1 | 1.000 ± 0.005. The projections have no bias, so this stands in for one |

A SwiGLU neuron computes `silu(g·x) · (u·x)`: one soft threshold times one linear read. That decides which condition goes where.

**First pair: repeating AND line end → `</think>`.** With `z = k(w·x − b c·x)`, `k = 4`:

```
A = silu(z)           · (n·x)        down column  +v/4
B = silu(z − 4 c·x)   · (n·x)        down column  −v/4
A − B ≈ clip(z, 0, 4) / 4 · (n·x)    nothing below the threshold or away from a line end, at most v
```

The repeat detector sits in the gate because it has to be exactly zero on ordinary text. The line-end read is clean enough to be the multiplied factor. The write `v` is the unembedding row of `</think>` passed back through the final norm's scale and sized so that the logit rises by 16 at a typical residual norm. Inside a loop `</think>` is about 9.5 logits below the top token at its best line end, so 16 clears it; in ordinary text the gate is shut and nothing is written.

**Second pair: once the thinking is closed, `<|im_end|>` instead.** The first pair alone has a gap that the generation test exposed: after the forced `</think>` the answer sometimes repeats too, the neurons fire again, write `</think>` a second time, and the model then emits nothing but `</think>` lines. The model's own logits cannot settle which stop token is due (inside a loop both sit about 10 logits under the top token), so the state needs its own read. But `m·x` is loose, and a loose read in a threshold would move the repeat detector. So the second pair puts both sharp conditions in its gate and uses the loose one only to mix two writes:

```
gate   z₂ = k(w·x − b c·x − 16 (c·x − n·x))      repeating AND at a line end
up     m·x                                        how closed the thinking reads
write  28 · v_end − v                             v_end raises <|im_end|> by 1 and </think> by 0
```

Where the first pair fires, the four neurons together write about `(1 − m·x)` of the `</think>` write and `m·x · 28` logits of `<|im_end|>`. An error in the closed read then costs a late or doubled close, not a wrong stop.

That is 4 neurons and 49,152 weights in a 9-billion-parameter model.

## Screening before generating

Because loop counts are so noisy, each design was first screened by teacher forcing: along each saved answer, find the first position where the edited argmax would differ from the unedited one, and with which token. Settings were chosen on half of the fitting answers and gated on the other half before any generation ran.

The screen caught the first design. It wrote both stop tokens wherever the repeat detector fired, and the screen showed it would end answers with `<|im_end|>` in mid-sentence ("= 714.28571", then nothing). The line-end condition and the `</think>`-only write came from that. I then revised the screen's own labels twice after reading its held-out outcomes, so its numbers (20 of 23 held-out loops broken, 52 of 53 finished answers untouched) describe the design; they do not confirm it. The generation tests below are the evidence.

With the neurons in the model, generation then left the saved answer at the screen's forecast position in 5 of 6 held-out answers for each pair.

## Results

All arms: greedy, thinking on, 2,048 new tokens, one PyTorch build, paired by prompt. "Right" reads the last number after the last `</think>`; "right and finished" also requires that the answer ended by itself, which is the fairer count for a model that loops.

| prompts | arm | right | right and finished | finished | loops | 3+ `</think>` | KL to the original |
|---|---|---:|---:|---:|---:|---:|---:|
| 96 fresh (GSM8K test rows 112–207), blind | unedited | 46 | 43 | 58 | 36 | 2 | 0.5778 |
| | two neurons | 55 | 54 | 80 | 12 | 12 | 0.5815 |
| | four neurons | 56 | 56 | 93 | 1 | 3 | 0.5835 |
| 96 check (rows 16–111), used to design the second pair | unedited | 41 | 36 | 60 | 28 | 1 | 0.5778 |
| | two neurons, random write (control) | 41 | 36 | 60 | 28 | 1 | 0.5778 |
| | two neurons | 42 | 41 | 75 | 12 | 11 | 0.5815 |
| | four neurons | 45 | 45 | 89 | 2 | 2 | 0.5835 |

Paired comparisons, one-sided exact sign tests:

| prompts | comparison | loops fixed / new | p | right gained / lost | p |
|---|---|---:|---:|---:|---:|
| fresh | four neurons against unedited | 35 / 0 | 3e-11 | 10 / 0 | 0.001 |
| fresh | two neurons against unedited | 25 / 1 | 4e-7 | 9 / 0 | 0.002 |
| fresh | four neurons against two | 11 / 0 | 5e-4 | 1 / 0 | 0.5 |
| check | four neurons against unedited | 26 / 0 | 1.5e-8 | 6 / 2 | 0.14 |
| check | two neurons against unedited | 17 / 1 | 7e-5 | 4 / 3 | 0.5 |
| check | four neurons against two | 10 / 0 | 0.001 | 4 / 1 | 0.19 |

**Attribution.** On the fresh prompts 57 of 96 answers are token-identical to the unedited model's, and 34 of the 35 fixed loops left the unedited text at a token where the neurons had just fired. On the check prompts with the first pair, 59 answers are identical, and 36 of the 37 that changed left the unedited text where the neurons fired, 35 of them with `</think>` as the first changed token. The second pair fired in 12 and 13 answers, each time on the answer's last step; none of those answers ended without a number and none ended inside open thinking.

**Control.** Same detector, the write replaced by a random direction of the same norm orthogonal to both stop rows: the detector fires in the same 37 answers and all 96 answers are token-identical to the unedited model's. The effect is the `</think>` write, not a disturbance of the last layer.

**Collateral.** Of the answers the unedited model finished, 57 of 58 (fresh) and 58 of 60 (check) are unchanged under the four neurons, with the same number right before and after.

**Accuracy.** The gain in right answers is the model stating an answer it had already reached inside the loop. On the check prompts the two answers "lost" were unfinished answers whose last number happened to match.

**Length.** Mean answer length on the fresh prompts falls from 1,048 to 667 tokens.

**Sampled.** At the model card's settings the four neurons change 2 of 96 answers; 41 right, 38 right and finished, 60 finished, against 41, 38, 59 unedited. No harm and no help: sampled answers that wander inside `<think>` do not repeat word for word, so the detector has nothing to fire on.

## Predictions, scored

Each rules file states its predictions before the run. Held and missed:

| prediction | outcome |
|---|---|
| repeat probe's held-out AUROC at least 0.98 | held (0.9988) |
| first pair: loops on the check prompts fall to 10 or fewer | missed narrowly (12) |
| first pair: at least 12 more finished answers; right within −2; KL within +0.005 | held (+15, +1, +0.0037) |
| first pair: one named way to fail is that the answer after a forced `</think>` loops in turn | happened (10 of the 12 loops left), and motivated the second pair |
| closed read where the detector fires: closed median ≥ 0.6, open 99% quantile ≤ 0.3 | missed narrowly (0.80; 0.31) |
| four neurons, fresh prompts: loops at most a quarter of the unedited count, +15 finished, right within −2 | held (1 of 36, +35, +10) |
| four neurons: no answer with three or more `</think>` | missed on both sets (2 and 3 answers; 11 and 12 with the first pair alone) |
| four neurons: KL within +0.006 | held, barely (+0.0057) |
| sampled: the first pair fires in 15 to 35 answers and 5 to 20 more finish | missed (fires in 2, +1 finished) |

## Limits

- **Greedy only.** Sampled, the model's unfinished thinking is a different problem that these neurons do not touch.
- **Repetitive but legitimate text is untested, and code is a warning.** The KL rise is not spread evenly: with the four neurons it is +0.037 nats on the code documents of the evaluation set and zero on chat, math, reasoning and argument. So the detector fires on some repetitive code. I have not yet tested whether it cuts short a generated answer that legitimately repeats lines (code, tables, lists, song lyrics).
- **Loops with no line break survive.** The detector acts only at line ends, so one paragraph repeating without a newline is not caught (2 of the 3 loops left across both sets).
- **One task family.** Probes were fit on GSM8K training answers and tested on GSM8K test questions.
- **Tied to these weights.** The reads are directions in this model's last-layer residual. Another model, or this one after more healing, needs its own fit: about 25 minutes of generation and capture on the iGPU, seconds of CPU for the closed-form fits, and about half an hour per 96-prompt test.
- **This is activation-informed weight writing, not weight reading.** Nothing here was inferred from the weights alone; the directions come from captured activations, as in ordinary probing and steering work. What the weight-space programming framing added was the form: a small explicit program with a condition and an action, written into neurons, instead of one direction removed everywhere.

## Files

- `loop_breaker.py`, `loop_breaker2.py`, `test_loop_breaker.py`, `test_loop_breaker2.py`: fitting, neuron construction, screens, generation tests. They import this folder's carve and healing tooling (`heal_block.py`, `moe_carve.py`, `slim_*.py`, `heal_probe.py`, `loop_edit.py`), which is not in the repository yet, so they document the method but do not run from a fresh clone.
- `loop_breaker_rules.json`, `loop_breaker2_rules.json`: design, predictions, amendments and outcomes, each dated.
- `loop_breaker2_prompts.json`: the 96 fresh prompts.
- `results/moe-carve-v2-20261029/heal_pieces/loop_breaker/` and `loop_breaker2/`: every arm's answers as token ids (`gen_*.jsonl`), per-arm summaries with the paired comparisons (`test_*.json`), the screens (`build*.json`) and the registered counts (`outcome.json`). `checks/` holds the unedited model's sampled answers. The per-step activation arrays and the neuron weights are not included.

## Related work

- Takehi et al., [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming): the loop-direction edit on Qwen3.5-0.8B that the two failed attempts copied.
- [Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations](https://arxiv.org/abs/2611.09382) and [Tree-Ising: Repairing Language Models with Bit-Level Ising Edits](https://arxiv.org/abs/2611.14210): follow-up studies of weight edits against looping in small Qwen models. The first edits a column in the final block toward a direction defined by tokens; the second makes bit-level edits along a loop-minus-end direction.
- [Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids?](https://arxiv.org/abs/2611.08070): a critique of the weight-reading claims.
