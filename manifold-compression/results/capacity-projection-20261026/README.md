# Healing capacity versus projection: matched 2×2 pilot

Status: complete locally, **$0 cloud spend**. Four fresh-adapter arms each received **32,975 answer-token presentations** over eight updates: **131,900 total training answer tokens**. An additional rank-compression diagnostic used no new training tokens. The previous 99,283-token healed checkpoint is unchanged; these fresh-start pilots are not replacements for it.

## What the experiment establishes

**The training path is a demonstrated bottleneck; final adapter rank alone does not explain the weak direct rank-16 result.** On the same frozen 1.399B-parameter projection, direct rank-16 training scores 5/39, rank-64 training scores 10/39, and compressing that trained rank-64 update back to rank 16 scores 11/39 without further training. The full-rank reconstruction control reproduces every one of the original 47 answers exactly. Thus a rank-16 final update can represent the better observed behavior; the original rank-16 training run did not find it.

This does **not** establish that the projection is lossless, that more optimization will recover every lost skill, or that the one extra answer from truncation generalizes. All projected variants still fail the four programs, and science remains substantially impaired. The truncated model still has one capped, repetitive, factually wrong science response.

The uncompressed controls reveal a second issue: this short curriculum itself trades away previously correct arithmetic and code. Stock closed accuracy falls 19/39 → 14/39 at rank 16 and → 17/39 at rank 64. Its science explanations improve in some respects, so the result is a tradeoff rather than uniform failure to learn. Projection cannot explain regressions that also occur without projection.

## Fixed-endpoint results

| Model / training | Closed | New arithmetic | Retention | Programs, original + edge checks | Repetition flags / 47 | Token-limit hits / 47 |
|---|---:|---:|---:|---:|---:|---:|
| Projected, unhealed | 6/39 | 3/32 | 3/7 | 0/4 | 6 | 8 |
| Stock, before healing | 19/39 | 12/32 | 7/7 | 4/4 | 0 | 5 |
| projected-r16 | 5/39 | 0/32 | 5/7 | 0/4 | 2 | 2 |
| projected-r64 | 10/39 | 4/32 | 6/7 | 0/4 | 2 | 2 |
| stock-r16 | 14/39 | 7/32 | 7/7 | 2/4 | 0 | 0 |
| stock-r64 | 17/39 | 10/32 | 7/7 | 3/4 | 0 | 0 |
| Projected rank 64 → 16 (diagnostic) | 11/39 | 5/32 | 6/7 | 0/4 | 1 | 1 |

The primary program I/O scores are 0/4 for both projected arms and 3/4 for both trained stock arms. The supplemental edge audit reduces stock rank 16 to 2/4: its positive-streak function passes the four original examples but returns zero for `[1]`. Stock rank 64 fixes the unique-character regression but returns one for a list with no positive values. Baseline stock passes all program checks. The edge audit was declared after observing stock-r16's flaw, then applied uniformly to every baseline, endpoint, and diagnostic variant; original primary grades remain intact. Passing these checks is not a proof for all inputs.

Increasing rank produces **+5 closed answers for the projected model and +3 for stock**, a difference of two questions on this small set. Within arithmetic, the rank gain is +4 projected versus +3 stock. This is not strong evidence that the rank benefit is unique to compression repair. The projected rank-64 improvement over the unhealed projection is only **3/32 → 4/32** on arithmetic; much of its total 6/39 → 10/39 gain is retention. Comparing only against the regressed rank-16 endpoint would overstate recovery from the original projection.

Development also shows that more updates are not monotonically beneficial. Projected rank 16 goes 0 → 1 → 1 correct arithmetic answers out of eight at updates 0/4/8; rank 64 goes 0 → 2 → 0. Stock rank 16 goes 5 → 2 → 3, and stock rank 64 goes 5 → 3 → 2. Development programs stay at 1/2 projected and 2/2 stock. The primary endpoint remains fixed at update eight; no checkpoint was selected using test results.

## Rank-compression diagnostic

The diagnostic was declared after seeing the projected rank comparison and before stock rank-64's trained evaluation. It is exploratory and separate from the primary experiment. For each learned update `B @ A`, double-precision thin QR and a small-core SVD construct balanced FP32 factors. Scaling remains `alpha/rank = 2`. The full-rank variant preserves all singular directions; the truncated variant retains the leading 16 per linear module.

The full-rank control matches all 47 answers, token counts, termination states and grades. The rank-16 approximation retains **79.23% of the squared Frobenius magnitude of the learned updates**—a weight-space measure, not a fraction of intelligence. It retains every correct closed answer from rank 64 and gains one multi-step arithmetic answer: 10/39 → 11/39. Arithmetic becomes 5/32; programs remain 0/4. One science loop terminates, leaving one loop and one cap instead of two.

Direct rank-16 and rank-64-then-truncated models have the same frozen projection and the same final number of adapter parameters: **15,247,872**. The latter used **60,991,488 trainable adapter parameters during training**. That makes training trajectory, initialization/direction coverage, or optimization plausible causes of the difference. This experiment does not distinguish those mechanisms from one another. It demonstrates that a better rank-16 solution exists for the observed evaluation; it does not prove all useful rank-64 behavior can always be compressed this way.

## Factual review and looping

The rank-16 projected endpoint fails the science questions, although most answers finish. Rank 64 correctly computes 12 coulombs / 4 seconds = 3 amperes and gives a correct compass explanation on development. It still assigns the wrong sign to an atom gaining electrons and wrongly changes its element. Its diffusion and friction responses repeat through the 512-token limit; the friction explanation incorrectly converts the lost kinetic energy primarily to potential energy.

Rank truncation preserves the correct current answer. Its diffusion response now finishes and has the right main conclusion, but the explanation lacks a clear account of thermal motion and balanced exchange. Charge remains false, and friction still loops. These outcomes are recorded in `science-manual-review.json`; no aggregate intelligence score is inferred from them.

Stock rank 16 repairs the charge derivation and gives a coherent equilibrium explanation; its current answer explicitly retracts an initial wrong number and supplies the right calculation. Stock rank 64 gives the correct core conclusions, but invents a specific number of protons and neutrons in the charge question. Correct final conclusions, factual coherence, termination, and repetition flags are distinct checks.

The development repetition flags for projected rank 16 fall 3 → 0 after 16,432 answer tokens, then rise to one at update eight. This independently supports cheap termination repair, but not a universal or permanent elimination of looping. The earlier successful 24-update checkpoint and its original benchmark results remain untouched.

## Controlled setup and limits

Four arms start with fresh zero-output LoRA adapters on either the unhealed shared-broad projection or the original Qwen3.5-2B text model. Rank-16 initial A directions are the first 16 of the rank-64 directions within each architecture; both use alpha/rank=2. Both baseline pairs reproduce identical complete development and test outputs. Source revision, weights, recipe, training data and training code are hash checked.

Every arm uses the exact first eight updates of the frozen expanded curriculum: fresh AdamW, learning rate 3e-5, betas (0.9, 0.95), zero weight decay, gradient clipping at one, and no warmup. Training uses CPU BF16 autocast with FP32 stored parameters; evaluation uses FP32, four threads, greedy nonthinking decoding, and no repetition penalties. Closed-answer/program/science test caps are 256/512/512 tokens; development science retains its original 1024 cap. The stock baseline therefore measures this decoding regime, not the model's unrestricted reasoning ceiling.

The 47-case test contains 32 new arithmetic operand sets, seven new retention questions, four known program regression tasks, and four newly worded applications of taught science concepts. Arithmetic training uses symbolic tasks while the held-out questions use word-problem forms. The science pool is small and resampled. Development code/science questions were previously inspected. This is one seed, a short budget, and no per-rank hyperparameter tuning; rank changes optimization geometry as well as parameter count. Equal tokens are not equal FLOPs. No alternative matched-size projection or second model architecture was tested in this round.

## What to do next

Prioritize **a better healing procedure before rebuilding the projection**. A useful next controlled run would expand the existing best adapter to rank 64 while preserving its starting function, then compare repair with explicit retention checks for previously correct math, code edge cases, factual explanations, and termination. Include varied problem wording and stronger verified replay; use fresh development data for early stopping and untouched cases for final testing. A stock control should demonstrate acceptable retention before committing to expensive scaling.

Training with a larger adapter and optionally compressing it afterward is now a concrete candidate. Do not promote this post-hoc truncated checkpoint merely because it scores one extra answer on an inspected small test. The remaining gap can still reflect projection damage, the training objective/data, insufficient training, or a combination; these results do not establish irreversible information loss or larger-model parity.

## Verification and artifacts

Seven setup/curriculum tests and two rank-factorization tests pass. Every primary arm passes baseline cache checks, exact in-process adapter reload, fresh reconstruction of four answers, and a final trained-cache check. The truncated adapter independently reproduces four answers and passes its cache gate. The full-rank SVD control reproduces all 47 complete generations exactly. All frozen training-code hashes are unchanged.

`summary.json` contains the paired fixed-endpoint comparison and category breakdowns. `program-edge-audit.json` retains supplemental code results beside the original grades. Each arm stores `step-0`, `step-4`, `step-8`, optimizer states, generation files, training logs, and gates. `rank-truncation/` contains both diagnostic adapters, spectra, control results and reload checks. `protocol.json`, `data.json`, `environment.json`, code hashes and the final artifact manifest record provenance. These are recipe-plus-adapter artifacts, not standalone Hugging Face exports.

From the repository root, reconstruct a primary endpoint with:

```python
import sys
sys.path.insert(0, "manifold-compression")
from load_capacity_projection import load
model = load("projected-r64", step=8)
```

For the diagnostic, use `truncate_capacity_adapter.load_variant("rank16-truncated")`. Both require this repository's pinned source and projection artifacts. No merged model was deployed or uploaded.

Two CPU workers ran primary arms with unchanged four-thread settings. As documented in `scheduling-amendment.json`, the original serial launcher stopped at its existing-directory guard after completing both projected arms; it did not overwrite the stock worker's artifacts. All four actual model runs completed successfully. There are no performance-speed claims from these shared-resource runs.
