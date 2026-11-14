# Allocation pilot: wider residual plus MLP pruning at the same budget

Status: complete locally, **$0 cloud spend**. Teacher-forced screen only; the
generation gates have not been run, and nothing here was healed or promoted.

**At the same stored-scalar budget, the P1 construction is clearly less
destructive than the current shared projection before any healing, but both
remain far from stock.** P1 keeps residual width 1,792 instead of 1,520 and
prunes every MLP from 6,144 to 4,200 neurons. On every text source of the
frozen broad suite, P1 is closer to stock than P0: KL to stock falls by 18–45%,
reference cross-entropy falls, and top-1 agreement rises, with every paired
interval excluding zero. Multiple-choice gains are smaller (MMLU +5.3 points;
HellaSwag and ARC-Easy +1.5 to +3.8 depending on normalization; ARC-Challenge
within noise). P1 also greatly reduces, but does not remove, P0's collapse onto
answer letter A on MMLU.

This supports P1 as a better starting point for the next experiment. It does
not show that P1 generates better answers, that the advantage survives
healing, or that residual width alone is the cause.

## What was compared

| Arm | Residual width | MLP width | Stored scalars | Construction |
|---|---:|---:|---:|---|
| Stock Qwen3.5-2B | 2,048 | 6,144 | 1,881,825,088 parameters | none |
| P0 (current shared-broad) | 1,520 | 6,144 | 1,399,093,089 | shared PCA basis |
| P1 | 1,792 | 4,200 | 1,399,041,649 | same basis, wider; MLP pruned and refit |

Both arms use the same broad 64k PCA basis (`../local-projection-20261025/broad-pca.pt`),
the same constant per-site RMS fit and the same pinned source. P1's stored
scalars are 51,440 below P0's; the budget was computed on the meta device and
matches the measured P0 count exactly. Neither arm was healed.

P1's MLP pruning ranks neurons by E[φ²]·|down column|², layer by layer on the
already-pruned model's own inputs, then refits each down projection by ridge
least squares (ridge 1e-3 of the mean Gram diagonal) to reproduce the
unpruned projected MLP output. The calibration set is the 438 broad rows
(65,759 tokens); excluding each row's first position leaves 65,321 Gram tokens
per layer. Calibration R² per layer is 0.94–0.998 with the refit (median 0.96)
versus 0.82–0.98 for the kept neurons' original weights (median 0.86). These
are calibration-fit numbers, not held-out evidence for the refit.

## Results

Same CPU evaluator for all three arms: frozen manifest
(`../tf-metrics-20261027/manifest.jsonl`, 3,606 rows), FP32, 16 threads, same
`tf_metrics.py` code hash. Text values are per scored token; KL is forward
KL(stock ‖ arm) over the full vocabulary.

| Text source | Stock CE | P0 CE | P1 CE | P0 KL | P1 KL | P0 top-1 | P1 top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| WikiText (117 chunks) | 2.562 | 4.161 | 3.840 | 1.461 | 1.156 | 0.458 | 0.506 |
| UltraChat fresh (207) | 1.366 | 2.081 | 1.917 | 1.096 | 0.901 | 0.595 | 0.632 |
| UltraChat held-out (28) | 1.685 | 2.399 | 2.203 | 1.210 | 0.978 | 0.539 | 0.588 |
| GSM8K answers (400) | 1.086 | 1.534 | 1.357 | 0.611 | 0.395 | 0.752 | 0.805 |
| HumanEval solutions (116) | 0.221 | 0.374 | 0.283 | 0.252 | 0.138 | 0.907 | 0.934 |

P0 → P1 paired differences (95% intervals, bootstrap over documents or
conversations): KL −0.30 [−0.32, −0.29] on WikiText, −0.20 [−0.21, −0.18] on fresh
chat, −0.23 [−0.28, −0.18] on held-out chat, −0.22 [−0.23, −0.21] on GSM8K and
−0.11 [−0.12, −0.10] on HumanEval. Cross-entropy and top-1 move the same way,
with every interval excluding zero. All intervals here are exploratory and
per source, with no correction for multiple comparisons and no pooled score.

Multiple choice, accuracy in percent (sum / per-token / per-character
normalization):

| Source | Stock | P0 | P1 | P0 → P1, per-token [95% CI] |
|---|---|---|---|---|
| MMLU letters (1,138) | 58.7 | 24.6 | 29.9 | +5.3 [+2.3, +8.2] |
| ARC-Challenge (600) | 40.2 / 44.5 / 41.5 | 21.0 / 23.2 / 22.8 | 22.7 / 24.3 / 22.8 | +1.2 [−1.5, +3.8] |
| ARC-Easy (400) | 69.25 / 64.5 / 67.0 | 36.0 / 35.5 / 34.5 | 39.0 / 37.0 / 38.25 | +1.5 [−2.0, +5.0] |
| HellaSwag (600) | 45.5 / 65.7 / 66.0 | 29.5 / 36.2 / 37.5 | 31.2 / 39.0 / 40.7 | +2.8 [+0.2, +5.5] |

Uniform-random baselines from the actual manifest are 25.0% (MMLU, HellaSwag),
25.03% (ARC-Challenge) and 25.06% (ARC-Easy). Unhealed, both projections sit
at or below chance on ARC-Challenge.

**Letter bias.** On MMLU, P0 picks A on 1,127 of 1,138 items (99.0%); its
24.60% accuracy (280 correct) is close to the always-A baseline of 24.52% (279
gold A answers). P1 picks A on 53.2% and B on only 11 items (1.0%), against
33.2% A for stock. P1's MMLU accuracy gain accompanies much weaker A dominance,
but this run cannot separate label or position sensitivity from knowledge; the
counterbalancing probe below is needed for that. MMLU accuracy alone should not
be read as recovered knowledge. On ARC and HellaSwag, the largest single-position
pick fraction is 26.0–29.75% under per-token normalization for all three arms;
it rises to 25.3–38.0% under summed likelihood and 26.0–34.25% per character
(P1's ARC-Challenge sum picks reach 36.3%). See
`../tf-metrics-20261027/choice-position-audit.json`.

**Against stock**, P1 still loses 20–30 points on ARC-Easy and MMLU, 17–20 on
ARC-Challenge and 14–27 on HellaSwag, and its KL to stock is 0.14–1.16
nats/token (`../tf-metrics-20261027/compare-stock-vs-p1.json`).

## What this does and does not show

- **The construction package wins, not one mechanism.** P1 changes residual
  width, prunes MLP neurons and refits the down projections. This comparison
  cannot say how much each contributes. The useful mechanism control is P1
  with the same kept neurons but their original `down[:, keep]` weights instead
  of the refit weights, which isolates the refit's held-out value.
- **Teacher-forced only.** No P0 or P1 answers were generated. P1 has not
  passed any generation gate, so nothing here justifies healing it or replacing
  the current parent.
- **Unhealed bases only.** Whether P1's advantage survives the same healing
  is untested. The retention study (`../retention-objective-20261027/README.md`)
  found that narrow healing improves task scores while moving the model further
  from stock, so a healed comparison must include broad metrics as well.
- **One construction, one calibration set, one width point.** Other
  residual/MLP splits, pruning criteria and calibration sets were not tried.
- **CPU numbers only.** These are not comparable with the CUDA scores in the
  retention study. The frozen scorer code is shared, but the device, runtime
  and kernels differ.
- **Exploratory statistics.** One construction and one evaluation pass;
  intervals are uncorrected for multiple comparisons.

## Verification

Controls (`controls.json`) passed before the build: rebuilding P0's recipe here
reproduces its basis exactly and its RMS scales exactly; the full-width
rotation through the same construction path reproduces stock (maximum logit
error 3.8e-5, cache check 3.4e-5). The build (`build-report.json`, 42 minutes,
16 threads) checkpointed every layer under the frozen protocol hash.
`p1-check.json` confirms the stored-scalar count, identical logits from two
independent rebuilds, and a passing cache/full-sequence check. Ten unit tests
cover the budget formula, refit math, resume, freeze gating and
reproducibility (`../../test_allocation_pilot.py`); nine more cover the scoring
code (`../../test_tf_metrics.py`).

The protocol (`protocol.json`) was frozen for the build only, reviewed by
Codex, with a 1.5-hour runtime bound. The timing measurement estimated 0.86
hours but was flagged NOISY by `dibs --measure` because another process was
active. Its `missing` field still records that P1 evaluation claims require
the generation gates.

Every CPU score file, comparison and audit is bound by SHA-256 in
`../tf-metrics-20261027/cpu-comparison-index.json`. After all inference,
`final-integrity.json` re-ran the freeze check (frozen code, source weights and
inputs unchanged) and binds the protocol, the P1 recipe and the P0 recipe
hashes. Scoring took 74, 81 and 82
minutes for stock, P0 and P1.

## Reconstruct

From `manifold-compression/`:

```python
from allocation_pilot import load_p1, load_p0
p1 = load_p1()   # results/allocation-pilot-20261027/p1-recipe.pt plus the pinned source
p0 = load_p0()
```

`p1-recipe.pt` holds the projection recipe, kept neuron indices and refit down
weights. It needs this repository's pinned source weights and is not a
standalone Hugging Face export.

## Next

1. Generate answers from unhealed stock, P0 and P1 on the fresh suite before
   any healing. First freeze quality tolerances relative to stock, with enough
   code and science coverage: "no regression against a compressed parent that
   scores zero on science" is not a sufficient floor.
2. If P1 still leads, run the refit control above, then heal P0 and P1 under one
   frozen recipe and judge them on both broad and generation metrics, with
   tolerances relative to stock.
3. Treat MMLU letter bias as its own diagnostic: counterbalance answer order on
   new items rather than recalibrating on these labels.
