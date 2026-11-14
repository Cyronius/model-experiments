# Broader and more calibration, Qwen3.5-2B

This combines the two calibration changes from the
[previous local screen](../local-projection-20261025/README.md): broader task
coverage and roughly four times as many calibration tokens. All compute is
local, with $0 cloud spend and no whole-model healing.

## Controlled comparison

| Calibration | About 64k tokens | About 256k tokens |
|---|---:|---:|
| Chat only | 8/56 correct | 14/56 correct |
| Broader mixture | 24/56 correct | **25/56 correct** |

More data adds only one net correct closed answer to the broader mixture:
two answers improve and one regresses. Sorting stays at 5/8, units improve
from 3/8 to 4/8, and grounding/memory stay at 8/8 each. Arithmetic, Python
execution and JSON construction remain at 0/8. The completed closed-answer
screen therefore shows little additional benefit from quadrupling the broader
corpus; the main earlier improvement came from changing the mixture.

This is a small fixed screen, so a one-answer difference does not establish
a general capability improvement. The observed gain is also smaller than
simply adding the separate gains: 25 rather than 30 correct answers. That is
a descriptive comparison, not a statistical interaction claim.

The long-output checks do not improve. All four programs finish and parse,
but only the search function passes all three return-value tests, and it uses
linear scanning instead of binary search. Thus **zero of four meet the full
requirements**, unchanged from the smaller broad corpus. Both science answers
are factually wrong and repeat through the 1,024-token limit. The photosynthesis
answer calls respiration a light-to-energy process; the spoon answer discusses
hardness instead of heat transfer. Across all 62 prompts, completion drops from
61 to 58, with token-limit hits increasing from one to four. Caps and repetition
remain separate measurements.

The result supports keeping broader calibration, but does not support another
increase in calibration volume as the immediate next priority. The 64k broad
recipe is a reasonable cheaper baseline for a correctness-healing comparison.
Neither this result nor the prior 24/56 result restores the 2B model's reasoning
or establishes what will transfer to 27B.

Stock scored 51/56 for content correctness. Formatting and integer-string
sorting are handled exactly as in the prior screen. The same 56 closed tasks,
four programming tasks and two science explanations are retained; this is a
small mechanism comparison, not a broad intelligence benchmark.

The combined arm retains the exact shared-projection architecture, residual
width 1,520, constant RMS corrections, and approximately 74.35% of the original
text-model parameters. There are no per-layer bridges, nonlinear decoders,
fitted linear decoders or token-dependent normalization predictors. A fresh
source model is loaded for evaluation. Greedy decoding, disabled thinking,
four CPU threads, FP32 evaluation, batch companions and token limits are
unchanged (256 closed, 512 code, 1,024 explanations).

## Frozen calibration corpus

There are **262,691 tokens in 1,737 complete conversations**:

| Source | Conversations | Tokens |
|---|---:|---:|
| Longer UltraChat | 131 | 131,517 |
| Verified Horizon math | 1,464 | 65,583 |
| Verified Horizon code I/O | 142 | 65,591 |

The target remains 50% longer chat, 25% math and 25% code I/O by token count.
Complete conversations slightly overshoot each quota. All source data were
already available locally. The original broad corpus's 438 conversations are
an exact prefix; 1,299 new conversations are appended without repeats. The
new examples follow the same seeded long-chat ordering and Horizon training
ordering. Held-out validation prompts are excluded, and an exact casefolded
test-message substring check finds no overlap. This does not establish absence
of semantic similarity or source-model pretraining overlap.

## Incremental activations and controls

The original broad activation packet is reused, and only the added
conversations run through the source model. Covariance sums combine additively
at all 49 residual normalization inputs. Sampled states retain the original
token weighting for fitting RMS constants. Native CPU BF16 autocast is used
for source activation passes, with FP32 covariance accumulation, FP64 sum and
eigensolve, and FP32 projection construction and evaluation.

The new incremental test compares split versus whole collection, covariance,
sample order, token weights and the resulting RMS constant. It passes. The
existing nine builder tests and real full-width/checkpoint-reload audits are
inherited from the preceding experiment, whose builder source hashes are
checked before preparation. Those copied control records are not a fresh
checkpoint-reload audit of this new arm. The new model must pass its own cache
consistency check before generation and exactly match the baseline parameter
and stored-scalar counts.

The new cache check passes, and the evaluated model has exactly 1,399,093,040
parameters and 1,399,093,089 stored scalars, matching the previous broad arm.
Mean sampled held-out retained energy increases from 93.29% to 93.82%, and
held-out chat KL falls from 1.1333 to 1.0211. These numerical improvements
coexist with the small answer gain and worse completion. The fitting weights
sum to exactly 262,691 tokens. No model selection used those held-out values.

## Reproduction

From the repository root, using the existing pinned environment and Dibs:

```bash
dibs run cpu --label "Broader and more calibration" --eta 90m -- \
  target/manifold-audit/env/bin/python manifold-compression/local_projection_broad_more.py run
```

`protocol.json` records the frozen data hash, controls and method.
`data.json` contains all calibration conversations and unchanged evaluations.
`additional-states.pt` contains only newly collected activations; its original
prefix packet remains in the previous experiment directory. The fitted recipe
and raw generations are saved separately. The final summary reports paired
gains/losses against each baseline, execution tests separately from program
requirements, completion separately from repetition, and manual science-answer
review. The artifact manifest hashes both new artifacts and reused dependencies.
