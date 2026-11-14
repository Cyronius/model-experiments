# Larger-model projection results — 2026-10-24

**Larger scale helped at 75% retention, but did not rescue the 50% projection.
Neither large architecture reached stock parity.** All nine primary arms
completed on one H200. No healing or MLP-neuron pruning was performed.

The experiment used the corrected final output matrix and one shared linear
PCA basis across residual sites. This tests that implementation, not every
possible manifold projection or a learned nonlinear reconstruction.

## Primary results

“75%” and “50%” mean approximately that fraction of the original **text-model
parameters retained**, at the same BF16 storage precision. All layers, heads,
head dimensions and MLP neurons were kept. Counts include vocabulary weights
and the output matrix; vision/MTP weights are excluded.

| Model / arm | Parameters | Strict checks / 64 | Stock successes retained | Ended / 72 | Word-loop flags / 72 | Held-out KL |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B stock | 1.882B | 50 | — | 70 | 0 | 0 |
| Qwen2, 74.35% retained | 1.399B | 5 | 5/50 | 45 | 12 | 0.789 |
| Qwen2, 49.28% retained | 0.927B | 1 | 1/50 | 1 | 9 | 2.868 |
| Qwen3.5-27B stock | 26.896B | 62 | — | 70 | 0 | 0 |
| Qwen27, 74.74% retained | 20.103B | 40 | 39/62 | 70 | 2 | 0.455 |
| Qwen27, 49.72% retained | 13.371B | 2 | 0/62 | 43 | 12 | 1.804 |
| Mistral Small 24B stock | 23.572B | 47 | — | 71 | 0 | 0 |
| Mistral24, 74.75% retained | 17.620B | 23 | 23/47 | 71 | 2 | 0.469 |
| Mistral24, 49.71% retained | 11.719B | 2 | 2/47 | 30 | 28 | 2.288 |

All six compressed arms failed the predeclared screen. It requires retaining
at least 95% of stock successes, no increase in word-loop flags, and at most
one fewer completed answer. Passing would only justify broader evaluation.

On the **49 checks both Qwen stock models passed**, the 75% models retained
5 successes for Qwen2 and 29 for Qwen27; both half-size models retained zero.
This supports a benefit from larger scale for these checkpoints at a modest
cut. It does not isolate a scaling law: depth, head counts and embedding tying
also differ between the Qwen sizes.

## What the scores mean

These are task-and-format checks, **not general benchmark accuracy**. They
include exact formatting and word limits. For example, Mistral stock emits
otherwise valid JSON inside code fences. Several compressed answers fail only
because of punctuation, capitalization or extra explanatory text. Conversely,
a word-limit check can pass nonsense: Qwen27 at half size passes two such
checks despite broken content.

Post-hoc review of every lost stock success at 75% retention found:

| Model | Clear content errors | Formatting only | Length only | Ambiguous ordering plus formatting |
|---|---:|---:|---:|---:|
| Qwen27 | 13 | 9 | 0 | 1 |
| Mistral24 | 11 | 12 | 1 | 0 |

The ambiguous Qwen list task preserves the relative order of existing items
but inserts the new item between them. The scorer assumes appending, which the
prompt does not state explicitly. The raw primary score is preserved, but this
is not counted as an unambiguous content error. The review was by the assistant,
after seeing the answers and without blinding; see
[case classifications](mc-scale-9b5ca68c9da9/manual-review.json).

The degradation is nevertheless substantive. At 75%, Qwen27 answers 29×17 as
507 and Mistral answers 423, versus the stock answer 493. Both lose Python list
aliasing behavior. In long-form generation, Qwen27 repeats a claim that binary
search is not a real algorithm. Mistral produces malformed C++ under a Python
code fence and repeats a regex fragment through the token cap.

Word-loop flags are imperfect: they miss punctuation-only loops and can flag
ordinary repeated structure. A separate post-hoc scan for eight or more
identical nonempty lines flags 15/10 Qwen2 answers and 0/24 Qwen27 answers at
75%/50%, respectively; Mistral has 0/1. Stocks have zero. These overlap the
word flags and must not be added to them. “Ended” means EOS appeared within the
cap, not that the answer was correct or complete in substance.

## Numerical controls and retained energy

The full-width rotation and cached-inference algebra gates passed for all
three models. BF16 rounding controls also passed on 512 held-out positions:

| Model | FP32 full-width mean KL | BF16 full-width mean KL | BF16 top-1 agreement |
|---|---:|---:|---:|
| Qwen2 | 3.68e-11 | 0.000944 | 99.61% |
| Qwen27 | 1.04e-10 | 0.000569 | 99.02% |
| Mistral24 | 1.74e-10 | 0.000596 | 99.61% |

The measured compression errors are much larger than these controls. This
supports attributing the failures to the compression approximations rather
than the full-width rotation or BF16 rounding alone. It does not prove every
possible implementation issue has been excluded. Raw `numerical_pass` fields
inside BF16 comparisons use a strict FP32 tolerance; the actual BF16 cache and
rounding gates are specified in the [protocol](README.md).

| Model | Pooled PCA energy retained at 75% | At 50% |
|---|---:|---:|
| Qwen2 | 94.33% | 86.77% |
| Qwen27 | 97.82% | 94.75% |
| Mistral24 | 97.33% | 93.40% |

High retained activation energy is not sufficient evidence of preserved
behavior. In particular, keeping 94.75% of pooled energy did not make the
half-size Qwen27 useful on this screen. These are calibration-energy fractions,
not fractions of knowledge or capability retained.

## Final-readout diagnostics

The primary retained-space output matrix is exact for a full orthogonal
rotation, but remains an approximation after truncation. An exact gain-folded
untied head removes an additional projection after the final normalization
gain. It still cannot directly restore discarded residual coordinates.

On unchanged teacher activations from the first two validation conversations
(32 positions), changing the readout made little difference for either Qwen
model. Mistral showed a larger difference. This prompted a separate Mistral
75% ablation using the same projected body, basis, RMS corrections and BF16
precision, with an identity adapter retained to match parameter count:

| Mistral75 variant | Held-out KL / 512 positions | Strict short checks / 64 | Stock successes retained |
|---|---:|---:|---:|
| Primary retained-space output matrix | 0.4687 | 23 | 23/47 |
| Exact gain-folded output head | 0.4453 | 29 | 29/47 |

All six recovered checks were formatting fixes; wrong multiplication and list
aliasing remained. The ablation completed all 64 short responses with no word
loop flags, as did the primary 75% arm on that subset. **It did not run the
eight long prompts**, so it supplies no evidence of a long-form loop repair.
This result is exploratory and is not substituted into the primary table.

A four-prompt Qwen2 batch-versus-single diagnostic also reproduced its stock
unit-conversion error in both modes. That small check found no evidence that
padding caused that particular error; it does not establish complete batch
invariance. Two new algebra tests verify the exact folded readout on retained
subspace states. The preflight suite had 47 passing tests; these two also pass.

## Recommended next step

Keep the output correction, reproducibility machinery and numerical gates.
Work first at **75% retention**, where both large models retain useful behavior
but still lose substantial content and instruction following. A ground-up
restart or an even larger model is not supported by these results.

The next controlled comparison should vary **calibration coverage and the
projection basis separately**: current shared basis versus a basis per block,
each with the current calibration set and a larger, diverse set of roughly
250K–1M tokens. Count residual transition matrices and readout weights in the
same total parameter budget. Measure held-out functional error as well as
activation reconstruction; collect later-layer calibration through the
compressed prefix where appropriate. Keep a fresh generation set for the
eventual decision, since the present checks have now been inspected repeatedly.

This is a proposed experiment, not an established fix. The present run used
only 65,968 Qwen calibration tokens and 89,157 Mistral tokens, so it cannot
rule out a material benefit from more coverage. SliceGPT explicitly uses a
different basis per block, and its main experiments use 1,024 sequences of
2,048 tokens; its reported calibration ablation also shows sample size matters.
[Primary paper, §§3.3 and A.3](https://arxiv.org/html/2401.15024v2)

After identifying the better initialization, test a bounded healing run with
task retention, EOS completion and repetition as stop criteria alongside KL.
The earlier local healing pilot regressed on Qwen, so simply training the
current half-size model longer is not a demonstrated route to parity. Learned
nonlinear reconstruction remains an open option; this scale screen did not
test it.

## Execution, cost and artifacts

The approved H200 job ran from 22:55:54 to 23:36:15 UTC, approximately **40.35
minutes** including setup. The pod was deleted; a separate API GET confirmed
HTTP 404. Final synchronization succeeded and **all 43 remotely manifested
artifacts matched their SHA-256 hashes**.

Estimated cost is **$3.58**, using the observed $5.29/hour GPU rate plus the
proposed temporary-disk estimate. This is below the $10 cap. The billing API
had not yet posted a charge at the final query, so $3.58 is an estimate, not an
invoice or a claim of zero billed usage. No persistent volume was created.

- [Machine-readable comparison](mc-scale-9b5ca68c9da9/comparison.json)
- [Lifecycle and termination record](mc-scale-9b5ca68c9da9/job-lifecycle.json)
- [Artifact hashes](mc-scale-9b5ca68c9da9/artifact-manifest.json)
- [Executed source](mc-scale-9b5ca68c9da9/executed-source/)
- [Qwen2 raw results](mc-scale-9b5ca68c9da9/qwen2/result.json), [Qwen27 raw results](mc-scale-9b5ca68c9da9/qwen27/result.json), [Mistral raw results](mc-scale-9b5ca68c9da9/mistral24/result.json)
- [Exact-readout ablation](mc-scale-9b5ca68c9da9/mistral24-exact-readout/result.json)

Basis tensors, RMS scales, projected configs, pinned source revisions, raw
generations and executed code are saved. **Full compressed HF weight copies
were not exported** during this short lease; the unhealed arms can be
reconstructed from the saved recipe and original pinned checkpoints. Supplemental
diagnostics shared the GPU with the primary run, so timing is not a clean
inference-performance benchmark. No additional paid run is authorized by this
job's approval.
