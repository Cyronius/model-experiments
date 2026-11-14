# Local Qwen2B projection experiments, 2026-10-25

This tests three alternatives to whole-model correctness healing: different
projection geometry, more/broader activation calibration, and more accurate
reconstruction. All compute is local, with **$0 cloud spend**. Results are a
small mechanism screen, not a broad intelligence benchmark or a prediction of
what will happen to the 27B model.

## Answer results

**Broader calibration is the strongest result; none of these changes reaches
base-model correctness.** The shared projection with the broader 64k-token
mixture scores 24/56, versus 8/56 for the same-size chat-only baseline and 51/56
for stock. There is no whole-model healing in these comparisons.

| Variant | Stored scalars retained | Closed answers correct |
|---|---:|---:|
| Stock | 100.00% | 51/56 |
| Shared, 16k chat | 74.35% | 8/56 |
| Shared, 64k chat | 74.35% | 8/56 |
| Shared, 256k chat | 74.35% | 14/56 |
| Shared, broader 64k | 74.35% | 24/56 |
| Per-layer, matched storage | 73.80% | 1/56 |
| Per-layer, same width; extra storage | 77.17% | 1/56 |
| 64k + token-dependent RMS | 74.35% | 8/56 |
| 256k + fitted linear decoder | 74.35% | 7/56 |

Scores assess content, with formatting recorded separately. Stock's strict
value/type score is 43/56 because it quotes all eight correctly sorted integer
lists. These are fresh synthetic tasks from seven families, not a general
intelligence percentage. The same-width per-layer diagnostic uses more storage
than the shared model; even that allowance does not recover its accuracy.

The broader arm scores 5/8 sorting, 3/8 units, 8/8 grounding and 8/8 memory,
but 0/8 arithmetic, Python execution and JSON construction. It has one
program that passes return-value tests but uses linear scanning when asked for
binary search: **zero of four programs meet the full reviewed requirements**,
versus two for stock. Both broad-arm science explanations are wrong and
repetitive, even though photosynthesis eventually stops. Across all 62 prompts,
token-limit hits fall from 31 with the 64k chat-only baseline to one with the
broader mixture. Completion is an improvement, not proof of factual correctness.

## Protocol

The pinned source is `Qwen/Qwen3.5-2B`, revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. The text model has
1,881,825,088 parameters and residual width 2,048. All 24 layers, attention
heads, recurrent states and MLP neurons remain. Only residual width changes.
The shared projection uses width 1,520 and 1,399,093,040 parameters, about
74.35% of the source. Stored-scalar accounting also includes normalization
buffers; tied vocabulary weights are counted once.

The per-layer alternative fits a separate basis at each layer. Dense bridges
carry the skip path into the next basis, while MLP updates project directly
into that destination. Bridges count against the parameter budget. Width
1,456 fits within the shared arm's storage budget. A second, explicitly larger
per-layer arm keeps width 1,520 to separate geometry from the bridge overhead.

Calibration uses complete conversations, with covariance measured at all
49 residual normalization inputs and every token:

| Corpus | Conversations | Tokens | Purpose |
|---|---:|---:|---|
| Small | 36 | 16,486 | Low-data shared basis |
| Large | 141 | 65,968 | Primary comparison basis |
| Extended | 583 | 262,293 | Nested extension of the same sampler |
| Broad | 438 | 65,759 | Half longer chat, quarters verified math and code I/O |
| Validation | 16 | 7,755 | Disjoint held-out conversations |

The extension reproduces the original three-tokenizer sampler and asserts the
64k corpus is an exact prefix. No held-out test prompt occurs in its source
conversations. The broad arm changes coverage, conversation length and the
number of independent conversations; it does not isolate those effects. PCA uses trace-normalized pooled,
uncentered second moments. Per-layer bases pool the layer's two sites, with
the final norm included at the last layer.

Calibration forward passes use CPU BF16 autocast, FP32 covariance accumulation
and FP64 eigensolves. Construction and answer evaluation use FP32. Every arm
starts from a fresh source load. Answers are greedy, thinking disabled, with
no repetition penalty, in fixed batches of four. There is no whole-model
language training. The small reconstruction networks are separately fitted
diagnostics, not deployed language models.

There are 56 closed tasks: eight each of arithmetic, sorting, Python execution,
unit conversion, JSON, grounding and memory. Four programming problems have
three sandboxed execution checks each, including input immutability for interval
merging. Two explanations inspect photosynthesis/respiration and equal-temperature
metal/wood spoons. Limits are 256, 512 and 1,024 tokens respectively. Hitting a
limit and showing repetitive degeneration are reported separately; these runs
do not establish whether an unfinished answer would finish by 4,096 tokens.

Sorting content credit accepts unambiguous integer strings, with strict typed
correctness retained separately. This clarification followed inspection of
stock, before inspection of compressed answers, and applies uniformly to all
saved outputs. Stock ordered all eight lists correctly but quoted the numbers.
The final summarizer regrades every arm using the same rule. Raw runtime
grades and generated text remain intact.

Repetition flags combine the original word-gram metric with the project's
existing rule of eight identical consecutive nonempty lines, which also catches
empty code-fence loops. These are screening flags, not a diagnosis: manual
review distinguishes redundant but finished text from sustained repetition.
Empty Python code can parse successfully but cannot pass the execution tests.
Passing program execution is also distinct from satisfying every instruction:
the broad arm's search function passes return-value checks using a linear scan,
but fails the explicit binary-search requirement. Both counts are reported;
every execution-passing program receives manual requirement review.

## Approximation experiments

The normalization arm replaces a constant RMS correction with a bounded ridge
predictor of the missing normalization factor. Its features are 16 retained
coordinates normalized by RMS, their squares, and log RMS. Calibration bounds
and fitting are independent of held-out answers.

The nonlinear diagnostic predicts dropped coordinates from retained ones at
the first, middle and final sites. A fixed ridge fit is followed by a small
64-hidden-unit SiLU residual network. It uses either 4,512 or 18,656 calibration
states per site and 512 held-out states. The larger-data run uses the same
64k projection basis, with fixed checkpoints at 400 and 1,600 fitting steps.
Held-out error does not select a checkpoint. This measures recoverable state
information, not answer correctness.

The fitted-linear follow-up uses the extended projection basis and all 262k
tokens' covariance at each of 49 sites. If `P` contains kept directions and
`R` dropped directions, its decoder is `D = P + R A.T`, with standardized
zero-intercept ridge 0.1. It verifies `P.T D = I`, then folds `D` into reader
weights and the final normalization matrix. Encoders, writer projections and
RMS scales are unchanged. Deployed shapes and parameter counts exactly match
the extended shared arm; calibration recipe matrices are not retained in the
deployed model. This follow-up was added after the larger-data linear diagnostic
improved; the held-out answer set and grading remain unchanged.

## Implementation checks

Nine tests pass, including full-width rotations, reduced-width formulas against
independent references, cache consistency, serialization with weights zeroed
before reload, decoder folding and substantive answer grading.

The real full-width per-layer control passes: max logit error 0.0000563,
mean KL approximately 1.06e-11, and 100% top-1 agreement on the checked positions.
Its cache check passes. The saved 5.56 GB projected per-layer checkpoint reloads
with exactly identical checked logits and no missing/unexpected keys. Every
answer arm also passes a cache consistency gate.

An initial whole-block transport implementation unnecessarily projected an
MLP update twice at reduced width. It was corrected **before any compressed
candidate answer evaluation** and covered by an independent reference test.
Old numerical audits are retained under `superseded-transport-audit/`; current
control and roundtrip files describe the corrected implementation.

## Representation diagnostics

The geometry can look good while answers fail. Mean retained energy on sampled
held-out states is 93.86% at 16k calibration tokens, 94.40% at 64k, and 94.60%
at 262k. The matched-budget per-layer basis retains 96.35%, and the same-width
per-layer basis 96.86%. These percentages are activation energy, not retained
intelligence. Shared-basis overlap rises from 0.911 for 16k versus 64k to 0.948
for 64k versus 262k (mean squared cosine of principal angles).

Token-dependent normalization cuts mean held-out RMS-factor MSE from
0.0003366 to 0.0000994, approximately 70%. This improvement also appears when
the final output-normalization site is excluded; it is not just a logit-scale
effect. Its deployed answer results must nevertheless justify the change.

More calibration samples help linear reconstruction much more than adding the
small nonlinear network. With the projection basis held fixed:

| Site | 4,512 states: linear recovery | 18,656 states: linear recovery | 18,656 states: linear + nonlinear, 1,600 steps |
|---|---:|---:|---:|
| First | 5.95% | 33.72% | 35.72% |
| Middle | 5.42% | 24.75% | 24.23% |
| Final | 13.98% | 29.05% | 28.49% |

These are fractions of **discarded** energy recovered on held-out states.
The nonlinear addition gives a small first-site benefit, but hurts at the
middle and final sites despite additional fitting. This does not establish
that all nonlinear decoders are ineffective; it limits the case for this
particular small decoder and fitting recipe. The full-covariance folded linear
decoder recovers a mean 27.60% across all 49 sites (range 20.37–49.84%).

## What this changes about the next experiment

The best-supported change is the calibration corpus. At the same token and
model-storage budgets, the broad mixture gains 16 closed answers over the 64k
chat-only baseline, with no lost correct answers. Increasing chat-only volume
to 262k gains seven grounding answers but loses one memory answer, a net six.
The broad mixture still gets zero arithmetic and Python-execution cases right.
Its improvements are concentrated in sorting, units, structured grounding and
completion behavior; they do not establish restored general reasoning.

Do not adopt the tested per-layer geometry, RMS predictor or folded linear
decoder on the strength of their representation metrics. The matched-budget
per-layer model retains more energy but loses answer accuracy; the folded
linear decoder improves held-out reconstruction and chat divergence while
reducing closed-answer accuracy from 14 to seven. Conversely, the broad
mixture has worse chat divergence than the shared baseline while answering
more tasks correctly. Select changes by independently checked task behavior.

A reasonable next build keeps the current shared projection, measures it on
a broader task/length mixture, then applies correctness healing with full
programs and complete, checked explanations as well as short answers. Re-run
the same correctness and long-form checks after each change. The larger
27B model's existing successful healing result remains separate evidence;
this 2B screen does not establish that the calibration gain transfers to it.

The 2B model at this compression level is badly damaged even in the best
arm. That creates a low-score floor for comparing more subtle mechanisms.
If revisiting geometry, first test at a gentler retention level with a usable
baseline and a loss that reflects downstream behavior. The current experiment
does not rule out other jointly fitted, nonlinear or task-sensitive projections.

## Reproduction and artifacts

Use the pinned existing environment `target/manifold-audit/env/bin/python` and
coordinate compute through Dibs. From the repository root, for an existing
calibration directory:

```bash
dibs run cpu --label "Projection comparisons" --eta 2h -- \
  target/manifold-audit/env/bin/python manifold-compression/local_projection_probe.py finish --threads 4
dibs run cpu --label "Extended calibration" --eta 90m -- \
  target/manifold-audit/env/bin/python manifold-compression/local_projection_more_data.py
dibs run cpu --label "Fitted linear decoder" --eta 60m -- \
  target/manifold-audit/env/bin/python manifold-compression/local_projection_linear.py
```

`protocol.json` records amendments and the frozen data hash. `summary.json`
contains final uniform grades and paired gains/losses. Per-arm generation files
preserve exact outputs, with separate answer-review files. `manual-review.json`
contains qualitative judgments. Geometry, RMS, reconstruction and linear-decoder
audits preserve the diagnostic measurements. `artifact-manifest.json` hashes
reports, source files, projection recipes, decoder states and the verified full
checkpoint. These research models use explicit recipe/state loaders; they are
not standalone Hugging Face exports.
