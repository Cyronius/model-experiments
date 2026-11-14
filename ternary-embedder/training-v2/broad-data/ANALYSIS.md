# Broad-data quality diagnosis and free follow-ups

The 3.77M-pair campaign did not substantially improve general embedding quality. Keep the previous three-layer ternary/float/Q4/Q8 family. The paid job is closed at an estimated $3.5242; pod deletion was verified and checkpoints are local. No further paid allocation was made for these follow-ups.

## What the training curves establish

Broad-data ternary gained 3.62 points on held-out HotpotQA and 1.99 on held-out SQuAD, while its observed wider scores lost 3.12 on Twitter similarity and 2.82 on SICK-R. Float shows the same pattern. Repeated-old-data controls also lose semantic quality. This supports a mismatch between the QA-ranking objective/selection rule and general semantic retention; it does not establish a ternary capacity ceiling.

The float repeat arm was selected for only 0.0103 points of internal validation improvement while losing about 0.98 on the fixed 12-task average. A source-level QA floor alone did not protect semantic quality. About 60% of the broad training mix is GooAQ plus PAQ; adding rows did not guarantee the task balance needed for finance and citation retrieval.

These are inferences from one seed and the existing development suite. Broad-versus-repeat retrieval confidence intervals include zero. See [curves](curve-diagnosis.json) and [controlled results](RESULTS.md).

## Smaller weight updates

Six candidates were frozen before scoring: step 500, a 25% trained / 75% parent weight blend, and a 50/50 blend, for each precision. Ternary remains ternary; architecture and parameter count are unchanged.

| Ternary candidate | Core retrieval | STSBenchmark | Fixed 12 mean | FiQA | SCIDOCS |
|---|---:|---:|---:|---:|---:|
| Previous parent | 45.227 | 82.452 | 70.481 | 24.835 | 16.789 |
| 25% trained-weight blend | 45.585 | 82.649 | 70.425 | 24.943 | 16.950 |

Only the quarter blend improved both core and STS point estimates, so it received the wider follow-up. Its core delta is +0.358 with a 95% paired-query interval of [-0.186, +0.887]. FiQA/SCIDOCS mean delta is +0.134 [-0.191, +0.459]. The wider average still slips; SICK-R loses 0.845. This is useful diagnostic evidence that smaller updates reduce damage, not a qualified replacement. No promotion or new speed claim is made.

See [all six diagnostics](../broad-retention/results/diagnostic-summary.json) and [wider follow-up with uncertainty](../broad-retention/results/followup-summary.json). Intervals use 5,000 paired-query resamples, seed 20261025, and do not include training-seed or candidate-selection uncertainty.

## Human similarity probe

A small, free probe trained 384 positive output-feature scales using human STS training labels, with the encoder frozen. Scales fold into the existing output projection, adding no layer or parameters. Exact model-visible input exclusions left 753 training pairs and 319 validation pairs; no test labels were used for training. Selection used semantic validation with QA retention guards. All four variants were frozen before observed-development scoring.

Core retrieval declined for all four adjusted models relative to their own starting checkpoints. The complete fixed 12-task evaluation and materialization checks passed. Additional FiQA/SCIDOCS scoring was skipped because every variant already failed retrieval retention. This limited output-only experiment cannot establish whether full-encoder human-supervised training would work.

| Adjusted source | Core | STS | Mean12 | Core delta | STS delta | Mean12 delta |
|---|---:|---:|---:|---:|---:|---:|
| parent-float | 46.529 | 83.956 | 71.913 | -0.493 | -0.034 | -0.043 |
| broad-float | 46.142 | 82.965 | 70.667 | -0.223 | 0.096 | -0.038 |
| parent-ternary | 44.894 | 82.431 | 70.485 | -0.333 | -0.021 | 0.004 |
| broad-ternary | 45.471 | 81.907 | 69.569 | -0.176 | 0.057 | -0.036 |

[Full source-matched diagnostics](../human-similarity-head/results/diagnostic-summary.json). Deltas are relative to each named source, not all relative to the original parent.

## Next experiment requirements

Before buying more GPU time, prepare a controlled recipe that combines retrieval training with direct semantic supervision, uses a smaller learning rate, and includes a semantic validation retention guard. Keep the three-layer architecture and separate float/ternary branches. Use a matched control to separate the new objective/data mix from simply taking smaller updates. Prior NLI-auxiliary and mined-negative experiments showed retrieval tradeoffs, so neither should be assumed to fix the problem.

Prioritize audited finance/citation and diverse human-relatedness training data over increasing mostly easy QA volume. These are experiment hypotheses, not demonstrated gains. More epochs or another layer are not supported as the next move by these results. A new paid job requires its own resource/rate/duration/cost proposal and affirmative approval under the user’s AGENTS.md spending rule; unused funds from the closed job are not a new allocation.

All follow-up tasks are already observed development. Full Touche2020 and TREC-COVID judgments remain unopened; the previously exposed Touche query 1 is excluded. No exploratory follow-up may open those labels. Default deployment remains unchanged.
