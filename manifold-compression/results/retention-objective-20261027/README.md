# Retention objective pilot

Completed 27 October 2026. Reviewed jointly by Codex and Claude.

Continuing the projected parent with the revised curriculum improves held-out arithmetic and replay tasks, but does not produce a replacement that satisfies every generation safeguard. Adding replay-only forward KL partially preserves the stock model's token distribution without a demonstrated multiple-choice or generation advantage. The current parent remains the default.

Codex and Claude agreed the protocol, divided implementation, reviewed each other's changes, and independently graded the blinded science answers before adjudication. The [joint generation review](mc-retention-0081cbfff083/review/generation-consensus.json) records the agreed findings and binds the underlying evidence by hash. The [pair plan](../../PAIR_PLAN_20261026.md) covers this objective study and Claude's separate allocation pilot.

## Fixed endpoints and training

The source is Qwen3.5-2B. A0 is the existing residual-width-1520 projected parent, already healed for 24 updates at rank 16. A1 and A2 start from identical, function-preserving rank-64 expansions of that parent. S0 is stock; S1 and S2 start from identical fresh rank-64 stock adapters. A1/S1 receive cross-entropy training; A2/S2 add exact forward KL from the stock teacher on verified replay answer tokens only, with temperature and weight both 1.

Every trained arm receives the same 65,839 answer tokens over 16 fixed updates, with learning rate 3e-5 and two warmup updates. The token mixture is approximately 35% arithmetic, 30% verified replay, 30% verified code and 5% science. The endpoint is fixed; update 8 is diagnostic only. The [frozen manifest](frozen.json), [data manifest](data-manifest.json) and [protocol](protocol.json) specify inputs and checks.

A0 to A1 combines additional training, greater adapter capacity and a revised curriculum. It does not isolate which curriculum component helps or compare the old and new recipes at equal dose. A1 to A2 and S1 to S2 isolate the KL addition. The projected and stock families have different prior training histories.

## Held-out generation

The development suite contains 130 cases and the test suite 260. Test answers and score summaries were reviewed only after both science reviews and the frozen development gates were complete. Arithmetic and replay use held-out instances of known task families. Programs use separate source groups; science tests applications within taught concepts.

| Fixed endpoint | Math /128 | Replay tasks /112 | Programs /12 | Science /8 | Repetition flags | Completed /260 |
|---|---:|---:|---:|---:|---:|---:|
| A0 projected parent | 15 | 74 | 5 | 0 | 2 | 259 |
| A1 projected CE | 49 | 101 | 4 | 0 | 0 | 260 |
| A2 projected CE plus KL | 48 | 97 | 5 | 0 | 1 | 260 |
| S0 stock | 51 | 96 | 10 | 4 | 0 | 242 |
| S1 stock CE | 81 | 112 | 8 | 2 | 0 | 260 |
| S2 stock CE plus KL | 75 | 106 | 9 | 2 | 0 | 260 |

A1, A2 and S1 passed the development gates; S2 failed program and science safeguards. On the fixed test endpoints, A1 loses one net program answer and A2 introduces a new repetition flag. Both stock training arms lose program and science answers. These test comparisons are descriptive checks of fixed endpoints, not a checkpoint search.

The lost projected-program case asks for all matches of string regex patterns. A0 correctly uses `re.findall`; A1 returns `re.search` results, and A2 calls `.findall` on strings. Both trained versions fail the supplied case. A2's new repetition flag is a 60-token water-polarity explanation repeated before an answer line; it ends normally. The frozen repetition rule still fails, but this is not an endless generation.

All trained arms pass the separate historical-program veto: none loses a previously correct case, adds a repetition flag or loses completion. Absolute correctness is 1/10, 3/10 and 2/10 for A0/A1/A2, and 8/10 for each stock arm. These gains do not count toward the primary result. The [historical review](mc-retention-0081cbfff083/review/regression-review.json) records every case; passing this veto does not override the fresh-test failures.

The A0 to A1 arithmetic and replay gains remain positive under both item-group and task-family bootstrap sensitivity analyses. The six replay answers lost by S2 relative to S1 are concentrated in Python and sorting: the item-resampled interval excludes zero, while the family-resampled interval reaches zero. These are exploratory comparisons from one training seed, without correction for multiple comparisons. See the [paired test analysis](mc-retention-0081cbfff083/review/test-paired-analysis.json) for intervals and gained/lost case IDs.

Science requires every stated rubric proposition and no core factual contradictions. Both original reviewers independently scored all projected endpoints 0/8. Original stock ratings differed, and the final counts followed joint adjudication; [the rating audit](mc-retention-0081cbfff083/review/science-rating-sensitivity.json) preserves that uncertainty. A failed strict explanation rubric does not mean an answer contains no correct facts.

## Broad evaluation

The frozen 3,606-row suite separates text likelihood, agreement with stock and multiple-choice correctness by source. CUDA endpoints are compared only with CUDA controls; the allocation pilot uses separate CPU comparisons. The subsets and prompt formats are diagnostic experiments, not official benchmark leaderboard scores.

Accuracy below is in percent, using per-token choice normalization throughout. MMLU choices are single letters. The result files also retain sum and per-character normalization.

| Fixed endpoint | ARC Challenge | ARC Easy | HellaSwag | MMLU letters |
|---|---:|---:|---:|---:|
| A0 projected parent | 27.0 | 45.8 | 43.3 | 25.9 |
| A1 projected CE | 29.0 | 46.8 | 45.0 | 27.9 |
| A2 projected CE plus KL | 29.3 | 45.5 | 45.0 | 27.2 |
| S0 stock | 44.5 | 64.5 | 65.7 | 58.7 |
| S1 stock CE | 46.3 | 67.0 | 65.3 | 57.4 |
| S2 stock CE plus KL | 45.0 | 66.25 | 65.5 | 56.8 |

The projected parent already has large deficits relative to stock on ARC, HellaSwag and text likelihood. Its MMLU result also has severe letter-selection bias: A0 selects A on 1,034 of 1,138 items, with 25.9% accuracy versus an always-A baseline of 24.5%. That result alone cannot establish absent knowledge. The [choice-position audit](mc-retention-0081cbfff083/tf-metrics/choice-position-audit.json) records this for every completed endpoint and normalization.

A1 improves multiple-choice scores modestly over A0, while moving farther from stock's predictions on all five text sources. Reference cross-entropy rises on GSM8K, HumanEval and both chat sources, but falls slightly on WikiText. Higher loss on reference solutions is not a measured decline in free-generation accuracy.

A2 partially limits this drift relative to A1. Its KL and reference cross-entropy improve on all five text sources, with paired intervals below zero. For example, GSM8K KL falls from 1.2832 to 1.1045 nats/token and HumanEval KL from 0.3726 to 0.3071. Both remain worse than A0's 0.5750 and 0.1826. Multiple-choice differences between A1 and A2 are small and mixed. See [the complete projected contrast](mc-retention-0081cbfff083/tf-metrics/compare-A1-vs-A2.json).

Stock S1 has source-dependent effects: chat and WikiText reference likelihood improve, while GSM8K and HumanEval likelihood worsen. ARC scores improve modestly; MMLU and HellaSwag change little. Changed teacher fidelity therefore does not automatically mean degraded task quality.

S2 reduces teacher KL relative to S1 on every text source. It improves GSM8K and HumanEval reference likelihood, but slightly gives back some of S1's improvements on fresh chat and WikiText; held-out chat likelihood is essentially unchanged. ARC Challenge accuracy falls by 1.17 percentage points under sum normalization and 1.33 points per token, with paired intervals below zero. Other multiple-choice changes are small. Thus KL consistently constrains distribution drift, but this does not ensure better task performance. See [the stock objective contrast](mc-retention-0081cbfff083/tf-metrics/compare-S1-vs-S2.json).

## Integrity and storage

All four trained endpoints passed cache/full-sequence checks and independent reconstruction with exact fixture logits and exact eight-token greedy outputs. The CUDA batching check passed tight FP32 agreement and the preregistered BF16 gradient bounds. Stock's broad text self-comparison has exact zero KL, full top-1 agreement and identical cross-entropy on all 868 text rows. The [final integrity audit](mc-retention-0081cbfff083/review/final-integrity.json) confirms all 51 frozen input hashes remain unchanged and all six broad score sets contain the complete 3,606 rows.

The base allocation contains 1,399,093,089 stored scalars, approximately 74.35% of the text source. Each evaluated projected rank-64 adapter adds 60,991,488 scalars, for 1,460,084,577 while unmerged. Stock rank-64 adapters add 67,276,800. A merged export or standalone compressed package was not validated; the equal-budget allocation pilot compares unhealed bases.

## Next work and cost

Keep the current parent as default and pause further optimization of this narrow curriculum. The completed [unhealed allocation comparison](../allocation-pilot-20261027/README.md) found that a wider residual stream with pruned and refit MLPs improves all five text diagnostics at the same size budget, while leaving large gaps to stock. The next step is generation from stock and both unhealed constructions, after freezing stock-relative quality tolerances and adequate code and science coverage. Any distribution-recovery study needs a separate protocol; this pilot did not meet the original trigger for broader KD. Broad metrics can screen candidates, but generation safeguards remain necessary before promotion.

The one A100-SXM4-80GB pod ran at $1.79/hour and was deleted on 27 October 2026 at 01:51 UTC. Final synchronization and artifact checksums were verified. The conservative elapsed compute-plus-disk estimate is **$3.87**, within the shared **$20** cap; this is an estimate, not a provider invoice. The [lifecycle record](mc-retention-0081cbfff083/job-lifecycle.json) and [budget ledger](../../runpod_budget_20261026.json) record cleanup and cost. No additional paid job is planned.
