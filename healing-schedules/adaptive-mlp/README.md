# Qwen3.5-2B adaptive MLP pruning

**Closed, November 25, 2026.** At the same 80.2M-parameter reduction (4.26% of the text model), ranking once and healing once matched iterative pruning on held-out WikiText with substantially fewer candidate evaluations. We are ending this line of experiments before scaling it to Qwen3.8 Flash Next.

## What survived pruning

Each round ranked candidate channel cuts by final-network cross-entropy. We physically removed matching gate, up, and down channels from the chosen MLP, then fitted a ridge correction to its down projection on 3,072 training tokens. The initial masked and physically resized models agreed within 0.00004 maximum absolute logit error on the pilot verification batch.

| Accepted cuts | Parameters removed | Validation perplexity increase | Test perplexity increase |
|---:|---:|---:|---:|
| 1 | 9.4M (0.50%) | 0.11% | 0.46% |
| 2 | 18.9M (1.00%) | 0.89% | 1.19% |
| 5 | 47.2M (2.51%) | 2.20% | 3.62% |
| 9 | 80.2M (4.26%) | 4.80% | 6.01% |

The 5% stopping budget used 64 validation passages, separate from the 16 ranking passages. Test results came from 128 passages evaluated after checkpoint selection. Layer 9 was selected three times and finished at width 1,536, down from 6,144. The last accepted cut removed 768 channels. The search stopped after the top-ranked 384-channel cut also exceeded the budget. Healing increased validation loss relative to the unhealed cut in five of the eight accepted rounds following the pilot.

## Equal-size comparison

| Method | Test perplexity increase vs. original |
|---|---:|
| Rank once, choose cuts independently, heal once | 5.94% |
| Rerank and heal between cuts | 6.01% |
| Remove the iterative run’s exact channels at once, heal once | 6.09% |

The independent one-shot run ranked 24 layer candidates. The iterative continuation scored 258 candidates across 11 attempts in addition to the initial pilot. Both used nine ridge fits on their accepted paths and the iterative search fitted three rejected cuts. These counts describe search work. Wall-clock speed and deployment throughput were not benchmarked.

Across the same test passages, independent one-shot perplexity was 0.07% lower than iterative perplexity (paired 95% bootstrap interval for one-shot relative to iterative, −0.62% to +0.48%). Holding the channel identities fixed gave interleaved healing a 0.07% relative advantage over one final healing stage. This effect was too small to give the complete iterative method a practical advantage here.

The evidence covers short WikiText passages, one dense 2B checkpoint, and local reconstruction. Downstream task accuracy and Flash Next expert pruning remain untested.

## Experiment records

- [Pilot and surgery verification](data/pilot.json)
- [Iterative rankings, accepted and rejected cuts, and per-passage losses](data/iterative.json)
- [One-shot controls and paired intervals](data/oneshot.json)
- [Pilot script](../followups/adaptive_mlp_pilot.py), [iterative search](../followups/iterate_mlp.py), and [one-shot comparison](../followups/oneshot_mlp.py)

Run the scripts in that order with the cached Qwen3.5-2B checkpoint and WikiText-2 raw splits available at the paths in the pilot script. The scripts write large weight edits under `target/adaptive-mlp-2b/`.
