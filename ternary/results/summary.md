# Ternary Bonsai 1.7B fingerprint: results (2026-10-15)

Parent: Qwen/Qwen3-1.7B. Subject: prism-ml/Ternary-Bonsai-1.7B-unpacked. Pinned SHAs in `models/ternary/revisions.json`. Wikitext-2 only (calibration on train, evaluation on test), 4 threads.

## T0–T3 (weights only)
- Decode is exact: 99.98% of g128 groups hold a single magnitude. Signs match the parent on 99.1% of the largest 20% of weights, so Qwen3-1.7B is (close to) the parent.
- Norm tensors: 110 of 113 differ from the parent. The change doesn't look like an equivalence rescaling (log-ratio vs column-scale correlation 0.18–0.53).
- Round-to-nearest agrees with Prism's codes on 61% of weights (best-fit rounding 62%), nowhere near 92%. The 92% in the forum thread was for rotated Bonsai 2 27B.
- Zero share: Prism 39.9%, best-fit rounding 47.4%, absmean rounding 31.7%.
- Prism's scale is 1.99x absmean and 1.73x the least-squares scale for its own codes. Relative weight error: Prism 0.93, best-fit ternary 0.44.

## T4: held-out relative output error, median over 28 layers x 7 linears
| Prism | Prism codes, refit scale | RTN, Prism scale | RTN absmean | best-fit | GPTQ, Prism scale | GPTQ, best-fit scale |
|---:|---:|---:|---:|---:|---:|---:|
| 0.711 | 0.479 | 0.207 | 0.258 | 0.174 | 0.116 | 0.091 |

- Prism's error is below every post-training method in 0 of 196 tensors.
- 8-path search on layers 0/7/14/21/27 gains only ~1–2% over GPTQ, so stronger solvers barely help here.

## T5: wikitext-2 test perplexity (32 windows x 1024)
| parent | Prism | RTN | best-fit | GPTQ, Prism scale | GPTQ, best-fit scale |
|---:|---:|---:|---:|---:|---:|
| 16.8 | 18.3 | 297,400 | 63,403 | 255,056 | 76,559 |

## Reading
Every post-training method tried collapses the model (perplexity 63k–297k) while Prism keeps 92% of it (18.3 vs 16.8). Prism's weights have much higher per-layer reconstruction error than any of them, so they were not chosen to reproduce the parent's layers. That, with the changed norms, means end-to-end optimization of the weights, i.e. fine-tuning, for the unrotated April models.

## Not established
- Whether the parent is the instruct model or Qwen3-1.7B-Base. Base would inflate the error and norm differences (not the perplexity result). Check: download Base and rerun T0/T1 against it.
- Whether the rotated Bonsai 2 behaves the same.
- Wikitext only; no task benchmarks.
