# Balanced-data round 2 — completed, candidate rejected

Approval: one A5000, at most $0.30/hour, at most two hours, total cap $0.75.
Actual allocated GPU rate was $0.27/hour. The pod ran for 1.3263 hours,
with estimated GPU compute cost $0.3581 plus temporary disk (roughly $0.011).
All 65 recovered artifacts passed SHA256 verification; pod deletion was
independently verified by HTTP 404. No paid resources remain. `plan.json` records the protocol before
external evaluation. The prior rejected checkpoint remains unpromoted.

Data: 50,000 public training triplets: 25,000 MS MARCO, 12,500 NLI, 12,500
duplicate-question examples. NLI coverage was corrected from the combinatorial
triplet table to one entailment/contradiction pair per distinct premise. Final
NLI coverage is 12,500 distinct premises. Query-group splitting and removal of
held queries from every training text role leave 47,409 training and 2,501
validation rows. No external benchmark labels enter optimization or selection.

Teacher encoding uses length buckets, up to 4096 padded tokens per 8B batch,
no autoregressive KV cache, and automatic OOM backoff. Exact text-vector reuse
avoided re-encoding unchanged data during the coverage correction. Both 8B
formats' targets were losslessly factored into shared passages and separate
query arrays, recovered locally, and SHA256-verified. The 0.6B targets were
also recovered at full precision. These caches are gitignored under work/round2.

Six matched 4000-update arms keep the student architecture and initialization
unchanged: 8B prompted with preservation weights 0.1 and 1.0; 8B plain with
weight 1.0; 0.6B plain with weight 1.0; 8B prompted with weight 1.0 plus a 0.2
verified-positive loss; and that final recipe using the independent float
control as the preservation reference. The student remains quantized in every
arm. The extra controls use the same allocation and original budget deadline.

Selection uses only the mixed training holdout: each source must remain within
0.01 MRR of the original; among eligible models within 0.01 MRR of the best
eligible model, prefer the smallest RMS change in cosine relationships on a
fixed unlabeled holdout-text sample. The original is a null control. All rules
are recorded before external tests. ArguAna is reserved as a generalization
check; it is not used to select a model.

The first startup selected the wrong workload because SSH did not inherit the
container environment variable. It was stopped on the same pod and dispatch
was corrected to an explicit script argument. The budget deadline did not reset.

The predeclared rule selected `balanced-prompted-anchor1.0`, update 500,
with mixed-source holdout MRR 0.89970 (original 0.86384). The two gold-positive
controls reached about 0.9077 but had greater cosine-relation drift. The chosen
checkpoint was frozen before external scoring; no other arm was substituted
after seeing external results.

The candidate failed acceptance: SciFact nDCG@10 50.2310 versus original
54.8532, NFCorpus 27.4150 versus 29.4029. The two-task mean fell from 42.1280
to 38.8230. Actual NPU mean was 38.7481, confirming that hardware arithmetic
was not the primary cause. Fixture CPU/NPU cosine agreement exceeded 0.9998;
short text embedding latency was about 1.4 ms, with unchanged architecture.

Full matched reference results are in `quality.json`.
ArguAna's official corpus lacks five documents referenced in test judgments;
these are counted as unreturned misses, retaining all 1406 queries and the
original ideal ranking denominator. The same handling applies to every model.

The original checkpoint remains the default. These results show training-holdout
gains did not transfer; they do not establish an architecture limit or an
improvement over either target. No further paid experiment is authorized.

The additional 12-task MTEB run was not used: the frozen candidate already
failed both primary retrieval gates and the reserved generalization check.
`evaluate_round2_mteb.py` remains available as a reproducible optional harness.
This avoids presenting selective secondary wins as overall success.

`weight-changes.json` records a post-evaluation diagnostic: around 1–9% of
ternary codes changed per dense matrix after 500 updates. This is descriptive,
not proof of the regression's cause. Any next selection protocol should use
full-corpus training/development retrieval and an independent validation source;
batch-local reciprocal rank was insufficient in both completed pilots. ArguAna
has now been observed and must not be called unseen in a future experiment.

Matched CPU results (128-token cap; nDCG@10 and STS Spearman ×100):

| Model | SciFact | NFCorpus | Mean core | ArguAna | STSBenchmark |
|---|---:|---:|---:|---:|---:|
| candidate | 50.23 | 27.41 | 38.82 | 39.76 | 81.69 |
| original | 54.85 | 29.40 | 42.13 | 42.71 | 81.58 |
| bge_micro | 61.33 | 31.10 | 46.21 | 51.81 | 82.50 |
| minilm | 62.39 | 31.45 | 46.92 | 48.76 | 82.03 |

Paired query bootstrap 95% intervals for candidate minus original:
- SciFact: -4.62 points, CI -6.79 to -2.51.
- NFCorpus: -1.99 points, CI -3.16 to -0.81.
- ArguAna: -2.95 points, CI -4.03 to -1.91.

## Subsequent local analysis and precision plan (2026-10-24)

The user elected to plan a family of ternary, int4 and int8 variants. The
independently trained float checkpoint is the common quality reference; the
ternary QAT master weights are not a usable float checkpoint. No further paid
resources were allocated for this analysis.

Accuracy-only local simulations quantize the six dense matrices per layer
using symmetric signed 4/8-bit codes, groups of 64 weights and bf16 absmax scales,
with the existing int8 activation quantizer. Embeddings, norms, biases and output
projection stay float. These are dequantized CPU simulations, not packed-runtime
latency or whole-model storage measurements.

| Model | SciFact | NFCorpus | Mean core | STSBenchmark |
|---|---:|---:|---:|---:|
| Existing ternary | 54.85 | 29.40 | 42.13 | 81.58 |
| Int4 simulation from float | 57.38 | 30.92 | 44.15 | 82.02 |
| Int8 simulation from float | 57.63 | 31.45 | 44.54 | 82.36 |
| Independent float | 57.40 | 31.47 | 44.44 | 82.44 |

The int8/float difference is small; it is not evidence of a reliable improvement
from quantization. Int4 retains most of the float gain without any retraining.
`../analysis/summary.json` retains paired query intervals and full results.

All six round-2 best checkpoints were scored post hoc on the two existing
retrieval development tasks to diagnose the failure. None exceeded the original.
Means: 8B plain 39.34; 0.6B plain 40.99; 8B prompted anchor 0.1 38.83;
8B prompted anchor 1 38.82; plus positive supervision 41.27; plus positive
supervision and float anchor 41.06. The selected arm's final update 4000 reaches
41.37 (better than its selected update 500 at 38.82, still below original 42.13).
The positive-supervision arm's final update scores 41.18. These are exploratory
results after observing test scores, not a new independent acceptance round.

Re-ranking the 2501 held-out training-source queries against all 4934 unique held
documents gives MRR 0.7855 original, 0.8112 selected, 0.8249 positive-supervision,
and 0.8510 8B prompted teacher. MS MARCO MRR improves 0.7237 -> 0.7843 -> 0.8053,
while NLI falls 0.7585 -> 0.7448 -> 0.7527. This establishes real training-domain
learning with poor external transfer; tiny validation batches are not the sole
explanation. The batch-local per-source guard missed the larger-pool NLI loss.

Recommended sequence:

1. Use full-corpus development retrieval on independent domains for selection,
   plus general sentence-quality checks. Treat SciFact, NFCorpus and ArguAna as
   already-observed development tasks; reserve new final tests before training.
2. Restore broad-domain replay/distillation while adding retrieval supervision.
   The follow-up saw about 13M tokens over 50K triplets, versus the original
   broad 4.25M-text corpus. Compare 150M/300M student-token budgets on varied,
   deduplicated data rather than repeating the current narrow mixture longer.
   Use domain-appropriate teacher instructions, verified positives, filtered
   mined negatives and larger meaningful candidate sets.
3. Compare two and three layers under that identical recipe and source backbone.
   Three layers cost 17,241,984 parameters (11.5% above two layers), with about
   50% more transformer work. The older three-layer gain is confounded by extra
   layer/pair losses; width-512 and looping experiments likewise confound shape
   and initialization and cannot establish universal architecture limits.
   Four layers (19,016,448 parameters) remain a later option if depth pays off.
4. Improve the float quality reference first. Produce int8/int4 variants from it,
   adding quantization-aware adaptation only if needed. Keep a separate ternary
   branch with broad replay and a controlled quantization training schedule.
   A 2.3-point recipe gap is recoverable headroom to investigate, not a promise
   that longer QAT closes it. Consider trainable row scales while retaining the
   existing ternary {-1,0,+1} format before changing native kernels.
5. Validate native outputs, quality, single-text and batch latency, and packed
   sizes for each precision tier. Dense layers contain only 3,538,944 parameters;
   11,928,576 parameters sit outside these weight quantizers. Most are token
   embeddings, so table quantization needs its own quality test before promising
   whole-model size ratios.

Research supports these directions, without guaranteeing our outcome:
[Qwen3 Embedding](https://arxiv.org/abs/2506.05176) uses broad multi-stage data,
contrastive training and false-negative masking;
[TAS-Balanced](https://arxiv.org/abs/2104.06967) studies informative topic-aware
negative sampling; [BitNet Reloaded](https://arxiv.org/abs/2407.09527) studies
small-model ternary capacity and learning-rate/quantizer sensitivity. Its results
are not an embedding-specific recipe or proof that our student needs double width.

## Direction selected on 2026-10-25

The user chose more and better data with direct positive-pair supervision for
next training, rather than further teacher-target distillation. The
[data-first corpus and plan](../data-first/README.md) supersede the teacher-loss
recommendations for the next experiment. Start with two layers so the changed
supervision/data can be measured; retain ternary/int4/int8 product variants.
