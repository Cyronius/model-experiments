# Training v2: data-first direct embedding supervision

The $10 campaign has finished training at an estimated **$9.5843**. All nine owned RunPod allocations are deleted. Final ternary retrieval is **45.23**, up from **42.13**; Q8 reaches **47.48**, versus **46.21** for BGE-micro and **46.92** for MiniLM. Q8 also exceeds both references on each development retrieval task, STSBenchmark, and the fixed 12-task mean (**72.02**). These are development point estimates, not full-MTEB superiority. The four frozen artifacts use three layers and **17,241,984 parameters**. FiQA/SCIDOCS judgments have now been opened for the separately frozen Q8 confirmation; no further tuning or selection uses them. See [final model files and report](../models/campaign-20261025/README.md), [campaign status](campaign/active-work.json), and [experiment interpretation](delivery/EXPERIMENT_NOTES.md). The original project default remains unchanged.

Frozen Q8's FiQA/SCIDOCS retrieval mean is **22.25**, versus **20.52** for BGE-micro and **29.17** for MiniLM. The broader goal of beating both references remains unmet.

The subsequent [broader-data experiment](broad-data/RESULTS.md) is complete at
**$3.5242**, with its pod deleted and checkpoints backed up. None of its four
training arms or fixed Q4/Q8 exports passed the wider development gate. Free
retention and human-similarity probes are documented in [the analysis](broad-data/ANALYSIS.md).
FiQA/SCIDOCS are observed development data for this campaign. Full Touche2020
and TREC-COVID qrels remain unopened; exposed Touche query 1 is excluded.
The prior model family remains selected. Both paid campaigns are closed.


Historical starting direction (superseded by the final campaign above): **more and better data, direct pair supervision**.
The approved data-first training job is complete: retrieval improved, targets
remain unmet, cost about $0.19, pod deleted. [Results and model files](data-first/RESULTS.md).
The user chose to stop teacher-target distillation for the next experiment.
The new [data-first plan and prepared corpus](data-first/README.md) documents one
million fetched pairs, tokenizer-aware benchmark exclusion, component-based
splits and local float/ternary training preflights. Keep two layers initially and
retain the ternary/int4/int8 product family. The historical recipe below records
prior work and is superseded for the next experiment.

The approved pilot is complete; its frozen candidate failed external quality
acceptance. See [pilot results and prepared next experiment](PILOT-RESULTS.md).
The original model remains the default.

Keep the deployed architecture unchanged: two layers, width 384, FFN 1536,
12 heads, mean pooling and 384-dimensional outputs; 15,467,520 parameters.
Beat bge-micro-v2 and MiniLM-L6-v2 first. Prompted Qwen 0.6B is a stretch target.
An 8B teacher can improve supervision without enlarging the deployed student.

## Matched baseline gates, measured 2026-10-24

Same CPU evaluator, 128-token cap, no query instructions for these baselines.
Full SciFact/NFCorpus corpora, 623 judged queries, and STSBenchmark test pairs.

| Model | SciFact nDCG@10 | NFCorpus nDCG@10 | Equal-task mean | STSBenchmark Spearman |
|---|---:|---:|---:|---:|
| Current student, CPU | 54.85 | 29.40 | 42.13 | 81.58 |
| bge-micro-v2 | 61.33 | 31.10 | 46.21 | 82.50 |
| MiniLM-L6-v2 | 62.39 | 31.45 | 46.92 | 82.03 |
| Qwen3-Embedding-0.6B, prompted | 67.73 | 35.50 | 51.62 | 91.20 |

`baselines.json` records exact fresh reference scores and checkpoint revisions.
The student and Qwen figures come from `../comparisons/qwen06b/`.

Predeclared first acceptance gate: exceed both reference models on each of
SciFact, NFCorpus and STSBenchmark. Then run the full existing 12-task subset
and report every category; do not describe a three-task win as overall MTEB
superiority. Finally revalidate actual NPU vectors and measure latency. A
statistically convincing improvement matters more than crossing a rounded
number; retain paired query intervals and repeat promising training seeds.

## Historical distillation recipe and controls

1. Start from the independent 150M-token **float-only control**, not the QAT
   checkpoint's master weights. That control scores 44.44 retrieval, versus
   42.13 for the old quantized recipe. The new run is continued training, not
   an equal-total-budget comparison with Grid 1.
2. Distill **query-to-candidate ranking distributions**, not the teacher's final
   vector coordinates. This supports both 1024-dimensional Qwen 0.6B and
   4096-dimensional Qwen 8B while preserving our 384-dimensional output.
3. Use public MS MARCO **train** triplets, including BM25 hard negatives.
   Cache instructed query vectors and unprompted passage vectors once per
   teacher. Positive and negative passages are deduplicated before softmax.
   No evaluation corpus, test queries, test relevance labels or STS test pairs
   enter training. Training-query groups are split before optimization, so
   repeated positives of a held-out query do not cross the train/validation boundary.
4. Preserve the original float student's geometry with a small frozen-vector
   anchor term. This is intended to limit similarity regressions; it is not
   evidence that regressions cannot happen.
5. Float continuation, then interpolate effective weights and activations
   toward the existing ternary/int8 arithmetic, then train fully quantized.
   The final checkpoint loads into the unchanged `Student` and NPU engine.
   No blended checkpoint is presented as a normal inference checkpoint.

The new pilot compares **0.6B versus 8B teachers with the same gradual QAT
schedule**, plus an **8B/abrupt-QAT control**. It uses the same 4096 public train
triplets, seed, batch size, 400 updates, and float starting checkpoint. The
purpose is to validate memory/throughput and seek a held-out training-split
signal, not to claim that 4096 triplets establish the best final quality.
Validation records teacher KL and positive-passage MRR/recall@1 among a fixed
batch's candidates. KL values from different teachers are not directly
comparable; use the common positive-passage metrics for that comparison.

The larger `run_grid.sh` is a separate, not-yet-approved 50k-triplet objective
ablation for the 0.6B teacher: in-batch only / hard negatives with abrupt QAT /
hard negatives with gradual QAT, 4000 updates each. All arms encode the same
positive and negative texts, even when negatives are omitted from the ranking
loss, so the negative-loss comparison keeps token exposure and compute matched.
Real and padded training tokens are logged.

Select schedules/checkpoints using only the MS MARCO training-split holdout.
Freeze a candidate before running the acceptance test set. Do not repeatedly
pick winners using SciFact, NFCorpus or STSBenchmark test labels. Keep a new,
unseen retrieval task for later generalization checks.

## Initial preflight, before the approved pilot

The following ran locally before approval:

- Fresh complete bge-micro-v2 and MiniLM reference evaluations.
- Qwen 0.6B target caching for 32 real MS MARCO training triplets.
- Eight-step end-to-end training smoke, including float/ramp/full-QAT phases,
  finite gradients and an exact standard-checkpoint reload.
- Four focused tests: ranking loss across different vector dimensions, exact
  float/ternary endpoints, checkpoint compatibility and phase scheduling.

The smoke is a pipeline check, **not an improved model**. Its tiny training
budget does not recover the abrupt quality change from quantization.
Those were the preflight limits. The subsequent approved A5000 pilot loaded
the 8B teacher, cached its targets, trained controls and evaluated a frozen
candidate; see PILOT-RESULTS.md for the complete outcome.

## Historical pilot commands

Use the project venv or a CUDA PyTorch environment; install `transformers`,
`datasets`, `huggingface_hub`, `numpy`, `scipy`, and `pyarrow`. Stage the recovered
`tern-embed/model.py`, float-control checkpoint and local tokenizer at their
existing relative paths. Teacher revisions are pinned in `cache_teacher.py`.
The 8B cache starts with batch 4, and teacher and student occupy the GPU in
separate processes. BF16 teacher weights alone are approximately 16 GB; actual
memory fit must be established by the pilot.

```bash
python -m unittest discover -s ternary-embedder/training-v2 -p test_rank_distill.py -v
bash ternary-embedder/training-v2/run_pilot.sh
```

`run_pilot.sh` does not rent a GPU or enforce a billing deadline. A separately
approved controller must enforce the pod's time/cost limit, download and verify
artifacts, and terminate the allocation. Do not invoke it on paid hardware
before that approval. The pilot proposal is one A5000 24 GB, up to $0.30/hour,
at most one hour, expected $0.30–$0.40 including temporary storage, total cap
$0.50. Abort rather than exceed the cap; unfinished work remains unfinished.
There is no authorization to run the larger grid or subsequent jobs.
