# Ternary embedder for the Strix NPU

A text-embedding model trained to fit the NPU: ternary weights, int8 activations, 2-3 layers.
Moved here from openflowlm-next on 2026-10-20. New models do not belong in OpenFlowLM.

## Status (2026-10-26)

- The **$10 campaign is finished training at an estimated $9.5843**; all nine
  owned RunPod allocations are deleted. The selected ternary model improves
  core retrieval from **42.13 to 45.23**. Final **Q8 reaches 47.48**, versus
  **46.21 BGE-micro** and **46.92 MiniLM**, and leads both on the individual
  development retrieval tasks, STSBenchmark, and the fixed 12-task mean
  (**72.02**). These point estimates are not full-MTEB superiority. The frozen
  ternary/float/Q4/Q8 family has **17,241,984 parameters** and three layers.
  Previously reserved FiQA/SCIDOCS judgments are now opened for a separate
  frozen-Q8 confirmation; no further selection uses them. See the
  [final artifacts and report](models/campaign-20261025/README.md) and
  [experiment interpretation](training-v2/delivery/EXPERIMENT_NOTES.md).


Frozen Q8's FiQA/SCIDOCS retrieval mean is **22.25**, versus **20.52** for BGE-micro and **29.17** for MiniLM. The broader goal of beating both references remains unmet.

The subsequent broader-data experiment completed for **$3.5242**, with all
four arms backed up and the pod deleted. It did not pass the wider quality
gate. The previous model family remains selected. See [results](training-v2/broad-data/RESULTS.md)
and [follow-up analysis](training-v2/broad-data/ANALYSIS.md). Both paid campaigns
are closed; another paid allocation requires separate approval.

The entries below record earlier experiments; their running-job status and
provisional model choices are historical.

- The user authorized a cumulative **$10 experimental campaign**. The matched
  additional-layer test is complete for about **$0.80**, with recovered artifacts
  and verified pod deletion. Appending a cloned third block lowered core
  retrieval by2.23 points for float and0.78 for ternary versus continued
  two-layer controls, and lowered wider quality. Keep two layers/15,467,520
  parameters for the next experiments. See [depth results](training-v2/depth3/RESULTS.md).
- Wider checks exposed forgetting: original versus data-first12-task means are
  71.40→70.67 float and70.16→69.49 ternary, despite better retrieval. Reference
  means are70.90 bge-micro and71.48 MiniLM; this is a fixed12-task subset, not
  full MTEB. Independent larger-batch/original-parent, human-NLI-supervision and filtered-negative
  comparisons are running, with off-pod recovery and independent budget
  watchdogs. See [baseline findings](training-v2/campaign/BASELINES.md),
  [larger batches](training-v2/batch1024/README.md),
  [human labels](training-v2/nli-aux/README.md),
  [hard negatives](training-v2/hardneg/README.md), and the
  [campaign budget ledger](training-v2/campaign/ledger.json).

- Current direction: broader, cleaner positive-pair data with direct contrastive
  supervision. No teacher targets in this campaign; architecture stays
  two-layer/384 initially. One million candidate pairs have been fetched and
  audited, with query/passage component splits and benchmark-text exclusion.
  A conservative 128-token cap-fit subset avoids truncating the supervision.
  Local float and ternary training smokes pass. The approved A5000 job completed in 40.81 minutes for about $0.19, with
  verified local checkpoint backups and pod deletion. External retrieval rose
  from 42.13 to 43.76 ternary, and 44.44 to 45.47 float; q8 reaches 45.54.
  The bge-micro/MiniLM target is not yet met. See
  [training results and artifacts](training-v2/data-first/RESULTS.md). See the [data-first plan](training-v2/data-first/README.md).

- Product plan: retain ternary and develop int4/int8 variants. Before data-first training,
  precision-control simulations scored 44.15 (W4A8) and 44.54 (W8A8) mean
  retrieval, versus 44.44 float and 42.13 original ternary. The new data-first
  scores are reported above.
  Embeddings/output projection stay float; native q4/q8 runtime and latency
  are not validated. All six earlier distillation arms failed to beat the original on the
  core retrieval tasks; the subsequent data-first run improved them. Further
  data/negative experiments continue after the failed cloned-layer comparison.
  See [analysis and precision plan](training-v2/round2/README.md#subsequent-local-analysis-and-precision-plan-2026-10-24).

- The approved Qwen-teacher pilot is complete (~$0.132 GPU compute plus
  temporary disk). Its validation-selected 8B-teacher candidate regressed
  external quality: 40.84 mean retrieval / 81.35 STSBenchmark versus the
  original model's 42.13 / 81.58. The original remains the default. Both
  pods are deleted. A separately approved balanced-data follow-up completed within its two-hour / $0.75 cap. Its frozen candidate regressed on both core retrieval tasks and was not promoted; see [round 2](training-v2/round2/README.md).
  See [pilot results](training-v2/PILOT-RESULTS.md).
- Direct CPU comparison is also complete: at eight threads with dynamic
  padding, our short-text latency is 1.67 ms versus Qwen's 28.13 ms (16.8x).
  On CPU our quality is 42.13 mean retrieval / 81.58 STSBenchmark, versus
  prompted Qwen's 51.62 / 91.20. Most of the speed advantage over Qwen
  persists without NPU acceleration; see the same comparison details below.
- Fresh local comparison with Qwen3-Embedding-0.6B is complete: actual NPU
  vectors score 41.95 mean retrieval / 81.55 STSBenchmark, versus prompted
  Qwen's 51.62 / 91.20 at a 128-token cap. Our NPU path takes 1.37 ms for
  the measured short sentence, versus 26.72 ms for dynamically padded Qwen
  on eight CPU threads. This is an implementation comparison, not a Qwen
  GPU/NPU speed claim. See [comparison details](comparisons/qwen06b/README.md).
- Linux port is running on hardware. Both sequence caps pass saved-reference
  checks; final-output-only host reads give a paired 1.6–2.0% forward gain.
  Fixed-text microbenchmarks reach 2580 emb/s (batch 4 / cap 32) and 765/s
  (batch 1 / cap 128), including tokenization. These are not full-corpus or
  matched comparisons to the earlier Windows measurements.
  See [Linux validation and reproduction](native-stage/LINUX.md).
- Plan: `embed-pipeline.md` (historical draft, followed by measured updates).
- Grid 1 trained and scored on a RunPod 3090 (2026-10-17, about $1.02). Results below.
- Grid 2 completed on 2026-10-22: 2 x 512 (bge-base layers 0 and 11, width cut)
  and 2 x 256 looped twice (ELECTRA layers 0 and 1). Both use Grid 1's winning
  final-vector-only recipe, 150M training tokens, seed 0, float first then ternary.
  Isolated RunPod A5000 `m3ruzkkbr88nf6`, $0.27/hr, about 53 minutes and
  $0.24 estimated compute cost (storage not included). The corpus matches Grid 1
  (4,252,720 texts, 2,337,599 pairs, 145,428,260 tokens).
  Runner: `tern-embed/grid2.sh`; local controller: `run_grid2.py`.
  All 54 artifact files are saved in `tern-embed/pulls/grid2`, SHA256-verified,
  and the pod is deleted. `grid2-summary.json` holds exact scores and per-task
  comparisons. Source snapshots and dependency versions are in the artifact bundle.
- A native ternary width probe is built and verified on the Strix NPU.
- Native full-output dense projections now pass 36 exact NPU checks using the
  trained 2a_384 weights and activations, including the FFN's six activation
  scale groups. A packed CPU oracle covers both layers through final vectors
  at sequence caps 8/32/128. This earlier projection gate is retained alongside
  the subsequent complete encoder.
- Both full encoder layers now run on 32 NPU cores in one dispatch, including
  quantization, attention, GELU and post-residual LayerNorms. CPU tokenization,
  input embeddings/norm, pooling and output projection are included in the
  measured text-to-vector time: **607.3 emb/s at batch 4 / cap 32**, **241.8/s
  at batch 1 / cap 128** for the final build. Bulk cap-128 encoding reaches
  326–333/s. Earlier timing passes were faster; raw observations and code
  identities are retained to show shared-machine variability.
  Actual NPU vectors score **41.9409** mean retrieval versus **42.1322** for
  the same checkpoint on CPU: only 0.1913 points of additional arithmetic loss.
  These are not a fresh comparison with the older MiniLM NPU benchmark.
  See [complete results and reproduction](native-stage/README.md).
- A matched 150M-token float-only control now isolates a retrieval cost of the
  current ternary training recipe: mean nDCG@10 is 44.44 float, 42.13 QAT and
  46.92 MiniLM on the same CPU evaluator. Roughly half the gap is already in
  the float student; roughly half comes from this quantized recipe. Int8
  activation quantization is not the main loss after QAT. See
  [controlled retrieval diagnosis](retrieval-diagnostics/README.md).
- Grid 2 does not clear the quality bar. Keep 2a_384 as the best balanced 2-layer
  candidate; neither added width nor looping helped in these recipes. The current quality goal is to beat both bge-micro-v2 and MiniLM-L6-v2
  while retaining a ternary variant and measuring the speed/quality tradeoff.
  Keep width 384 as the starting point; controlled depth experiments remain open.
- Reviewed `C:/code/xdna2-notes` on 2026-10-23. The hardware plan now explicitly
  budgets kernel padding, memtile DMA channels, dispatch overhead and numerical
  quality. Four-tile columns/four replicas are proposals, not validated layouts.
  See `embed-pipeline.md`, "XDNA evidence review". A subsequent native-width
  experiment below provides direct hardware evidence.

## Hardware width decision (2026-10-23)

**Keep 384 for the custom kernel.** Native 384 and 1536 input widths run without
padding to 512/2048, with exact integer output checks. In the same one-core
ternary primitive, 512 took 21% longer than 384, and 2048 took 28% longer than
1536. These are repeated arithmetic-probe timings, not embedding latencies.

The older generic GEMM's padding is a software constraint we can avoid. The
512 model also has 1.78x the dense layer work and packed weight bytes. Next is
a faster native-384 encoder using the now-validated fused stage.
See [native-width results and reproduction](npu-width-probe/README.md) and
[raw aggregate](npu-width-probe/width-result.json). Full-array routing and actual
NPU retrieval quality are now validated in `native-stage/`; replicas and the
~750k tok/s estimate remain unvalidated.

## Grid 1 results

Ternary weights, int8 activations, 12-task MTEB subset, 128 tokens.

| run | shape, recipe | STS | Retrieval | Classif. | Rerank | Pair |
|---|---|---:|---:|---:|---:|---:|
| 2a_256 | 2 x 256, final vector | 76.9 | 34.0 | 72.6 | 57.6 | 65.0 |
| 2a_384 | 2 x 384, final vector | 79.3 | 41.8 | 75.7 | 60.4 | 66.6 |
| r0 | 3 x 384 (bge-micro-v2) | 79.0 | 43.3 | 75.7 | 60.4 | 66.3 |
| Ternlight-base | 2 x 384 (reference) | 76.0 | 35.5 | 76.4 | 59.5 | 58.7 |
| all-MiniLM-L6-v2 | the bar (float) | 78.9 | 46.9 | 80.0 | 63.5 | 67.9 |

- The recipe beats Ternlight at the same shape. Width matters most in this grid.
- The later matched-budget control shows a meaningful retrieval cost for the
  ternary recipe; the earlier STS proxies did not establish that it was free.
- The extras (layer-by-layer targets, pairs) did not help at this budget.
- 2a_384 passes the bar on STS only. Retrieval is 5.1 below. Full table: `embed-pipeline.md`, step 3.

## Grid 2 results (2026-10-22)

Same 12-task subset, 128-token cap, 150M-token recipe and seed as Grid 1.

| run | shape | STS | Retrieval | Classif. | Rerank | Pair |
|---|---|---:|---:|---:|---:|---:|
| 2d_512 | 2 x 512, bge-base cut from 768 | 78.9 | 41.2 | 75.6 | 59.8 | 65.9 |
| 2f_256 | 2 x 256, shared layers applied twice | 76.4 | 33.2 | 72.2 | 56.9 | 64.7 |

- 2d_512 trails 2a_384 in every category. This compares the planned initialization
  recipes, not width alone: the source encoder and width reduction also differ.
- Looping trails plain 2a_256 in every category while requiring twice as many
  layer passes. It does not rescue the narrow model at this training budget.
- Both fail the STS-plus-mean-retrieval target against MiniLM. The apparent STS
  tie for 2d_512 is rounding: 78.8620 versus MiniLM's 78.9365.
- The subsequent matched-budget 2a_384 float control is complete; see
  `retrieval-diagnostics/README.md`. Longer ternary budgets and an isolated
  retrieval-objective comparison remain untested. Depth remains a fallback.

## Related

- The fast bge-large result is a different thing: batch tiers took one text from 1.650 s to 0.090 s.
  See `C:\code\xdna2-notes\findings\F-0329-embedding-batch-tiers.md`.
- Historical `open_npue` measured MiniLM at 1305 emb/s, batch 64 on a different
  workload. Its source is archived; the current openflowlm-next checkout no
  longer includes the backend. Do not compare that figure directly with these timings.
- Qwen3-Embedding on the LLM engine is openflowlm-next work (`feat/qwen3-embedding`).

## Where the pieces are

- `embed-pipeline.md`: the plan.
- `tern-embed/`: training code (own git repo). `pulls/grid1` holds the Grid 1 results.
- `quality-floor/`: the MTEB harness (mteb 1.39.7).
- `native-stage/`: complete one-dispatch NPU encoder, offline text CLI, retrieval
  evaluation and retained earlier projection/hybrid controls.
- `retrieval-diagnostics/`: matched-budget float control, ranking margins and quantization ablations.
