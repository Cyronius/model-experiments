# Ternary embedder for the Strix NPU

A text-embedding model trained to fit the NPU: ternary weights, int8 activations, 2-3 layers.
Moved here from openflowlm-next on 2026-10-20. New models do not belong in OpenFlowLM.

## Status (2026-10-23)

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
  candidate; neither added width nor looping helped in these recipes. Retrieval
  quality can improve, but matching MiniLM is not an absolute acceptance gate:
  the objective is the measured speed/quality tradeoff. Keep width 384.
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
