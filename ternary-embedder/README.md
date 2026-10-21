# Ternary embedder for the Strix NPU

A text-embedding model trained to fit the NPU: ternary weights, int8 activations, 2-3 layers.
Moved here from openflowlm-next on 2026-10-20. New models do not belong in OpenFlowLM.

## Status (2026-10-20)

- Plan: `embed-pipeline.md` (draft from 2026-10-16, NPU engine not built).
- Grid 1 trained and scored on a RunPod 3090 (2026-10-17, about $1.02). Results below.
- No NPU kernel, no engine, no serving. Plan steps 1, 2 and 4 are not started.
- Decision owed: whether 2 x 384 (about 750k tok/s) is good enough, or 3 x 384 (about 500k tok/s).

## Grid 1 results

Ternary weights, int8 activations, 12-task MTEB subset, 128 tokens.

| run | shape, recipe | STS | Retrieval | Classif. | Rerank | Pair |
|---|---|---:|---:|---:|---:|---:|
| 2a_256 | 2 x 256, final vector | 76.9 | 34.0 | 72.6 | 57.6 | 65.0 |
| 2a_384 | 2 x 384, final vector | 79.3 | 41.8 | 75.7 | 60.4 | 66.6 |
| r0 | 3 x 384 (bge-micro-v2) | 79.0 | 43.3 | 75.7 | 60.4 | 66.3 |
| Ternlight-base | 2 x 384 (reference) | 76.0 | 35.5 | 76.4 | 59.5 | 58.7 |
| all-MiniLM-L6-v2 | the bar (float) | 78.9 | 46.9 | 80.0 | 63.5 | 67.9 |

- The recipe beats Ternlight at the same shape. Width matters most. Ternary costs little.
- The extras (layer-by-layer targets, pairs) did not help at this budget.
- 2a_384 passes the bar on STS only. Retrieval is 5.1 below. Full table: `embed-pipeline.md`, step 3.

## Related

- The fast bge-large result is a different thing: batch tiers took one text from 1.650 s to 0.090 s.
  See `C:\code\xdna2-notes\findings\F-0329-embedding-batch-tiers.md`.
- Today's NPU embedder `open_npue` (MiniLM 1305 emb/s) stays in openflowlm-next.
- Qwen3-Embedding on the LLM engine is openflowlm-next work (`feat/qwen3-embedding`).

## Where the pieces are

- `embed-pipeline.md`: the plan.
- `tern-embed/`: training code (own git repo). `pulls/grid1` holds the Grid 1 results.
- `quality-floor/`: the MTEB harness (mteb 1.39.7).
