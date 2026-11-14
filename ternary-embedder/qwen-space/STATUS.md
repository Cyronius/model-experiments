# Plan A status (frozen 3-layer student + head into hosted Qwen3-Embedding-8B space)

Plan: `qwen-space/PLAN.md` revision 4. Shared contract: `shared-qwen8b/README.md`.

## Spend (hosted API, Plan A cap $5)

Counts come from `shared.spend("A")` (uncached calls only; the `spend` table
in `training-v2/work/shared-qwen8b/api-cache/cache.sqlite`).

| Date | Stage | What | Tokens | Dollars |
|---|---|---|---|---|
| 2026-11-12 | planning | endpoint check (7 calls) | 7,944 | 0.00008 |
| 2026-11-12 | 0 | shared client test (1 call) | 255 | 0.0000026 |
| 2026-11-12 | 1.1-1.2 | protocol check, 4 sources x 200 + 2 x 200 queries | 122,003 | 0.00122 |
| 2026-11-12 | 1.3 | dev-task docs (128 + 504 contracts) and prompted queries | 6,957,630 | 0.06958 |
| | | **total after Stage 1** (`shared.spend("A")` 7,079,888 tok + planning) | **7,087,832** | **$0.0709** |
| 2026-11-12 | 5 (early) | Touché subsample docs (504 contract) + 48 queries + 206 long NQ docs | 23,090,337 | 0.23090 |
| 2026-11-12 | 2 | (no hosted calls) | 0 | 0 |
| | | **total after Stage 2** (`shared.spend("A")` 30,170,225 tok + planning) | **30,178,169** | **$0.3018** |

## Log

### 2026-11-12 Stage 0 (done)

- Claimed `code`, `jaswanth`, `reserved` (10:27). Plan B later claimed
  `leakage-jaswanth` (10:33); Plan A uses it once `DONE`.
- `shared-qwen8b/shared.py`: `embed`, `spend`, `is_validation`, `leak_key`,
  `cut_to_contract`, `n_content_tokens`, `dev_task`, `query_input`. Cache is
  SQLite WAL (`api-cache/cache.sqlite`), key = sha256 of
  json([provider, model, instruction, text]); spend per plan in the same DB.
  Cap enforced before each request. 401/403 raises `KeyRevoked`.
  `cut_to_contract` moves a cut that would fall inside a word back to the
  word start, so the kept text re-tokenizes to <= N tokens (noted in docstring).
- `qwen-space/common.py`: student loading (q8/float/q4, native-512 probe),
  features (single / windowed / native), SS384 vectors, metric wrappers.
  `qwen-space/manifest.py`: locked MANIFEST.json writer.
- Jaswanth downloaded (dibs network, disk claim 8 GB): revision
  `ef5ade98...`, 104 files, **1,300,000 rows** (the plan said 1,275,000),
  4096-d int8 + abs_max; dequantized norms 0.9993-1.0011. First 2,000 rows
  have text identical to RyanMarcus rows 0-1999, cosine min 0.9996.
  `qwen-space/stage0/jaswanth-verify.json`.
- `qwen-space/RESERVED.md` written: BEIR NQ (robro612 vectors) and
  Touché-2020 (through the API), judged + 100,000 random docs, seed 20261112.
  Texts downloaded to `training-v2/work/shared-qwen8b/reserved/`
  (nq.zip md5 d4d3d2e4... as on the card; webis-touche2020.zip md5
  46f650ba5a527fc69e0a6521c5a23563).

### 2026-11-12 Stage 1 (done) — see `qwen-space/stage1/RESULTS.md`

- Steps 1-2 done: `qwen-space/stage1/protocol_check.py` ->
  `training-v2/work/shared-qwen8b/protocol-check.json`. All four sources
  accepted (median / p5 cosine at 4096): Jaswanth 0.9975 / 0.9966,
  RyanMarcus 0.9977 / 0.9960, NQ 0.9972 / 0.9959, round-2 cache 0.9975 /
  0.9960. Query protocol on 200 queries: full string 0.9969 vs
  custom_instruction 0.9854 -> full string, `custom_instruction` empty.
- Steps 3-5: `dibs run network -- .venv/bin/python qwen-space/stage1/baselines.py`
  (log `training-v2/work/qwen-space/stage1.log`). Core nDCG@10 at 504/1024:
  TT 59.76, SS384 Q8 windowed 48.28, BM25 50.55; at 128: TT 56.66, SS384 47.19,
  BM25 44.41. Sanity: historical Q8 pipeline 47.475 (expected 47.48) — pass.
- Stage 4 pre-checks run early (`qwen-space/runtime/format_check.py`): int8
  embeddings cost 0.077 core (<= 0.1, use int8); 7-bit per-row codes derived from
  the Q8 reconstruction cost -0.03 (<= 0.3, use them).

### 2026-11-12 Stage 2 (done) — see `qwen-space/stage2/RESULTS.md`

- Index of all 1.3M Jaswanth texts (`stage2/prep_index.py`); selection
  (`stage2/prep_data.py`, stats in `stage2/data-stats.json`): 117,168 dropped
  over 504 tokens, 266 rows (197 distinct texts) dropped for leakage using
  Plan B's `leakage/jaswanth-removed.tsv` (not yet marked DONE at selection;
  `stage2/run.py` re-filters against the final list before fitting), 1 empty;
  1,182,565 eligible, 9,202 validation, 300,000 training (seed 0). Kept
  length: median 33 content tokens, p95 344; only 22.5% exceed 126.

### Stage 2 rules fixed before any Stage 2 result (written 2026-11-12)

- Reading mode per arm (windowed vs native-512 probe): lower validation loss
  (mean over prefixes 256/512/1024/4096 of 1 - cosine) on the Jaswanth
  validation split.
- Ridge penalty: lowest validation loss over lambda in {1e-5..1e-1} x n.
- MLP: 6 epochs, AdamW lr 1e-3 one-cycle, batch 1024, wd 0.01, seed 0;
  H3 adds the margin term (weight 1.0) on round-2 triplets whose query is
  not in the validation split, 256 triplets per step.
- Selected arm: the highest core-mean TS (504 contract, stored dimension 1024)
  among H1-H3, each in its better reading mode. (Lesson: select on
  full-corpus retrieval.) Dev tasks are not clean; Stage 5 confirms.
- Data curve: selected arm at 50k / 100k / 300k (first n of the seed-0 sample).
- Calibration: a*s+b fit on the round-2 cache validation split (queries with
  is_validation true; each query against all validation passages), 1024 prefix.
- G2 uses the calibrated variant if it passes, else the raw one.

### 2026-11-12 Stage 2 results at 300k (before the full-set extension)

- Run: `dibs run cpu -- .venv/bin/python qwen-space/stage2/run.py 16`
  (log `training-v2/work/qwen-space/stage2.log`; snapshot `stage2/result-300k.json`).
- Core TS (504 contract, 1024 dims): H1 ridge 40.50 windowed / 41.66 native;
  **H2 MLP 45.04 windowed / 46.37 native**; H3 (margin) 44.49 / 45.99;
  H4 float-feature ridge 40.26 (Q8 ridge 40.50: the quantized backbone costs nothing).
  Every arm's native-512 probe had the lower validation loss, so native was
  chosen for all; selected arm H2 native.
- Data curve (H2 native): 50k 41.71, 100k 44.58, 300k 46.37 (val loss
  0.256 / 0.236 / 0.215): still rising -> extend to the full pool (plan arm 5).
- Correction: `run.py` first picked the calibrated G2 variant by a coding
  error although it fails; the fixed rule (calibrated only if it passes) gives
  the raw variant. Fixed in code and in result.json; G2 fails under both.
- Rule added before the full-set result: the full-set refit replaces the
  300k head only if its validation loss is lower; then the final evaluation,
  calibration, mixed-index test and gates are recomputed with it.

### 2026-11-12 Stage 2 final (full-set extension) and checkpoint

- `dibs run cpu -- .venv/bin/python qwen-space/stage2/prep_full.py 16` (873,363
  more texts, native features 73 min) then `.../full_fit.py 16`.
- Full set 1,173,363: val loss 0.201, core TS 504@1024 **47.12** (@4096 48.63).
  Replaces the 300k head (lower val loss).
- Gates: **G1 pass** (-1.15), **G2 fail** (raw: recall ratio 0.790 < 0.85,
  top-10 share 22.9% > 15%; calibrated variant 0.170 / 0.5%), **G3 pass**
  (+2.45). BM25 beats TS by 3.43 > 3: lexical/hybrid tier competitive.
- Fallback: G2 narrow fail -> marked **at risk**, continue to Stages 3-4.
- Checkpoint report given to the owner in the session on 2026-11-12 (see
  the session transcript); work continued without waiting.
- Stage 6 is not triggered (no gate failed by the stop rule; G1 passed).

### 2026-11-12 Stage 3 (done) — see `qwen-space/stage3/RESULTS.md`

- `dibs run cpu -- .venv/bin/python qwen-space/stage3/run.py`. No practical
  policy clearly beats random on core; query-driven best by AUC (53.48 vs
  random 53.28; +1.1 over all three tasks). Head confidence and length are worse
  than random. ~80% must be refined to reach 95% of the refined score.
  Recommendation: query-driven (search-log signal; no runtime signal needed).

### 2026-11-12 Stage 4 (done except timings) — see `qwen-space/runtime/RESULTS.md`

- `qwen-space/runtime/build.sh` -> `libtde.so` (GLIBC <= 2.16, only libc/libm).
  `pack.py` -> `training-v2/work/qwen-space/runtime/model.tde` (sha f6dc1bf0...).
- Gates (`check.py`): tokenizer 0/19,519 mismatches; min cos vs reference
  0.99981 on both paths; dev TS diff -0.03 core; paths agree 0.9999998. Pass.
- Q4 30-minute check: **dropped** — no clean `--measure` slot inside the box
  (another session's build reservation until ~18:06). No numbers.
- Timings: pending a clean measurement slot (`runtime/bench.py`).
- Note: a stale `--measure` wait of mine briefly blocked my own CPU job; released.

### 2026-11-12 Stage 5 (done; timings pending) — see `qwen-space/stage5/RESULTS.md`

- Release assembled at `/home/cyrus/models/doc-embed-3l-qwen8b` (placeholder
  name; owner to confirm) by `qwen-space/release/assemble.py`.
- Confirmation subsamples built and hashed in RESERVED.md before scoring
  (`stage5/build_reserved.py`); NQ vectors extracted (`fetch_nq_vectors.py`,
  full 22 GB file deleted); Touché + 206 long NQ docs embedded through the API.
- Activation written 21:30Z (`stage5/activation.json`), then `confirm.py`.
- Fresh venv (Python 3.14, numpy + tokenizers only): all Stage 4 gates
  repeated and passed (`stage5/fresh-env.json`); torch not importable.
- Confirmation (1024 dims, nDCG@10): NQ TT 92.67 / **TS 75.78** / SS384 82.36 /
  BM25 66.85, TS-SS384 -6.58 [-7.69, -5.48] (NQ was in backbone training);
  Touché TT 49.33 / **TS 44.16** / SS384 36.69 / BM25 42.19, TS-SS384 +7.47
  [+3.02, +11.83]. Mixed p=10%: NQ recall ratio 0.93, share 30.5%; Touché 0.40, 6.5%.
- Release README written (`/home/cyrus/models/doc-embed-3l-qwen8b/README.md`),
  timing section filled after the measured run.
- Cleanup: deleted student features, Jaswanth-derived texts/targets, extra heads,
  scratch model copies, the fresh-env outputs and venv (15 GB -> 1.3 GB). Kept:
  `stage2/head_selected.npz` (+ `head_H2_q8_native_full.npz`, same weights,
  different metadata), `stage2/data.npz`, dev and confirmation teacher vectors
  (also in the shared cache), `jaswanth-index.npz`, logs. Nothing under
  `training-v2/work/shared-qwen8b/` deleted.
- **Plan A is released** (placeholder name pending the owner's choice). Plan A
  no longer needs the DeepInfra key (except for the optional drift canary and a
  Stage 6 if approved); the key file is left in place for Plan B.

### Stage 6

- Proposal written (`qwen-space/stage6/PROPOSAL.md`) because G2 failed; not run;
  waits for the owner's approval.

## Open questions for the owner (plan section 7)

1. API key and $5 cap: settled 2026-11-12.
2. Stored dimension: default 1024 (Stage 1 table shows the cost of each).
3. Direct DeepInfra or OpenRouter: default direct DeepInfra.
4. Azure CPU details: pending `azure_probe.py`.
5. Licences of the public data: training on Jaswanth (Apache-2.0 tag,
   derived from datasets with their own terms) is the owner's call.
6. (new, Stage 2) The data curve still rises at 1.17M texts. More data means the
   12.5M RyanMarcus set (no stated licence; contract preconditions apply). Approve?
7. (new, Stage 2) G2 fails: provisional vectors crowd top-10 slots. Options:
   accept with a documented refine-first policy (Stage 3), or a calibration
   fit on long documents (needs teacher query vectors for a long-document
   training source), or a hybrid BM25 tier for provisional documents.
8. (new) Release name: placeholder `doc-embed-3l-qwen8b`.

- 17:20-19:10 the measured timing run (`runtime/bench.py`) waited behind other
  sessions' shared build reservations (one stale since 18:16 but still listed,
  more builds kept joining); cancelled at 19:10 so my pending measurement would
  not hold up other sessions. **Timings not measured**; README says so and gives
  the command.

## Next action

1. When the machine is quiet, run (then put the numbers in the release README
   "Speed" section and `runtime/RESULTS.md`, marked dev box, CLEAN only):
   `dibs run --measure --label "doc-embed timings" --eta 25m -- .venv/bin/python qwen-space/runtime/bench.py /home/cyrus/models/doc-embed-3l-qwen8b/model.tde /home/cyrus/models/doc-embed-3l-qwen8b/libtde.so qwen-space/runtime/bench.json`
2. Owner decisions: release name (placeholder `doc-embed-3l-qwen8b`), Stage 6
   proposal, more data (12.5M set licence), data licences, G2 handling.
3. Owner runs `azure_probe.py` on the App Service instance.
