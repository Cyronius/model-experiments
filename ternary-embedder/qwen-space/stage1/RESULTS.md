# Stage 1: hosted-space check and baselines (2026-11-12)

Scripts: `protocol_check.py` (steps 1-2), `baselines.py` (steps 3-5). Teacher vectors through
`shared.embed(plan="A")` and the shared cache; dev-task vectors are in
`training-v2/work/qwen-space/dev/` (`{task}_T_docs_{128,504}.npy`, `{task}_T_queries.npy`, float16 as
returned) and are also in the shared API cache, so Plan B gets them as cache hits.

## Protocol check (`training-v2/work/shared-qwen8b/protocol-check.json`)

200 texts per source, embedded through the hosted endpoint exactly as stored (no instruction),
compared with the source's own vectors.

| Source | median cos 4096 | p5 4096 | median cos 1024 | p5 1024 | Verdict |
|---|---:|---:|---:|---:|---|
| Jaswanth (20 files x 10 rows) | 0.9975 | 0.9966 | 0.9977 | 0.9966 | accepted |
| RyanMarcus (row groups 667, 1065) | 0.9977 | 0.9960 | 0.9979 | 0.9964 | accepted |
| robro612 NQ (0 of 200 needed the 512-token cut) | 0.9972 | 0.9959 | 0.9974 | 0.9960 | accepted |
| Round-2 local cache (positives < 127 Qwen tokens) | 0.9975 | 0.9960 | 0.9976 | 0.9962 | accepted |

All are at the ~0.997 ceiling seen in the endpoint check (bf16 locally vs the hosted precision).
11 of the 200 RyanMarcus texts carry a literal `query: ` prefix and still match (sent as stored).

**Query protocol** (200 round-2 queries vs `prompted-queries.npy`): the full string
`Instruct: {instruction}\nQuery:{query}` with `custom_instruction` empty gives median 0.9969
(p5 0.9958); the bare query with `custom_instruction` gives 0.9854 (p5 0.9722, min 0.873).
**Chosen: full string, `custom_instruction` empty** (recorded in `protocol-check.json`).

## Baselines (nDCG@10; core = mean of SciFact and NFCorpus)

Contracts: "128" = first 126 WordPiece content tokens, "504" = first 504 (student windowed, 4 x 126).
Cuts never fall inside a word (`shared.cut_to_contract`). Queries: teacher prompted with the task
instructions in `shared.DEV_PROMPTS`; for SS384 the student embeds the bare query.

| Arm | SciFact | NFCorpus | ArguAna | core |
|---|---:|---:|---:|---:|
| TT 128 @512 | 74.76 | 36.98 | 72.07 | 55.87 |
| TT 128 @1024 | 74.96 | 38.36 | 73.17 | 56.66 |
| TT 128 @2048 | 75.28 | 38.58 | 73.42 | 56.93 |
| TT 128 @4096 | 75.32 | 38.78 | 73.44 | 57.05 |
| TT 504 @512 | 76.84 | 39.80 | 75.79 | 58.32 |
| **TT 504 @1024** | 78.27 | 41.24 | 76.63 | **59.76** |
| TT 504 @2048 | 78.16 | 41.16 | 76.67 | 59.66 |
| TT 504 @4096 | 77.90 | 41.44 | 76.57 | 59.67 |
| BM25 128 | 58.96 | 29.86 | 52.39 | 44.41 |
| BM25 504 | 68.93 | 32.17 | 49.37 | 50.55 |
| SS384 Q8 128 (single window) | 62.43 | 31.96 | 48.84 | 47.19 |
| **SS384 Q8 504 (windowed)** | 65.29 | 31.26 | 45.69 | **48.28** |
| SS384 float 128 | 62.14 | 31.96 | 48.59 | 47.05 |
| SS384 float 504 (windowed) | 65.31 | 31.44 | 45.33 | 48.37 |
| SS384 Q8, historical pipeline (raw text, truncate at 128) | 62.97 | 31.98 | 48.83 | 47.48 |

Recall@10 for every cell is in `result.json`.

- **Sanity check (step 5): passed.** The historical Q8 pipeline reproduces 47.475 core (expected
  47.48). Cutting at a word boundary (the contract) costs 0.29 core at 128 tokens (47.19), because
  a partly kept final word is dropped.
- **Stored dimension:** at the 504 contract 1024 dims lose nothing against 4096 (59.76 vs 59.67
  core); 512 dims cost about 1.4 points. At 128 tokens, 1024 dims cost 0.4. 1024 stays the default.
- Longer documents help the teacher by ~3 points (57.05 -> 59.67 at 4096) and BM25 by ~6, but the
  windowed student by only ~1 (47.19 -> 48.28; NFCorpus and ArguAna drop).
- Dev documents are long: share over 126 content tokens 98% (SciFact), 97% (NFCorpus), 76% (ArguAna);
  share over 504: 9.3%, 9.8%, 2.2%.

## Spend

Plan A total after Stage 1: 1,124 calls, 7,079,888 tokens, **$0.0708** (from `shared.spend("A")`,
includes the Stage 0 test and the protocol check), plus $0.00008 for the planning-time endpoint check.

## `result.json` keys (read by Plan B; do not rename)

- `tasks.{SciFact,NFCorpus,ArguAna}`: `docs`, `queries`, `doc_content_tokens`,
  `TT_{128,504}_{512,1024,2048,4096}`, `BM25_{128,504}`, `SS384_{q8,float}_{128,504}`,
  `SS384_q8_raw_truncate128`; each arm is `{ndcg10, recall10}`.
- `core_mean_ndcg10.{same arm keys}`; `contracts`; `query_protocol`; `instructions`;
  `spend_plan_A_after`.
- Per-query rows for paired intervals: `training-v2/work/qwen-space/stage1-rows.json`, keyed
  `{task}|{arm}`.
