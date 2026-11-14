# Stage 5: confirmation and release (2026-11-12)

## 1. Confirmation on the reserved tasks (opened once, frozen model only)

Protocol: `qwen-space/RESERVED.md`. Subsample id-list hashes were recorded there before any score;
activation record `activation.json` (21:30Z) freezes `model.tde` f6dc1bf0..., `libtde.so` caeb4ddf...,
head d684692a.... Script: `confirm.py` (`dibs run cpu -- .venv/bin/python qwen-space/stage5/confirm.py
/home/cyrus/models/doc-embed-3l-qwen8b`). TS uses the released C runtime. Teacher queries: generic
web-search instruction written into the input string. 504 contract.

| | NQ (judged + 100k, 3,452 q) | Touché-2020 (judged + 100k, 48 q) |
|---|---:|---:|
| TT @1024 | 92.67 | 49.33 |
| TT @4096 | 93.26 | 50.03 |
| **TS @1024** | **75.78** | **44.16** |
| TS @4096 | 77.99 | 44.56 |
| SS384 Q8 504 (windowed) | 82.36 | 36.69 |
| BM25 504 | 66.85 | 42.19 |
| TS - SS384 @1024, 95% paired interval | **-6.58 [-7.69, -5.48]** | **+7.47 [+3.02, +11.83]** |
| cos(provisional, refined) @1024 mean / p5 | 0.706 / 0.614 | 0.742 / 0.669 |

nDCG@10. Absolute NQ numbers are high because the corpus is a 104k subsample (judged documents
plus 100k random); compare arms with each other only. **NQ caveat (from RESERVED.md):** NQ training
pairs were part of the backbone's earlier training, which helps SS384 directly (student queries and
student documents both trained on NQ) and TS only through the documents; the large negative gap on
NQ is partly that. Touché has 48 queries; its interval is wide.

Mixed index (raw scores, 1024 dims, five seeds; recall ratio as in Stage 2):

| p | NQ nDCG / recall ratio / top-10 share | Touché nDCG / recall ratio / top-10 share |
|---|---|---|
| 0 | 92.67 / - / - | 49.33 / - / - |
| 1% | 92.43 / 0.97 / 6.4% | 49.22 / 0.47 / 1.0% |
| 10% | 90.15 / **0.93** / **30.5%** | 47.61 / **0.40** / **6.5%** |
| 50% | 81.93 / 0.91 / 68.7% | 44.07 / 0.55 / 29.3% |

The G2 picture changes with the domain: on NQ provisional documents keep their relevant hits
(0.93) but crowd unrelated top-10 slots (30.5% at p = 10%); on Touché they rarely crowd (6.5%) but
lose most relevant hits (0.40, on few queries). Neither task meets both G2 limits.

**Reading.** On a domain absent from both training sets (Touché), provisional vectors beat the
student's own space by 7.5 points and beat BM25. On NQ, which the backbone was trained on, they
trail the student's own space by 6.6. On the dev tasks they trail by 1.2. Provisional-to-refined
cosine is 0.71-0.79 everywhere: the head reaches the teacher's neighbourhood, not its exact point.

## 2-3. Release folder and drift canary

`/home/cyrus/models/doc-embed-3l-qwen8b` (placeholder name, owner to confirm), assembled by
`qwen-space/release/assemble.py`: `model.tde`, `tokenizer.json`, `libtde.so`, `tde/`, `src/` +
`build.sh`, `config.json`, `README.md`, `azure_probe.py`, `drift_canary.py`, `canary.json` +
`canary-vectors.npy` (50 fixed SciFact documents with their hosted vectors from 2026-11-12; the
script re-embeds them, about 15k tokens, and reports min/median cosine at 4096 and 1024).

## 4. Fresh environment

`fresh_env.py`: a new venv (Python 3.14.8; numpy 2.5.4 and tokenizers 0.23.3 with their
dependencies, no torch) imports the package from the release folder (`python -I`) and encodes the
1,000 mixed-length documents and all dev documents on both code paths; the comparison runs in the
repository venv. Result `fresh-env.json`: min cos vs reference 0.99981 on both paths, paths agree
0.9999998, dev TS core diff -0.03 on both, token ids identical to transformers on all dev contract
texts, `contract_text` identical to `shared.cut_to_contract`; `torch`/`transformers` never imported
and not importable. All Stage 4 checks pass.

## Spend

No hosted calls in Stage 5 beyond the Touché/NQ vectors listed in STATUS.md ($0.231, made
before activation). Plan A total: **$0.3018** of the $5 cap.

## `result.json` keys

`activation`, `tasks.{nq,touche}.{docs, queries, dropped_queries_without_positive, TT_1024, TT_4096,
TS_1024, TS_4096, SS384_q8_504, BM25_504, paired_TS_minus_SS384_1024, cos_prov_ref_1024,
mixed_index_1024.p{1,10,50}.{ndcg10,recall_ratio,top10_share}, mixed_index_variant}`,
`spend_plan_A_after`.
