# Reserved confirmation tasks (shared by Plan A and Plan B)

Written 2026-11-12 by Plan A's agent, before any head, table or mixer was
fitted and before any leakage list was built (claim `reserved` in
`training-v2/work/shared-qwen8b/CLAIMS/`). Style follows
`training-v2/delivery/RESERVED_PROTOCOL.md`. Both plans confirm on these tasks,
each opening them once, for its own frozen final model only. Nothing here may
be used for fitting, selection, calibration or any tuning.

## Tasks

### 1. BEIR NQ (public 8B vectors)

- Source vectors: Hugging Face `robro612/nq_qwen3_embedding_8b`, commit
  `985ee9aebdbb6aa21035b2b93d35f6634d243456` (documents fp16, 512-token cut,
  no prompt; queries with `Instruct: Given a web search query, retrieve
  relevant passages that answer the query\nQuery:{query}`). Used only if the
  source passes the protocol check in `protocol-check.json` (median cosine at
  least 0.99 and 5th percentile at least 0.97). If it fails, teacher vectors
  for the subsample below are made through the hosted API instead (shared
  cache), and this is noted in each plan's report.
- Texts: BEIR `nq.zip` (md5 `d4d3d2e48787a744b6f6e691ff534307`), document =
  title and text joined by one space, as in the source card and the dev
  loader.
- Corpus subsample: all documents named in `qrels.test.tsv` (the judged
  documents, any grade), plus 100,000 others drawn uniformly without
  replacement from the remaining document ids with
  `numpy.random.default_rng(20261112).choice(sorted_remaining_ids, 100000, replace=False)`,
  where `sorted_remaining_ids` is the lexicographically sorted list of all
  document ids not judged. The selected id list (sorted, newline-joined, UTF-8,
  trailing newline) has its SHA-256 written into the section "Recorded hashes"
  below by whichever plan builds it first, before any model scores on it are
  computed. The id list is stored as
  `training-v2/work/shared-qwen8b/reserved/nq-subsample-ids.txt`.
- All 3,452 test queries.
- Document contract: 504 (the student sees at most 504 content tokens; the
  teacher vectors are the source's own, cut at 512 Qwen tokens). Documents
  longer than the 504 contract are re-embedded through the API under the
  contract, so both sides see the same text.
- Note: NQ training pairs were part of the backbone's earlier training. That
  affects `TS` and `SS384` alike, so their difference is still meaningful, but
  absolute student scores on NQ are flattered.

### 2. Touché-2020 (BEIR `webis-touche2020`, v2), through the API

Chosen over TREC-COVID and Climate-FEVER because its text (args.me debate
portals) is not in any training source (Jaswanth / RyanMarcus: amazon reviews,
MS MARCO, ELI5, S2ORC, SPECTER, sentence compression, NPR), and its documents
are mostly longer than 126 tokens, so it tests the 504 contract. TREC-COVID's
CORD-19 abstracts overlap S2ORC/SPECTER; Climate-FEVER's Wikipedia documents
are mostly short. No candidate has both 300 or more test queries and long
documents; Touché has 49 test queries, so its confidence intervals will be
wide. This is accepted and must be stated in reports.

- Texts: BEIR `webis-touche2020.zip` from the UKP BEIR mirror (record md5 at
  download), document = title and text joined by one space.
- Queries: all test queries except query id `1`, which was exposed earlier in
  this project (see `training-v2/README.md`). Instruction (Qwen3-Embedding's
  generic one, as for NQ): `Given a web search query, retrieve relevant
  passages that answer the query`, written into the input string.
- Relevance: grades above 0 are relevant (TREC convention of
  `retrieval-diagnostics/diagnose.py::evaluate`).
- Corpus subsample: all judged documents plus 100,000 others, drawn exactly as
  for NQ with `numpy.random.default_rng(20261112)`. Id list stored as
  `training-v2/work/shared-qwen8b/reserved/touche-subsample-ids.txt`, hash
  recorded below before any scores are computed.
- Teacher vectors through the shared API cache under the 504 contract (and
  the 128 contract if a plan reports it), so each text is paid for once.

Previously opened tasks that are therefore **not** reserved: SciFact,
NFCorpus, ArguAna (development), FiQA and SCIDOCS (opened 2026-10-25). TREC-COVID
qrels remain unopened and stay available for future use.

## What is reported

For each task: nDCG@10 and recall@10 for `TT`, `TS`, `SS384` and `BM25` at
stored dimension 1024 and 4096, the paired bootstrap interval of `TS - SS384`
(`retrieval-diagnostics/diagnose.py::paired`), and the mixed-index test of
Plan A Stage 2 (p = 1, 10, 50%, five seeds). Use `round2_metrics.evaluate`
(missing judged documents count as misses). Self-document exclusion and tie
handling are the existing ones.

## Activation

Before computing any score on these tasks, each plan writes an activation
record into its own folder (`qwen-space/stage5/activation.json` for Plan A):
the frozen model file hash, the head hash, the subsample id-list hashes
below, and the date. After activation no change to the model is allowed.

## Recorded hashes

(Filled in by the plan that builds the subsample, before any scores.)

- NQ subsample ids SHA-256: `727d1332b43a58b68e13c1c73a1ef96ca45542457666a963b711cc6e2485027c` (104201 documents, 3452 queries; built 2026-11-12 by Plan A)
- Touché subsample ids SHA-256: `3e466e760f246100a226001b7e8e573c5a5fcf878922dcfb6f5b0a8a17d0eaad` (102059 documents, 48 queries; built 2026-11-12 by Plan A)
- Touché zip md5: `46f650ba5a527fc69e0a6521c5a23563`
