# Fast CPU document embedder in Qwen space: implementation plan

Written 2026-11-12 (revision 4: the owner supplied the API key and the hosted
endpoint was checked with it) for an implementing agent (Claude Opus 5.5,
medium effort). Work from `/home/cyrus/code/model-experiments/ternary-embedder`.
Paths below are relative to that directory unless they start with `/`.

**This is Plan A.** A second plan, Plan B (`ngram-space/PLAN.md`, an n-gram
lookup student), trains toward the same hosted teacher and may run at the same
time under another agent. The two plans share downloads, the hosted-API cache,
the validation split, the leakage list and the confirmation tasks. The rules
for that shared material are in `shared-qwen8b/README.md` (**the contract**).
Read it before Stage 0. Where this plan and the contract disagree about a
shared file, follow the contract and note it in `STATUS.md`.

Keep `qwen-space/STATUS.md` current: after each stage, add the date, what ran,
exact commands, results, failures, API tokens and dollars spent, and the next
action. If your context is reset, that file plus this plan must be enough to
resume.

## 1. What we are building and why

A small document embedder that runs on CPU only and writes vectors into the same
space as hosted Qwen3-Embedding-8B. The intended system:

1. At ingest, the small model embeds each document on the server CPU. This
   **provisional vector** makes the document searchable at once.
2. Later, a batch job re-embeds documents through the hosted model. That
   **refined vector** replaces the provisional one in the same index. The batch
   may cover only the documents where refinement pays off (Stage 3).
3. Queries are embedded by the hosted model.

The question this plan answers: **can a tiny CPU model match Qwen's
representational space well enough for provisional documents?**

Decisions the owner has already made. Do not reopen them:

- Documents only. Anything about a small query encoder is a later project; do
  no query-encoder work here. Teacher query vectors are used only to evaluate.
- Inputs up to 512 tokens must be supported.
- CPU only. Do not use the NPU anywhere in this plan.
- The large model is **hosted**, on DeepInfra, or possibly through OpenRouter.
- Q4 weights get one 30-minute check (Stage 4) and nothing more.

**Teacher: Qwen3-Embedding-8B as served by DeepInfra.** Reasons: on DeepInfra
8B costs $0.01 per million tokens and 4B costs $0.02; 8B is the stronger model;
and public 8B text-vector data exists (section 2). The space to match is
whatever the hosted endpoint returns, so that endpoint is the reference for
every check. Do not download teacher weights or run a teacher locally.

Deployment target: an Azure App Service Premium v3 **P2V3** instance: 4 vCPUs,
16 GB RAM, 250 GB storage, Linux, shared with other Python processes. Its CPU
model and instruction sets are unknown and can differ between instances. vCPUs
are usually hyperthreads, so assume about two physical cores until measured.
Consequences, all binding:

- No `-march=native`. The existing `cpu-kernel/libtern_cpu.so` was built that
  way on a Zen 5 and will crash on other CPUs.
- Runtime CPU detection with two code paths: AVX-512 VNNI, and AVX2.
- No torch and no transformers at runtime. Allowed: numpy, `tokenizers`, and
  our own shared library.
- No OpenMP. The kernel is single-threaded; the caller runs a small pool of
  worker threads (default 2).
- Timings from this dev box (Ryzen AI Max+ 395) do not transfer to Azure. Never
  present them as server numbers.

## 2. What already exists (read these first)

| File | Why it matters |
|---|---|
| `README.md` | Project history and status. |
| `models/campaign-20261025/README.md` | The frozen model family and its scores. |
| `comparisons/qwen06b/README.md` | How Qwen was evaluated here; prompts; data loader. |
| `cpu-kernel/README.md` | Existing AVX-512 VNNI kernels and their time split. |
| `training-v2/round2/README.md`, `training-v2/PILOT-RESULTS.md` | Two failed Qwen-teacher distillation attempts and their lessons. |
| `training-v2/delivery/EXPERIMENT_NOTES.md` | What helped and what did not. |

### The student (verified 2026-11-12)

- **Backbone:** 3 layers, width 384, 12 heads, FFN 1536, mean pooling,
  17,241,984 parameters, BERT WordPiece vocabulary of 30,522, `max_pos` 128.
  Code: `tern-embed/model.py` (`Student`, `quant_act`). `Student.forward`
  returns `(states, atts, unit_vectors)`; the pooled 384-d state before the
  `out` projection is the masked mean of `states[-1]`.
- **Checkpoints:** `models/campaign-20261025/model-float.pt`, `model-q8.pt`,
  `model-q4.pt`, `model-ternary.pt`. Q8 and Q4 derive from the float checkpoint
  and load through `training-v2/export_pair_precision.py::load_reference`
  (see `Encoder(path, precision="q8")` in `training-v2/evaluate_round2_mteb.py`).
  All linears use int8 activations (`quant_act`).
- **Scores at a 128-token cap** (nDCG@10; "core" = mean of SciFact and NFCorpus):
  Q8 47.48, float 47.02, Q4 46.67, ternary 45.23, MiniLM-L6 46.92,
  Qwen-0.6B prompted 51.62.
- **CPU kernels:** `cpu-kernel/tern_cpu.cpp` has `vpdpbusd` linears for ternary
  codes only. Batch 1 on 2 cores: 0.67 ms at a 32-token cap, 2.34 ms at 128.
  About half of that time is torch attention, LayerNorm and GELU overhead.
  The 2-bit packed kernel is about 9% faster than the same codes stored one
  per byte. Q8 and Q4 have no packed kernel yet.
- **Evaluation helpers:** `comparisons/qwen06b/compare.py::data(name)` loads a
  BEIR task from `comparisons/qwen06b/data/` (scifact, nfcorpus, arguana, fiqa,
  scidocs are present). `training-v2/round2_metrics.py::evaluate(scores, data)`
  returns metrics and per-query rows; `retrieval-diagnostics/diagnose.py::paired`
  gives paired bootstrap intervals. Reuse these. Do not write a new metric.
- **Query template used so far** (`training-v2/cache_teacher.py`):
  `Instruct: {instruction}\nQuery:{query}`; documents carry no instruction.
- **Environments:** `.venv` is Python 3.12 with CPU torch 2.14.1 and
  transformers. A ROCm torch env exists at
  `/home/cyrus/code/model-experiments/target/manifold-rocm/env` if student
  feature extraction on CPU proves too slow (rule 2 below applies).

### The hosted teacher (public metadata, plus a live check on 2026-11-12)

- DeepInfra native endpoint:
  `POST https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Embedding-8B` with a
  bearer key. Request fields in the published schema: `inputs` (list of
  strings), `normalize`, `dimensions`, `custom_instruction`. Token limit
  32,768 per input. Price $0.01 per million input tokens. The served version
  is labelled `excl-v1`, not a Hugging Face revision, and the precision is not
  published.
- **Measured with the owner's key** (7 calls, 7,944 tokens, $0.00008; 24 short
  MS MARCO passages and 16 queries from the round-2 cache; script and numbers
  in `qwen-space/endpoint-check-20261112/`):
  - The response holds `embeddings`, `input_tokens`, `request_id` and
    `inference_status` (`status`, `runtime_ms`, `cost`, `tokens_input`). Use
    `inference_status.cost` and `input_tokens` for the spend count.
  - Vectors are 4096-d and already unit length, with `normalize` omitted and
    with `normalize: true` alike (explicit `false` was not tried; it does not
    matter here).
  - A repeated call returned bit-identical vectors.
  - `dimensions: 1024` returns the first 1024 values of the 4096-d vector,
    re-normalized (cosine 1.0000 against the local cut). Local prefix cuts are
    therefore exact.
  - **The hosted model matches the Hugging Face weights.** Against the local
    cache (revision `1d8ad4ca...`, bf16): passages, sent as bare strings,
    median cosine 0.9976 (minimum 0.9968) at 4096 and 0.9977 at the 1024
    prefix. About 0.997 is the ceiling to expect in the Stage 1 protocol
    check; a source well below it differs in protocol, not precision.
  - A bare input gets no hidden instruction: bare queries match the local
    plain-query cache at 0.9976 and the prompted cache at only 0.85.
  - Queries: the full string `Instruct: {instruction}\nQuery:{query}` matches
    the local prompted cache at 0.9971. The `custom_instruction` field gives a
    different vector (0.984 against both), so it builds some other template.
    **Write the instruction into the input string and leave
    `custom_instruction` empty.**
- Through OpenRouter (`qwen/qwen3-embedding-8b`) three providers serve this
  model: Nebius (bf16), DeepInfra (precision not stated) and SiliconFlow (fp8).
  Vectors from different providers are not guaranteed to match. If OpenRouter
  is used, pin the provider to DeepInfra and disable fallbacks.

### Free text-vector data (found 2026-11-12; none of it verified against the hosted endpoint yet)

| Source | What it is | Notes |
|---|---|---|
| Hugging Face `Jaswanth-0821/qwen3_embedding_8b_distill` | 1,275,000 texts with 4096-d Qwen3-Embedding-8B vectors, int8 codes plus `abs_max_val` per row; 6.4 GB in 104 parquet files; tagged Apache-2.0 | Dequantize as `codes * abs_max_val / 127`, then re-normalize. Its first rows are identical to the next source, so it looks like the first part of that set. **Primary training data.** |
| Hugging Face `RyanMarcus/qwen-8B-embedding-distill` | 12,468,795 text-vector pairs, 4096-d float32, unit norm; texts cut to 2,048 characters; random samples of amazon-reviews, msmarco-bm25, eli5, s2orc, specter, sentence-compression, npr | One 192 GB parquet file, more than the free disk. Read row groups over HTTP range requests if more data is ever needed. No licence stated. Some texts carry a literal `query: ` prefix. |
| Hugging Face `robro612/nq_qwen3_embedding_8b` | BEIR NQ: 2,681,468 document vectors (fp16, 4096-d, no prompt, 512-token cut), 3,452 query vectors with the generic web-search instruction, qrels and exact top-k lists at several prefix dimensions; revision `1d8ad4ca...` in bf16 | `documents.npy` is 20.5 GiB. A ready-made evaluation task. CC BY-SA 4.0. |
| Local `training-v2/work/round2/teacher-cache/` | `passages.npy` (50000, 2, 4096) float32: positive and negative passage of 50,000 triplets; `prompted-queries.npy` and `plain-queries.npy` (50000, 4096). Encoded in bf16 with a cap of 128 Qwen tokens; `manifest.json` names the sources | Short texts: half MS MARCO passages, the rest NLI sentences and Quora questions. The only source with triplets and teacher query vectors. |

If the owner later prefers the 4B model: `Sherlock-Comms/wikipedia-en-2026-07-01-qwen3-embed-4b`
holds 17.5 million Wikipedia passage vectors from Qwen3-Embedding-4B, already
cut to 1024 dimensions.

### Lessons from earlier work that apply here

- Fine-tuning the backbone on a narrow set of teacher targets made retrieval
  worse both times (42.13 fell to 38.8-41.3) although training-holdout scores
  rose. Start head-only with the backbone frozen.
- Batch-local metrics misled model selection. Select only on full-corpus
  retrieval.
- Unjudged hard negatives hurt. Do not add them.
- Width 256 lost about 8 points; do not try narrower models. Appending a cloned
  layer hurt.

## 3. Ground rules

1. **Spending.** The only paid service is the hosted embedding API. Expected
   total is under $1; the hard cap for this whole plan is **$5**. On
   2026-11-12 the owner supplied a DeepInfra key in reply to the question
   about this cap ("here, use this ... i will delete this key once we're done
   with it"); this plan treats that as approval to spend up to the cap on the
   hosted API and on nothing else. Already spent: $0.0001 (the check in
   section 2).
   - The key is in `/home/cyrus/.config/deepinfra/api_key` (mode 600, outside
     every repository). Load it only into the environment of the process that
     calls the endpoint:
     `export DEEPINFRA_API_KEY="$(cat /home/cyrus/.config/deepinfra/api_key)"`.
   - Never print it, and never copy it into a repository, a command line, a
     log, a result file, `STATUS.md`, the cache, the manifest or a message.
   - A 401 or 403 means the owner has deleted the key: stop hosted calls and
     say so.
   - When this plan is released or abandoned, tell the owner the key is no
     longer needed by Plan A. Do not delete the key file yourself while Plan B
     may still be using it.

   Count tokens and dollars per stage in `STATUS.md` and stop at the cap. The
   cap is for this plan only: count the calls this plan's agent pays for, not
   cache hits. Plan B keeps its own count and cap. No other paid compute
   (RunPod and the like) without separate approval.
2. **Shared machine.** Load the `dibs` skill before the first large download
   or timing run. Timings you will report run under `dibs run --measure ...`
   and record CLEAN or NOISY. If you use the GPU for student feature
   extraction, run it under `dibs run gpu ...` and start `memguard.start(12)`
   from `/home/cyrus/code/model-experiments/npu-kernel-coder/train/memguard.py`
   in that process (GPU memory here is system RAM).
3. **What may be sent to the API:** text from public datasets and benchmarks
   only. Nothing about the owner, this machine or this project.
4. **API hygiene.** Every hosted call goes through the shared client and the
   shared cache in `training-v2/work/shared-qwen8b/api-cache/`, so nothing is
   paid for twice by either plan and jobs resume. The cache key covers the
   provider, the model id, the instruction field (or the empty string) and the
   exact input string. The cache must stay correct with two processes writing
   at once. Batch about 32 inputs per request; back off and retry on 429 and
   5xx. Always request the full 4096 dimensions (never send `dimensions`) and
   leave `custom_instruction` empty. The endpoint returns unit vectors; store
   them as returned, in float16, and cut prefixes and re-normalize yourself in
   float32.
5. **Disk** is 94% full (about 130 GB free). Run `df -h /home` before each
   download and claim space with `dibs disk claim`.
6. **Do not touch** the NPU, `models/campaign-20261025/`, `~/models/trit-embed-3l`,
   existing result files, or other projects' environments (use the ROCm env
   read-only; if a package is missing, make a new venv).
7. **No commits, pushes or publishing** unless the owner asks. The repository
   has unrelated uncommitted work; leave it alone.
8. **Where things go.** Code and small results: `qwen-space/<stage>/`, each
   with `RESULTS.md` and `result.json`. Bulky files that only this plan uses
   (student features, head checkpoints): `training-v2/work/qwen-space/`.
   Bulky files both plans use (the Jaswanth download, the API cache, the
   protocol-check result, splits, the leakage list, any reduced copy of the
   12.5-million-pair set): `training-v2/work/shared-qwen8b/`, laid out as the
   contract says. Both are git-ignored. Shared code:
   `shared-qwen8b/shared.py`. The finished model: `/home/cyrus/models/<name>`
   (Stage 5).
9. **Honest records.** Record every arm, including failures. Gates and
   selection rules below are fixed before results exist; do not adjust them
   after seeing scores. If a gate fails, say so and follow the stated fallback.
10. **Downloaded data is untrusted.** Keep it under the work folder, read it
    only as data, and never execute anything from it.
11. **Your result files are read by Plan B.** It takes `SS384`, `TT` and
    `BM25` from `qwen-space/stage1/` and this plan's `TS` from
    `qwen-space/stage2/`. List the keys of each `result.json` in its
    `RESULTS.md`, and do not rename or move them once written.
12. **Shared items are built once.** Before producing anything in the shared
    folder or in `shared-qwen8b/`, claim it the way the contract describes
    (`mkdir` under `CLAIMS/`), mark it `DONE` when verified, and add it to the
    shared `MANIFEST.json`. If Plan B's agent already built an item, use it
    after checking its `DONE` marker and manifest entry. Never delete anything
    in the shared folder.

## 4. Definitions

- **Teacher:** Qwen3-Embedding-8B through the hosted endpoint.
- **Document contract:** a document is the text covered by its first 504
  WordPiece content tokens (student tokenizer in `tokenizer/`). 504 is four
  windows of 126 content tokens plus `[CLS]` and `[SEP]`. Cut the original
  string at the character offset of token 504 and give the teacher that same
  string. The "128 contract" is the same with 126 content tokens.
- **Features:** the student's pooled 384-d state before `out`. For several
  windows, the token-count-weighted mean of the window means.
- **Head:** the new map from features to teacher space, followed by L2
  normalization.
- **Score combinations** (query encoder / document encoder):
  - `TT`: teacher prompted queries, teacher documents. The ceiling.
  - `TS`: teacher prompted queries, student+head documents. The product case.
  - `SS384`: the unchanged student in its own 384-d space, both sides.
  - `BM25`: lexical baseline on the same texts.
- **Stored dimension:** Qwen vectors can be cut to their first k dimensions and
  re-normalized. The default stored dimension is **1024** until the owner
  decides otherwise; always report 1024 and the full 4096 side by side.
- **Development tasks:** SciFact and NFCorpus (core) and ArguAna (secondary).
  All were used for earlier selection in this project, so they are not a clean
  test. Stage 5 uses fresh tasks for confirmation.

## 5. Stages

### Stage 0: setup (short)

1. Create `qwen-space/` subfolders and `training-v2/work/qwen-space/`. Start
   `STATUS.md`. Confirm the spending approval and the key (rule 1).
2. Shared code, `shared-qwen8b/shared.py` (claim `code`; if it already exists
   and is marked done, use it and do not rewrite it). It holds only what both
   plans need, with the interface the contract lists: the cached hosted client
   (rule 4) with per-plan token and dollar counts, the validation-split test,
   the text normalizer used for leakage matching, the document-contract
   cutter, and the dev-task loaders.
3. Plan A code, `qwen-space/common.py`: a feature extractor (single window
   and windowed) for a `Student` or `load_reference` model, and thin wrappers
   over the existing `evaluate` and `paired`.
4. The Jaswanth dataset (6.4 GB; claim `jaswanth`), into
   `training-v2/work/shared-qwen8b/jaswanth/`. Confirm the dequantized vectors
   have 4096 dimensions and that its first rows match the first rows of the
   RyanMarcus set (fetch a few through the Hugging Face rows API).
5. The confirmation tasks, `qwen-space/RESERVED.md` (claim `reserved`; Stage 5
   step 1 says what goes in it). It must exist before the leakage list of
   Stage 2 is built, and both plans use the tasks it names.

### Stage 1: check the hosted space, then baselines

Goal: prove which free data lives in the hosted model's space, and produce the
reference numbers every later stage compares against.

1. **Protocol check.** For each free source in section 2, embed 200 of its
   texts through the hosted endpoint exactly as stored (no instruction) and
   compare with the source's vectors. Report the median and 5th-percentile
   cosine at 4096 and at the 1024 prefix.
   - Accept a source if the median is at least 0.99 and the 5th percentile at
     least 0.97.
   - Between 0.95 and 0.99 median: usable for a first fit only. The final head
     must then be refit on vectors made through the API (at least 100,000
     documents; about 30 million tokens, roughly $0.30).
   - Below 0.95: reject it and find out why (instruction, truncation, pooling)
     before using it for anything.
   - For the NQ set, cut inputs the way its card describes before comparing.
   - Write the result to `training-v2/work/shared-qwen8b/protocol-check.json`
     (claim `protocol-check`): per source the sample size, both cosines at
     both dimensions and the verdict; plus the date, the endpoint, the served
     version label and, after step 2, the query protocol chosen. Plan B reads
     this file instead of repeating the check.
2. **Query protocol.** Embed 200 of the cached round-2 queries two ways: as
   the full string `Instruct: {instruction}\nQuery:{query}` with no
   `custom_instruction`, and as the bare query with `custom_instruction` set
   to the instruction. Compare each with `prompted-queries.npy`. Use the
   variant with the higher median cosine for all teacher queries from here on
   and record it. If neither reaches 0.99, report it and use the first. The
   16-query check in section 2 already points to the full string (0.997
   against 0.984); this step confirms it on 200. Also add the local round-2
   cache to `protocol-check.json` as a source (texts under 128 Qwen tokens
   only), so its vectors can be used without paying for them again.
3. **Dev-task teacher vectors through the API** (about 6 million tokens, a few
   cents): documents of SciFact, NFCorpus and ArguAna under the 504 contract
   and under the 128 contract; their queries with the task instructions in
   `comparisons/qwen06b/compare.py::PROMPTS` (for ArguAna use
   `Given a claim, find documents that refute the claim`).
4. **Baselines**, nDCG@10 and recall@10 per task at both contracts:
   - `TT` at stored dimensions 512, 1024, 2048 and 4096;
   - `SS384` for `model-q8.pt` and `model-float.pt`: single window at the 128
     contract, windowed at the 504 contract;
   - `BM25` (the `bm25s` package, or `rank_bm25`, installed into `.venv`).
5. Sanity check: `SS384` for Q8 at the 128 contract should reproduce about
   47.48 core mean. Investigate any difference above 0.3 before continuing.

Output: `qwen-space/stage1/RESULTS.md` with the protocol-check table, the
baseline table and the spend so far.

### Stage 2: head-only Qwen-space test

Goal: learn whether the frozen student plus a new head can produce provisional
vectors that work with teacher queries, at up to 512 tokens. No backbone
training.

**Training data.** From the accepted free data (the Jaswanth set first):

- Keep texts that fit the 504 contract in full, so teacher and student saw the
  same text. Report how many are dropped and the length distribution kept.
- **Remove benchmark leakage.** Drop any training text that matches a document
  of the dev tasks or of the confirmation tasks in `qwen-space/RESERVED.md`:
  compare normalized text (lower case, collapsed whitespace) on the first 200
  characters in both directions, and titles exactly. The training sources
  include s2orc and specter, which overlap scientific benchmarks. This list is
  shared (claim `leakage-jaswanth`): save the SHA-256 of every removed text and
  the counts per benchmark under `training-v2/work/shared-qwen8b/leakage/`. If
  it already exists, use it.
- **Validation split:** the contract's rule, a fixed function of the text
  (about 0.8%, roughly 10,000 texts of the whole Jaswanth set). Save the ids
  under `training-v2/work/shared-qwen8b/splits/` if they are not there yet.
  Never train on a validation text.
- **Training:** start with a random 300,000 of the remaining texts (seed 0).
  Dev tasks are never used for fitting.
- Student features come from the Q8 model (`load_reference`).

**Two ways to read up to 504 tokens.** Run every arm below with both and keep
the better one by validation loss:

- *Windowed (default):* windows of 126 content tokens, each with its own
  `[CLS]` and `[SEP]`, no overlap; features are the token-weighted mean. Needs
  no backbone change.
- *Native 512 probe:* extend `pos` from 128 to 512 rows using rows 128-511 of
  the bge-micro-v2 position table (cached under `cache/hub`), no training.
  Expect it to be worse; it shows whether a native long path would need
  backbone training.

**Arms:**

1. **H1, ridge regression.** Closed-form linear map with bias from 384-d
   features to the 4096-d teacher vector. Choose the penalty on the validation
   split. Minutes of work; the baseline.
2. **H2, small MLP.** `384 -> 2048 -> 4096` with GELU plus a linear skip path.
   Loss: mean over prefixes {256, 512, 1024, 4096} of
   `1 - cosine(prefix(pred), prefix(target))`. AdamW on precomputed features, a
   few epochs.
3. **H2 with a margin term** (only if the round-2 cache passed the protocol
   check): add, on the cached triplets with the teacher's prompted query
   vector `q`, the squared error between `q.pred(pos) - q.pred(neg)` and
   `q.t(pos) - q.t(neg)`, weight 1.0.
4. **Float-feature control.** H1 on features from `model-float.pt`, to show
   what the quantized backbone costs.
5. **Data curve.** The best arm refit on 50,000, 100,000 and 300,000 texts.
   Only if the curve is still rising at 300,000, extend to the full set.

**Evaluation** on all three dev tasks at the 128 and the 504 contract, at
stored dimension 1024 and at 4096:

- `TS` next to `TT`, `SS384` and `BM25` from Stage 1, with paired intervals for
  `TS - SS384`.
- Mean and 5th-percentile cosine between provisional and refined vectors.
- **Mixed-index test.** Mark a random share p of documents provisional and the
  rest refined; rank all of them against teacher queries. Use p = 1%, 10%, 50%
  and five seeds each. Report: overall nDCG@10; recall@10 of relevant
  provisional documents compared with the same documents when refined; and the
  share of top-10 slots held by provisional documents.
- **Calibration.** Fit one affine correction to provisional scores
  (`a * score + b`) on the training-source validation split only, and report
  the mixed-index test with and without it.

**Gates** (fixed in advance; core mean; stored dimension 1024; 504 contract):

- **G1, alignment cost:** `TS >= SS384 - 2.0`, with `SS384` windowed.
- **G2, mixed index at p = 10%, calibration allowed:**
  (a) recall@10 of relevant provisional documents is at least 0.85 of what the
  same documents get when refined; (b) provisional documents hold at most 15%
  of top-10 slots.
- **G3, long inputs:** `TS` at the 504 contract is no more than 0.5 below `TS`
  at the 128 contract.

**Fallbacks:**

- A gate fails narrowly (G1 by 4 points or less, or G2 or G3): write it up,
  mark the result as at risk, and continue to Stages 3 and 4.
- `TS < SS384 - 6`: stop. Report that a shared space is not reachable
  head-only, and write the Stage 6 proposal.
- `BM25` beats `TS` by more than 3 points: not a failure, but say plainly in
  the report that a lexical or hybrid provisional tier is competitive.

Output: `qwen-space/stage2/RESULTS.md` and the selected head checkpoint.
**Checkpoint: report these results to the owner**, then continue without
waiting.

### Stage 3: which documents to refine

Goal: a policy for spending the hosted model where it helps, using only vectors
already computed.

Simulate on the dev tasks at the 504 contract: refine a fraction f of documents
(0, 5, 10, 25, 50, 100%), chosen by each policy, and measure mixed-index
nDCG@10.

Policies:

- random (baseline);
- lowest head confidence first: the norm of the head output before
  normalization;
- cheap text signals: token count, share of `[UNK]` and of words split into
  three or more pieces;
- query-driven: documents that entered the provisional top-20 for past
  queries. Use half the queries as history and score on the other half;
- oracle: largest `1 - cosine(provisional, refined)`. An upper bound only.

Report the curves, the area under each, and the fraction at which the best
practical policy reaches 95% of the fully refined score. Recommend one policy
and name the signal the runtime must return for it.

### Stage 4: CPU runtime

Goal: a torch-free library that turns text into provisional vectors on any
x86-64 Linux server with AVX2, faster where AVX-512 VNNI exists.

Design, in `qwen-space/runtime/`:

1. **One C++ shared library with a C ABI**, single-threaded, no OpenMP, no
   dependency beyond libc and libm. Statically link libstdc++ and libgcc.
   Baseline flags `-O3 -march=x86-64-v3`; put the AVX-512 VNNI kernels in a
   separate translation unit compiled with the matching `-mavx512*` flags and
   pick the path once at load with `__builtin_cpu_supports`. An environment
   variable forces the AVX2 path for testing. Check with `objdump -T` that no
   symbol needs glibc newer than 2.28 and record the result.
2. **Whole forward pass in C++:** embedding lookup (int8 table with per-row
   scale, as in the `trit-embed-3l` release; the campaign Q8 file keeps
   embeddings in float32, so first confirm in Python that int8 embeddings cost
   at most 0.1 core nDCG, else store them as float16), embedding LayerNorm,
   three layers (quantized linears, float attention with masking, LayerNorm,
   tanh GELU), masked mean pooling, windowing for long inputs, the head, L2
   normalization. Attention, LayerNorm and GELU stay in float32. The activation
   quantizer must match `quant_act` (power-of-two scale per token per 256
   channels); reuse `quant_rows` from `cpu-kernel/tern_cpu.cpp`.
3. **Linear kernels.** Start from the v2 layout in `tern_cpu.cpp` (outputs
   across SIMD lanes, weights as the unsigned operand, signed int8 activations,
   offset removed with the precomputed activation sum).
   - Use **one scale per output row** so no rescaling happens inside the inner
     loop.
   - Use weight codes in **[-64, 63]** stored with offset 64. Then the AVX2
     sequence `vpmaddubsw` + `vpmaddwd` cannot saturate (127 x 128 x 2 fits in
     int16) and both paths compute the same integers.
   - First confirm in Python that this 7-bit, per-row format costs at most 0.3
     core nDCG against `model-q8.pt` (`SS384`, 128 contract). If it costs more,
     fall back to 8-bit codes with scale groups of 64 on the VNNI path and a
     two-pass AVX2 path, and note the extra cost.
4. **Python package** (`ctypes` + numpy + `tokenizers`): load
   `tokenizer/tokenizer.json` directly, `encode(texts) -> float32 array`, plus
   the confidence signal chosen in Stage 3. Calls release the GIL; the package
   offers a worker pool with a configurable size (default 2). Assert that it
   never imports torch or transformers.
5. **Model file:** one self-describing file (safetensors or a small custom
   format with a JSON header) holding embeddings, quantized linears with
   scales, norms, biases and the head. Record the source checkpoint hash.

Checks (gates for this stage):

- Token ids identical to the transformers tokenizer on every dev text.
- On 1,000 documents of mixed length, minimum cosine to the Python reference
  (same quantized format) is at least 0.999 on **each** code path.
- Dev `TS` within 0.15 nDCG of the reference on each code path.
- The two code paths agree with each other to minimum cosine 0.9999.

Measurements, under `dibs run --measure`, tokenization included, one thread
pinned to one core, for 16, 64, 128, 256 and 504-token documents, on both code
paths: median and p95 latency, documents per second, and the same with 2 and 4
worker threads; peak RSS and load time. Engineering targets on this dev box for
a 504-token document on one thread: at most 12 ms on the AVX-512 VNNI path and
at most 40 ms on the AVX2 path. These come from an operation-count estimate;
report what you measure.

**Q4 check: 30 minutes, hard stop.** Do not build a Q4 model or a full Q4
kernel. Once the linear kernel works, time one linear (the 384 -> 1536 FFN
matrix over 504 token rows, synthetic input) twice: weights one per byte, and
the same weights packed two per byte and unpacked in the loop, the way the
existing 2-bit path unpacks. Do this on both code paths if time allows, AVX2
first. If packing is not at least 15% faster, or the 30 minutes run out, Q4 is
dropped: record the two numbers and move on. Earlier evidence points to a drop
(2-bit packing gained about 9%). Only a win of 15% or more justifies asking
the owner whether to pursue Q4 further.

**Azure probe.** Write `qwen-space/runtime/azure_probe.py`, a single script the
owner runs on the App Service instance. It prints the CPU model, core and
thread counts, the relevant CPU flags, glibc version and available memory; then,
if the package is installed, runs a 20-second benchmark at 64 and 504 tokens
with 1, 2 and 4 threads and prints one JSON block. Give the owner the exact
commands. Server numbers come only from this script.

### Stage 5: confirmation and release

1. **Confirmation on fresh tasks.** `qwen-space/RESERVED.md` is written in
   Stage 0, before anything is fitted, in the style of
   `training-v2/delivery/RESERVED_PROTOCOL.md`. It is shared: Plan B confirms
   on the same tasks, so the two students can be compared. It names two tasks
   this project has never evaluated on:
   - BEIR NQ with the public 8B vectors (`robro612/nq_qwen3_embedding_8b`), if
     that source passes the protocol check. Subsample the corpus to the judged
     documents plus 100,000 random ones, fixed by a seed and a hash written
     down in advance. Its queries use the generic web-search instruction.
     Note in the file that NQ training pairs were part of the backbone's
     earlier training; that affects `TS` and `SS384` alike.
   - One more retrieval task whose text is not in the training sources,
     encoded through the API (subsample the same way if large). Prefer a task
     with at least 300 test queries and documents mostly longer than 126
     tokens. Candidates to weigh: TREC-COVID, Touché-2020, Climate-FEVER.
   Open them once, for this plan's frozen final model only, and report `TT`,
   `TS`, `SS384`, `BM25` and the mixed-index test. Teacher vectors for these
   tasks go through the shared cache, so they are paid for once.
2. **Release folder** `/home/cyrus/models/<name>` (placeholder name
   `doc-embed-3l-qwen8b`; ask the owner before settling it): model file,
   tokenizer, Python package source and built library, `config.json` (teacher,
   provider, date checked, stored dimension, document contract, weight format,
   query protocol), and a README with measured quality, the mixed-index
   results, dev-box timings marked as such, and the Azure probe instructions.
3. **Drift canary.** Store 50 fixed public texts with their hosted-teacher
   vectors in the release folder, plus a script that re-embeds them through
   the API and reports the cosine. The hosted model can change without notice;
   this lets the owner detect it.
4. Run the package from a fresh virtual environment containing only numpy and
   `tokenizers`, from the release folder, and repeat the Stage 4 checks.
5. **Clean up scratch:** list everything under `training-v2/work/qwen-space/`
   with sizes. Delete student features and intermediate checkpoints once the
   released model passes its checks. Keep the small result files. Do **not**
   delete anything under `training-v2/work/shared-qwen8b/`: Plan B trains on
   the same files. Record in `STATUS.md` that Plan A is released (or
   abandoned); the contract removes the shared folder only when both plans
   have said so, or when the owner says so.

### Stage 6 (proposal only): backbone fine-tuning

Do not run this. Write it only if a Stage 2 gate fails as described. The
proposal states: which gate failed and by how much; the training set (the
public 12.5-million-pair set makes teacher vectors free, so the cost is
training time only; use the reduced copy in the shared folder if Plan B has
built it, at whatever dimension it was stored); the loss (the Stage 2 losses plus a term that keeps the 384-d
output close to the frozen model); selection on full-corpus retrieval over
independent domains; training time; whether it needs paid GPU time and the
estimated cost. Then stop and wait for approval.

## 6. Out of scope

- Any query encoder, and any measurement of query cost.
- Index or database integration, and the batch re-embedding service itself.
- Running a teacher model locally.
- Any NPU build of this model.
- Multilingual or code-specific support. The student is English-only with a
  BERT vocabulary; say so in the release README.
- Changing the backbone architecture.
- Q4 beyond the 30-minute check.

## 7. Open questions for the owner

Carry these in `STATUS.md` and raise them at the Stage 2 checkpoint. Defaults
let work continue meanwhile.

1. **API key and the $5 cap.** Settled 2026-11-12 (rule 1): the owner supplied
   the key and a check succeeded. The owner gave the same key to Plan B with
   its own $5 cap, so the total exposure is $10; expected spend across both is
   under $2.
2. **Stored dimension and vector store.** Default 1024; Stage 1 shows what
   each dimension costs.
3. **Direct DeepInfra or OpenRouter.** Default direct DeepInfra. Either way
   the provider must stay DeepInfra, or the space may shift.
4. **Azure CPU details** from `azure_probe.py`. Until then both code paths are
   first-class.
5. **Licences of the public data.** The 12-million-pair set states none, and
   both distillation sets derive from datasets with their own terms. Training
   on them is the owner's call.
