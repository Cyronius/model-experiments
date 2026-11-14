# N-gram lookup document embedder in Qwen space: implementation plan (Plan B)

Written 2026-11-12 (revision 3, amended the same day: the owner supplied the
API key, and Plan A's endpoint check is recorded in section 3) for an
implementing agent (Claude Opus 5.5,
medium effort). Work from `/home/cyrus/code/model-experiments/ternary-embedder`.
Paths below are relative to that directory unless they start with `/`.

**This is Plan B.** Plan A may run at the same time under another agent.

**Read these two files first:**

1. `qwen-space/PLAN.md` ("Plan A", revision 3). This plan is its companion. It
   keeps Plan A's system design, hosted teacher, ground rules, definitions and
   evaluation, and replaces only the student model. Where this plan is silent,
   Plan A applies.
2. `shared-qwen8b/README.md` ("the shared contract"), adopted by both plans. It
   fixes where the data, API cache, splits, shared code and confirmation tasks
   live, and how to avoid building or paying for anything twice. Where this
   plan and the contract disagree about a shared item, the contract wins.

Keep `ngram-space/STATUS.md` current: after each stage, add the date, what ran,
exact commands, results, failures, API tokens and dollars spent, and the next
action. If your context is reset, that file plus the three documents above must
be enough to resume.

## 1. What we are building and why

The same product as Plan A section 1: a CPU-only document embedder whose
provisional vectors live in the space of hosted Qwen3-Embedding-8B, are
searched with that model's query vectors, and are later replaced by its own
document vectors.

Plan A's student is a 3-layer token-level transformer. Plan B's student has
**no token-level transformer**:

1. Each token position looks up learned vectors for its unigram, bigram and
   trigram and adds them. This is memory reads and additions only.
2. Positions are pooled into chunks of 16 or 32 tokens.
3. A small mixing layer runs over the chunk vectors (at most 32 of them), so
   it sees the whole document at chunk resolution, including chunk order.
4. The chunks are pooled, mapped to teacher space and L2-normalized.

The question this plan answers: **what is the smallest model of this kind that
gives useful provisional vectors, and how much faster than Plan A's student is
it on CPU?** Word order inside a chunk is carried by the n-gram tables;
position across chunks is carried by the mixer.

Decisions the owner has already made. Do not reopen them:

- Everything in Plan A's decision list: documents only; up to 512 tokens; CPU
  only; no NPU; the teacher is Qwen3-Embedding-8B as served by DeepInfra, never
  run locally.
- Representation width is **512 or 1024. Never below 512.**
- A **whole-chunk mixer**, not a transformer over the token stream, and not an
  iterative denoising ("diffusion") procedure.
- Table rows **start from seeds and are then calibrated** on teacher document
  vectors. Unseeded arms are run only as controls, to measure what seeding buys.
- BM25 is a retrieval-layer baseline and possible fallback tier. Nothing
  BM25-like goes into the model.
- The model file must stay small. Default budget: **2 GB on disk**, memory
  mapped. Report results at about 0.5, 1 and 2 GB.
- **The public 8B text-vector data is shared with Plan A**, including the
  12.5-million-pair set, under the shared contract.

## 2. The model (binding specification)

### 2.1 Input

- Tokenizer: the existing WordPiece tokenizer in `tokenizer/` (30,522 entries,
  lowercasing), the same one Plan A uses. No `[CLS]` or `[SEP]`.
- Document contract: Plan A's (section 4 there). A document is the text of its
  first 504 WordPiece content tokens; the teacher sees that same string. Using
  the same contract is what lets the two plans share teacher vectors.

### 2.2 Representation space

Let `D` = 4096 be the teacher dimension, `mu` the mean teacher document vector
over the training documents, and `V` (D x d) the top `d` principal directions
of the centered teacher document vectors. The model's `d`-dimensional
representation `z` is read out as `normalize(V z + mu + MLP(z))`.

- `d` is 512 or 1024. The two are nested: the 512 model's coordinates are the
  first 512 coordinates of the 1024 model at initialization.
- `V` and `mu` start from the PCA and are trainable. `MLP` is
  `d -> 2048 -> D` with GELU, its last layer zero-initialized. Rungs L0 and L1
  have no MLP.
- At runtime only the stored dimensions of the output are computed.

### 2.3 Tables

- **Unigram table:** one row per token id: a vector `u[t]` (width `d`) and a
  non-negative scalar weight `w[t]`.
- **Bigram and trigram tables:** exact tables for the selected n-grams only
  (Stage 1). One residual vector `r[g]` per n-gram, width `w_ng` = 512 (the
  first 512 coordinates of the representation; zero-padded when `d` = 1024). An
  n-gram not in the table contributes nothing. **No hashing of the tail.**
- An n-gram belongs to the position of its last token.
- All rows and weights are trainable. Keep the weights positive, for example
  through a softplus.

For a set of positions `S` (a chunk, or the whole document):

```
h(S) = ( sum_{i in S} w[t_i] * u[t_i]  +  sum_{g ending in S} r[g] ) / W(S)
W(S) = sum_{i in S} w[t_i]
```

### 2.4 Chunks, mixer, pooling

- Chunks are consecutive runs of `c` tokens (`c` = 16 or 32), so at most 32
  chunks. The last chunk may be short. Chunk vector `h_j = h(chunk j)`, chunk
  weight `W_j`.
- Mixer: `K` pre-norm transformer encoder layers over the chunk vectors, width
  512, 8 heads, FFN 2048, with a learned chunk-position embedding. When `d` =
  1024, project down to 512 before the mixer and back up after it. The mixer
  output is **added** to `h_j` through a zero-initialized projection, so a
  model with a fresh mixer computes exactly what it computed without one.
- Pooling: `z = sum_j a_j * h'_j` with `a_j` proportional to
  `W_j * exp(s(h'_j))`, where `h'_j` is the chunk vector after the mixer and
  `s` is a linear scorer initialized to zero. At initialization this is the
  weighted mean over the whole document.

### 2.5 The ladder

Each rung starts from the trained rung before it and computes the same function
at its first step.

| Rung | Adds | Per-document arithmetic at 504 tokens (estimate) |
|---|---|---|
| L0 | unigram table, weighted mean, linear readout | about 500 row additions |
| L1 | bigram and trigram residual tables | about 1,500 row additions |
| L2 | the MLP in the readout | plus about 4 million multiply-adds |
| L3 | chunking, chunk positions, learned pooling (`K` = 0) | negligible extra |
| L4 | one mixer layer (`K` = 1) | plus about 100 million (c=16) or 50 million (c=32) |
| L5 | two mixer layers (`K` = 2) | plus about 200 million or 100 million |

For scale, Plan A's 3 x 384 student at 504 tokens is roughly 3.3 billion
multiply-adds. These are operation counts, not timings.

## 3. What is shared with Plan A

**Shared items** (built once by whichever agent gets there first, under the
contract's claim rule):

| Shared item | Claim | Described in |
|---|---|---|
| `shared-qwen8b/shared.py`: `embed`, `is_validation`, `leak_key`, `cut_to_contract`, `dev_task` | `code` | contract section 4; Plan A Stage 0 |
| The Jaswanth download (1,275,000 texts with 8B vectors) | `jaswanth` | Plan A Stage 0 |
| `qwen-space/RESERVED.md`, the confirmation tasks | `reserved` | Plan A Stage 5 step 1; written before any leakage list or fitting |
| `protocol-check.json`: verdict per public source, and the query protocol | `protocol-check` | Plan A Stage 1 steps 1-2 |
| Leakage list for the Jaswanth set; validation ids | `leakage-jaswanth` | Plan A Stage 2; contract section 2 |
| Reduced copy of the 12.5-million-pair set | `ryanmarcus-p1024` | contract section 2, with its two preconditions |
| Hosted-API cache | none; always through `shared.embed` | contract section 2 |

**Plan A's own results that this plan reads** (never its code; neither plan
imports the other's folder):

| Needed here | Comes from | If it is missing |
|---|---|---|
| `SS384` at both contracts | `qwen-space/stage1/` | Use the published 128-contract figure (Q8, 47.48 core) as the reference for gate B3 and say so. |
| `TT` and `BM25` | `qwen-space/stage1/` | Compute them here. The teacher vectors come from `shared.embed` (cache hits if Plan A ran, a few cents if not). |
| Plan A's `TS` | `qwen-space/stage2/` | Compare against `SS384` only and say so. |

Implement the mixed-index test and calibration in this plan's folder exactly as
Plan A Stage 2 describes (p = 1%, 10%, 50%, seeds 0-4). Use
`training-v2/round2_metrics.py::evaluate` and
`retrieval-diagnostics/diagnose.py::paired` directly; do not write a new
metric.

Facts verified on 2026-11-12:

- `tokenizer/tokenizer.json` is WordPiece with 30,522 entries and a lowercasing
  BERT normalizer.
- The local candidate corpora are JSONL rows of text pairs with `query` and
  `positive` fields (`training-v2/work/data-first/candidates/*.jsonl`,
  `training-v2/work/broad-data/candidates/<source>/`,
  `training-v2/work/broad-data/ready/train.jsonl`).
- `gguf-py` is not installed in `.venv` or in the ROCm environment (Stage 5).
- No stage of either plan had run. `shared-qwen8b/shared.py` did not exist,
  nothing from Hugging Face had been downloaded and no API call had been made.
- The facts about the public datasets and the hosted endpoint come from Plan
  A's section 2. They were read from public metadata, not measured here.

**Rule for shared items.** Look in `training-v2/work/shared-qwen8b/` (claims,
`DONE` markers, `MANIFEST.json`). If an item is done, use it and never rebuild
it. If it is claimed but not done, wait or do other work. If it is unclaimed
and you need it, claim it and build it exactly as the contract and Plan A's
plan describe, then record it in the manifest. The only file this plan may
write inside `qwen-space/` is `RESERVED.md`, under its claim.

**Teacher queries** are embedded the one way `protocol-check.json` records
(the query protocol), so that both plans hit the same cache entries. Plan A's
endpoint check of 2026-11-12 (16 queries) points to the full input string
`Instruct: {instruction}\nQuery:{query}` with the `custom_instruction` field
left empty. Assume that until the file records the verdict.

**What the endpoint returns** (Plan A's check, 7 calls, numbers in
`qwen-space/endpoint-check-20261112/result.json`): unit vectors, whatever
`normalize` says; repeated calls bit-identical; a bare string gets no hidden
instruction, so seeds for single tokens and n-grams are sent as bare strings;
`dimensions: 1024` equals the re-normalized 1024 prefix, so always request the
full vector and cut locally. The hosted model matches the Hugging Face bf16
weights at a median cosine of 0.9976, so a correct public source scores about
0.997 against the endpoint, not 1.0.

## 4. Ground rules

All ten ground rules of Plan A section 3 apply, with these substitutions and
additions:

- **Spending.** The only paid service is the hosted embedding API; no other
  paid compute (RunPod and the like) without separate approval. Plan B has
  its **own hard cap of $5**, separate from Plan A's; expected total is under
  $1. The owner supplied a DeepInfra key on 2026-11-12, after being shown this
  cap, with the words "use this for your deepinfra key (I will delete this key
  when we're done with it)". This plan treats that as approval to spend up to
  the cap on the hosted API and on nothing else. The owner did not repeat the
  $5 figure, so report the running total at every checkpoint.
- **The key.** It is in `/home/cyrus/.config/deepinfra/api_key` (outside the
  repository, mode 600). Load it only into the environment of the process
  that calls the endpoint:
  `export DEEPINFRA_API_KEY="$(cat /home/cyrus/.config/deepinfra/api_key)"`.
  Never print it, and never copy it into the repository, a log, `STATUS.md`,
  the manifest, the cache or a message to another session. Do not delete the
  key file: Plan A uses the same one (contract rule 7). A 401 from the
  endpoint means the owner has deleted the key: stop hosted calls and say so.
  One test call on 2026-11-12 (11 tokens, $0.00000011) returned status 200
  with the fields `embeddings`, `input_tokens`, `request_id` and
  `inference_status` (which carries `cost` and `tokens_input`).
- **Counting.** Count tokens and dollars per stage in `STATUS.md` and stop at
  the cap. Every
  hosted call goes through `shared.embed(..., plan="B")` and the shared cache,
  so nothing either plan paid for is paid for again. With Plan A's cap, the
  owner's total exposure across both plans is $10.
- **What may be sent to the API:** text from public datasets and benchmarks
  only, including single tokens and n-grams taken from them.
- **Where things go.** Code and small results: `ngram-space/<stage>/`, each
  with `RESULTS.md` and `result.json`. Bulky files that only this plan uses:
  `training-v2/work/ngram-space/` (git-ignored). Shared files:
  `training-v2/work/shared-qwen8b/`. The finished model:
  `/home/cyrus/models/<name>`.
- **Training runs on the local GPU** in the ROCm environment, under
  `dibs run gpu ...` with `memguard.start(12)`, as Plan A rule 2 describes.
  Runs longer than about 20 minutes must checkpoint and resume.
- **Lessons that bind here** (Plan A section 2): select only on full-corpus
  retrieval, never on batch-local metrics; do not add mined hard negatives;
  dev tasks are never used for fitting.
- This model has millions of free parameters per megabyte of table.
  Overfitting is the main risk. Every trained arm reports training and
  validation loss side by side.

## 5. Definitions

Plan A's definitions (`TT`, `TS`, `SS384`, `BM25`, stored dimension, the 128
and 504 contracts, development tasks) are used unchanged. Added here:

- **Agreement@10:** for a prompted teacher query, the share of the `TT` top-10
  documents that also appear in the `TS` top-10, ranked over the full corpus;
  averaged over queries. It needs no relevance labels, so every query counts.
  Report it with a paired bootstrap interval between arms.
- **Dev agreement:** agreement@10 over all queries of SciFact, NFCorpus and
  ArguAna, at the 504 contract unless stated.
- **Validation loss:** the training loss on the shared validation split.
- **Coverage:** the share of token positions in validation text whose bigram
  (or trigram) is present in the table.
- **Seed:** the initial value of a table row.

Model selection inside an arm (checkpoint, regularization strength) uses
validation loss only. Comparisons between arms use dev agreement and `TS`
nDCG@10 at stored dimension 1024.

## 6. Stages

### Stage 0: setup

1. Create `ngram-space/` subfolders and `training-v2/work/ngram-space/`. Start
   `STATUS.md`. Load the `dibs` skill. Load the key as section 4 describes
   and confirm that one hosted call succeeds.
2. Resolve the shared items of section 3 in this order, building any that are
   unclaimed: `code`, `jaswanth`, `reserved`, `protocol-check`,
   `leakage-jaswanth` (with the validation ids). Record what you found and
   what you built. The stages below assume the protocol-check verdict for the
   Jaswanth set is "accepted". If it is "first-fit only", continue, and see
   the note at the end of Stage 3. If it is "rejected", stop and report.
3. Write `ngram-space/common.py`: tokenization to ids without special tokens,
   n-gram key packing (ids are below 2^15, so a bigram fits 30 bits and a
   trigram 45 bits of a uint64), and imports from `shared-qwen8b/shared.py`.
4. **Training set.** From the Jaswanth set: keep texts that fit the 504
   contract in full, drop the shared leakage list and the shared validation
   split, and report the counts and the token-length distribution. Tokenize
   once and store the ids as a uint16 memory map in the work folder.

### Stage 1: counting pass and n-gram selection (CPU only, no teacher)

Goal: decide which n-grams get table rows.

1. Corpus for counting: the training texts of Stage 0, plus the local
   candidate corpora (count over both the `query` and the `positive` text of
   each row, after removing exact duplicate strings). Cap the pass at about
   one billion tokens, sampled across sources.
2. Count unigrams exactly. Count bigrams and trigrams with any exact or
   near-exact method that fits in 32 GB of RAM. The simplest: a first pass into
   a count-min sketch (2^28 uint32 counters per n-gram order), then an exact
   second pass over only the keys whose sketch count is at least 20.
3. Selection: drop n-grams containing `[UNK]`, and n-grams made only of
   punctuation and the 150 most frequent tokens. Rank the rest by count.
   Produce nested tiers per order: **T1** = top 250,000, **T2** = top
   1,000,000, **T3** = top 2,000,000.
4. For each selected n-gram, record its most frequent original surface string
   (original casing and spacing), found with the tokenizer's character
   offsets. This is what the teacher will read in Stage 2.
5. Report: coverage of the shared validation texts at each tier and order,
   separately for texts up to 128 tokens and longer texts; unigram frequencies
   `p[t]`; table bytes at each tier for `w_ng` = 512.

Output: `ngram-space/stage1/RESULTS.md` and the tier files.

### Stage 2: seeds from the hosted teacher

Goal: starting values that already point the right way, for a few cents.

1. Through `shared.embed`, with no instruction, embed as stand-alone texts:
   all 30,522 vocabulary entries (strip the `##` prefix of continuation
   pieces; skip special tokens) and the T1 bigrams and trigrams using their
   surface strings. About 0.53 million short items, roughly 3 million tokens
   and $0.03. Send 200 items first and check the usage figures before the
   rest.
2. Fit `mu` and `V` (1024 directions) on 300,000 training vectors of the
   Jaswanth set (dequantized and re-normalized).
3. **Linear ceiling.** Replace each dev document's teacher vector by
   `normalize(V_d V_d^T (t - mu) + mu)` for `d` = 512 and 1024 and report `TS`
   and dev agreement. No model of this plan with a linear readout can beat
   these numbers.
4. Build seeds. With `c(x) = V^T (teacher(x) - mu)`:
   - `u[t] = c(t)` and `w[t] = a / (a + p[t])` with `a` = 0.001.
   - Bigram `g = (t1, t2)`:
     `r[g] = (w[t1] + w[t2]) * c(g) - w[t1] * u[t1] - w[t2] * u[t2]`.
   - Trigram: the same idea. Set `r[g]` so that the model's output on the
     stand-alone trigram equals `c(g)`, after subtracting the unigram terms and
     any bigram residuals already in the table.
   The principle: **at initialization the model reproduces the teacher on
   every seeded item taken alone**, up to the rank-`d` projection. Assert this
   numerically on 1,000 random seeded items.

Output: `ngram-space/stage2/RESULTS.md`, seeds under the work folder.

### Stage 3: lookup-only rungs

Goal: how good is a model that only reads tables, and what do n-grams, seeds
and width each buy.

Training: the Stage 0 training set, starting with a random 300,000 texts.
Loss: the mean over prefixes {256, 512, 1024, 4096} of
`1 - cosine(prefix(pred), prefix(target))`, the same as Plan A's H2 loss, so
the two students are compared on equal terms. Float32; table rows use a sparse
optimizer, dense parts AdamW. Resolve n-gram rows on the fly (for example
`torch.searchsorted` over the sorted key array); do not store per-position row
indices for the whole set. Regularization: a pull toward the seed,
`lambda * ||row - seed||^2` on rows touched in the batch, with `lambda` chosen
from {0, 1e-4, 1e-3} on validation loss; token dropout 10%. At most 10 epochs,
early stopping on validation loss.

Arms:

| Arm | Rung | Width | Tables and initialization |
|---|---|---|---|
| S0 | L0 | 1024 | seeds only, **no training** (floor) |
| S1 | L0 | 1024 | random initialization, trained (control for seeding) |
| S2 | L0 | 512 and 1024 | unigram seeds, trained |
| S3 | L1 | better of S2 | + T1 bigrams, residuals start at zero |
| S4 | L1 | same | + T1 bigrams, residuals seeded (try scaling the seeds by 0.25, 0.5, 1.0 before training; pick on validation loss) |
| S5 | L1 | same | best bigram arm + T1 trigrams, same initialization choice |
| S6 | L1 | same | best of S3-S5 extended to T2 (see the rule below) |
| S7 | L2 | same | best L1 arm + readout MLP |
| S8 | L2 | the other width | the S7 recipe at the other width |

Rule for S6: run it only if gate B1 passes and Stage 1 shows T2 raising
coverage by at least 10 points over T1. If the seeded initialization won,
first embed the extra T2 items (about 8 million tokens, $0.08).

**Data curve and the 12.5-million-pair set.** Train the best L1 arm on
300,000, 600,000 and all of the Jaswanth training texts and plot dev agreement
against the number of texts. If it still rises by at least 1.0 point from
600,000 to all, the larger set is worth having. Before streaming it, both
preconditions of the contract must hold: **the owner has said yes to training
on that set** (it states no licence), and the owner has confirmed the stored
dimension, or has accepted that a later change means streaming 192 GB again.
If either is missing, report the curve at the Stage 3 checkpoint, ask, and
carry on with the Jaswanth set meanwhile. Once both hold, build
`ryanmarcus-p1024` as the contract describes (claim it first; it is a long
stream), build its own leakage list, and continue the curve at 3 million and
at all usable texts, applying the same contract filter and validation split.
Those rows carry only the first 1024 dimensions, so their
loss uses prefixes {256, 512, 1024}; Jaswanth rows keep the 4096 term. Use the
largest set the curve justifies for every later arm. Report the curve either
way: it tells the owner how much the data is worth to each plan.

Evaluation, for every arm: `TS` and dev agreement at the 504 and the 128
contract, at stored dimension 1024 and at 4096, next to the Stage 2 ceilings,
`TT`, `SS384`, `BM25` and Plan A's `TS` when available. For S0 and the best
L0, L1 and L2 arms also run the mixed-index test with and without calibration.

Gates and rules (fixed in advance):

- **B1, n-grams earn their memory:** the best L1 arm beats the best L0 arm by
  at least 1.0 point of dev agreement with the paired 95% interval excluding
  zero, and its core `TS` is not more than 0.3 lower. If B1 fails, drop the
  n-gram tables and continue with unigrams only.
- **B2, rung rule (used for every later rung too):** adopt the larger rung only
  on the same test as B1 against the rung below it. Otherwise keep the
  smaller one.
- Width: keep 1024 over 512 only if it passes the B1 test against 512.
- Seeding: report S1 against S2, and S3 against S4, plainly. If seeding does
  not help after training, say so and stop buying seeds.

If the Jaswanth verdict was "first-fit only": after selecting the arm, estimate
what it costs to re-embed its training texts through the API (tokens times
$0.01 per million), report that to the owner and ask before spending above the
cap. Do not release a model calibrated only on vectors that did not match the
hosted endpoint.

Output: `ngram-space/stage3/RESULTS.md`. **Checkpoint: report to the owner**,
then continue.

### Stage 4: the whole-chunk mixer

Goal: learn what chunk positions and the mixer add.

1. Arms, each warm-started from the one before, on the Stage 3 training set:
   - L3 with `c` = 16;
   - L4 with `c` = 16 and with `c` = 32;
   - L5 with the better `c`;
   - for the selected rung only: the other representation width, and one
     control trained from the Stage 2 seeds in a single run (no rung-by-rung
     warm start).
   During training, start the chunk grid at a random offset in `[0, c)` so the
   model does not depend on where chunk boundaries fall.
2. Evaluate every arm as in Stage 3, with the full mixed-index test for the
   selected rung.
3. **Boundary robustness.** For the selected rung, embed the dev documents with
   grid offsets 0, 5 and 11. Report the mean and 5th-percentile cosine between
   the offset variants of the same document, and the spread of `TS`.
4. **Probes relative to the teacher.** Build about 200 document pairs for each
   of: negation, swapped word order, same sentences in a different order, same
   entity name with a different meaning. Embed them through the API (a few
   thousand tokens). For each pair report the teacher's cosine and the
   student's. The useful number is how much of the teacher's separation the
   student keeps; a pair the teacher itself does not separate says nothing
   about the student. Report only; not a gate.

Gates (fixed in advance; core mean of SciFact and NFCorpus; stored dimension
1024; 504 contract unless stated):

- **B2** decides between L2, L3, L4 and L5.
- **B3, usefulness:** selected rung `TS >= SS384 - 6.0` (windowed `SS384`). If
  it fails, stop before Stage 6 and report. The fallbacks to offer: use this
  model only for the refinement signal of Stage 6, or rely on Plan A's student.
- **B4, mixed index:** Plan A's gate G2, unchanged.
- **B5, long inputs:** Plan A's gate G3: `TS` at the 504 contract is no more
  than 0.5 below `TS` at the 128 contract.
- **B6, boundaries:** the spread of `TS` across grid offsets is at most 0.3.
- As in Plan A: if `BM25` beats `TS` by more than 3 points, it is not a
  failure, but say plainly that a lexical or hybrid provisional tier is
  competitive.

**Recommendation rule against Plan A** (when Plan A's `TS` exists): recommend
Plan B's model if its `TS` is within 2.0 of Plan A's at the 504 contract and
Stage 6 gate B7 holds. Otherwise present both as a speed and quality trade-off
and let the owner choose.

Output: `ngram-space/stage4/RESULTS.md`. **Checkpoint: report to the owner.**

### Stage 5 (off by default): seeds from the Qwen3.8 n-gram table

Do not run this unless the owner asks for it after seeing Stage 3.

The original reason for it was to seed n-grams that a local teacher could not
afford to read. With the hosted teacher, a seed for every row of the largest
tier costs well under a dollar, so that reason is gone. What remains is a
curiosity: whether rows trained for next-token prediction carry anything the
teacher's own seeds and millions of calibration texts do not. The cost is a
7 to 30 GB temporary download, porting a hash function exactly, and the
donor's license (`qwen-community-1.0`; the Qwen embedding models are
Apache-2.0), which the owner must read before donor-derived values enter a
released model.

If it is requested, bound it to one working day:

1. Background: `/home/cyrus/code/model-experiments/qwen38-flash-next-ngram/`
   (write-up and `surgery_qwen38.py`). The table is the GGUF tensor
   `per_layer_token_embd.weight`: 16 hash heads, 8 for bigrams and 8 for
   trigrams, each about 20 million rows of 160 dimensions, laid out by the keys
   `qwen4exp.ple.head_offsets`, `qwen4exp.ple.head_vocab_sizes` and
   `qwen4exp.ple.heads_per_ngram`, indexed in that model's own token ids.
2. Take the hashing routine from the llama.cpp source for architecture
   `qwen4exp` **verbatim** into a small harness. Do not re-derive it. A wrong
   hash looks exactly like a table with no usable content.
3. Install `gguf` into `.venv`. Read the shard headers first and confirm that
   `gguf-py` can dequantize the table tensor's quantization type; if it cannot,
   stop and report. Then download only the byte ranges of the heads you need
   from the unsloth UD-Q3_K_XL shards (start with 2 heads per n-gram order,
   about 7 GB) and dequantize the rows.
4. For an n-gram of ours: tokenize its surface string with the Qwen3.8
   tokenizer, look up the rows of the token n-grams it contains, average them,
   and concatenate across heads. That is its donor feature.
5. **Early exit.** Fit a ridge regression from donor features to the trained
   residual rows of the selected model on 80% of the T1 n-grams and test on the
   other 20%. If the mean cosine between predicted and true rows is below 0.2,
   stop: the donor rows do not carry what we need. Record the result.
6. Otherwise use the donor predictions as seeds for T2 rows in place of
   teacher seeds, and compare after identical calibration. Adopt only under
   the B1 test. Delete the downloaded table bytes.

### Stage 6: CPU runtime, quantization and the refinement signal

Skip if B3 failed. The platform constraints of Plan A section 1 and its Stage 4
are binding here too: no `-march=native`, runtime selection between an AVX-512
and an AVX2 path, no torch or transformers at runtime, no OpenMP, one C++
shared library with a C ABI, a `ctypes` + numpy + `tokenizers` Python package.
If Plan A's runtime exists, copy its build setup, CPU detection and packaging
approach into this plan's folder; do not import or link against Plan A's
folder.

1. **Quantization.** Table rows as int8 with one float scale per row; unigram
   weights as float32; mixer and readout in float32. Confirm in Python that
   this costs at most 0.2 `TS` against the float model; if it costs more, use
   float16 rows and report the size.
2. **Model file:** one self-describing file, **memory mapped read-only** so
   several worker processes share its pages and only touched rows are
   resident. Record the source checkpoint hash.
3. **N-gram index:** open addressing with linear probing over uint64 keys,
   power-of-two capacity, load factor at most 0.5, built offline and stored in
   the file.
4. **Forward pass in C++:** ids to keys, software prefetch of the rows, scaled
   row accumulation per chunk, chunk division, the mixer (plain FMA matrix
   kernels are enough at 32 rows), pooling, readout restricted to the stored
   dimensions, L2 normalization.
5. **Checks:** token ids identical to the transformers tokenizer on every dev
   text; minimum cosine to the Python reference at least 0.999 on 1,000
   documents of mixed length on each code path; dev `TS` within 0.15 on each
   path.
6. **Measurements** under `dibs run --measure`, tokenization included, one
   pinned thread, documents of 16, 64, 128, 256 and 504 tokens, both code
   paths: median and p95 latency and documents per second, also with 2 and 4
   worker threads; the share of time spent in tokenization; file size, peak
   resident memory, load time, and the first-call latency with a cold page
   cache. Measure the selected rung and the best lookup-only rung.
7. **Gate B7, speed:** at 504 tokens on one thread, the selected rung is at
   least 5 times faster than Plan A's runtime on the same code path. If Plan
   A's runtime does not exist yet, use its engineering targets (12 ms and
   40 ms), which gives 2.4 ms on the AVX-512 path and 8 ms on the AVX2 path.
   If a mixer rung misses B7, try `c` = 32 and then int8 mixer linears before
   falling back to the best lookup-only rung.
8. **Refinement signal.** The owner wants to spend the hosted model where it
   helps. Compute per document: the norm of the readout before normalization,
   token count, bigram and trigram coverage, and the share of `[UNK]`. Report
   each signal's rank correlation with `1 - cosine(provisional, refined)` on
   the dev documents, and repeat the simulation of Plan A Stage 3 here for
   three policies (random, the best signal, oracle) at the same fractions. The package returns the chosen signal with each
   vector.
9. **Azure probe.** Write `ngram-space/runtime/azure_probe.py` with the same
   output format as Plan A's probe (Plan A Stage 4), so the owner can run a
   20-second benchmark of this model on the App Service instance. Dev-box timings are never presented as
   server numbers.

### Stage 7: confirmation and release

1. **Confirmation** on the two tasks named in the shared
   `qwen-space/RESERVED.md` (the same tasks Plan A confirms on), opened once
   for this plan's frozen final model: `TT`, `TS`, `SS384`, `BM25`,
   agreement@10 and the mixed-index test.
2. **Release folder** `/home/cyrus/models/<name>` (placeholder name
   `doc-embed-ngram-qwen8b`; ask the owner before settling it): model file,
   tokenizer, package source and built library, `config.json` (teacher,
   provider, date checked, stored dimension, document contract, rung, widths,
   table tiers, training sources) and a README with measured quality next to
   Plan A's student, the mixed-index results, dev-box timings marked as such,
   and the probe instructions. Include Plan A's drift canary (the same 50
   texts and script).
3. Run the package from a fresh virtual environment with only numpy and
   `tokenizers` and repeat the Stage 6 checks.
4. **Clean up scratch:** list everything under `training-v2/work/ngram-space/`
   with sizes and delete seeds and intermediate checkpoints once the released
   model passes its checks. **Never delete anything under
   `training-v2/work/shared-qwen8b/`**; the shared contract says when that may
   go.

## 7. Out of scope

- Everything out of scope in Plan A section 6.
- A token-level transformer or convolution over the token stream.
- Hashed tables for the tail, character n-grams and word-level features.
- A different tokenizer. The WordPiece tokenizer keeps the document contract
  shared with Plan A; revisit only if Stage 1 coverage is poor.
- Quantization-aware training.

## 8. Open questions for the owner

Carry these in `STATUS.md` and raise them at the checkpoints. Defaults let work
continue meanwhile.

1. **API key and Plan B's own $5 cap.** Settled 2026-11-12 for Plan B: the
   owner supplied the key (section 4) and a test call succeeded. Plan A's
   session reports that the owner gave it the same key in reply to its
   question about the $5 caps (contract rule 7), so the key covers both
   plans, each under its own cap. The exposure is $10; expected spend across
   both is under $2.
2. **Model file budget.** Default 2 GB on disk, memory mapped. Is that right
   for the 16 GB shared server?
3. **Training on the 12.5-million-pair set** (Plan A's question 5). The owner
   has said the plans should share it, but it states no licence and derives
   from datasets with their own terms. The contract requires an explicit yes
   to training on it before anyone streams it.
4. **Stored dimension** (Plan A's question 2). Default 1024. It decides how
   wide the reduced copy of the 12.5-million-pair set must be.
5. **Order of work with Plan A.** Either plan can start first; shared items
   are built by whichever agent needs them first.
6. **Stage 5** stays off unless asked for.
