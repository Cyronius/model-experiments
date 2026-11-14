# Stage 6 proposal: backbone fine-tuning (NOT run; waiting for approval)

Written 2026-11-12 after Stage 2 and Stage 5. The release in `/home/cyrus/models/doc-embed-3l-qwen8b`
stands on its own; this proposal is the next step only if the owner wants a better provisional tier.

## Why (which gate failed and by how much)

- **G2, mixed index at p = 10% (core, 504 contract, 1024 dims): failed on both parts.**
  Recall ratio of relevant provisional documents 0.790 (limit 0.85, short by 0.06); provisional
  share of top-10 slots 22.9% (limit 15%, over by 7.9 points). Calibration did not help.
- G1 passed (-1.15 against the -2.0 limit), G3 passed (+2.45). The stop rule did not trigger.
- Reserved confirmation adds weight: on NQ, TS trails SS384 by 6.58 [-7.69, -5.48] (partly because
  the backbone was trained on NQ); on Touché TS leads by 7.47 [+3.0, +11.8]. Provisional-to-refined
  cosine is only 0.71-0.79.
- The head-only data curve is still rising (41.7 / 44.6 / 46.4 / 47.1 core at 50k / 100k / 300k /
  1.17M), but slowly; the frozen 384-d features cap what a head can reach.

## Training set

- **First choice, no new approval needed for data:** the Jaswanth set already on disk
  (1,173,363 training texts after the 504 contract and leakage filters, full 4096-d targets; the
  shared validation split and leakage list unchanged). Teacher vectors are free.
- **If more data is wanted:** the reduced RyanMarcus copy (`ryanmarcus-p1024/`, 12.5M pairs,
  1024-d prefix targets) — not built yet by either plan. Building it needs the owner's yes on its
  licence (none stated) and accepts 1024-d targets (enough for the 1024 stored dimension). About 13 GB
  of vectors plus text, streamed from the 192 GB parquet by row group; free except time.
- Long documents are rare in both (22.5% of kept Jaswanth texts exceed 126 tokens). If the
  native-512 path is trained, add the 504-contract cuts of long public texts with teacher vectors
  through the API (e.g. 200k S2ORC/SPECTER-style abstracts already in the sets, or new public
  passages, about 80M tokens, ~$0.80 under Plan A's remaining $4.70).

## Loss

1. The Stage 2 loss: mean over prefixes {256, 512, 1024, 4096} of 1 - cos(prefix(head(f)),
   prefix(target)), on the head output, back-propagated into the backbone.
2. **Anchor term** keeping the 384-d output close to the frozen model:
   `w_a * mean(1 - cos(out_new(x), out_frozen(x)))` on the same batch (start w_a = 1.0; also try 0.3),
   so the student's own space (SS384) and its retrieval behaviour do not collapse. This answers the
   earlier lesson that fine-tuning on narrow teacher targets cut retrieval from 42.1 to 38.8-41.3.
3. Optional (only if G2 is the target): a score-distribution term on round-2 triplets (teacher query
   q, provisional p vs refined t): penalise `mean(q.p) - mean(q.t)` over random pairs, to stop
   provisional vectors scoring high against unrelated queries. No unjudged hard negatives.
- Train the position rows 128-511 (they are untrained bge-micro-v2 rows today), quantisation-aware
  for the Q8/7-bit runtime format (`quant_act` straight-through), learning rate 2e-5 for the backbone,
  1e-4 for the head (initialised from the selected head), 1-2 epochs.

## Selection

Full-corpus retrieval only (no batch-local metrics): `TS` core at 504/1024 plus the mixed-index
recall ratio and top-10 share at p = 10%, on SciFact, NFCorpus and ArguAna, **plus** independent
domains not used by the head fit: FiQA and SCIDOCS (already opened in 2026-10, fine for development)
with teacher vectors through the API (~$0.15). Also report SS384 to watch the anchor. NQ and Touché
stay closed (they are now opened for Plan A's release; a new confirmation set would be needed,
e.g. TREC-COVID whose qrels are still unopened).

## Time and cost

- Compute on this machine: the backbone is 17M parameters. Forward plus backward on 1.17M texts
  (median 33 tokens, some to 506) is about 3x the feature-extraction cost measured here (35-73 min
  per pass on the shared CPU), so roughly **2-4 h per epoch on CPU**, or well under 1 h per epoch on
  the local ROCm GPU under `dibs run gpu` with memguard. **No paid GPU needed. Estimated cost $0**
  for compute; hosted API about $0.15-0.95 depending on the optional long-document targets (within
  Plan A's cap).
- Then repeat Stage 4 packing/checks (minutes) and a fresh confirmation set.

Stop here and wait for the owner's approval.
