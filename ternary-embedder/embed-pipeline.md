# Plan (draft): a layer-pipelined embedding engine on the NPU

**Status:** draft, 2026-10-16. Nothing built. Stops for the owner's decision on the target (see the end).
**Worktree:** `C:\code\openflowlm-embed`, branch `feat/embed-pipeline` (untied, from `upstream/main` 99fe7e4d).
**Spec impact:** new requirements, named when step 2 starts. No embedding-engine spec exists yet
(`specs/open-engine` has only `OPEN-EMBED-STRIDE`; `specs/server-api` has `SERVER-EMBED-TASK-PROMPT`).

## Goal

The fastest text-embedding engine on the Strix NPU. The owner's idea: pipeline the model across the
array, so each part of the array runs a different layer or task, and data flows between them.

## The baseline today (`open_npue`, from `docs/docs/benchmarks/embeddings_results.md`)

| model | emb/s at batch 64 | real tok/s | one text |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 1305 | 28.8k | 9.3 ms |
| bge-small-en-v1.5 | 565 | 12.5k | 18.8 ms |
| bge-base-en-v1.5 | 311 | 6.9k | 25.3 ms |

- Each layer is four GEMM dispatches. Attention, layer norm and pooling run on the host.
- The texts are about 22 real tokens (tok/s ÷ emb/s), padded to a compiled `seq` of 64. So about
  two thirds of the NPU arithmetic is on padding.
- MiniLM does about 1305 × 64 × 10.6M weights × 2 ≈ 1.8 TFLOP/s of padded work. That is a few
  percent of the array's ~50 TOPS peak. Most of the time goes somewhere other than multiplies.

## What the earlier NPU work showed (covenant-agent tiny planner, R30-R34)

That was a few-million-parameter planner with a diffusion decoder, not a general embedding model.
The lessons transfer; the numbers do not.

- Ternary decoder kernel: 0.103 ms per layer pass; the 16-pass diffusion draft took 1.65 ms against
  9.93 ms in bf16 (6×). A refiner step took 0.44 ms against 2.19 ms.
- Encoder in bf16 on all 32 cores (each core owns rows and runs the whole layer): 1.30 ms per pass
  over 475 rows.
- After ternary, the multiplies cost almost nothing. Scale, GELU, norms, input quantization and
  attention took nearly all of the time. The chip has no fp32 vector multiplier.
- Program memory (16 KB per core) was full at 15.6 KB, because each core held the code for a whole layer.

## Why 32 one-layer stages do not fit, and what does

On-chip memory: 32 compute tiles × 64 KiB = 2 MiB, and 8 mem tiles × 512 KiB = 4 MiB.

| model | layer weights | int8 | int4 | ternary (2-bit) |
|---|---:|---:|---:|---:|
| MiniLM-L6 (h 384, 6 layers) | 10.6M | 10.1 MiB | 5.1 MiB | 2.5 MiB |
| bge-small (h 384, 12 layers) | 21.2M | 20.2 MiB | 10.1 MiB | 5.1 MiB |
| bge-base (h 768, 12 layers) | 84.9M | 81 MiB | 40.5 MiB | 20.2 MiB |

- One MiniLM layer at ternary is 432 KiB. One tile has 64 KiB. One FFN matrix alone is 144 KiB.
  So a layer cannot live in a tile.
- One MiniLM layer at ternary fits in one column's mem tile (432 of 512 KiB).
- So the stage is a **column**, not a tile: one layer per column. This is the only mapping where
  no weight byte leaves the chip after load.
- Weight residency is not optional for this design. If each column streamed its layer from DDR
  for every 64-row slot, the 8 columns together would need tens of GB/s, more than the NPU gets.
- Inside a column, the 4 tiles split the layer by task. Per token, the multiply work is:
  QKV 24%, attention + O-proj 11%, FFN-up 32%, FFN-down 32%. The tasks are:
  1. QKV projection.
  2. Attention (split by head) + O-proj + norm.
  3. FFN-up + GELU.
  4. FFN-down + norm.
- Each tile holds only its own task's code. That removes the 16 KB program-memory wall from the
  covenant work.
- Activations move tile to tile and column to column on chip. Only token embeddings go in
  (host lookup), and only the pooled 384-value vector comes out.

## The model is shaped by the chip (owner, 2026-10-16: train from scratch or almost scratch)

We train our own model, so the chip sets the shape. **Ternary weights are decided.**

- **Shape:** hidden 384, FFN 1536, **8 layers**, ternary weights (2-bit packed), int8 activations.
  - About 14M layer weights, 8 × 432 KiB = 3.4 MiB. One layer per column fills all 8 columns.
  - The host looks up token embeddings (30.5k × 384, kept in 16-bit on the host).
  - Pooling and L2 normalization run in the last tile of column 8.
- **Why 384 × 8:** at hidden 448 or 512 a ternary layer no longer fits one 512 KiB mem tile.
  At int4, only 4 layers fit. Ternary buys twice the depth.
- **Ops:** keep LayerNorm and tanh-GELU first, so the inherited weights still fit (the covenant
  kernels already do tanh-GELU). RMSNorm and ReLU² are cheaper on the NPU; try them later as an
  ablation, because vector work is the expected bottleneck.
- **1-bit weights: no.** They enter the same int8 × int4 multiply as ternary, and the multiplies
  cost about 0, so there is no speed gain. Half the memory does not help: 8 copies of 2 × 256
  already use every tile. BitNet b1.58 beat binary BitNet because the zero matters. 1-bit
  *output* vectors (Hamming search, then rescoring) are a separate, search-side option.
- **Slot length** is now a training choice too: 64 for queries and sentences, 128-512 for passages.
  Attention cost and activation memory grow with it.

### Layer per tile (owner, 2026-10-16)

With our own model, one layer per tile is possible. The weights live in the column's mem tile,
not in the tile: 4 layers share one 512 KiB mem tile, 108 KiB each.

| layers per column | hidden | layers | layer weights | per layer (ternary) |
|---|---:|---:|---:|---:|
| 1 (task split over 4 tiles) | 384 | 8 | 14.2M | 432 KiB |
| 2 (2 tiles per layer) | 256 (ELECTRA-small's width) | 16 | 12.6M | 192 KiB |
| 4 (one layer per tile) | 192 | 32 | 14.2M | 108 KiB |

- 32 × 192 and 8 × 384 have the same weight count, so the same multiplies per token. But vector
  work (norms, scaling, quantizing, softmax) sets the speed, and it grows with layers × width:
  6144 against 3072. So 32 × 192 is about 1.5× slower per pass. See the estimate below.
- **For one layer per tile:** all 32 stages are the same, so the balance is perfect. The task split
  wastes about 20% (its stages run 11-32% of a layer). Step 1 also becomes one tile, one kernel.
- **Against:** each tile runs a whole layer, so the 16 KB program-memory limit comes back
  (covenant `te.py` fit at 15.6 KB only with rolled loops, which cost speed). Four tiles read
  weights from one mem tile at once; check its DMA channels. Width 192 is narrow, and ternary
  hurts narrow layers more. No pretrained 192-wide model exists to inherit from directly (weight
  selection from a 384-wide model is the nearest option).
- **Weights inside the tile** (no mem tile) leave about 36 KiB for weights after activation
  buffers. That forces hidden 96-112 and a 4-5M-weight model. Not recommended.
- **Decide by training.** See the grid in step 3. Quality picks the row.

### The other axes (owner, 2026-10-16)

**Different tiles compute different parts of the vector.** Two forms:
- *Split inside a layer:* attention by head, FFN by slices of the 1536-wide middle, partial sums
  passed down the AIE cascade link. The 8 × 384 column design already needs this. Cost: two
  joins per layer (after attention, after FFN), and LayerNorm needs whole-row sums.
- *Independent towers:* each row of 8 tiles is its own 8-layer, 192-wide model; the 4 outputs are
  joined at the end (768 dims). No traffic between towers. Same weights and multiplies as 32 × 192,
  so the same throughput. It only cuts one-text latency (8 stages, not 32). Quality is likely
  lower (each tower is shallow and narrow); mixing between towers every few layers is the middle
  ground. **Dropped:** the target is throughput, and towers only help latency.

**Looping (shared weights, several passes).** The mem tiles cap the weights at about 14M.
Looping is the only way to add depth past that cap. It is paid in throughput: 2 loops = half the
embeddings per second, because every pass costs the full multiplies.
- Per stage: each tile or column runs its layer twice. Every stage stays balanced.
- Ring: the 32-tile chain closes into a loop (a 4 × 8 grid has one), and a slot goes around twice.
- Early exit: a slot leaves when its vector stops changing (cosine between passes above a
  threshold). Easy texts pay one pass. This fits the ring: a slot leaves, a new one enters.
- Evidence: ALBERT-style sharing loses some quality against the same depth unshared, but beats a
  shallower unshared model of the same size. Recent looped models add small per-loop deltas to
  close the gap.

**Diffusion versions.** An encoder makes the whole vector in one forward pass. There is no
token-by-token decode, so diffusion's main win (many tokens per pass) has nothing to speed up here.
Two parts of it still apply:
- *Training objective:* masked denoising with a random mask ratio. BERT's MLM is the one-step case
  of this. It matters only for the pretraining part, and the plan distills instead.
- *Iterative refinement:* repeated passes over shared weights until the result settles. That is
  looping with early exit, above. Covenant's diffusion draft was this (2 layers × 4 loops).
So "diffusion" is not a separate axis for an embedder; it is looping plus exit. A diffusion
*text generator* on this pipeline is a different workstream.

### Shallow replicas (Ternlight evidence, covenant R21/R27)

Ternlight-mini (wenshutang/ternlight) is a ternary embedder distilled from all-MiniLM-L6-v2:
**2 layers**, 256 wide, 4 heads, FFN 1024, 9.5M parameters (82% of them the word table).
Ternlight-base is 2 layers at 15.4M. On covenant's zero-shot tool-reading tests, mini scored
88.9 against the teacher's 86.6 on decidable decoys, but 70.3 against 76.4 on lookalikes. On the
plain constant suite, Ternlight beat its teacher (66.6 against 62.8). Its card gives Spearman 0.835
against the teacher. This is task-specific evidence, not MTEB. On MTEB (step 0b) both tiers
miss the teacher by a wide margin. That rules out Ternlight, not 2 layers: see step 0b.

- For throughput, speed goes with vector work per token, which grows with layers × width:
  2 × 256 = 512, against 3072 for 8 × 384. About 6× less work per token.
- A 2-layer model is 2 × 192 KiB = 384 KiB: one full copy per column. **8 copies, one per
  column, each on 4 tiles (2 tiles per layer).** No 32-stage chain, no cross-column traffic.
- So the deep pipeline is a way to fit a big model on chip. If a shallow model passes the
  quality bar, copies of it beat any deep pipeline on throughput.
- At ~1.3M tok/s, the host side matters: the tokenizer needs several threads, and input traffic
  is about 0.3 GB/s (int8 token vectors after the host lookup).
- Ternlight's arithmetic (one scale per matrix, int8 activations per word) is close to the `tt`
  kernel's, not identical. Its license is not checked.

### Training: almost scratch, not scratch

- **Initialize** from bge-small-en-v1.5 or all-MiniLM-L12-v2. Both are hidden 384 / FFN 1536 /
  12 layers, so keeping 8 of 12 layers is an exact shape match. For width 256, ELECTRA-small
  (12 layers, FFN 1024) is the match. ELECTRA is a pretraining method and a backbone, not an
  embedding model, so it needs the distillation step like the others.
- **Distill** from a stronger teacher's vectors (bge-base, or larger) on unlabeled text. The
  teacher's vectors are computed once. This avoids MLM pretraining and contrastive pair mining.
- **Ternary last.** Train the float student, then convert to ternary with the R33 method, and keep
  the float master. Covenant R32 (ternary from the start, tiny model) lost about 8 points on
  held-out sets; R33 (float, then convert) stayed within 0.7.
- **Pure random init** needs MLM pretraining plus contrastive training on hundreds of millions of
  pairs. At this size it costs more and usually ends worse. Not recommended.
- **Cost (owner: keep it cheap).** Rank shapes on ~300M tokens, not 3B; train only the winner
  longer. Teacher vectors (bge-small) are computed once, about 20 min, and shared by every run.
  Community RTX 3090 at $0.22/hr (covenant R27 cost $0.45). Per run: 2 × 256 about 10 min,
  the 14M-weight shapes about 1-1.5 h, looped ones 2-3 h. The 2- and 4-layer runs are small
  enough for the laptop CPU at $0.
- **Quality bar:** the student must beat all-MiniLM-L6-v2 on a small MTEB subset (STS and one
  retrieval task). If it does not, the speed record uses a worse model than the one that already
  runs at 1305/s.

## Costs and risks

- **Throughput = the slowest stage.** Attention grows with sequence length; FFN does not. A
  mapping balanced at 64 rows goes out of balance at 512. Fixed slot size, or one design per bucket.
- **Pack short texts.** Put several texts in one 64-row slot with a block-diagonal attention mask
  (covenant packed lines the same way). This recovers the ~2/3 of work spent on padding today.
- **Fill.** The pipeline needs 8 slots in flight. One text alone sees the sum of 8 stage times.
  That is still far below today's 9.3 ms, because today is overhead-bound.
- **Model lock-in.** The design fits one shape. bge-small and larger do not fit on chip even at
  ternary; they need DDR weight streaming, and then the gain comes from fusion, not residency.
- **Quality.** See the quality bar above. No speed number counts before it holds.
- **Host tokenizer.** At the target rate it must do several hundred thousand tokens per second.

## Estimate (not a measurement)

**Calibration:** covenant `tt` (R31) is a ternary layer at width 256 on this chip. Its compute
took 0.053 ms a pass at 4 rows per core; the multiplies cost about 0. Scaled by element counts
to an encoder layer at width 192, that is about **7 µs per row per layer on one tile**. A 384-wide
layer split over a column's 4 tiles is about 4.8 µs per row (75% balance assumed).

Real tokens per second, NPU only, slots 90% full. The range is 1.5× worse (desktop load) to
2× better (bigger slots, more fusing). One text = 22 tokens.

| design | passes | real tok/s | emb/s | one text |
|---|---|---:|---:|---:|
| 32 × 192 ring | 1 | ~125k (85-250k) | ~5.5k | ~5 ms |
| 32 × 192 ring | 2 | ~62k (42-125k) | ~2.8k | ~10 ms |
| 32 × 192 ring | 1-2, half exit after 1 | ~83k | ~3.8k | 5-10 ms |
| 16 × 256, 2 tiles per layer | 1 | ~160k (105-320k) | ~7.2k | ~2-3.5 ms |
| 3 × 384 copies (bge-micro shape) | 1 | ~500k | ~23k | <1 ms |
| 2 × 384 copies (Ternlight-base shape) | 1 | ~750k | ~34k | <1 ms |
| 2 × 256, 8 copies (Ternlight-mini shape) | 1 | ~1.3M (0.85-2.5M) | ~58k | ~0.25 ms |
| 8 × 384 columns | 1 | ~187k (125-375k) | ~8.5k | ~1-2.5 ms |
| 8 × 384 columns | 2 loops per stage | ~94k | ~4.3k | ~2-5 ms |
| today, MiniLM-L6 | 1 | 28.8k | 1305 | 9.3 ms |

- The copy rows scale the 8 × 384 row by layers × width and assume the whole array stays busy.
  A 3 × 384 layer (432 KiB) needs one mem tile each, so 8 columns hold 2 whole copies plus
  2 spare layers; spreading a layer's weights across columns may recover the rest. Not checked.
- Looping divides throughput by the pass count; early exit wins back part of it.
- One text goes through every stage on one tile each, so a 32-stage ring has ~5 ms latency
  per pass. The column design spreads each layer over 4 tiles, so it is several times faster.
- Lever we control in training: vector operations per layer (one norm per layer instead of
  two, no biases, ReLU², fused quantize). This matters more for the deep, narrow shapes.
- Throughput as the target relaxes the on-chip cap: with large slots, weights could stream from
  main memory, which allows bigger models. Each is slower per token, so this matters only if
  every on-chip model misses the quality bar. Not checked.
- Not checked: mem tile feeding 4 tiles at once, a 64-row slot's K/V in 64 KiB (may need int8
  K/V or 32-row slots), and tile-to-tile handoff cost.

## Steps

Each step has a gate. If a gate fails, stop and report.

0. **Profile today's path** (MiniLM, batch 64): split the wall time into tokenizer, host ops,
   dispatch, and NPU busy. Gate: confirms where the time goes. Needs `dibs` for the NPU.
0b. **Quality floor, $0, CPU.** Score Ternlight-mini, Ternlight-base, all-MiniLM-L6-v2 and
   bge-small on the MTEB subset of the quality bar. Gate: if a 2-layer model lands near the
   teacher, the shallow-copies design leads the grid.

   **Result, 2026-10-16: gate failed.** Harness in `.claude/quality-floor/` (mteb 1.39.7, all
   models cut at 128 tokens, no query prompts). The harness reproduces the published STSBenchmark
   scores (MiniLM 82.0, bge-small 85.9).

   | type (tasks) | MiniLM-L6 | bge-small | Ternlight-mini | Ternlight-base |
   |---|---:|---:|---:|---:|
   | STS (7) | 78.9 | 82.8 | 73.2 | 76.0 |
   | Retrieval (SciFact, NFCorpus) | 46.9 | 50.0 | 33.8 | 35.5 |
   | Classification (Banking77) | 80.0 | 81.8 | 74.5 | 76.4 |
   | Reranking (AskUbuntu) | 63.5 | 62.6 | 57.8 | 59.5 |
   | Pair classification (TwitterSemEval) | 67.9 | 73.2 | 59.5 | 58.7 |

   Two ternary layers lose 3-6 points on STS and 11-13 on retrieval against their own teacher.
   Covenant's "matches its teacher" was true only for short tool texts. This run cannot split the
   loss between depth, ternary and Ternlight's own recipe.

   **Owner (2026-10-16): do not rule out 2 layers on Ternlight alone; its training is weak.**
   Ternlight's own docs (`docs/model-internals.md`, `eval/quality/RESULTS.md`) support that:
   - It copies only the final pooled vector (1 - cos), plus a small in-batch spread term. No
     layer-wise distillation of hidden states or attention.
   - Corpus, initialization and training budget are not stated. Its only eval is agreement with
     the teacher on 100 MS MARCO queries; no human-labeled STS, no retrieval, no long text.
   - Its float student, before ternary, already agrees with the teacher at only Spearman 0.883.
     Ternary takes that to 0.835. So most of the loss is the recipe or the size, not ternary.

   **Depth curve, float models with known recipes (same harness, same day):**

   | model | layers × width | STS | Retrieval | Classif. | Rerank | Pair |
   |---|---|---:|---:|---:|---:|---:|
   | potion-base-8M (static) | 0 | 72.5 | 37.4* | 72.6 | 54.1 | 57.5 |
   | Ternlight-mini (ternary) | 2 × 256 | 73.2 | 33.8 | 74.5 | 57.8 | 59.5 |
   | Ternlight-base (ternary) | 2 × 384 | 76.0 | 35.5 | 76.4 | 59.5 | 58.7 |
   | paraphrase-MiniLM-L3-v2 | 3 × 384 | 78.7 | 39.3 | 78.2 | 59.6 | 61.5 |
   | paraphrase-MiniLM-L6-v2 | 6 × 384 | 79.9 | 40.0 | 78.6 | 60.4 | 64.6 |
   | **bge-micro-v2** | **3 × 384** | **78.8** | **46.2** | **81.2** | 58.1 | **67.8** |
   | all-MiniLM-L6-v2 (the bar) | 6 × 384 | 78.9 | 46.9 | 80.0 | 63.5 | 67.9 |
   | bge-small-en-v1.5 | 12 × 384 | 82.8 | 50.0 | 81.8 | 62.6 | 73.2 |

   \* potion reads whole documents; every other model stops at 128 tokens.

   - **Ternlight is at the level of a 0-layer static model.** Its 2 layers add almost nothing,
     which points at its training, as the owner said.
   - **Depth costs little with a fixed recipe:** the paraphrase pair, 3 against 6 layers, differs
     by about 1 point (3 on pair classification).
   - **A 3-layer model passes the bar.** bge-micro-v2 (distilled from bge-small) matches
     all-MiniLM-L6-v2 on 4 of 5 types; only reranking is lower (58.1 against 63.5).
   - No well-trained 2-layer model was in the set, so 2 layers is still open.
1. **One stage.** One 192-wide layer on one tile, weights read from the mem tile, random ternary
   weights (numbers meaningless, arithmetic checked against a float64 reference, as R31). Measure
   its time and program size, then 4 tiles on one mem tile at once. The stage time sets the
   pipeline's throughput. Gate: set from step 0.
2. **The chain.** 8 columns, packed slots, pooling on chip. Write the spec requirements here.
3. **The model grid**, in parallel with steps 1-2. Same teacher vectors, data and ternary
   conversion for every run. Run 0 and the 2-layer track first: about 15-40 min a run, about
   4 h in all on a community 3090, about $0.85 (owner approved, 2026-10-16).

   **Running** since 2026-10-17 01:54 UTC: RunPod pod `78w5gy5b0etaqt` (community 3090, $0.22/hr,
   stop guard 06:53 UTC), code `C:\code\tern-embed` @ b061288 (`grid.sh`). The student rebuild
   matches its source model in float (max diff 2.7e-6 bge-micro-v2, 7.6e-6 ELECTRA-small), and
   every grid configuration passed a CPU smoke run first. Layer choices: ELECTRA students start
   from its first 2 layers; bge-micro and bge-base students from their first and last.
   - Corpus: 4,252,720 texts, 2,337,599 pairs, 145M tokens, mean 34.2 tokens, 3% at the 128 cap.
   - First launch (04:00 UTC) was too slow: three runs side by side took 0.12 (2a_256), 0.23
     (2a_384) and 0.72 (r0) s/step, so r0 alone needed ~3.4 h. Random batches padded every row to
     128 tokens (3-4x the work), all in fp32. Fixed in tern-embed: length-bucketed batches, bf16
     autocast (int8 codes times powers of two stay exact in bf16), and a 3-slot job queue.
     2d (2 x 512) and 2f (looped) now wait for the best of 2a-2c, as the plan says.
   - Relaunched 05:05 UTC, done 06:14 UTC. All 142 files pulled to `tern-embed/pulls/grid1`
     and checked by md5 (archive md5 18e17c3e...).

   **Grid 1 results** (all ternary weights + int8 activations, 12-task subset, 128 tokens):

   | run | shape, recipe | STS | Retrieval | Classif. | Rerank | Pair |
   |---|---|---:|---:|---:|---:|---:|
   | 2a_256 | 2 × 256, final vector | 76.9 | 34.0 | 72.6 | 57.6 | 65.0 |
   | 2b_256 | + layer by layer | 76.7 | 31.8 | 72.3 | 57.4 | 64.8 |
   | 2c_256 | + pairs | 75.2 | 32.7 | 70.6 | 57.6 | 63.2 |
   | **2a_384** | **2 × 384, final vector** | **79.3** | 41.8 | **75.7** | **60.4** | **66.6** |
   | 2b_384 | + layer by layer | 78.9 | 41.3 | 75.3 | 58.8 | 65.7 |
   | 2c_384 | + pairs | 77.9 | 42.5 | 74.8 | 59.4 | 64.7 |
   | r0 | 3 × 384 (bge-micro-v2), full recipe | 79.0 | **43.3** | 75.7 | 60.4 | 66.3 |
   | Ternlight-base | 2 × 384 ternary (reference) | 76.0 | 35.5 | 76.4 | 59.5 | 58.7 |
   | all-MiniLM-L6-v2 | the bar (float) | 78.9 | 46.9 | 80.0 | 63.5 | 67.9 |

   - **Our recipe beats Ternlight at its own shape:** 2a_384 against Ternlight-base is +3.3 STS,
     +6.3 retrieval, +7.9 pair, +0.9 rerank, -0.7 classification. The owner's call was right.
   - **The extras did not help at this budget.** Layer-by-layer targets (2b) and pairs (2c) are
     level or worse than plain final-vector distillation (2a), except +0.7 retrieval for 2c_384.
   - **Width matters most.** 256 is 2.4 STS, 7.8 retrieval and 3.1 classification below 384.
   - **Ternary is nearly free here.** STSB-dev and teacher cosine, float at step 8000 against the
     final ternary model: 2a_256 0.851/0.890 -> 0.850/0.887; 2a_384 0.868/0.958 -> 0.863/0.942;
     r0 0.875/0.945 -> 0.867/0.924. (The float point is half trained, so this is a lower bound.)
   - **Against the bar:** 2a_384 passes STS (79.3 against 78.9) but is below on retrieval (-5.1),
     classification (-4.3), reranking (-3.1) and pair classification (-1.3). The loss is capacity
     and training, not ternary.
   - Cost: pod 01:53-06:30 UTC at $0.22/hr, about $1.02, of which about 2 h idle behind dibs
     blocks. Pod deleted 06:30 UTC after the md5 check; the corpus and teacher vectors were not
     kept (about 25 min to rebuild on a new pod). Runs 1-6 after, only if needed: about 9
   GPU-hours, about $2.
   0. 3 × 384: ternary conversion of bge-micro-v2 (R33 method), distilled from bge-small. It is
      the smallest model that passed the bar in float, so this measures ternary's cost where it
      matters. Proposed next after step 0b.
   **The 2-layer track (owner, 2026-10-16: keep 2 layers; Ternlight's training was the problem).**
   Runs alongside run 0. Every run is float first, then ternary, and is scored on the same 12
   tests. Each fixes one of Ternlight's weak points, so the table shows which one matters:

   **Width 256 is the target** (owner: "great quality and insane speeds"). 2a-2c run at both
   widths, 256 and 384, so the cost of going narrow is measured at every recipe step.

   | run | start from (256 / 384) | distillation | data |
   |---|---|---|---|
   | 2a | 2 of ELECTRA-small's layers / 2 of bge-micro-v2's 3 | final vector (bge-small) | sentences + 128-token passages |
   | 2b | as 2a | + layer by layer (teacher states and attention) | as 2a |
   | 2c | as 2a | as 2b | + query-passage pairs, in-batch negatives |
   | 2d | 2 × 512 from 2 of bge-base's layers, cut from 768 | best of 2a-2c | best of 2a-2c |
   | 2f | 2 × 256, looped twice | best of 2a-2c | best of 2a-2c |

   Estimated throughput: 2 × 256 ~1.1-1.3M tok/s, 2 × 384 ~750k, 2 × 512 ~560k, looped
   2 × 256 ~600k.

   - Ternlight-base (2 × 384) is the reference row: 2a-2c beat it, or our recipe is no better.
   - Teacher costs stay low: bge-small's final vectors are computed once; the layer-by-layer
     targets come from bge-micro-v2, which is only 3 layers.
   - 2 × 512 uses 2 mem tiles per layer, so 2 copies fill the 8 columns exactly.
   - **What decides 2 against 3 layers:** the best 2-layer ternary run against run 0 (ternary
     3 × 384). 2 layers buys about 1.5× throughput (750k against 500k tok/s); the owner weighs
     that against the quality gap.
   1. 8 × 384, column per layer (init from bge-small or MiniLM-L12).
   2. 32 × 192, tile per layer (weight selection from a 384-wide model).
   3. 16 × 256, 2 tiles per layer (init from ELECTRA-small: its 12 layers, 4 of them repeated).
   4. Run 1 with 2 loops per stage.
   5. Run 2 as a ring, 2 passes.
   6. The better of runs 4-5, with early exit.

   Report quality and multiplies per token for each. Gate: the quality bar above. The winner picks
   the NPU design for step 2.
4. **Serve.** Wire it into `/v1/embeddings` as a new backend; benchmark with `oflm bench-embed`.

## Related work (2026-10-17)

Session `openflowlm-next-8f` is adding Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B on the LLM
engine: worktree `C:\code\openflowlm-qwen3-embed`, branch `feat/qwen3-embedding`, an LLM-backed
`AutoEmbeddingModel` subclass. Its requirement IDs are `OPEN-HIDDEN-STATE`, `SERVER-EMBED-QWEN3`,
`SERVER-RERANK`; avoid them. Step 4 here builds on its backend shape if it lands first. Its CPU
reference vectors (a possible stronger teacher later) are in `C:\models\qwen3-embed\ref`.

## Decision for the owner

- **Target:** decided (owner, 2026-10-16): throughput. Latency is not a goal.
- **Model:** decided: our own, sized to fit on chip. Open: the init source (bge-small or
  all-MiniLM-L12-v2), the teacher, and the slot length.
