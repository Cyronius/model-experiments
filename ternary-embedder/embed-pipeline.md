# Plan (draft): a layer-pipelined embedding engine on the NPU

**Original plan:** draft, 2026-10-16. Historical proposals below are superseded
where later measured updates say so.
**Update, 2026-10-22:** Grid 1 is complete (results below). The owner authorized
the remaining shallow experiments: 2d_512 and 2f_256, final-vector-only recipe,
150M tokens each, seed 0. Grid 2 is complete: neither candidate improves the
existing options or clears the quality bar. Isolated RunPod A5000
`m3ruzkkbr88nf6` ran for about 53 minutes at $0.27/hr (about $0.24 compute,
excluding storage). All 54 artifact files were downloaded and SHA256-verified
before deleting the pod. See `README.md`, `grid2-summary.json` and `run_grid2.py`;
the complete NPU encoder is not implemented.
**Update, 2026-10-23:** Native ternary probes for K 384, 512, 1536 and 2048
compiled and ran on Strix, with exact integer outputs. Width 384 has no forced
512 padding in the custom kernel; keep 384 for hardware development. The
512 primitive took 21% longer, and 2048 took 28% longer than 1536. See
`npu-width-probe/README.md` for measurements and their limited scope. The full
encoder and full-array layout are still unimplemented.
**Further update, 2026-10-23:** Complete streamed native-384 dense projections
now pass 36 exact NPU checks against trained 2a_384 activations. FFN-down
preserves all six 256-channel activation-scale groups. A separate packed CPU
oracle checks both layers through final vectors at sequence caps 8/32/128
(worst vector element error 8.65e-7). Quantization/dequantization and nonlinear
ops remain on the CPU, so this does not complete the one-stage hardware gate.
See `native-stage/README.md`, `oracle-result.json` and `projection-result.json`
in that directory for reproduction, scope, compiled budgets and raw samples.
**Current project:** `C:\code\model-experiments\ternary-embedder`.
**Quality-control update, 2026-10-23:** The matched 150M-token float-only
2a_384 control is complete. On one consistent CPU evaluator, mean retrieval
nDCG@10 is 46.920 MiniLM, 44.436 float-only and 42.132 original QAT. The
current quantized training recipe costs 2.304 points (paired query-bootstrap
95% CI 1.320-3.277), while a 2.484-point gap is already present in float.
Int8 activation quantization has no detected retrieval cost after QAT in this
subset. This supersedes the earlier inference that the loss was "not ternary".
See `retrieval-diagnostics/README.md` and its retained reports. The isolated
A5000 cost about $0.193 compute, excluding storage; 26 artifacts were
SHA256-verified before pod deletion. No further training is queued.
**Historical worktree:** `C:\code\openflowlm-embed`, branch `feat/embed-pipeline` (untied, from `upstream/main` 99fe7e4d); no longer present locally.
**Complete encoder update, 2026-10-23:** Both native-384 encoder layers now run
on 32 NPU cores (eight columns by four rows) in one dispatch. All 12 projections,
quantizers, attention, GELU and eps=1e-12 post-residual LayerNorms run on the
array. Tokenization, input embeddings/norm, pooling and the output projection
remain on the CPU and are included in the timings. Final-build text-to-vector
medians: 6.587 ms for four texts at cap 32 (607.3/s), 4.135 ms for one text at
cap 128 (241.8/s). Bulk cap-128 encoding reaches 326–333/s. Earlier builds and
timing passes were faster; their code identities and observations are retained
alongside the final build to show shared-machine variability. Actual NPU mean
retrieval is 41.9409 versus 42.1322 on CPU for the same checkpoint. Program text
is 16096 B/core; full routing, batch isolation and NPU retrieval have passed. Weights
still stream and K/V uses DDR scratch; replicas, resident weights and pooled
NPU output are unimplemented. The older MiniLM 1305/s NPU result is a different
batch/workload and has not been rerun; fresh CPU MiniLM timing is reported
separately. See `native-stage/README.md`, `fused-result.json` and
`fused-retrieval.json`. The owner considers the 46.92 / 44.44 / 42.13 quality
gap modest: the acceptance decision is speed versus quality, not a requirement
to beat MiniLM's retrieval score before measuring performance.
**Spec impact:** new requirements, named when step 2 starts. No embedding-engine spec exists yet
(`specs/open-engine` has only `OPEN-EMBED-STRIDE`; `specs/server-api` has `SERVER-EMBED-TASK-PROMPT`).

## Goal

The fastest text-embedding engine on the Strix NPU. The owner's idea: pipeline the model across the
array, so each part of the array runs a different layer or task, and data flows between them.

## XDNA evidence review (2026-10-23)

Owner goal: a smart embedder with very high throughput. Keep 2a_384 as the current
two-layer quality candidate, while measuring the hardware design before fixing
the replica count or claiming a throughput. The notes repository is
`C:/code/xdna2-notes` (`C:/code/xdna-notes` does not exist on this machine).
The following are historical measurements on their stated stacks, not results
for this embedder. The implementation choices below are proposals.

- **A native 384 kernel is necessary to avoid known software padding.**
  [F-0347](../../xdna2-notes/findings/F-0347-dit-gemm-shape-padding.md) records
  `dit_gemm` requirements M % 512, K % 512, N % 1024. Calculated for our FFN,
  reusing those rules would turn 384 -> 1536 into 512 -> 2048 (1.78x the useful
  arithmetic), and 1536 -> 384 into 1536 -> 1024 (2.67x), before row padding.
  These are kernel constraints, not a hardware requirement that width be 512.
  [F-0346](../../xdna2-notes/findings/F-0346-dit-gemm-small-k-low-rate.md) also
  shows poor small-K rates: do not extrapolate from large GEMMs.
- **Four tiles per column and four complete replicas are unvalidated layouts.**
  [F-0313](../../xdna2-notes/findings/F-0313-memtile-split-reaches-dma-floor.md)
  measured gains from two or three compute cores per column, but a four-core
  streaming graph failed to route.
  [F-0217](../../xdna2-notes/findings/F-0217-memtile-dma-channel-cap.md) records
  six input and six output DMA channels per memtile; the failed graph needed
  nine output channels. This is not a universal four-core ban: a resident-weight
  graph may have different channel requirements. Count every split, join,
  weight and activation stream before choosing a topology. The embedding raw
  note also measured a 16-core matmul faster than its 32-core version.
- **Fuse the two-layer walk and pooling in one context where resources allow.**
  [R-0003](../../xdna2-notes/recommendations/R-0003-layer-as-one-context.md) and
  [R-0008](../../xdna2-notes/recommendations/R-0008-context-switching-and-dispatch-cost.md)
  show why dispatch/context count must be budgeted. Historical direct C++ calls
  cost about 0.077 ms versus 0.47 ms through IRON's Python path
  ([F-0179](../../xdna2-notes/findings/F-0179-iron-python-dispatch-overhead.md));
  those floors must be remeasured for our image. Aim for one batched inference
  dispatch after weight load, not a host round trip for each projection or norm.
- **Choose batch/sequence buckets and avoid padded work.**
  [F-0329](../../xdna2-notes/findings/F-0329-embedding-batch-tiers.md) measured
  1.650 -> 0.090 s for one bge-large text using batch tiers, with bit-identical
  vectors. This is a different engine/model, so the 18.3x gain is not transferable.
  It supports measuring several request sizes. For packed short texts, preserve
  per-text attention masks, position numbering and pooling boundaries.
- **Ternary matmuls alone do not guarantee speed or preserve full-model quality.**
  [F-0317](../../xdna2-notes/findings/F-0317-ternary-2bit-gemv.md) and the
  [tiny-planner ablations](../../xdna2-notes/raw/openflowlm-next/2026-10-09-wt-tiny-planner-decoder-on-npu-dispatch-floor-ternary.md)
  put much of the remaining time in streams, attention, scaling, quantization
  and normalization. Balance these tasks, not just weight counts. Keep 2-bit
  packed weights and measure a DMA-only/compute-disabled build alongside the
  real kernel to identify the actual bottleneck.
- **Gate faster arithmetic on embedding quality.** F-0329 records bge-small
  failing its bfp16 MTEB gate by 0.5010 points. Our training simulates ternary
  linear weights and int8 inputs, but does not prove equivalence to every NPU
  GELU, softmax or norm implementation. Check intermediate states, final vectors
  and the same MTEB subset before accepting approximations. Set the intended
  rounding mode explicitly
  ([F-0074](../../xdna2-notes/findings/F-0074-core-rounding-floor.md)); avoid
  naive fp32 vector multiplication
  ([F-0070](../../xdna2-notes/findings/F-0070-fp32-vector-multiply-emulated.md)).

Native-width arithmetic gate is complete (`npu-width-probe/README.md`). Next
hardware gate: compile and validate a complete native-384 stage, with explicit
L1/program/memtile/DMA budgets, then compare feasible tile mappings on the same
real activations. Measure whole-layer time, reductions/joins and host overhead.
Only then extend to two layers, pooling, packed batches and replicas. Continue
to report the CPU model's quality separately from any NPU numerical degradation.
The ~750k tok/s estimate below is still an unvalidated planning estimate.

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
- **Original quality target (superseded on 2026-10-23):** beat MiniLM on STS and
  retrieval. The current decision instead weighs the measured retrieval loss
  against complete-pipeline throughput. The historical MiniLM 1305/s number
  needs a matched-workload rerun before claiming an NPU speed advantage.

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
   - **Early STS/teacher-cosine proxies (not a retrieval control).** Float at step 8000 against the
     final ternary model: 2a_256 0.851/0.890 -> 0.850/0.887; 2a_384 0.868/0.958 -> 0.863/0.942;
     r0 0.875/0.945 -> 0.867/0.924. The float point is half trained; these proxies
     do not isolate quantization or establish its retrieval cost.
   - **Against the bar:** 2a_384 passes STS (79.3 against 78.9) but is below on retrieval (-5.1),
     classification (-4.3), reranking (-3.1) and pair classification (-1.3).
     The later matched-budget control below separates a float-student gap from
     a further loss caused by the current quantized training recipe.
   - Cost: pod 01:53-06:30 UTC at $0.22/hr, about $1.02, of which about 2 h idle behind dibs
     blocks. Pod deleted 06:30 UTC after the md5 check; the corpus and teacher vectors were not
     kept (about 25 min to rebuild on a new pod). Runs 1-6 after, only if needed: about 9
   GPU-hours, about $2.
   **Grid 2 results, 2026-10-22:** 2d_512 and 2f_256 used the winning 2a recipe
   (final-vector distillation only), 150M tokens, batch 256, seed 0, 50% float
   followed by 50% ternary. The corpus rebuilt to the same 4,252,720 texts,
   2,337,599 pairs and 145,428,260 tokens. Both new shapes passed GPU smoke
   training, checkpoint reload and STS evaluation before their full runs.

   | run | STS | Retrieval | Classif. | Rerank | Pair |
   |---|---:|---:|---:|---:|---:|
   | 2d_512 | 78.9 | 41.2 | 75.6 | 59.8 | 65.9 |
   | 2f_256 | 76.4 | 33.2 | 72.2 | 56.9 | 64.7 |

   - 2d_512 starts from bge-base layers 0 and 11, cut from 768 to 512. It is
     below 2a_384 in all five categories. This tests the planned wider recipe,
     not width in isolation, because its initialization also differs.
   - 2f_256 starts from ELECTRA layers 0 and 1, applied twice with shared weights.
     It is below 2a_256 in every category despite doubling the layer passes.
   - Training took 607 s for 2d_512 and 785 s for 2f_256. Final STSB-dev / held
     teacher cosine: 0.8630 / 0.9403 and 0.8480 / 0.8782 respectively.
   - Both fail the STS-plus-mean-retrieval target. 2d_512's rounded STS tie with
     MiniLM hides a small deficit (78.8620 versus 78.9365).
   - Artifacts: `tern-embed/pulls/grid2` (54 files, SHA256 verified), including
     final/float/last checkpoints, logs, task results, source and environment.
     Exact comparisons: `grid2-summary.json`. Pod deleted; no new training queued.
   - Keep 2a_384 as the best balanced two-layer ternary candidate. The matched
     float control described next is complete. Longer ternary budgets and an
     isolated retrieval-objective comparison remain untested. Depth is a fallback.

   **Matched float control and retrieval diagnosis, 2026-10-23:** Same
   initialization, seed 0, corpus construction/counts, batch 256, vector loss,
   learning-rate schedule and 17,134 updates (150M-token target); quantization
   stays off throughout the control. Before the QAT switch their float training
   traces reproduce closely. Current CPU retrieval scores:

   | model | SciFact | NFCorpus | mean |
   |---|---:|---:|---:|
   | MiniLM-L6 | 62.386 | 31.453 | 46.920 |
   | float-only 2a_384, 150M tokens | 57.403 | 31.470 | 44.436 |
   | original QAT 2a_384, 150M tokens | 54.856 | 29.409 | 42.132 |

   The float gap is 2.484 points; the current ternary training recipe adds
   2.304. Turning off activation quantization with ternary weights does not
   recover the gap. Converting the 75M float checkpoint without adaptation
   drops mean retrieval from 43.797 to 25.369; QAT recovers most of that damage.
   Turning off quantization on QAT-trained master weights is not a float-only
   control (mean score 1.402). Corpus centering also hurts retrieval, so a
   uniform spacing correction is not sufficient. This single-seed study does
   not establish a ceiling for shallow or ternary models. Details, intervals,
   ranking examples and source/environment snapshots are in
   `retrieval-diagnostics/`. Archived GPU scores above and current CPU scores
   differ slightly; all new paired comparisons use the same current CPU stack.

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
