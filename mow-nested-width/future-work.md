# Future work

Ideas parked during the main MoW build. Nothing here blocks the current grid.

---

## 1. Why does 5–8 digit addition never get a single answer fully right?

**Status:** open question. Deferred until the main grid finishes.

### What we already know

The model partly learns hard addition but never completes a whole answer. Loss sits
around 1.35. Exact-match is 0%.

A loss of 1.35 means each single digit is right roughly half the time. An 8-digit
answer needs every digit right. Half, multiplied by itself eight times, is under one
percent. So exact-match rounds to zero.

**MLP width has already been ruled out.** Five runs, same task and recipe, width the
only difference:

| MLP width | loss on 5–8 digit addition | exact-match |
|---|---|---|
| 16 | 1.2815 | 0% |
| 32 | 1.4631 | 0% |
| 64 | 1.1380 | 0% |
| 128 | 1.4225 | 0% |
| 512 | 1.3961 | 0% |

Width changed by 32 times and nothing moved. The narrowest model beat the widest one.
Loss shows no trend. So the limit is somewhere else.

Data lives in `mow/runs_widthtest/`.

### Things to test, cheapest first

**1. Training length.** ~~The model saw about 96,000 hard-addition examples.~~
**Partly tested at 24k steps. Doubling the budget did not move hard addition.**
See the depth-by-steps table under test 2. Longer runs (50k, 200k) on hard addition
alone are still untested.

**2. Number of layers.** ~~The model has 4. Carrying a digit is a chain, and each link
may need its own layer.~~ **Tested. Depth did not help at a fixed step budget.**

Run 8 layers first, and only work back to 6 and 4 if 8 helped. It did not, so 6 was
never run. A 4-layer control was run next to it because the task mix grew from 4
cells to 6 when recall was added, which makes the width table above an unsafe
baseline.

The 12k pair showed 8 layers behind 4, but the 8-layer loss was still falling twice
as fast at the end. So the pair was re-run at 24k steps, which also answers test 1.

| layers | steps | arith_hard loss | arith_hard exact | overall loss | wall |
|---|---|---|---|---|---|
| 4 | 12k | 1.5236 | 0.8% | 0.5591 | 13.3 min |
| 8 | 12k | 1.7085 | 0% | 0.6481 | 22.2 min |
| 4 | 24k | 1.5330 | 0% | 0.4644 | 19.1 min |
| 8 | 24k | 1.4699 | 0% | 0.4616 | 35.6 min |

All four used dense width 512, `d_model` 128, seed 0. Data lives in
`mow/runs_layertest/`; the runner is `mow/run_layertest.py`. The 0.8% exact-match is
one example out of 128, so it is noise, not partial success.

Depth only looked harmful because 8 layers converge slower. At 24k the two depths
land in the same place. Hard-addition loss stays near 1.5 across all four runs.

**The 24k budget went somewhere — just not to addition.** Overall loss improved
because `recall_hard` improved. Nothing else moved.

| cell | L4 12k | L4 24k |
|---|---|---|
| recall_hard loss | 1.8238 | 1.2496 |
| recall_hard exact | 23% | 51% |
| arith_hard loss | 1.5236 | 1.5330 |
| arith_hard exact | 0.8% | 0% |

So memorization keeps absorbing extra training. Hard addition does not.

### Where this leaves the question

Every capacity knob has now been tested and none of them moves hard addition:

| knob | range tested | effect on arith_hard |
|---|---|---|
| MLP width | 16 to 512 (32x) | none |
| depth | 4 to 8 layers (2x) | none, once converged |
| steps | 12k to 24k (2x) | none |
| d_model | 128 to 256 (2x) | worse |

So the answer is not more capacity. The next section says what it is instead.

---

## 1b. It is a positional addressing failure, not a capacity limit

Diagnosed on the `lt_L4_24k` checkpoint by decoding the 512 held-out arith_hard
examples and scoring each answer digit separately.

Digit accuracy is U-shaped by position. The first digit emitted and the last few
are far above chance. Everything in the middle is at chance, which is 10%.

| answer digit (0 = least significant, emitted first) | accuracy | n |
|---|---|---|
| 0 | 54.1% | 512 |
| 1 | 9.6% | 512 |
| 2 | 7.8% | 512 |
| 3 | 11.5% | 512 |
| 4 | 28.3% | 512 |
| 5 | 42.1% | 504 |
| 6 | 48.7% | 413 |
| 7 | 82.0% | 261 |
| 8 | 100.0% | 23 |

**This rules out the carry-chain story.** Digit 0 is the units digit. It needs no
carry at all, and the model still gets it wrong 46% of the time. Meanwhile digit 7
needs the whole carry chain and is right 82% of the time. Depth cannot be the
missing piece when the shallowest digit fails hardest.

**It also rules out error compounding.** Teacher-forced accuracy, where the model is
handed the correct previous digits, matches greedy decoding position for position.
The model is not being derailed by its own mistakes. It cannot compute the digits.

**The pattern matches token addressing.** The digits it gets right are the ones next
to a delimiter. In `A<rev a>+<rev b>=`, the least-significant digits sit beside `A`
and `+`, and the most-significant digits sit beside `+` and `=`. Middle digits have
no landmark, so reaching the i-th digit of a span means counting positions. Position
information here is a learned absolute embedding (`model.py:254`), which gives the
model no way to express "i places after the `+`".

**Alignment across operands of different length is not the cause.** If it were,
equal-length operands would do much better. They do not:

| \|len(a) - len(b)\| | n | teacher-forced digit accuracy |
|---|---|---|
| 0 | 132 | 31.7% |
| 1 | 188 | 37.5% |
| 2 | 111 | 33.2% |
| 3 | 81 | 22.2% |

Diagnostic scripts are in the scratchpad, not the repo: `diag_arith.py` and
`diag_align.py`. Move them into `mow/` if this gets picked up again.

### Confirmed: zero-padding the operands solves the task completely

Both operands zero-padded to 8 digits, so every operand digit sits at a constant
absolute index. Nothing else changed. Same 4 layers, same 24k steps, same seed,
same width, same `d_model`.

| | control | padded operands |
|---|---|---|
| arith_hard loss | 1.5330 | 0.0001 |
| arith_hard exact | 0% | **100%** |
| overall loss | 0.4644 | 0.2104 |
| overall exact | 75.1% | 91.7% |

Every digit position is now 100%, including the middle ones that were at chance.
All 512 held-out examples decode exactly right, at every answer length from 5 to 9
digits. `recall_hard` is unchanged (1.2496 vs 1.2611), so nothing else moved.

Run dir `mow/runs_layertest/lt_L4_24k_pad8/`. Enabled by `DataConfig.arith_pad_digits`,
which defaults to 0 and so leaves every grid run untouched.

### What this means

Hard addition was never a capacity problem. Width, depth, steps and `d_model` all
failed because none of them addresses the actual limit: with learned absolute
position embeddings, the model cannot locate the i-th digit of a variable-length
span. Give it fixed positions and a 0.82M-parameter, 4-layer model solves 8-digit
addition perfectly.

The width sweep's conclusion still stands, but its scope was wrong. It showed width
does not fix hard addition. The reason is that nothing about the MLP could.

---

## 1c. Switching the grid to padded operands: what it took

Padding alone is not enough. The grid was re-run with `arith_pad_digits=8` (78 runs,
`mow/runs/`, old results archived to `mow/runs_unpadded/`) and 44 of 78 runs came
back with a broken arith_hard cell.

### The stuck-digit lottery

Training reaches one of two outcomes: 100% exact, or one digit position frozen at
chance for the rest of the run. Failure losses cluster at 0.28 and 0.56, and one
stuck digit out of ~8 answer tokens costs `ln(10)/8.3 = 0.277` nats. So those are
exactly one and exactly two stuck positions.

It is a seed lottery, not a width or condition effect. arith_hard exact-match on the
dense baselines, learned positions:

| width | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 8 | 100% | 100% | 100% |
| 16 | 100% | 100% | 12.5% |
| 32 | 100% | 7.8% | 73.4% |
| 64 | 5.5% | 100% | 12.5% |

No cheap recipe change fixes it. Each rescues a different run:

| (width, seed) | default 12k | 24k steps | wd 0.1 | warmup 1000 |
|---|---|---|---|---|
| w16, s2 | 12.5% | 100% | 12.5% | 12.5% |
| w32, s1 | 7.8% | 13.3% | 3.9% | 87.5% |
| w64, s0 | 5.5% | 100% | 100% | 5.5% |

**Cause.** Learned absolute positions give no sharing across output positions. The
circuit producing answer digit i must be learned separately for every i, so each
digit is an independent lottery ticket. Padding fixed addressing but not this.

### Rotary position embeddings fix it

Added as `ModelConfig.pos_encoding` (`learned` | `rope`), defaulting to `learned`.
Rotary makes attention depend on the query-key offset, so one "read the operand
digit k places back" circuit serves every digit position at once.

arith_hard exact-match across all four combinations, 4 widths x 3 seeds:

| width | rope + pad | learned + pad | rope + unpad | learned + unpad |
|---|---|---|---|---|
| 8 | 100/100/100 | 100/100/100 | 0/0/0 | 0/0/0 |
| 16 | 100/100/100 | 100/100/12.5 | 0/0/0 | 0/0/0 |
| 32 | 100/100/100 | 100/7.8/73.4 | 0/0/0 | 0/0/0 |
| 64 | 100/100/100 | 5.5/100/12.5 | 0/0/0.8 | 0/0/0 |

**Both changes are needed, and they fix different things.**

- Padding fixes addressing. Rotary cannot: the offset to operand b's i-th digit
  depends on `len(a)`, so it varies per example and no fixed relative offset reaches
  it. Rotary without padding is 0%, exactly like the original.
- Rotary fixes the lottery. Padding cannot: it pins positions but still leaves each
  output position to learn its own circuit.

Rotary also drops the 32x128 absolute position table, so it is 4096 parameters
cheaper. recall_hard is unaffected either way, as expected.

Run data: `mow/runs_ropetest/` (rope + pad), `mow/runs_ropenopad/` (rope + unpad),
`mow/runs_fixsearch/` (recipe search).

### Recommendation

Set `pos_encoding="rope"` as the default and re-run the 78-run grid, roughly 3
hours. Padding is already the default. Every condition shares the trunk equally, so
comparisons stay fair, and the current `mow/runs/` is not usable for arith_hard.

### Open decision for the main study

`arith_hard` currently contributes a permanent 0% cell to the grid. Two options:

- **Keep the task as is.** The 0% is now explained rather than mysterious, and that
  explanation is worth a paragraph in the README.
- **Switch addition to padded operands.** This gives the difficulty axis a real
  gradient, which is what section 1 wanted for figure 3. It invalidates every
  existing arith cell in `mow/runs/` and needs a full grid re-run.

A third path fixes the cause rather than the task: rotary or relative position
embeddings, which would let variable-width operands work. That is a larger change
and it alters the trunk every condition shares.

**3. Digit count.** Find where it breaks. Train separate runs on 2, 3, 4, 5, and 6
digit addition.

- Cost: about 10 minutes each.
- What this gives: the exact digit count where exact-match falls off. That tells us
  whether the problem is a hard wall or a slow decline.

**4. Model size overall.** ~~Raise `d_model` from 128 to 256 and re-test.~~
**Tested. `d_model` 256 is worse than 128, not better.**

4 layers, 24k steps, seed 0, `d_ff` held at 512 so this moves the residual stream
rather than the MLP again. Control is the `d_model` 128 run from test 2.

| d_model | params | arith_hard loss | arith_hard exact | overall loss | wall |
|---|---|---|---|---|---|
| 128 | 0.82M | 1.5330 | 0% | 0.4644 | 19.1 min |
| 256 | 2.16M | 1.7706 | 0% | 0.5942 | 23.2 min |

Every cell got worse, `recall_hard` most of all (51% exact down to 27%). Unlike the
8-layer case this is not clearly undertraining: over the last 3000 steps the 256
model's arith_hard loss fell 0.016 per 1000 steps, slower than the 128 model's
0.024. It is not on track to overtake.

Run dir `mow/runs_layertest/lt_L4_d256_24k/`.

### Why it matters

If a cause is found and fixed, hard addition becomes a second difficulty axis with a
real gradient. Right now only the memorization family gives one. Two independent
axes would make figure 3 much stronger.

If no cause is found at this model size, that is worth writing down too. It would say
the task needs a bigger model than this study uses.

---

## 2. Items already deferred in the main plan

From `mow-experiment-plan.md`, Phase 4. Repeated here so there is one list.

- **MoE-of-MoW hybrid.** Compose `MoEMLP` over `MoWMLP`, with two routers (expert and
  width). The classes were built to allow this. Only worth doing if the phase 1
  routers train stably.
- **Wall-clock benchmark at batch size 1.** Compare slicing a nested prefix against
  gathering and scattering experts. This is the deployment argument, and it cannot be
  answered by FLOP counting alone.
- **Speculative decoding.** Use the narrowest tier as the draft model.

---

## 3. Re-run the whole grid on a GPU

The current numbers come from CPU only. There is no usable GPU on this machine.

`config.PRESETS["full"]` already holds the larger trunk from the original plan. On a
CUDA machine the same grid should run in about 20 minutes instead of hours.

Worth doing to check whether any finding here is an artifact of the small model size.
The width result in particular deserves a re-test at larger scale.

---

## 4. The suite has no width axis at the `small` preset

Found while deciding whether a saturated `arith_hard` is worth keeping. It is the
wrong question: no cell in the suite responds to width, so dropping or keeping
addition changes nothing.

Two independent measurements agree.

**Dense ladder** — separate models, one per width, rope + padded, mean of 3 seeds:

| cell | h8 | h16 | h32 | h64 |
|---|---|---|---|---|
| arith_easy | 0.0001 | 0.0001 | 0.0000 | 0.0000 |
| arith_hard | 0.0006 | 0.0001 | 0.0002 | 0.0002 |
| dyck_easy | 0.0001 | 0.0000 | 0.0000 | 0.0002 |
| dyck_hard | 0.0014 | 0.0001 | 0.0001 | 0.0010 |
| recall_easy | 0.0042 | 0.0039 | 0.0036 | 0.0028 |
| recall_hard | 2.2655 | 2.2751 | 2.1481 | 2.0826 |
| **overall** | **0.3787** | 0.3799 | 0.3587 | **0.3478** |

**Fixed-tier sweep** — one nested model evaluated at each tier, which is figure 1's
own data, `mow_pretrain12k`, mean of 3 seeds:

| cell | h8 | h16 | h32 | h64 |
|---|---|---|---|---|
| arith_hard | 0.2834 | 0.2815 | 0.2795 | 0.2788 |
| dyck_hard | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| recall_easy | 0.0096 | 0.0063 | 0.0046 | 0.0033 |
| recall_hard | 2.1419 | 2.1202 | 2.1033 | 2.0775 |
| **overall** | **0.4474** | 0.4429 | 0.4394 | **0.4348** |

Eight times the width buys 8% on the dense ladder and 3% on the nested sweep. Every
cell is either free at h8 or impossible at h64. Nothing sits in between.

The dense ladder rules out nesting as the explanation: those are independent models,
never trained to share a prefix, and they are just as flat.

### Why this breaks figures 1 through 5

- **Figure 1, nesting tax.** The tax is the gap between a nested tier and its dense
  equivalent. Both are flat, so the figure compares two flat lines.
- **Figure 2, frontier.** The router will drop to the narrowest tier at almost no
  loss, giving 25% FLOPs for nearly free. True, but a degenerate frontier.
- **Figures 3-5, routing vs difficulty.** There is no width-difficulty signal in the
  data for the router to discover.

### Why it happens

The window where width matters sits between trivial and impossible, and no cell is
in it. `recall_hard` memorizes the m=32 and m=64 tables, which is 5120 associations
of two 64-way symbols, about 61 kbit. The widest tier has 256*64 = 16k MLP
parameters per layer. It is not close, so recall fails at every tier. Everything
else is solved at h=8, so it succeeds at every tier.

### What would fix it, cheapest first

1. **Calibrate the recall tables.** `recall_easy` (m=16, 256 associations) is free at
   every tier; `recall_hard` (m=32, 64) is impossible at every tier. The
   discriminating size is between them. A sweep of m in {20, 24, 28} at the two
   extreme widths, 6 runs of about 11 minutes, locates it.
2. **Raise the trunk to `PRESETS["full"]`** (`d_ff` 1024, tiers 256-1024). At 262k
   MLP parameters per layer, m=64 becomes plausibly memorizable and the tier ladder
   should separate. This is section 3's GPU plan.
3. **Add a width-hungry cell by construction** rather than by tuning an existing one.

Option 1 is the cheapest test and does not need a GPU.

### Calibration result: m=32 is the width axis

Sweep of pure recall table sizes at the two extreme widths, rope + padded, seed 0,
12k steps. `mow/runs_recalcal/`.

| m | associations | h8 loss | h8 exact | h64 loss | h64 exact |
|---|---|---|---|---|---|
| 16 | 256 | 0.0000 | 100% | 0.0000 | 100% |
| 20 | 400 | 0.0098 | 100% | 0.0000 | 100% |
| 24 | 576 | 0.3286 | 69.5% | 0.0002 | 100% |
| 28 | 784 | 0.7041 | 51.6% | 0.0086 | 100% |
| 32 | 1024 | 1.1947 | 20.3% | 0.2174 | 82.8% |

**The earlier diagnosis was half wrong, in a useful way.** m=32 is not out of reach
at h64 — it reaches 82.8%. What made `recall_hard` flat was mixing m=32 with m=64 in
the same cell. The m=64 table (4096 associations) is genuinely impossible at this
scale, and carrying it dragged the whole cell to the floor at every tier, hiding the
gradient that m=32 alone provides.

**m=32 alone is the pick.** It is the only size where neither end saturates, so all
four tiers have room to separate: 20.3% to 82.8% exact, and 1.19 to 0.22 in loss.
m=24 and m=28 both peg h64 at 100% and would waste the top of the ladder.

Applied as `DataConfig.recall_hard_m = (32,)`.

### Configuration the grid should now run

| setting | value | why |
|---|---|---|
| `arith_pad_digits` | 8 | otherwise addition is 0% at every setting (section 1b) |
| `pos_encoding` | `rope` | otherwise addition is a per-digit lottery (section 1c) |
| `recall_hard_m` | `(32,)` | otherwise there is no width axis at all (this section) |

All three are defaults now. Setting them to `0`, `"learned"` and `(32, 64)`
reproduces the original behaviour exactly.
