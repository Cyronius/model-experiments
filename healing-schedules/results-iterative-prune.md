# Results: prune in small steps, healing between them

Plan and predictions: [plan-iterative-prune.md](plan-iterative-prune.md), written before any run and
pre-registered in `protocol.json` (`followup_iterative_prune`). Run 2026-11-16/17 on one rented
H100 NVL ($20 cap). Raw records: `runs/iterative-prune/`.

## Question

Does removing 25% of Qwen3.5-2B in steps, healing after each, beat removing 25% at once and healing
once, at the same total training compute?

## Answer

**No, not in output quality.** At 6e15 training FLOPs every staged arm ends within seed noise of
one-shot pruning. **Staging does keep the inside of the model closer to the teacher**: the more
steps, the closer every layer's hidden state stays to the teacher's.

## Arms

Every arm ends at the same geometry: the light student (residual 2048→1792, FFN 6144→4800,
1,473,244,736 parameters). Each one is healed with joint-cosine (lr 3e-5, batch 1), and the budget is
split equally across stages. Stages continue the data stream where the last one stopped. The
staged arms re-prune the healed model using its own activation statistics.

| Arm (6e15 FLOPs) | Val CE | Test CE | Tokens | ARC-E | ARC-C | MMLU | GSM8K | Stream error | Stream cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| One-shot, seed 17 | 3.142 | 3.117 | 323,584 | .520 | .305 | .310 | .250 | .221 | .889 |
| One-shot, seed 2 | 3.122 | **3.093** | 323,584 | .480 | .325 | .315 | .188 | .224 | .887 |
| S2 (12.5 → 25%) | 3.143 | 3.135 | 313,344 | .490 | .270 | .310 | .188 | .214 | .890 |
| S3 (10 → 20 → 25%) | 3.134 | 3.117 | 311,296 | .500 | .305 | .320 | .281 | .201 | .895 |
| S5 (5% steps to 25%) | **3.122** | 3.099 | 303,104 | .500 | .295 | .315 | .188 | **.196** | **.897** |
| One-shot 1× (local, earlier) | 3.142 | | | .525 | .315 | .320 | .250 | .221 | .889 |

One-shot seed 17 is the second pod's rerun on the fast kernel. The first pod's run on the torch
fallback gave val 3.141 and test 3.117, the same result.

The benchmarks are the quick evaluation: the test split, 200 questions per multiple-choice task and
32 GSM8K problems (1024 tokens), all on the frozen subsets. With 200 questions, one standard error
is about ±0.035; one GSM8K problem is 0.031. No benchmark difference in this table is meaningful.

The stream columns come from `block_probe.py`. Stream error is the relative error of the student's
residual stream against the teacher's, at the kept coordinates, averaged over the 24 layers. Stream
cosine is the matching cosine.

## Damage at each cut

Val CE right after each cut (before healing) and after that stage's healing:

| Arm | Stage cuts: val CE after the cut → after healing |
|---|---|
| One-shot | 25%: 3.660 → 3.122–3.142 |
| S2 | 12.5%: 2.948 → 2.828; 25%: 3.205 → 3.143 |
| S3 | 10%: 2.855 → 2.758; 20%: 3.056 → 3.002; 25%: 3.151 → 3.134 |
| S5 | 5%: 2.671 → 2.634; 10%: 2.773 → 2.751; 15%: 2.914 → 2.891; 20%: 3.030 → 2.989; 25%: 3.140 → 3.122 |

The teacher's val CE is 2.550. A healed model takes the next cut well: every step costs 0.38 nats
or less, against 1.11 for the one-shot 25% cut. But each stage's short heal only recovers part
of that, and the arms converge on the same final loss.

## Against the predictions

| Prediction (written before the runs) | Outcome |
|---|---|
| S3 ends within 0.03 of one-shot at 6e15 | **Confirmed.** 3.134 against 3.122 and 3.142 for the two one-shot seeds. |
| Each cut of a healed model costs ≤ 0.4 val CE | **Confirmed.** The largest was 0.38 (S2's second cut); 5% cuts cost 0.10–0.14. |
| S5 is no better than S3 | **Confirmed within noise.** S5 is 0.012 better on val CE and 0.018 on test CE, smaller than the 0.02 gap between one-shot seeds. Finer steps did not hurt either. |
| If staging helps, it helps more at 10× | **Not tested.** The 10× arms did not fit before the deadline. |

## What staging changes: the internals

The block probe compares every layer with the teacher. Measures:

- **Stream:** the student's residual stream against the teacher's, at the kept coordinates.
- **Block:** each block fed the teacher's own input.

| Model | Stream error | Stream cosine | Layer-0 block error | Mean block error |
|---|---:|---:|---:|---:|
| Unhealed one-shot | .274 | .857 | .079 | .034 |
| One-shot 1× (seed 2) | .224 | .887 | .082 | .036 |
| One-shot 1× (seed 17) | .221 | .889 | .081 | .036 |
| One-shot 10× (local) | .262 | .878 | .123 | .044 |
| S2 | .214 | .890 | .075 | .035 |
| S3 | .201 | .895 | .074 | .034 |
| S5 | .196 | .897 | .074 | .034 |

Staged models end with blocks that still compute what the teacher's blocks compute (block error at
or below the unhealed level). Their hidden states also track the teacher's at every depth. One-shot
healing reaches the same output by letting the blocks co-adapt and drift. More training (10×) makes
that drift larger.

**The staged arms keep almost the same coordinates.** Overlap of the final selection with the
one-shot selection:

| Arm | Residual coordinates | FFN neurons (mean over layers) |
|---|---:|---:|
| S2 | 97.0% | 96.0% |
| S3 | 97.3% | 95.9% |
| S5 | 97.8% | 96.2% |

So the difference between the arms lies in the weights, not in which parts they kept.

### Where the closeness is: by depth

Stream error (relative error against the teacher's hidden state) at selected layers:

| Layer | 0 | 5 | 8 | 11 | 14 | 17 | 20 | 23 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unhealed | .079 | .204 | .244 | .319 | .308 | .340 | .343 | .399 |
| One-shot, seed 17 | .083 | .226 | .254 | .278 | .233 | .217 | .215 | .216 |
| One-shot, seed 2 | .084 | .233 | .257 | .283 | .236 | .218 | .214 | .216 |
| S5 | **.073** | **.186** | **.211** | **.239** | **.210** | .203 | .205 | .214 |
| One-shot 10× | .134 | .297 | .327 | .349 | .264 | .225 | .216 | .204 |

- **Beyond seed noise.** The two one-shot seeds agree within .003–.007 at every layer.
  S5 is about .04 lower through layers 5–11, roughly 15–18% closer to the teacher.
- **Gone by the output.** From layer 17 on, every model sits near .21. The healing loss targets
  only the final-layer hidden state and the output distribution, so every arm gets there. How the
  middle layers get there is left free, and that is where the arms differ.
- **Two routes to the same output.**
  - One-shot healing compensates for a large cut by changing every block. Its early layers end
    slightly further from the teacher than the unhealed model, and 10× training moves them further.
  - S5 absorbs small cuts one at a time, and its blocks stay at or below the unhealed block error.
- **Direction, not scale.** One-shot healing restored the early layers' hidden-state size to the
  teacher's (norm ratio about 1.0). S5's stayed about 9% smaller (about 0.91, like the unhealed
  model). S5's advantage is in direction: cosine .902 against .890 at layer 5.

## What this means

- For the final loss at this budget, one-shot pruning is as good as gradual pruning and simpler.
  This agrees with Minitron at light sparsity.
- **Teacher-like blocks did not buy a better output here.** One argument for block-wise healing
  before a joint run is that it leaves blocks close to the teacher's and so gives the joint run a
  better start. S5 produced that condition another way: small cuts, each healed jointly, with no
  block-level loss. It still ended at one-shot's loss. At light sparsity and equal compute, block
  closeness alone is not worth output quality. Block-local losses could get closer than S5 did,
  and that remains untested; it is a separate experiment.
- Gradual pruning may still matter where block-level fidelity is used directly: grafting or merging
  teacher components (for example dense→MoE expert transplants), or layer-local distillation.
  These are untested.
- The heavy cut (63.7%, the 0.8B size) is where staging has the most room, as the plan says. One-shot
  heavy healing stalls near 4.8 val CE.

## Not run

S3-10× and one-shot-10× were skipped. Each needs about two hours on the H100, and the worker's time
estimate ran past the deadline. Testing them needs a new pod: about $8–10 for the pair.

## Cost

$12.34 of the $20 cap: $4.95 for the first pod and $7.39 for the second. The ledger is
`runpod_budget_iterprune_20261116.json`.

## Run notes

- **First pod (hs-dec55376b8e8, $4.95):**
  - Qwen3.5's linear attention ran on the torch fallback at about 10% GPU use, because
    flash-linear-attention was missing from the requirements. I installed fla during the run.
  - The worker then crashed on a status() keyword clash after the first arm.
  - That arm's training result is kept above (†).
- **Second pod (hs-3cafb4128afa):**
  - fla was installed from the start. A stage of 2e15 trains in about 4 minutes, and a 1× arm takes
    about 25 minutes including the quick evaluation and the block probe.
  - Failing arms are now recorded and skipped instead of ending the job.
- **Kernel difference:** fla changes the canary by at most 0.002 nats (tolerance 0.01), well
  below the effects measured here.
- **Tests on the pod:** the CPU unit tests fail there (29 of 35) because fla's Triton kernels
  reject CPU tensors. The tests need fla disabled on the CPU. All tests pass locally, where fla is
  not installed.
