# Plan: prune in small steps, healing between them

Written 2026-11-16, before any run. Run 2026-11-16/17: results in
[results-iterative-prune.md](results-iterative-prune.md).

## Question

Does removing 25% of Qwen3.5-2B in 10% steps, healing after each step, give a better model than
removing 25% at once and healing once, when both get the same total training compute?

## What we already have to compare against

Light student: one-shot 25% cut (residual 2048→1792, FFN 6144→4800, 1.47B parameters), healed with
joint-cosine (lr 3e-5, batch 1, seed 17).

| Run | Training FLOPs | Val CE | Gap closed | ARC-E (200) | ARC-C (200) | MMLU (200) | GSM8K (32) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unhealed | 0 | 3.660 | 0% | | | | |
| One-shot, 1× | 6e15 | 3.142 | 46.7% | .525 | .315 | .320 | .250 |
| One-shot, 10× | 60e15 | 3.099 | 50.5% | .485 | .305 | .325 | .313 |

Gap closed = (3.660 − CE) / (3.660 − 2.550). The block probe for both one-shot runs is in
`runs/long-joint-cosine/block-probe.json`.

## Design

### Stages

Every stage removes about 10% of the teacher's non-embedding parameters (1,373,265,728). The residual
width shrinks along the same slope as the light student (256 coordinates per 25%, aligned to 64), and
the FFN width takes up the rest. The last stage lands on the light student's geometry exactly, so the
final model has the same size and shape as the one-shot model; only which coordinates and neurons it
kept, and its weights, differ.

| Stage | Cut | Residual | FFN | Non-embedding kept | Total parameters |
|---|---:|---:|---:|---:|---:|
| 1 | 10% | 1920 | 5760 | 89.9% | 1,711,154,880 |
| 2 | 20% | 1856 | 5056 | 80.0% | 1,560,054,400 |
| 3 | 25% | 1792 | 4800 | 74.9% | 1,473,244,736 (= light) |

The last step is 5%, not 10%, to land on 25%.

### Each stage

1. **Prune** the previous stage's healed model (stage 1 prunes the stock 2B). Selection is the same
   activation rule as the light student (Minitron-style energies, optional norm rescaling chosen on the
   controller split), but the statistics come from the model being pruned. A healed model has adapted
   its blocks to one another (the block probe showed this), so its own activations say what it relies
   on; the teacher's would not.
2. **Heal** with joint-cosine at lr 3e-5, batch 1, warmup 32 updates. The optimizer starts fresh (the
   shapes changed), and the learning rate decays to zero within the stage. The loss is unchanged:
   final-layer residual match, logit KL and 0.1·CE against the full teacher, at the student's kept
   coordinates.
3. **Continue the data stream** where the previous stage stopped (`--data-offset`), so no tokens repeat.

### Arms

All arms end at the light geometry and are compared at equal total training FLOPs. Pruning compute
(activation statistics) is reported separately; it is small and runs on the CPU.

| Arm | Stages | Healing split | Total | Compare with |
|---|---|---|---:|---|
| **S3-1×** | 10 / 20 / 25% | equal thirds (2e15 each) | 6e15 | one-shot 1× (3.142) |
| S2-1× | 12.5 / 25% | equal halves | 6e15 | one-shot 1× |
| S5-1× | 5 / 10 / 15 / 20 / 25% | equal fifths | 6e15 | one-shot 1× |
| One-shot 1×, seed 2 | 25% | one stage | 6e15 | seed noise for every 1× comparison |
| **S3-10×** | 10 / 20 / 25% | equal thirds (20e15 each) | 60e15 | one-shot 10× (3.099) |

Order: the seed-2 baseline and S3-1× first. S2 and S5 follow to see whether finer steps help. S3-10×
runs only if a 1× arm beats one-shot by more than the seed noise, or if the 1× results are too close
to call.

**Changed before launch (2026-11-16):** the whole plan runs on one rented H100 (`run_pod.py`,
`pod_worker.py`, $20 cap). Every one-shot baseline is rerun on the pod, so all comparisons use the same
hardware, and a one-shot 10× arm joins S3-10×. The arms run in this priority order whenever the deadline
allows, with no gate on the 1× results: oneshot-1x-seed17, S3-1×, oneshot-1x-seed2, S2-1×, S5-1×,
S3-10×, oneshot-10x-seed17.

Equal shares are the simplest split with nothing to tune. A smaller model is cheaper per token, so at
equal FLOPs the staged arms see fewer tokens than one-shot. Report tokens next to FLOPs.

### Measurements

- **Damage at each prune:** val CE right after each cut, before healing. If a healed model tolerates
  the next 10% far better than the stock model tolerates 25% (3.66), that is the mechanism working.
- **Final model:** val CE curve, test CE, gap closed, and the quick evaluation (`evaluate.py --splits
  test --mc-examples 200 --gsm8k-examples 32`, about 20 min), all on the same frozen subsets as the
  table above.
- **Block probe** of the final model against the teacher (`block_probe.py`). Do blocks stay closer to
  the teacher's when each cut is small?
- **Overlap of kept coordinates and neurons** with the one-shot selection: how different are the two
  25% models?

## Predictions (written before any run)

- **Primary:** at 6e15, S3 ends within 0.03 of one-shot (3.142), in either direction. Minitron found
  one-shot structured pruning with retraining as good as or better than iterative at equal tokens, and
  25% is a light cut. Gradual pruning has helped most at high sparsity.
- **Damage:** each 10% cut of a healed model costs less val CE than the 25% one-shot cut did (+1.11
  from the teacher), so 0.4 or less per stage.
- **Finer steps:** S5 is no better than S3 at 1×. Each stage gets only 1.2e15, close to where
  joint-cosine passes the 25% rung (0.53e15) and short of the 40% rung (0.76e15), so the healing is
  shallow.
- **If staging helps at all,** it helps more at 10× than at 1×, and the next test is the heavy cut.

## Code to write

1. **`pruning.py --from-artifact`:** prune an already-pruned, healed model.
   - Source: a run's `final/` folder plus its parent `pruning.json`.
   - Activation statistics come from the source model on the same calibration rows.
   - Selection runs within its current coordinates and neurons.
   - Composed indices: `residual_indices` and `ffn_indices` still map to the teacher's coordinates,
     which the distillation targets and the block probe need.
   - `pruning.json` records the parent artifact and run, with hashes.
   - Today's verified-source check (stock 2B config plus download manifest) stays for the first stage.
2. **Tests:**
   - Index composition: pruning to the full size is the identity, and two-step indices match the
     selected teacher columns.
   - A CPU round trip on a tiny model: prune, heal a few steps, re-prune, then load in `run.py`.
3. **`drivers/iterative-prune.sh`:**
   - Runs one arm's stages in order: prune, heal, quick eval of the final model, block probe.
   - Each stage detached and resumable, the same machine-friendly way as the long run: dibs, nice 10,
     idle I/O, a 24G memory cap, 2-hour segments with 10-minute GPU breaks.
4. **Protocol:** a `followup_iterative_prune` entry in `protocol.json`, recording this design and the
   predictions before the first run.

## Cost

| Item | Time on this machine |
|---|---|
| Seed-2 baseline, S3-1×, S2-1×, S5-1× | about 1 h each (30 min training, plus prunes, evals and breaks) |
| S3-10× | about 6–7 h |
| Prunes | a few CPU minutes each, while the GPU is idle |

All local, no paid compute.

## If it works: the heavy cut

Repeat at the heavy student's 63.7% cut (residual 1024, FFN 3584, the 0.8B size) in 10% steps: seven
stages ending on that geometry. Use its own learning rate; 1e-4 was right there. One-shot heavy healing
stalled near 4.8 val CE, so a gradual path has the most room to help there.

## Open choices

Defaults are noted. Change them before the first run, not after.

- **Statistics source for re-pruning:** the healed model (default) or the teacher restricted to the
  kept coordinates. Ablate only if S3 looks interesting.
- **Healing split:** equal (default) or weighted toward the final stage.
- **Final-stage schedule:** a cosine per stage (default), or one WSD run whose stable phase spans all
  stages and decays only at the end.
