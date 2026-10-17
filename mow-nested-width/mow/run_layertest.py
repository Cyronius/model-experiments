"""Capacity tests for hard (5-8 digit) addition: depth, steps, and d_model.

Future-work item 1, tests 1, 2 and 4. The width sweep in runs_widthtest/ moved MLP
width by 32x and arith_hard loss did not respond, so the binding constraint is
somewhere else. This runner walks the remaining cheap knobs one at a time.

The baseline recipe matches the width sweep: dense MLP at the full d_ff=512 width,
d_model=128, n_head=4, wd 0.1, seed 0. The width sweep ran when the task mix had 4
cells; it now has 6 (recall was added), so arith_hard sees fewer examples per step
than it did then. That makes the old numbers an unsafe baseline, which is why every
comparison here is against a control re-run under current conditions.

Results so far, all on arith_hard:
  depth  4 vs 8 layers      no effect once converged (8 is just slower to converge)
  steps  12k vs 24k         no effect; the extra budget went to recall_hard instead

d_model is the last cheap candidate. It is deliberately raised without touching
d_ff, since width is already ruled out over a 32x range -- so this isolates the
residual stream rather than confounding it with MLP capacity. Attention still
changes with it (head dim 32 -> 64 at n_head=4), which no single-knob test can avoid.

    python -m mow.run_layertest --layers 4 --steps 24000 --d-model 256
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig

RUNS = Path(__file__).resolve().parent / "runs_layertest"

HIDDEN = 512          # == d_ff; the width sweep's 100%-FLOPs point
BASE_D_MODEL = 128    # runs at the baseline width keep their original short names


def run_name(n_layer: int, steps: int, d_model: int, pad: int) -> str:
    d = "" if d_model == BASE_D_MODEL else f"_d{d_model}"
    p = f"_pad{pad}" if pad else ""
    return f"lt_L{n_layer}{d}_{steps // 1000}k{p}"


def build_cfg(n_layer: int, steps: int, d_model: int, pad: int) -> ExperimentConfig:
    return ExperimentConfig(
        name=run_name(n_layer, steps, d_model, pad),
        tag="capacity",
        data=DataConfig(arith_pad_digits=pad),
        model=ModelConfig(n_layer=n_layer, d_model=d_model, n_head=4, d_ff=512,
                          mlp="dense", dense_hidden=HIDDEN),
        train=TrainConfig(seed=0, steps=steps, weight_decay=0.1,
                          mode="dense", eval_every=1000, exact_every=4000),
        notes="future-work item 1: depth x steps x d_model x operand padding",
    )


def _run_one(job):
    from .train import train, is_done
    n_layer, steps, d_model, pad = job
    name = run_name(n_layer, steps, d_model, pad)
    out = RUNS / name
    if is_done(out):
        return name, "skipped", 0.0
    t0 = time.time()
    try:
        r = train(build_cfg(n_layer, steps, d_model, pad), out)
        hard = r["per_cell"]["arith_hard"]
        return name, f"ok loss={hard['loss']:.4f} acc={hard['acc']:.3f}", time.time() - t0
    except Exception as e:
        import traceback
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(traceback.format_exc())
        return name, f"FAILED: {e}", time.time() - t0


def _init_worker(threads: int):
    os.environ["MOW_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="8,4")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--d-model", dest="d_model", type=int, default=BASE_D_MODEL)
    p.add_argument("--pad-digits", dest="pad", type=int, default=0,
                   help="diagnostic: zero-pad both addition operands to this width")
    p.add_argument("--threads", type=int, default=12)
    a = p.parse_args(argv)

    layers = [int(s) for s in a.layers.split(",")]
    jobs = [(n, a.steps, a.d_model, a.pad) for n in layers]
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"=== capacity test: n_layer in {layers} | {a.steps} steps "
          f"| d_model {a.d_model} | hidden {HIDDEN} | pad {a.pad} ===", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(jobs), initializer=_init_worker,
                             initargs=(a.threads,)) as ex:
        for name, status, secs in ex.map(_run_one, jobs):
            print(f"  {name}: {status} ({secs/60:.1f} min) "
                  f"| elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
