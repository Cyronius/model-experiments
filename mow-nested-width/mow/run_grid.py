"""Sweep runner. Enumerates every condition x seed, runs them in a process pool,
and skips anything that already has a done.json so a killed sweep can resume.

Two waves, because the routed-MoW runs warm-start from the nesting pretrain of
their own seed:
  wave A  dense baselines, nesting pretrain, all MoE conditions   (independent)
  wave B  routed MoW at each lambda                               (needs wave A)

Total optimization budget is held equal across conditions: MoW spends
PRETRAIN_STEPS + ROUTER_STEPS, and every non-MoW condition gets that same sum.
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import (DataConfig, ExperimentConfig, ModelConfig, TrainConfig,
                     preset_model)

RUNS = Path(__file__).resolve().parent / "runs"

PRETRAIN_STEPS = 8000
ROUTER_STEPS = 4000
TOTAL_STEPS = PRETRAIN_STEPS + ROUTER_STEPS      # budget every baseline matches

# Compute-pressure sweep. The scale that matters is the loss gap between the
# narrowest and widest tier: lambda multiplies a FLOPs fraction in [0.25, 1], so
# lambda ~ (loss gap) / 0.75 is where compute and quality trade off evenly.
LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.15, 0.5, 1.5)
FINE_LAMBDAS = (0.0, 0.15)


@dataclass
class Spec:
    name: str
    tag: str
    preset: str = "small"
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    depends: Optional[str] = None      # run name whose ckpt warm-starts this one
    seed: int = 0

    @property
    def dirname(self) -> str:
        return f"{self.name}_s{self.seed}"


def conditions(seed: int, preset: str = "small") -> list:
    """Every width here is derived from the preset, so retargeting the ladder is
    a single edit in config.PRESETS rather than constants scattered per condition."""
    base = preset_model(preset)
    D_FF, TIERS = base.d_ff, tuple(base.tiers)
    EXP_H, FINE_H = D_FF // 4, max(1, D_FF // 16)
    out = []

    # --- dense baselines, one per tier width (figure 1's reference points)
    for h in TIERS:
        out.append(Spec(f"dense_w{h}", "dense", model=dict(mlp="dense", dense_hidden=h),
                        train=dict(mode="dense", steps=TOTAL_STEPS), seed=seed))

    # --- nesting pretrain; the fixed-tier sweep in its final.json is figure 1
    out.append(Spec("mow_pretrain", "mow_pretrain", model=dict(mlp="mow"),
                    train=dict(mode="mow_pretrain", steps=PRETRAIN_STEPS), seed=seed))

    # --- nesting pretrain at the FULL budget, used as figure 1's reference and as
    # the fixed-tier curve in figure 2. The 8k pretrain above cannot serve that role:
    # comparing it against 12k dense baselines reports a training-budget difference
    # as though it were a cost of nesting.
    out.append(Spec("mow_pretrain12k", "mow_pretrain_full", model=dict(mlp="mow"),
                    train=dict(mode="mow_pretrain", steps=TOTAL_STEPS), seed=seed))

    # --- routed MoW: compute-quality frontier
    for lam in LAMBDAS:
        out.append(Spec(f"mow_router_lam{lam:g}", "mow_router",
                        model=dict(mlp="mow", router=True),
                        train=dict(mode="routed", steps=ROUTER_STEPS, lambda_flops=lam),
                        depends="mow_pretrain", seed=seed))

    # --- parameter-matched MoE: 4 experts x 128 == one 512 MLP
    for k in (1, 2):
        out.append(Spec(f"moe_top{k}", "moe_topk",
                        model=dict(mlp="moe", n_experts=4, expert_hidden=EXP_H,
                                   moe_mode="topk", top_k=k, router=True),
                        train=dict(mode="routed", steps=TOTAL_STEPS), seed=seed))

    # --- adaptive-k MoE: the direct equivalence test against routed MoW
    for lam in LAMBDAS:
        out.append(Spec(f"moe_adaptive_lam{lam:g}", "moe_adaptive",
                        model=dict(mlp="moe", n_experts=4, expert_hidden=EXP_H,
                                   moe_mode="adaptive", k_choices=(1, 2, 3, 4), router=True),
                        train=dict(mode="routed", steps=TOTAL_STEPS, lambda_flops=lam),
                        seed=seed))

    # --- fine-grained adaptive-k, balance loss on and off (figure 5).
    # k spans 1..16 so this condition can reach the full FLOPs range; the plan's
    # k in {1,2,3,4} would have capped it at 25% and made figure 2 unfair.
    for lam in FINE_LAMBDAS:
        for bal in (1e-2, 0.0):
            tagname = "bal" if bal > 0 else "nobal"
            out.append(Spec(f"moe_fine_{tagname}_lam{lam:g}", f"moe_fine_{tagname}",
                            model=dict(mlp="moe", n_experts=16, expert_hidden=FINE_H,
                                       moe_mode="adaptive", k_choices=(1, 2, 4, 8, 16),
                                       balance_coef=bal, router=True),
                            train=dict(mode="routed", steps=TOTAL_STEPS,
                                       lambda_flops=lam), seed=seed))
    return out


def build_cfg(spec: Spec, device: str = "cpu") -> ExperimentConfig:
    model = preset_model(spec.preset, **spec.model)
    tkw = dict(seed=spec.seed, eval_every=500, exact_every=2000)
    tkw.update(spec.train)
    if spec.depends:
        tkw["init_from"] = str(RUNS / f"{spec.depends}_s{spec.seed}" / "ckpt.pt")
    return ExperimentConfig(name=spec.dirname, tag=spec.tag, model=model,
                            data=DataConfig(), train=TrainConfig(**tkw),
                            device=device)


def _run_one(spec: Spec):
    from .train import train, is_done
    out = RUNS / spec.dirname
    if is_done(out):
        return spec.dirname, "skipped", 0.0
    device = os.environ.get("MOW_DEVICE", "cpu")
    t0 = time.time()
    try:
        train(build_cfg(spec, device), out)
        return spec.dirname, "ok", time.time() - t0
    except Exception as e:                       # keep the sweep alive
        import traceback
        (out).mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(traceback.format_exc())
        return spec.dirname, f"FAILED: {e}", time.time() - t0


def _init_worker(threads: int, device: str = "cpu"):
    os.environ["MOW_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MOW_DEVICE"] = device


def run_wave(specs, workers: int, threads: int, label: str, device: str = "cpu"):
    if not specs:
        return []
    print(f"\n=== wave {label}: {len(specs)} runs, {workers} workers x {threads} threads ===",
          flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(threads, device)) as ex:
        for name, status, secs in ex.map(_run_one, specs):
            results.append((name, status, secs))
            done = len(results)
            print(f"  [{done}/{len(specs)}] {name}: {status} ({secs:.0f}s) "
                  f"| elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"=== wave {label} finished in {(time.time()-t0)/60:.1f} min ===", flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--only", default=None, help="regex filter on run name")
    p.add_argument("--wave", default="all", choices=["a", "b", "all"])
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",")]
    specs = [s for seed in seeds for s in conditions(seed)]
    if a.only:
        rx = re.compile(a.only)
        specs = [s for s in specs if rx.search(s.name)]

    wave_a = [s for s in specs if not s.depends]
    wave_b = [s for s in specs if s.depends]

    if a.list:
        for s in specs:
            print(f"  {s.dirname:34s} tag={s.tag:16s} "
                  f"steps={s.train.get('steps')} dep={s.depends}")
        print(f"\n{len(specs)} runs ({len(wave_a)} wave A, {len(wave_b)} wave B)")
        return

    RUNS.mkdir(parents=True, exist_ok=True)
    res = []
    if a.wave in ("a", "all"):
        res += run_wave(wave_a, a.workers, a.threads, "A", a.device)
    if a.wave in ("b", "all"):
        res += run_wave(wave_b, a.workers, a.threads, "B", a.device)

    failed = [r for r in res if r[1].startswith("FAILED")]
    print(f"\n{len(res)} runs, {len(failed)} failed")
    for n, s, _ in failed:
        print(f"  {n}: {s}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
