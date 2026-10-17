"""Phase 0 of the dynamic-width plan: what does a mid-conversation tier switch
cost when the old context's KV cache is reused instead of re-prefilled?

For every ordered tier pair (A, B): the prompt runs at tier A, the answer at
tier B, and the answer attends to tier-A KV. Reported against the two honest
baselines -- pure B (what a re-prefill would give) and pure A (what refusing to
switch would give). The refresh sweep recomputes the last k prompt tokens at B.

Runs on the mow_pretrain12k checkpoints (the full-budget nested models).

    python -m mow.run_kvswitch --seeds 0,1,2,3,4,5,6,7,8,9
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from .config import ExperimentConfig
from . import data as D
from .eval import kv_switch_exact, kv_switch_loss, mixed_tier_logits
from .model import Ctx, Transformer

RUNS = Path(__file__).resolve().parent / "runs"
OUT = Path(__file__).resolve().parent / "runs_kvswitch"

EXACT_SEEDS = 3          # greedy decode is the slow part; loss runs on all seeds
REFRESH_KS = (2, 4, 8)   # k=1 (the minimum) is the main table's setting


def load_model(seed: int):
    base = RUNS / f"mow_pretrain12k_s{seed}"
    cfg = ExperimentConfig.load(base / "config.json")
    model = Transformer(cfg.model)
    ck = torch.load(base / "ckpt.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()
    return cfg, model


def sanity(model, batch, n_layer: int) -> None:
    """The mixed forward must equal the plain forward when no switch happens."""
    x = torch.from_numpy(batch.x[:64])
    split = torch.full((64,), 7)
    for t in (0, 3):
        ref, _ = model(x, Ctx(tiers=[t] * n_layer))
        mix = mixed_tier_logits(model, x, t, t, split)
        assert torch.allclose(ref, mix, atol=1e-5), f"mixed!=pure at tier {t}"
    # split beyond the sequence end must reduce to pure tier A
    ref, _ = model(x, Ctx(tiers=[0] * n_layer))
    mix = mixed_tier_logits(model, x, 0, 3, torch.full((64,), x.shape[1]))
    assert torch.allclose(ref, mix, atol=1e-5), "split=T should be pure tier A"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--threads", type=int, default=12)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    seeds = [int(s) for s in a.seeds.split(",")]

    OUT.mkdir(exist_ok=True)
    res = {"loss": {}, "exact": {}, "refresh": {}}
    t0 = time.time()

    for si, seed in enumerate(seeds):
        cfg, model = load_model(seed)
        n_tiers, n_layer = len(cfg.model.tiers), cfg.model.n_layer
        eval_sets = D.make_eval_sets(cfg.data)
        if si == 0:
            sanity(model, eval_sets["arith_hard"], n_layer)
            print("sanity: mixed forward matches plain forward", flush=True)

        L, E, R = {}, {}, {}
        for ta in range(n_tiers):
            for tb in range(n_tiers):
                key = f"{cfg.model.tiers[ta]}->{cfg.model.tiers[tb]}"
                per = {c: kv_switch_loss(model, eval_sets[c], ta, tb)
                       for c in D.CELLS}
                L[key] = {"overall": sum(per.values()) / len(per), "per_cell": per}
                if si < EXACT_SEEDS:
                    pe = {c: kv_switch_exact(model, eval_sets[c], ta, tb,
                                             cfg.model.block_size,
                                             limit=cfg.data.n_exact_per_cell)
                          for c in D.CELLS}
                    E[key] = {"overall": sum(pe.values()) / len(pe), "per_cell": pe}
                if ta != tb and si < EXACT_SEEDS:
                    R[key] = {}
                    for k in REFRESH_KS:
                        per_k = {c: kv_switch_loss(model, eval_sets[c], ta, tb,
                                                   refresh_k=k) for c in D.CELLS}
                        R[key][k] = sum(per_k.values()) / len(per_k)
        res["loss"][seed] = L
        res["exact"][seed] = E
        res["refresh"][seed] = R
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    with open(OUT / "results.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---- aggregate report
    tiers = [8, 16, 32, 64]
    def mean_over(d, key, pick):
        vals = [pick(d[s][key]) for s in d if key in d[s]]
        return sum(vals) / len(vals)

    print("\n=== overall answer loss, mean over seeds (rows: context tier A, cols: answer tier B) ===")
    print("%8s" % "" + "".join("%10d" % t for t in tiers))
    for tav in tiers:
        row = "%8d" % tav
        for tbv in tiers:
            row += "%10.4f" % mean_over(res["loss"], f"{tav}->{tbv}", lambda v: v["overall"])
        print(row)

    print("\n=== recall_hard loss (the only width-sensitive cell) ===")
    print("%8s" % "" + "".join("%10d" % t for t in tiers))
    for tav in tiers:
        row = "%8d" % tav
        for tbv in tiers:
            row += "%10.4f" % mean_over(res["loss"], f"{tav}->{tbv}",
                                        lambda v: v["per_cell"]["recall_hard"])
        print(row)

    print(f"\nwrote {OUT / 'results.json'} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    sys.exit(main() or 0)
