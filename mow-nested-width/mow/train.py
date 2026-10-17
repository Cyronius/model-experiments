"""Training loop, aux losses, and the tier-sampling schedules.

Three modes:
  dense        one fixed width, no router.
  mow_pretrain nesting pretrain. Each step runs four uniform-tier passes plus one
               pass with an independently sampled tier per layer, and averages
               them. The mixed pass is not decoration: a router picks tiers per
               layer per token, so a trunk trained only on uniform-tier forwards
               would meet ragged combinations for the first time at routing time.
  routed       router active. Temperature anneals, then straight-through takes
               over, and the FLOP budget loss applies compute pressure.

Total optimization budget is matched across conditions by the caller: MoW spends
pretrain_steps + router_steps, so the dense and MoE baselines get that same sum.
Without that, the frontier figure would be comparing training budgets.
"""

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig, PAD
from . import data as D
from .eval import evaluate, sweep_fixed_tiers, is_routed
from .model import Ctx, Transformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, tc) -> float:
    if step < tc.warmup:
        return tc.lr * (step + 1) / max(tc.warmup, 1)
    p = (step - tc.warmup) / max(tc.steps - tc.warmup, 1)
    cos = 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * cos)


def schedule_ctx(step: int, cfg: ExperimentConfig) -> Ctx:
    """Soft-to-hard: geometric temperature anneal, then straight-through."""
    tc, mc = cfg.train, cfg.model
    switch = max(int(tc.hard_after_frac * tc.steps), 1)
    if step >= switch:
        return Ctx(temp=mc.router_temp_min, hard=True)
    p = step / switch
    temp = mc.router_temp0 * (mc.router_temp_min / mc.router_temp0) ** p
    return Ctx(temp=temp, hard=mc.router_hard)


def make_optimizer(model, tc):
    """1-D params (biases, LayerNorm, tier scales) skip weight decay. The router
    gets full LR while the trunk can be slowed with trunk_lr_mult."""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_router = ".router." in n or n.startswith("router.")
        key = (is_router, p.dim() >= 2)
        groups.setdefault(key, []).append(p)
    out = []
    for (is_router, decay), ps in groups.items():
        out.append({"params": ps,
                    "weight_decay": tc.weight_decay if decay else 0.0,
                    "lr_mult": 1.0 if is_router else tc.trunk_lr_mult})
    return torch.optim.AdamW(out, lr=tc.lr, betas=(tc.beta1, tc.beta2))


def masked_ce(logits, y, m):
    ls = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                         reduction="none")
    return (ls * m.reshape(-1)).sum() / m.sum().clamp_min(1)


def load_init(model, path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in missing if "router" not in k]
    assert not bad, f"warm start missing non-router weights: {bad[:5]}"
    assert not unexpected, f"warm start had unexpected weights: {unexpected[:5]}"


# ---------------------------------------------------------------- the loop

def train(cfg: ExperimentConfig, out_dir, log_every: int = 500) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")

    # 12 threads is the measured single-process optimum on this box (24 is slower).
    # run_grid lowers this so several runs can share the cores.
    torch.set_num_threads(int(os.environ.get("MOW_THREADS", "12")))
    set_seed(cfg.train.seed)
    device = cfg.device
    tc, mc = cfg.train, cfg.model

    model = Transformer(mc).to(device)
    if tc.init_from:
        load_init(model, tc.init_from)
    if tc.freeze_trunk:
        for n, p in model.named_parameters():
            if ".router." not in n:
                p.requires_grad_(False)

    opt = make_optimizer(model, tc)
    eval_sets = D.make_eval_sets(cfg.data)
    rng = random.Random(cfg.train.seed * 7919 + 13)

    csv_path = out / "metrics.csv"
    fields = ["step", "wall_s", "lr", "temp", "hard", "train_loss", "val_loss",
              "val_acc", "mlp_frac", "entropy", "k_entropy", "collapsed"] + \
             [f"{c}_{k}" for c in D.CELLS for k in ("loss", "acc", "frac")]
    fcsv = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    writer.writeheader()

    n_tiers = len(mc.tiers)
    ema, t0 = None, time.time()
    print(f"[{cfg.name}] {model.n_params()/1e6:.3f}M params | mode={tc.mode} "
          f"| steps={tc.steps} | lam={tc.lambda_flops}", flush=True)

    for step in range(tc.steps):
        lr = lr_at(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_mult"]

        b = D.sample_batch(rng, tc.batch_size, cfg.data)
        x = torch.from_numpy(b.x).to(device)
        y = torch.from_numpy(b.y).to(device)
        m = torch.from_numpy(b.mask).to(device)
        valid = x != PAD

        opt.zero_grad(set_to_none=True)
        ctx = schedule_ctx(step, cfg)

        if tc.mode == "mow_pretrain":
            passes = []
            if tc.pretrain_uniform:
                passes += [[t] * mc.n_layer for t in range(n_tiers)]
            if tc.pretrain_mixed:
                passes.append([rng.randrange(n_tiers) for _ in range(mc.n_layer)])
            for tiers in passes:                      # accumulate, one step
                logits, _ = model(x, Ctx(tiers=tiers))
                (masked_ce(logits, y, m) / len(passes)).backward()
            with torch.no_grad():
                logits, _ = model(x, Ctx(tiers=[n_tiers - 1] * mc.n_layer))
                loss_val = masked_ce(logits, y, m).item()
        else:
            logits, infos = model(x, ctx)
            lm = masked_ce(logits, y, m)
            flop_pen, balance, _ = model.routing_losses(infos, valid, tc.lambda_flops)
            (lm + flop_pen + balance).backward()
            loss_val = lm.item()

        if tc.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()

        ema = loss_val if ema is None else 0.98 * ema + 0.02 * loss_val

        last = step == tc.steps - 1
        if step % tc.eval_every == 0 or last:
            ectx = Ctx(temp=mc.router_temp_min, hard=True) if is_routed(model) else \
                   (Ctx(tiers=[n_tiers - 1] * mc.n_layer) if mc.mlp == "mow" else Ctx())
            do_exact = last or step % tc.exact_every == 0
            r = evaluate(model, eval_sets, ectx, cfg, device, do_exact=do_exact)
            row = {"step": step, "wall_s": round(time.time() - t0, 1), "lr": lr,
                   "temp": ctx.temp, "hard": int(ctx.hard), "train_loss": ema,
                   "val_loss": r["loss"], "val_acc": r.get("acc"),
                   "mlp_frac": r["mlp_frac"], "entropy": r.get("entropy"),
                   "k_entropy": r.get("k_entropy"), "collapsed": r.get("collapsed")}
            for c in D.CELLS:
                e = r["per_cell"][c]
                row[f"{c}_loss"] = e["loss"]
                row[f"{c}_acc"] = e.get("acc")
                row[f"{c}_frac"] = e["mlp_frac"]
            writer.writerow(row); fcsv.flush()
            if step % log_every == 0 or last:
                acc = f" acc {r['acc']:.3f}" if "acc" in r else ""
                ent = f" H {r['entropy']:.3f}" if r.get("entropy") is not None else ""
                flag = "  COLLAPSE" if r.get("collapsed") else ""
                print(f"  [{cfg.name}] {step:6d} loss {r['loss']:.4f}{acc} "
                      f"frac {r['mlp_frac']:.3f}{ent}{flag} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    fcsv.close()

    # ---- final report
    ectx = Ctx(temp=mc.router_temp_min, hard=True) if is_routed(model) else \
           (Ctx(tiers=[n_tiers - 1] * mc.n_layer) if mc.mlp == "mow" else Ctx())
    final = evaluate(model, eval_sets, ectx, cfg, device, do_exact=True)
    final["name"] = cfg.name
    final["tag"] = cfg.tag
    final["seed"] = tc.seed
    final["lambda_flops"] = tc.lambda_flops
    final["n_params"] = model.n_params()
    final["wall_s"] = round(time.time() - t0, 1)
    final["total_steps"] = tc.steps
    final["d_ff"] = mc.d_ff
    final["tiers"] = list(mc.tiers)

    if mc.mlp == "mow":
        final["fixed_tiers"] = sweep_fixed_tiers(model, eval_sets, cfg, device)

    torch.save({"model": model.state_dict(), "step": tc.steps}, out / "ckpt.pt")
    with open(out / "final.json", "w") as f:
        json.dump(final, f, indent=2)
    (out / "done.json").write_text(json.dumps({"ok": True, "wall_s": final["wall_s"]}))
    print(f"[{cfg.name}] done in {final['wall_s']:.0f}s | loss {final['loss']:.4f} "
          f"| acc {final.get('acc', float('nan')):.3f} | frac {final['mlp_frac']:.3f}",
          flush=True)
    return final


def is_done(out_dir) -> bool:
    return (Path(out_dir) / "done.json").exists()
