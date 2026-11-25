"""Train dense / loop / oploop on the circuit task; evaluate accuracy, halting and routing by gate level.

Example:
  python train_circuit.py --kind oploop --K 8 --name oploop-k8
Results: results/circuit/<name>.json (+ a line in results/circuit/results.jsonl).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from circuits import N_OPS, generate
from model import Net, halt_step, halting_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "npu-kernel-coder", "train"))
import memguard  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="oploop", choices=["dense", "oploop"])
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--name", required=True)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--T", type=int, default=20)
ap.add_argument("--nodes", type=int, default=48)
ap.add_argument("--labels", type=int, default=64)
ap.add_argument("--depth", type=int, default=12)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--beta", type=float, default=0.01, help="cost per expected step")
ap.add_argument("--beta-start", type=float, default=0.45, help="fraction of training before the step cost ramps in")
ap.add_argument("--curriculum", type=float, default=0.4, help="fraction of training over which max depth ramps 2 -> --depth")
ap.add_argument("--freeze-frac", type=float, default=0.0, help="fraction of batches trained with halted tokens frozen")
ap.add_argument("--aux", type=float, default=0.01)
ap.add_argument("--tau", default="2.0,0.5")
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "circuit"))
args = ap.parse_args()

memguard.start(12)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = os.environ.get("DEV", "cuda")
vocab = {"op": N_OPS, "lab": args.labels + 1, "la": args.labels + 1, "lb": args.labels + 1}
net = Net(args.kind, vocab, d=args.d, hidden=args.hidden, K=args.K, T=args.T).to(dev)
n_params = sum(p.numel() for p in net.parameters())
print(f"{args.name}: {n_params / 1e6:.2f}M params", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
tau0, tau1 = map(float, args.tau.split(","))


def batch(n, lo, hi, g=rng):
    c = generate(n, args.nodes, lo, hi, args.labels, g)
    return {k: torch.from_numpy(v).to(dev) for k, v in c.items()}


def lr_at(s):
    w = 500
    return args.lr * (s / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / (args.steps - w))))


def ent(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


@torch.no_grad()
def evaluate(lo, hi, T, n=int(os.environ.get("EVAL_N", 2048)), seed=123):
    net.eval()
    g = np.random.default_rng(seed)
    max_lv = hi + 1
    modes = ["readout", "freeze"] + (["shuffle", "random"] if args.kind == "oploop" and args.K > 1 else [])
    stats = {m: {"correct": np.zeros(max_lv), "steps": np.zeros(max_lv)} for m in modes + ["last"]}
    count = np.zeros(max_lv)
    choice_rows, halt_rows = [], []
    for _ in range(n // 256):
        c = batch(256, lo, hi, g)
        gate = c["level"] > 0
        lv = c["level"][gate].cpu().numpy()
        count += np.bincount(lv, minlength=max_lv)
        for mode in modes:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = net(c, T=T, freeze=mode == "freeze", route=mode if mode in ("shuffle", "random") else None)
            hs = halt_step(out["lam"])
            pred = out["logits"].argmax(-1)
            pick = pred[-1] if mode == "freeze" else pred.gather(0, hs[None])[0]
            ok = (pick == c["val"])[gate].float().cpu().numpy()
            stats[mode]["correct"] += np.bincount(lv, ok, max_lv)
            stats[mode]["steps"] += np.bincount(lv, (hs[gate] + 1).float().cpu().numpy(), max_lv)
            if mode == "readout":
                halt_rows.append((hs[gate].cpu().numpy(), lv))
                ok = (pred[-1] == c["val"])[gate].float().cpu().numpy()
                stats["last"]["correct"] += np.bincount(lv, ok, max_lv)
                stats["last"]["steps"] += np.bincount(lv, np.full(len(lv), T, float), max_lv)
                if "choice" in out:
                    ch = out["choice"].permute(1, 2, 0)[gate].cpu().numpy()  # [tokens, T]
                    choice_rows.append((ch, c["op"][gate].cpu().numpy(), lv, hs[gate].cpu().numpy()))
    r = {"T": T, "depth": [lo, hi], "count_by_level": count[1:].tolist()}
    for mode, s in stats.items():
        nz = np.maximum(count, 1)
        r[mode] = {"acc": float(s["correct"][1:].sum() / count[1:].sum()),
                   "acc_by_level": (s["correct"] / nz)[1:].round(4).tolist(),
                   "steps_by_level": (s["steps"] / nz)[1:].round(2).tolist(),
                   "mean_steps": float(s["steps"][1:].sum() / count[1:].sum())}
    if choice_rows:
        ch = np.concatenate([x[0] for x in choice_rows])
        ops = np.concatenate([x[1] for x in choice_rows])
        lvc = np.concatenate([x[2] for x in choice_rows])
        hs = np.concatenate([x[3] for x in choice_rows])
        K = net.K
        usage = np.bincount(ch.ravel(), minlength=K) / ch.size
        # H(op | step) and H(op | step, gate type), over steps up to each token's halt.
        h_step, h_step_type, w = 0.0, 0.0, 0
        for t in range(T):
            live = hs >= t
            if live.sum() == 0:
                continue
            h_step += ent(np.bincount(ch[live, t], minlength=K)) * live.sum()
            h_step_type += sum(ent(np.bincount(ch[live & (ops == o), t], minlength=K)) * (live & (ops == o)).sum()
                               for o in range(N_OPS))
            w += live.sum()
        seqs = {tuple(row[:s + 1]) for row, s in zip(ch, hs)}
        r["router"] = {"usage": usage.round(3).tolist(), "H_op_given_step": h_step / w,
                       "H_op_given_step_type": h_step_type / w, "H_max": math.log(K),
                       "distinct_seqs_frac": len(seqs) / len(ch)}
    hs_all = np.concatenate([x[0] for x in halt_rows])
    lv_all = np.concatenate([x[1] for x in halt_rows])
    r["corr_halt_level"] = float(np.corrcoef(hs_all, lv_all)[0, 1]) if hs_all.std() > 0 else None
    net.train()
    return r


os.makedirs(args.out, exist_ok=True)
log = []
t0 = time.time()
net.train()
for step in range(1, args.steps + 1):
    for gopt in opt.param_groups:
        gopt["lr"] = lr_at(step)
    frac = step / args.steps
    tau = tau0 * (tau1 / tau0) ** frac
    beta = args.beta * min(1.0, max(0.0, (frac - args.beta_start) / 0.2))
    hi = args.depth if args.curriculum <= 0 else min(args.depth, 2 + int((args.depth - 1) * frac / args.curriculum))
    c = batch(args.bs, 1, hi)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        out = net(c, tau=tau, freeze=bool(rng.random() < args.freeze_frac))
    gate = (c["level"] > 0).float()
    T = out["logits"].shape[0]
    ce = F.cross_entropy(out["logits"].reshape(-1, 2), c["val"].repeat(T, 1, 1).reshape(-1),
                         reduction="none").view(T, *c["val"].shape)
    p = halting_dist(out["lam"])
    task = ((p * ce).sum(0) * gate).sum() / gate.sum()
    exp_steps = ((p * torch.arange(1, T + 1, device=dev)[:, None, None]).sum(0) * gate).sum() / gate.sum()
    loss = task + beta * exp_steps
    if "aux" in out:
        loss = loss + args.aux * out["aux"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if step % 200 == 0:
        print(f"step {step} task {task.item():.4f} E[steps] {exp_steps.item():.2f} beta {beta:.4f} "
              f"tau {tau:.2f} depth<={hi} {time.time() - t0:.0f}s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        r = evaluate(1, args.depth, args.T)
        print(json.dumps({"step": step, "acc": r["readout"]["acc"], "acc_freeze": r["freeze"]["acc"],
                          "acc_last": r["last"]["acc"], "steps": r["readout"]["mean_steps"],
                          "corr": r.get("corr_halt_level"), "router": r.get("router")}), flush=True)
        log.append({"step": step, **r})

final = {"args": vars(args), "params": n_params, "train_seconds": time.time() - t0,
         "in_dist": log[-1],
         "ood": evaluate(args.depth + 1, 2 * args.depth, 2 * args.T),
         "log": log}
print("OOD:", json.dumps({k: final["ood"][k] for k in ("readout", "freeze", "last")}), flush=True)
torch.save(net.state_dict(), os.path.join(args.out, f"{args.name}.pt"))
json.dump(final, open(os.path.join(args.out, f"{args.name}.json"), "w"), indent=1)
summ = {"name": args.name, "kind": args.kind, "K": args.K, "params": n_params,
        "acc": final["in_dist"]["readout"]["acc"], "acc_freeze": final["in_dist"]["freeze"]["acc"],
        "steps": final["in_dist"]["readout"]["mean_steps"], "corr": final["in_dist"].get("corr_halt_level"),
        "ood_acc": final["ood"]["readout"]["acc"], "ood_acc_last": final["ood"]["last"]["acc"],
        "ood_steps": final["ood"]["readout"]["mean_steps"], "ood_acc_freeze": final["ood"]["freeze"]["acc"],
        "acc_shuffle": final["in_dist"].get("shuffle", {}).get("acc"), "acc_random": final["in_dist"].get("random", {}).get("acc")}
with open(os.path.join(args.out, "results.jsonl"), "a") as f:
    f.write(json.dumps(summ) + "\n")
print(json.dumps(summ), flush=True)
