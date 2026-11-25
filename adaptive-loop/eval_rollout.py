"""Closed-loop test of a time-step pedal model: start at the true DC point, feed the input one
sample at a time, and let the model's own predicted voltages be its next input for 480 samples
(10 ms at 48 kHz). Compares node waveforms with the solver's.

  HALT_RULE=cdf python eval_rollout.py --name step-oploop-k8 --kind oploop --K 8
Writes results/analog-step/<name>-rollout.json and a line in results/analog-step/rollout.jsonl.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from gen_analog import LABELS, ROLES, TYPES
from model import Net

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--kind", default="oploop")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--T", type=int, default=24)
ap.add_argument("--fourier", type=int, default=6)
ap.add_argument("--sets", default="roll,roll_ood")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--asinh", type=float, default=0.0, help="model predicts asinh(dv/S)")
ap.add_argument("--edge-frac", type=float, default=0.0)  # training-only flags, accepted so run scripts can pass the same args
ap.add_argument("--edge-thr", type=float, default=0.1)
ap.add_argument("--no-freeze", action="store_true")
args = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
dev = os.environ.get("DEV", "cuda")
out_dir = os.path.join(here, "results", os.environ.get("OUT_DIR", "analog-step"))
vocab = {"ctype": len(TYPES), "t1": LABELS + 1, "t2": LABELS + 1, "t3": LABELS + 1}
net = Net(args.kind, vocab, d=args.d, hidden=args.hidden, K=args.K, T=args.T, n_out=1, cont_dim=3,
          fourier=args.fourier).to(dev)
net.load_state_dict(torch.load(os.path.join(out_dir, f"{args.name}.pt"), map_location=dev))
net.eval()
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda" and not os.environ.get("NOAMP"))
OUT = ROLES.index("out")


@torch.no_grad()
def rollout(name):
    d = np.load(os.path.join(here, "cache", "analog-step", f"{name}.npz"))
    B = len(d["type"])
    S = torch.from_numpy(d["states"]).to(dev)  # [B, 481, 40]
    U = torch.from_numpy(d["vin"]).to(dev)     # [B, 481]
    nidx = torch.from_numpy(d["nidx"]).to(dev)
    vs = torch.from_numpy(d["vs"]).to(dev)
    tok = {("ctype" if k == "type" else k): torch.from_numpy(d[k]).to(dev)
           for k in ("type", "t1", "t2", "t3", "pad", "cont", "node")}
    pred = torch.zeros_like(S)
    hold = S[:, :1].expand_as(S)  # baseline: every node stays at its DC value
    steps_sum, steps_n = 0.0, 0
    t0 = time.time()
    for s0 in range(0, B, args.bs):
        sl = slice(s0, min(B, s0 + args.bs))
        c = {k: v[sl] for k, v in tok.items()}
        c["cont"] = c["cont"].clone()
        ni, nm = nidx[sl].clamp_min(0), nidx[sl] >= 0
        rows = torch.arange(ni.shape[0], device=dev)
        v = S[sl, 0].clone()
        pred[sl, 0] = v
        for s in range(S.shape[1] - 1):
            vn = v.gather(1, ni)
            c["cont"][..., 0] = torch.where(nm, vn / 9, c["cont"][..., 0])
            c["cont"][..., 1] = torch.where(nm, vn, c["cont"][..., 1])
            c["cont"][rows, vs[sl], 0] = U[sl, s + 1]
            c["cont"][rows, vs[sl], 1] = U[sl, s]
            with torch.autocast(**amp):
                o = net(c, freeze=not args.no_freeze)
            lam = o["lam"]
            p_halt = 1 - torch.cumprod(1 - lam, 0)
            hs = ((p_halt >= 0.5) | (torch.arange(lam.shape[0], device=dev)[:, None, None] == lam.shape[0] - 1)).float().argmax(0)
            dv = o["logits"][-1, ..., 0] if not args.no_freeze else \
                o["logits"].gather(0, hs[None, ..., None])[0, ..., 0]
            steps_sum += float((hs[nm] + 1).float().sum())
            steps_n += int(nm.sum())
            if args.asinh:
                dv = args.asinh * torch.sinh(dv.float().clamp(-12, 12))
            v = v.clone()
            v.scatter_(1, ni, torch.where(nm, vn + dv.float(), vn))  # pads write their old value back to node 0
            v[:, 0] = 0.0
            pred[sl, s + 1] = v
    secs = time.time() - t0
    n_nodes = (nidx >= 0).sum(1)
    valid = torch.arange(40, device=dev)[None] < n_nodes[:, None]  # node slots in use
    err = (pred - S).pow(2).mean(1).sqrt()      # [B, 40] rmse over time
    herr = (hold - S).pow(2).mean(1).sqrt()
    role = torch.full((B, 40), -1, device=dev, dtype=torch.long)
    for b in range(B):  # node role per node index
        m = nidx[b] >= 0
        role[b, nidx[b][m]] = torch.from_numpy(d["role"][b]).to(dev)[m]
    is_out = role == OUT
    ac = (S - S.mean(1, keepdim=True)).pow(2).mean(1).sqrt()
    r = {"set": name, "circuits": B, "seconds": secs, "mean_steps": steps_sum / max(steps_n, 1),
         "node_rmse_mean": float(err[valid].mean()), "node_rmse_median": float(err[valid].median()),
         "node_w50": float((err[valid] < 0.05).float().mean()), "hold_node_w50": float((herr[valid] < 0.05).float().mean()),
         "out_rmse_median": float(err[is_out].median()), "out_rel_median": float((err[is_out] / ac[is_out].clamp_min(1e-3)).median()),
         "hold_out_rel_median": float((herr[is_out] / ac[is_out].clamp_min(1e-3)).median()),
         "by_role": {ROLES[k]: {"rmse_median": float(err[role == k].median()), "w50": float((err[role == k] < 0.05).float().mean())}
                     for k in range(len(ROLES)) if (role == k).any()}}
    np.savez_compressed(os.path.join(out_dir, f"{args.name}-{name}-traj.npz"), pred=pred[:32].cpu().numpy(),
                        true=S[:32].cpu().numpy(), role=role[:32].cpu().numpy())
    return r


res = {s: rollout(s) for s in args.sets.split(",")}
json.dump(res, open(os.path.join(out_dir, f"{args.name}-rollout.json"), "w"), indent=1)
summ = {"name": args.name, **{f"{s}_{k}": v for s, r in res.items() for k, v in r.items()
                              if k in ("node_w50", "hold_node_w50", "out_rel_median", "hold_out_rel_median", "mean_steps", "seconds")}}
with open(os.path.join(out_dir, "rollout.jsonl"), "a") as f:
    f.write(json.dumps(summ) + "\n")
print(json.dumps(summ), flush=True)
