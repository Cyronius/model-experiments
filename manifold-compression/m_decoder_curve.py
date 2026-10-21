# M2 learning curve: how much of the cut part a decoder recovers as fit data grows, on a large single-site collection (2026-10-17).
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from common import RESULTS, WORK, save_json, set_threads

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="Qwen3.5-0.8B-big")
ap.add_argument("--ks", default="256,512")
ap.add_argument("--fracs", default="0.08,0.25,0.5,1.0")
ap.add_argument("--hidden", type=int, default=2048)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--threads", type=int, default=6)
args = ap.parse_args()
set_threads(args.threads)

meta = json.loads((WORK / args.work / "meta.json").read_text())
T1 = meta["T"] - 1
seq = torch.arange(meta["n_tokens"]) // T1
val = (seq % 10 == 9).nonzero().squeeze(1)
fit = (seq % 10 != 9).nonzero().squeeze(1)
fit = fit[torch.randperm(len(fit), generator=torch.Generator().manual_seed(0))]
out = {"work": args.work, "n_fit_tokens": len(fit), "curve": {}}

for s in meta["stored_sites"]:
    X = torch.from_numpy(np.load(WORK / args.work / f"site{s:03d}.npy").astype(np.float32))
    mu = X[fit].mean(0)
    Xc = X - mu
    ev, V = torch.linalg.eigh(Xc[fit].T @ Xc[fit] / len(fit))
    V = V[:, ev.argsort(descending=True)]
    for k in map(int, args.ks.split(",")):
        Vk = V[:, :k]
        Z, R = Xc @ Vk, None
        R = Xc - Z @ Vk.T
        zs, rs = Z[fit].std(0), R[fit].pow(2).mean().sqrt()
        for frac in map(float, args.fracs.split(",")):
            rows = fit[:int(len(fit) * frac)]
            torch.manual_seed(0)
            net = nn.Sequential(nn.Linear(k, args.hidden), nn.GELU(), nn.Linear(args.hidden, R.shape[1]))
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
            for _ in range(args.steps):
                idx = rows[torch.randint(len(rows), (1024,))]
                loss = (net(Z[idx] / zs) - R[idx] / rs).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            with torch.no_grad():
                def r2(ix):
                    p = net(Z[ix] / zs) * rs
                    return (1 - (p - R[ix]).pow(2).sum() / R[ix].pow(2).sum()).item()
                res = {"n": len(rows), "train_r2": r2(rows[:20000]), "val_r2": r2(val)}
            out["curve"].setdefault(meta["site_names"][s], {}).setdefault(k, []).append(res)
            print(f"{meta['site_names'][s]} k={k} n={len(rows)}: train R2 {res['train_r2']:.3f}  val R2 {res['val_r2']:.3f}", flush=True)
            save_json(out, RESULTS / meta["model"].split("/")[-1] / "decoder_curve.json")
