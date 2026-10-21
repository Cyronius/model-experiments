# M2 payoff: does a well-fed decoder lower KL, not just raise R²? Single-site KL with PCA vs PCA + decoder, on the large collection (2026-10-18).
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from common import WORK, add_session_args, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap)
ap.add_argument("--work", default="", help="activation dir under work/; default the model's own")
ap.add_argument("--sites", default="", help="comma list; default the dir's stored_sites")
ap.add_argument("--ks", default="256,512")
ap.add_argument("--hidden", type=int, default=2048)
ap.add_argument("--steps", type=int, default=3000)
args = ap.parse_args()

S = session_from(args)
work = WORK / (args.work or S.tag)
meta = json.loads((work / "meta.json").read_text())
T1 = meta["T"] - 1
seq = torch.arange(meta["n_tokens"], device=S.device) // T1
fit = (seq % 10 != 9).nonzero().squeeze(1)
S.eval_batches()
out = {"model": args.model, "sites": {}}


def train(Z, R, zs, rs, k):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(k, args.hidden), nn.GELU(), nn.Linear(args.hidden, R.shape[1])).to(S.device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for _ in range(args.steps):
        idx = fit[torch.randint(len(fit), (1024,), device=S.device)]
        loss = (net(Z[idx] / zs) - R[idx] / rs).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


sites = [int(x) for x in args.sites.split(",")] if args.sites else meta.get("stored_sites", range(meta["n_sites"]))
for s in sites:
    X = torch.from_numpy(np.load(work / f"site{s:03d}.npy").astype(np.float32)).to(S.device)
    mu = X[fit].mean(0)
    Xc = X - mu
    ev, V = torch.linalg.eigh(Xc[fit].T @ Xc[fit] / len(fit))
    V = V[:, ev.argsort(descending=True)]
    name = meta["site_names"][s]
    for k in map(int, args.ks.split(",")):
        Vk = V[:, :k]
        Z = Xc @ Vk
        R = Xc - Z @ Vk.T
        zs, rs = Z[fit].std(0), R[fit].pow(2).mean().sqrt()
        net = train(Z, R, zs, rs, k)
        pca = lambda h, mu=mu, Vk=Vk: mu + ((h - mu) @ Vk) @ Vk.T
        dec = lambda h, mu=mu, Vk=Vk, net=net, zs=zs, rs=rs: mu + ((h - mu) @ Vk) @ Vk.T + net(((h - mu) @ Vk) / zs) * rs
        res = {"pca": S.kl(s, edit={s: pca}), "decoder": S.kl(s, edit={s: dec})}
        out["sites"].setdefault(name, {})[k] = res
        a, b = res["pca"]["all"]["mean"], res["decoder"]["all"]["mean"]
        print(f"{name} k={k}: KL PCA {a:.4f}  PCA+decoder {b:.4f}  ({100 * (a - b) / a:.0f}% lower)", flush=True)
        save_json(out, S.results / "decoder_kl.json")
