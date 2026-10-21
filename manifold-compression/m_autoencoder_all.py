# Same test as m_autoencoder_kl.py, but every site is compressed at once, so errors compound (2026-10-18).
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from common import WORK, add_session_args, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap)
ap.add_argument("--work", default="")
ap.add_argument("--sites", default="")
ap.add_argument("--ks", default="64,256")
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


def mlp(i, o, hidden, zero_last=True):
    net = nn.Sequential(nn.Linear(i, hidden), nn.GELU(), nn.Linear(hidden, o))
    if zero_last:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


def train(Xc, Z, R, xs, zs, rs, k, warp, hidden):
    torch.manual_seed(0)
    dec = mlp(k, R.shape[1], hidden).to(S.device)
    enc = mlp(Xc.shape[1], k, hidden).to(S.device) if warp else None
    params = list(dec.parameters()) + (list(enc.parameters()) if warp else [])
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for _ in range(args.steps):
        idx = fit[torch.randint(len(fit), (1024,), device=S.device)]
        z = Z[idx] / zs
        if warp:
            z = z + enc(Xc[idx] / xs)
        loss = (dec(z) - R[idx] / rs).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return enc.eval() if warp else None, dec.eval()


n_sites = meta["n_sites"]
sites = [int(x) for x in args.sites.split(",")] if args.sites else list(range(n_sites))
ARMS = (("decoder", False, args.hidden), ("decoder2x", False, 2 * args.hidden), ("warp", True, args.hidden))
for k in map(int, args.ks.split(",")):
    edits = {"pca": {}, **{label: {} for label, _, _ in ARMS}}
    for s in sites:
        X = torch.from_numpy(np.load(work / f"site{s:03d}.npy").astype(np.float32)).to(S.device)
        mu = X[fit].mean(0)
        Xc = X - mu
        ev, V = torch.linalg.eigh(Xc[fit].T @ Xc[fit] / len(fit))
        V = V[:, ev.argsort(descending=True)]
        xs = Xc[fit].pow(2).mean().sqrt()
        Vk = V[:, :k]
        Z = Xc @ Vk
        R = Xc - Z @ Vk.T
        zs, rs = Z[fit].std(0), R[fit].pow(2).mean().sqrt()
        edits["pca"][s] = lambda h, mu=mu, Vk=Vk: mu + ((h - mu) @ Vk) @ Vk.T
        for label, warp, hidden in ARMS:
            enc, dec = train(Xc, Z, R, xs, zs, rs, k, warp, hidden)

            def edit(h, enc=enc, dec=dec, mu=mu, Vk=Vk, zs=zs, rs=rs, xs=xs):
                x = h - mu
                z = (x @ Vk) / zs
                if enc is not None:
                    z = z + enc(x / xs)
                return mu + (x @ Vk) @ Vk.T + dec(z) * rs

            edits[label][s] = edit
        del X, Xc, Z, R
        print(f"k={k} site {s} trained", flush=True)
    res = {}
    with torch.no_grad():
        for label, fs in edits.items():
            res[label] = S.kl(0, edit=fs)
    out["sites"][k] = res
    print(f"ALL SITES k={k}: " + "  ".join(f"{key} {res[key]['all']['mean']:.4f} (copy {res[key].get('copy', {}).get('mean', float('nan')):.4f})" for key in res), flush=True)
    save_json(out, S.results / "autoencoder_all.json")
