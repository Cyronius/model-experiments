# Placement test: rebuild one MLP block at hidden width m from chosen neurons plus least-squares output weights, no gradient training (2026-10-18).
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import WORK, Session, save_json, set_threads, yield_if_asked

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--work", default="Qwen3.5-0.8B-mlp")
ap.add_argument("--layers", default="6,15")
ap.add_argument("--ms", default="256,512,1024,1792")
ap.add_argument("--ks", default="0,512", help="input width the neurons see; 0 = full width")
ap.add_argument("--methods", default="random,importance,omp,omp_mixed")
ap.add_argument("--n-fit", type=int, default=60000, help="tokens used for least squares")
ap.add_argument("--n-select", type=int, default=20000, help="tokens used for greedy selection")
ap.add_argument("--omp-batch", type=int, default=32)
ap.add_argument("--ridge", type=float, default=1e-3)
ap.add_argument("--n-eval", type=int, default=2)
ap.add_argument("--threads", type=int, default=12)
args = ap.parse_args()
set_threads(args.threads)

meta = json.loads((WORK / args.work / "meta.json").read_text())
T1 = meta["T"] - 1
seq = torch.arange(meta["n_tokens"]) // T1
gen = torch.Generator().manual_seed(0)
fit_all = (seq % 10 != 9).nonzero().squeeze(1)
fit_all = fit_all[torch.randperm(len(fit_all), generator=gen)]
fit_rows, sel_rows = fit_all[:args.n_fit], fit_all[:args.n_select]
val_rows = (seq % 10 == 9).nonzero().squeeze(1)[:20000]

S = Session(args.model, n_fit=16, n_eval=args.n_eval, bs=4)
r = S.runner
S.eval_batches()
out_path = S.results / "placement.json"
results = {"model": args.model, "n_fit": len(fit_rows), "layers": {}}
if out_path.exists():
    results = json.loads(out_path.read_text())
done = {(int(li), key, meth, int(m)) for li, v in results["layers"].items() for key, by in v.items() if key.startswith("k=")
        for meth, by_m in by.items() for m in by_m}
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def features(x, G, Uw):
    return F.silu(x @ G.T) * (x @ Uw.T)


def lstsq(Phi, Y):
    """Ridge least squares with a bias; returns (W [m, d], b [d])."""
    pm, ym = Phi.mean(0), Y.mean(0)
    Pc = Phi - pm
    A = Pc.T @ Pc
    A += args.ridge * A.diagonal().mean() * torch.eye(A.shape[0])
    W = torch.linalg.solve(A, Pc.T @ (Y - ym))
    return W, ym - pm @ W


def r2(pred, y):
    return (1 - (pred - y).pow(2).sum() / (y - y.mean(0)).pow(2).sum()).item()


def omp_select(Phi, Y, m):
    """Batch orthogonal matching pursuit: repeatedly add the neurons that best explain what the current fit misses."""
    Pn = (Phi - Phi.mean(0)) / ((Phi - Phi.mean(0)).norm(dim=0) + 1e-8)
    chosen = torch.zeros(Phi.shape[1], dtype=torch.bool)
    R = Y - Y.mean(0)
    while chosen.sum() < m:
        score = (Pn.T @ R).pow(2).sum(1)
        score[chosen] = -1
        take = min(args.omp_batch, m - int(chosen.sum()))
        chosen[score.topk(take).indices] = True
        W, b = lstsq(Phi[:, chosen], Y)
        R = Y - (Phi[:, chosen] @ W + b)
    return chosen.nonzero().squeeze(1)


for li in map(int, args.layers.split(",")):
    layer = r.lm.layers[li]
    norm, mlp = layer.post_attention_layernorm, layer.mlp
    Gt, Ut = mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()
    Dt = mlp.down_proj.weight.detach()
    site = 2 * li + 1
    with torch.no_grad():
        Hall = torch.from_numpy(np.load(WORK / args.work / f"site{site:03d}.npy", mmap_mode="r")[
            torch.cat([fit_rows, val_rows]).numpy()].astype(np.float32))
        Xall = norm(Hall)
        Yall = mlp(Xall)
    nf = len(fit_rows)
    X, Y, Xv, Yv = Xall[:nf], Yall[:nf], Xall[nf:], Yall[nf:]
    Xs, Ys = X[:args.n_select], Y[:args.n_select]
    res = results["layers"].get(str(li)) or {"kind": r.types[li], "teacher_width": Gt.shape[0],
                                            "kl_drop": S.kl(site, mlp_edit={li: lambda h, u: torch.zeros_like(u)})}
    results["layers"].pop(str(li), None)
    log(f"L{li}: drop-MLP KL {res['kl_drop']['all']['mean']:.4f}")

    torch.manual_seed(0)
    Gr = torch.randn_like(Gt) * Gt.norm(dim=1, keepdim=True).mean() / Gt.shape[1] ** 0.5
    Ur = torch.randn_like(Ut) * Ut.norm(dim=1, keepdim=True).mean() / Ut.shape[1] ** 0.5

    for k in map(int, args.ks.split(",")):
        if k:
            mu = X.mean(0)
            ev, V = torch.linalg.eigh((X - mu).T @ (X - mu) / len(X))
            Vk = V[:, ev.argsort(descending=True)[:k]]
            proj = lambda x, mu=mu, Vk=Vk: mu + ((x - mu) @ Vk) @ Vk.T
        else:
            proj = lambda x: x
        Xp, Xvp, Xsp = proj(X), proj(Xv), proj(Xs)
        with torch.no_grad():
            Phi_t_sel = features(Xsp, Gt, Ut)
            imp = Phi_t_sel.pow(2).mean(0) * Dt.pow(2).sum(0)
        key = f"k={k or 'full'}"
        res.setdefault(key, {})
        for method in args.methods.split(","):
            for m in map(int, args.ms.split(",")):
                if (li, key, method, m) in done:
                    continue
                yield_if_asked()
                with torch.no_grad():
                    if method == "random":
                        G, Uw = Gr[:m], Ur[:m]
                    elif method == "importance":
                        idx = imp.topk(m).indices
                        G, Uw = Gt[idx], Ut[idx]
                    elif method == "omp":
                        idx = omp_select(Phi_t_sel, Ys, m)
                        G, Uw = Gt[idx], Ut[idx]
                    elif method == "omp_mixed":
                        Gd, Ud = torch.cat([Gt, Gr]), torch.cat([Ut, Ur])
                        idx = omp_select(features(Xsp, Gd, Ud), Ys, m)
                        G, Uw = Gd[idx], Ud[idx]
                        n_teacher = int((idx < Gt.shape[0]).sum())
                    W, b = lstsq(features(Xp, G, Uw), Y)
                    fit = r2(features(Xvp, G, Uw) @ W + b, Yv)
                approx = lambda h, u, G=G, Uw=Uw, W=W, b=b: features(proj(norm(h)), G, Uw) @ W + b
                kl = S.kl(site, mlp_edit={li: approx})
                entry = {"val_r2": fit, "kl": kl}
                if method == "omp_mixed":
                    entry["n_teacher_neurons"] = n_teacher
                res[key].setdefault(method, {})[str(m)] = entry
                log(f"L{li} {key} {method:10s} m={m:5d}: R2 {fit:.3f}  KL {kl['all']['mean']:.4f}"
                    + (f"  ({n_teacher}/{m} teacher)" if method == "omp_mixed" else ""))
                results["layers"][li] = res
                save_json(results, out_path)

log("done")
