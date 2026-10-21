# M4: push the state off (or along) the manifold at one site and track whether later layers shrink or grow the deviation (2026-10-17).
import argparse
import time

import torch

from common import KLAccumulator, add_session_args, kl_per_token, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap, bs=4)
ap.add_argument("--site-stride", type=int, default=2)
ap.add_argument("--eps", default="0.05,0.2", help="perturbation size as a fraction of the site's typical centered norm")
ap.add_argument("--on-k", type=float, default=0.0625, help="'on-manifold' = span of the top fraction of PCs")
ap.add_argument("--off-k", type=float, default=0.5, help="'off-manifold' = span of PCs below this fraction")
args = ap.parse_args()

S = session_from(args)
r = S.runner
meta = S.fit_meta()
d, n_sites = meta["d"], meta["n_sites"]
eps_list = list(map(float, args.eps.split(",")))
sites = list(range(0, n_sites - 1, args.site_stride))
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def basis(s):
    X = S.fit_states(s)
    Xc = X - X.mean(0)
    ev, V = torch.linalg.eigh(Xc.T @ Xc / len(Xc))
    V = V[:, ev.argsort(descending=True)]
    return V, Xc.norm(dim=1).mean().item()


results = {"model": args.model, "site_names": meta["site_names"], "eps": eps_list, "runs": {}}
out_path = S.results / "perturb.json"
S.eval_batches()
log("eval cache built")

for s in sites:
    V, scale = basis(s)
    dirs = {"on": V[:, :int(args.on_k * d)], "off": V[:, int(args.off_k * d):]}
    results["runs"][s] = {}
    for kind, B in dirs.items():
        for eps in eps_list:
            g = torch.Generator().manual_seed(s)
            acc = KLAccumulator()
            dev = torch.zeros(n_sites)
            rel = torch.zeros(n_sites)
            cnt = 0
            for b in S.eval_batches():
                h = S.state(b, s)
                delta = torch.randn(*h.shape[:2], B.shape[1], generator=g).to(S.device) @ B.T
                delta = delta / delta.norm(dim=-1, keepdim=True) * eps * scale
                logits, rec = r.run(h, s, b["ctx"], edit={s: lambda x: x + delta},
                                    record=set(range(s, n_sites)), positions=b["pos"])
                acc.add(kl_per_token(b["ref_lp"].to(S.device), logits), b["doms"], b["focus"])
                for t, ht in rec.items():
                    ref = S.state(b, t)
                    diff = (ht - ref)[:, 1:]
                    dev[t] += diff.norm(dim=-1).sum().item()
                    rel[t] += (diff.norm(dim=-1) / ref[:, 1:].norm(dim=-1)).sum().item()
                cnt += h.shape[0] * (h.shape[1] - 1)
            inj = eps * scale
            results["runs"][s][f"{kind}@{eps}"] = {
                "kl": acc.report(),
                "gain": (dev[s:] / cnt / inj).tolist(),
                "rel_dev": (rel[s:] / cnt).tolist(),
                "rel_ratio": (rel[-1] / rel[s]).item(),
            }
            gain = results["runs"][s][f"{kind}@{eps}"]["gain"]
            log(f"{meta['site_names'][s]} {kind}@{eps}: relative deviation final/injected "
                f"{results['runs'][s][f'{kind}@{eps}']['rel_ratio']:.2f}  abs gain {gain[-1]:.2f}  "
                f"KL {results['runs'][s][f'{kind}@{eps}']['kl']['all']['mean']:.4f}")
    save_json(results, out_path)

log("done")
