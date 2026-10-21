# Run the fit split through the model and store the residual state at every site, one .npy per site (2026-10-17).
import argparse
import time

import numpy as np
import torch

from common import add_session_args, batches, save_json, session_from, yield_if_asked

ap = argparse.ArgumentParser()
add_session_args(ap, bs=2)
ap.add_argument("--sites", default="", help="comma list of sites to store; default all")
ap.add_argument("--suffix", default="", help="store under work/<model>-<suffix> instead of work/<model>")
args = ap.parse_args()

S = session_from(args)
r, fit = S.runner, S.data["fit"]
T = S.T
N = len(fit) * (T - 1)
d = r.cfg.hidden_size
if args.suffix:
    S.work = S.work.with_name(f"{S.tag}-{args.suffix}")
S.work.mkdir(parents=True, exist_ok=True)
sites = [int(x) for x in args.sites.split(",")] if args.sites else list(range(r.n_sites))
out = {s: np.lib.format.open_memmap(S.work / f"site{s:03d}.npy", mode="w+", dtype=np.float16, shape=(N, d))
       for s in sites}

t0, row = time.time(), 0
for k, (doms, ids, focus) in enumerate(batches(fit, args.bs)):
    emb, ctx = r.prepare(ids)
    _, states = r.run(emb, 0, ctx, record=set(sites), positions=torch.tensor([0]))
    n = ids.shape[0] * (T - 1)
    for s, h in states.items():
        out[s][row:row + n] = h[:, 1:].reshape(n, d).cpu().numpy().astype(np.float16)
    row += n
    print(f"batch {k + 1}/{-(-len(fit) // args.bs)}  {time.time() - t0:.0f}s", flush=True)
    yield_if_asked()

for m in out.values():
    m.flush()
save_json({
    "model": args.model, "T": T, "n_tokens": N, "d": d, "n_sites": r.n_sites, "stored_sites": sites,
    "layer_types": r.types, "site_names": [r.site_name(s) for s in range(r.n_sites)],
    "token_domain": [dom for dom, _, _ in fit for _ in range(T - 1)],
}, S.work / "meta.json")
print("done", N, "tokens x", r.n_sites, "sites")
