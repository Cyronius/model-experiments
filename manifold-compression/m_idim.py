# M5: intrinsic dimension per site at several neighbourhood sizes (Levina-Bickel MLE), as a lower bound on usable width (2026-10-17).
import argparse
import json

import numpy as np
import torch

from common import RESULTS, WORK, save_json, set_threads

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
ap.add_argument("--n-points", type=int, default=4000)
ap.add_argument("--ks", default="5,10,20,40,80")
ap.add_argument("--threads", type=int, default=12)
args = ap.parse_args()
set_threads(args.threads)

tag = args.model.split("/")[-1]
meta = json.loads((WORK / tag / "meta.json").read_text())
ks = list(map(int, args.ks.split(",")))
g = torch.Generator().manual_seed(0)
rows = torch.randperm(meta["n_tokens"], generator=g)[:args.n_points].sort().values.numpy()


def mle_id(X, k):
    """Levina-Bickel estimate with k neighbours, averaged over points (MacKay-Ghahramani inverse averaging)."""
    D = torch.cdist(X, X)
    D.fill_diagonal_(float("inf"))
    nn_d = D.topk(k, largest=False).values.clamp_min(1e-12)
    inv = torch.log(nn_d[:, -1:] / nn_d[:, :-1]).sum(1) / (k - 1)
    return (1 / inv.mean()).item()


results = {"model": args.model, "ks": ks, "site_names": meta["site_names"], "id": {}}
for s in range(meta["n_sites"]):
    X = torch.from_numpy(np.load(WORK / tag / f"site{s:03d}.npy", mmap_mode="r")[rows].astype(np.float32))
    X = X - X.mean(0)
    results["id"][s] = {k: mle_id(X, k) for k in ks}
    print(meta["site_names"][s], " ".join(f"k{k}:{v:.1f}" for k, v in results["id"][s].items()), flush=True)
save_json(results, RESULTS / tag / "idim.json")
