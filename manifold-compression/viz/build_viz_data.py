# Export a 3D view of the residual-stream manifold at every site of Qwen3.5-0.8B, plus measured stats, for manifold_viewer.html (2026-10-18).
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import RESULTS, WORK, _local_or_download, build_data  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TAG = "Qwen3.5-0.8B"
N_POINTS = 1500
N_TRACKS = 40
OUT = Path(__file__).resolve().parent / "viz_data.json"

meta = json.loads((WORK / TAG / "meta.json").read_text())
T1, n_sites, d = meta["T"] - 1, meta["n_sites"], meta["d"]
domains = meta["token_domain"]
N = meta["n_tokens"]

# Same tokens at every site, spread evenly over domains, so a dot can be followed through depth.
rng = np.random.default_rng(0)
by_dom = {}
for i, dom in enumerate(domains):
    by_dom.setdefault(dom, []).append(i)
per = N_POINTS // len(by_dom)
rows = np.sort(np.concatenate([rng.choice(v, min(per, len(v)), replace=False) for v in by_dom.values()]))
tracks = rng.choice(len(rows), N_TRACKS, replace=False)

tok = AutoTokenizer.from_pretrained(_local_or_download(meta["model"]))
fit = build_data(tok, T=meta["T"], n_fit=16, n_eval=4)["fit"]
labels = []
for r in rows:
    seq, pos = divmod(int(r), T1)
    ids = fit[seq][1]
    p = pos + 1
    before = tok.decode(ids[max(0, p - 10):p]).replace("\n", " ⏎ ")[-48:]
    labels.append({"t": tok.decode([ids[p]]).replace("\n", "⏎"), "c": before, "p": p, "d": domains[r]})


def load(s):
    return torch.from_numpy(np.load(WORK / TAG / f"site{s:03d}.npy", mmap_mode="r")[:].astype(np.float32))


sites, prev = [], None
pooled = torch.zeros(d, d)
for s in range(n_sites):
    X = load(s)
    mu = X.mean(0)
    Xc = X - mu
    C = Xc.T @ Xc / len(Xc)
    ev, V = torch.linalg.eigh(C)
    order = ev.argsort(descending=True)
    ev, V = ev[order].clamp_min(0), V[:, order]
    pooled += C / ev.sum()
    Y = (Xc[rows] @ V[:, :3]).numpy()
    Y /= np.sqrt((Y ** 2).sum(1).mean())
    if prev is not None:
        # Rotate (and maybe mirror) to line up with the previous site, so stepping through layers doesn't flip the cloud.
        U, _, Wt = np.linalg.svd(Y.T @ prev)
        Y = Y @ (U @ Wt)
    prev = Y
    frac = (ev / ev.sum()).numpy()
    cum = np.cumsum(frac)
    sites.append({
        "name": meta["site_names"][s],
        "xyz": np.round(Y, 3).tolist(),
        "var3": round(float(cum[2]), 4),
        "var1": round(float(frac[0]), 4),
        "k90": int(np.searchsorted(cum, 0.90) + 1),
        "k99": int(np.searchsorted(cum, 0.99) + 1),
        "norm": round(float(X.norm(dim=1).mean()), 3),
    })
    print(meta["site_names"][s], sites[-1]["var3"], sites[-1]["k90"], flush=True)

# One shared basis for all sites, for drawing token paths through depth on the same axes.
evp, Vp = torch.linalg.eigh(pooled / n_sites)
B = Vp[:, evp.argsort(descending=True)[:3]]
paths = np.zeros((N_TRACKS, n_sites, 3))
for s in range(n_sites):
    X = load(s)
    Z = ((X[rows[tracks]] - X.mean(0)) / X.norm(dim=1).mean()) @ B
    paths[:, s] = Z.numpy()
paths /= np.sqrt((paths ** 2).sum(-1).mean())


def res(name):
    p = RESULTS / TAG / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


width, idim, pert, bnd = res("width"), res("idim"), res("perturb"), res("boundary")
for s, site in enumerate(sites):
    st = {}
    w = width.get("single", {}).get(str(s))
    if w:
        st["kl_quarter"] = round(w["256"]["kl"]["all"]["mean"], 4)
        st["kl_half"] = round(w["512"]["kl"]["all"]["mean"], 4)
    if str(s) in idim.get("id", {}):
        st["idim"] = round(idim["id"][str(s)]["20"], 1)
    p = pert.get("runs", {}).get(str(s))
    if p:
        st["pull_on"] = round(p["on@0.1"]["rel_ratio"], 2)
        st["pull_off"] = round(p["off@0.1"]["rel_ratio"], 2)
    layer, half = divmod(s, 2)
    b = bnd.get("layers", {}).get(str(layer))
    if b and half == 0 and s < n_sites - 1:
        st["own_r2"] = round(max(b["r2_linear"], b["r2_mlp"]), 2)
        st["sub_kl"] = round(b["kl_predict"]["all"]["mean"], 4)
        st["drop_kl"] = round(b["kl_drop"]["all"]["mean"], 4)
        st["copy_p99"] = round(b["kl_predict"].get("copy", {}).get("p99", float("nan")), 3)
    site["stats"] = st

OUT.write_text(json.dumps({
    "model": meta["model"], "d": d, "layer_types": meta["layer_types"], "n_tokens": N,
    "labels": labels, "tracks": tracks.tolist(), "paths": np.round(paths, 3).tolist(), "sites": sites,
}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote", OUT, OUT.stat().st_size // 1024, "KB")
