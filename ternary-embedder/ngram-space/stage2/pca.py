"""Stage 2 step 2: fit mu and V (top 1024 principal directions) on 300,000
training vectors of the Jaswanth set (dequantized, re-normalized).

The 300,000 rows are a random subset of the Stage 0 training rows (seed 0); the
same subset is Stage 3's first training set (train300k_rows.npy).

Usage: python -I stage2/pca.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

OUT = C.WORK / "stage2"


def targets(rows):
    jw = C.WORK / "jaswanth"
    codes = np.load(jw / "codes.npy", mmap_mode="r")
    am = np.load(jw / "absmax.npy")
    rows = np.sort(rows)
    x = codes[rows].astype(np.float32) * (am[rows, None] / 127.0)
    return rows, C.unit(x)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train = np.load(C.WORK / "stage0" / "train_rows.npy")
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(train, 300_000, replace=False))
    np.save(OUT / "train300k_rows.npy", sub)
    _, X = targets(sub)
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / len(Xc)
    ev, evec = np.linalg.eigh(cov.astype(np.float64))
    o = np.argsort(-ev)
    ev, evec = ev[o], evec[:, o]
    V = evec[:, :1024].astype(np.float32)
    # deterministic sign: largest-magnitude loading positive
    s = np.sign(V[np.abs(V).argmax(0), np.arange(1024)])
    V *= s
    np.save(OUT / "mu.npy", mu.astype(np.float32))
    np.save(OUT / "V.npy", V)
    np.save(OUT / "eigvals.npy", ev.astype(np.float32))
    tot = ev.sum()
    # reconstruction of validation vectors
    val = np.load(C.WORK / "stage0" / "val_rows.npy")
    _, Y = targets(val)
    res = {"rows": 300_000, "mu_norm": float(np.linalg.norm(mu)),
           "explained_var_512": float(ev[:512].sum() / tot), "explained_var_1024": float(ev[:1024].sum() / tot)}
    for d in (512, 1024):
        Vd = V[:, :d]
        R = C.unit((Y - mu) @ Vd @ Vd.T + mu)
        cs = (R * Y).sum(1)
        res[f"val_cos_d{d}"] = {"mean": float(cs.mean()), "p5": float(np.percentile(cs, 5))}
        for k in (1024,):
            ck = (C.unit(R[:, :k]) * C.unit(Y[:, :k])).sum(1)
            res[f"val_cos_d{d}_prefix{k}"] = {"mean": float(ck.mean()), "p5": float(np.percentile(ck, 5))}
    res["seconds"] = time.time() - t0
    json.dump(res, open(OUT / "pca.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
