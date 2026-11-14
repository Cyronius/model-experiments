"""Stage 2 step 4: seeds.

c(x) = V^T (teacher(x) - mu) with the 1024 PCA directions (the 512 model uses
the first 512 coordinates; nested).
  u[t] = c(t), w[t] = a / (a + p[t]), a = 0.001 (special/unused tokens: u = 0)
  bigram g=(t1,t2): r[g] = (w1 + w2) c(g) - w1 u1 - w2 u2           (first 512 coords)
  trigram g=(t1,t2,t3): r[g] = (w1+w2+w3) c(g) - sum w u - r[(t1,t2)] - r[(t2,t3)]
      (bigram residuals only if those bigrams are in the table)
so that the model on any seeded item alone outputs normalize(V c + mu), the
rank-d projection of the teacher vector (for d = 1024: exact on the first 512
representation coordinates for n-grams, which is all a 512-wide residual can
carry).

This module provides the c vectors (written once) and functions that compute
residual seeds from a model's current unigram and bigram rows, so a rung that
adds trigrams to a trained bigram model is seeded relative to that model.

Usage: python -I stage2/seeds.py   (writes c vectors + runs the assertion)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

S1 = C.WORK / "stage1"
S2 = C.WORK / "stage2"
A = 0.001


def basis():
    return np.load(S2 / "mu.npy"), np.load(S2 / "V.npy")


def coords(vecs16, mu, V, bs=50000):
    out = np.zeros((len(vecs16), V.shape[1]), np.float32)
    for b in range(0, len(vecs16), bs):
        x = vecs16[b:b + bs].astype(np.float32)
        ok = np.abs(x).sum(1) > 0
        out[b:b + bs][ok] = (x[ok] - mu) @ V
    return out


def w_seed():
    p = np.load(S1 / "unigram_p.npy")
    return (A / (A + p)).astype(np.float32)


def bigram_seeds(keys, c_g, u, w, wng=512):
    t1 = (keys >> 15) & 0x7FFF
    t2 = keys & 0x7FFF
    W = w[t1] + w[t2]
    return (W[:, None] * c_g[:, :wng] - w[t1, None] * u[t1, :wng] - w[t2, None] * u[t2, :wng]).astype(np.float32)


def trigram_seeds(keys, c_g, u, w, bi_keys=None, bi_rows=None, wng=512):
    t1 = (keys >> 30) & 0x7FFF
    t2 = (keys >> 15) & 0x7FFF
    t3 = keys & 0x7FFF
    W = w[t1] + w[t2] + w[t3]
    r = W[:, None] * c_g[:, :wng] - sum(w[t, None] * u[t, :wng] for t in (t1, t2, t3))
    if bi_keys is not None:
        o = np.argsort(bi_keys)
        sk = bi_keys[o]
        for bk in ((t1 << 15) | t2, (t2 << 15) | t3):
            j = np.searchsorted(sk, bk).clip(max=len(sk) - 1)
            f = sk[j] == bk
            r[f] -= bi_rows[o[j[f]]]
    return r.astype(np.float32)


def main():
    mu, V = basis()
    res = {}
    cv = coords(np.load(S2 / "seedvec_vocab.npy"), mu, V)
    np.save(S2 / "c_vocab.npy", cv)
    for n in (2, 3):
        c = coords(np.load(S2 / f"seedvec_n{n}_T1.npy", mmap_mode="r"), mu, V)
        np.save(S2 / f"c_n{n}_T1.npy", c)
    w = w_seed()
    np.save(S2 / "w_seed.npy", w)
    u = cv
    k2 = np.load(S1 / "ranked_n2_keys.npy")[:250_000]
    k3 = np.load(S1 / "ranked_n3_keys.npy")[:250_000]
    c2 = np.load(S2 / "c_n2_T1.npy")
    c3 = np.load(S2 / "c_n3_T1.npy")
    r2 = bigram_seeds(k2, c2, u, w)
    r3 = trigram_seeds(k3, c3, u, w, k2, r2)
    res["seed_norms"] = {"u_mean": float(np.linalg.norm(u, axis=1).mean()),
                         "r2_mean": float(np.linalg.norm(r2, axis=1).mean()),
                         "r3_mean": float(np.linalg.norm(r3, axis=1).mean()),
                         "w_min": float(w.min()), "w_median": float(np.median(w))}
    # numerical assertion with the torch model on 1,000 random seeded items
    import torch
    sys.path.insert(0, str(C.PLAN))
    from bmodel import Student, make_batch, inv_softplus
    tok = C.tokenizer()
    rng = np.random.default_rng(0)
    vocab_ok = np.nonzero(np.abs(u).sum(1) > 0)[0]
    items = [("u", int(i)) for i in rng.choice(vocab_ok, 400, replace=False)] + \
            [("b", int(i)) for i in rng.choice(250_000, 300, replace=False)] + \
            [("t", int(i)) for i in rng.choice(250_000, 300, replace=False)]
    for d in (512, 1024):
        m = Student(d=d, n2_keys=k2, n3_keys=k3)
        with torch.no_grad():
            m.u.weight.copy_(torch.from_numpy(u[:, :d]))
            m.theta.weight.copy_(inv_softplus(torch.from_numpy(w))[:, None])
            m.n2.rows.weight.copy_(torch.from_numpy(r2))
            m.n3.rows.weight.copy_(torch.from_numpy(r3))
            m.V.copy_(torch.from_numpy(V[:, :d]))
            m.mu.copy_(torch.from_numpy(mu))
        seqs, want = [], []
        for kind, i in items:
            if kind == "u":
                seqs.append(np.array([i])); want.append(u[i])
            elif kind == "b":
                seqs.append(np.array(C.unpack(k2[i], 2))); want.append(c2[i])
            else:
                seqs.append(np.array(C.unpack(k3[i], 3))); want.append(c3[i])
        want = np.stack(want)[:, :d]
        if d == 1024:
            # n-gram residuals carry the first 512 coords only: compare there,
            # and compare the full z for unigrams
            pass
        with torch.no_grad():
            _, z = m(make_batch(seqs, "cpu"))
        z = z.numpy()
        kinds = np.array([k for k, _ in items])
        err = {}
        for k in ("u", "b", "t"):
            sel = kinds == k
            cols = slice(0, d) if k == "u" else slice(0, 512)
            e = np.abs(z[sel][:, cols] - want[sel][:, cols]).max()
            rel = e / np.abs(want[sel][:, cols]).max()
            err[k] = {"max_abs": float(e), "max_rel": float(rel)}
            assert rel < 1e-4, (d, k, e, rel)
        # end-to-end: output vs projected teacher
        Vd = V[:, :d]
        proj = C.unit(want @ Vd.T + mu)
        out = C.unit((z @ Vd.T) + mu)
        cs = (proj * out).sum(1)
        err["min_cos_output_vs_projection"] = float(cs[kinds == "u"].min()) if d else None
        res[f"assert_d{d}"] = err
    json.dump(res, open(S2 / "seeds.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
