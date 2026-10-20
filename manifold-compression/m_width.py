# M1 + M2: project the residual stream to width k (PCA = the free, SliceGPT-like part) and check how much a small MLP recovers of what was cut (2026-10-17).
import argparse
import time

import torch
import torch.nn as nn

from common import add_session_args, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap, bs=4)
ap.add_argument("--ks", default="0.25,0.375,0.5,0.75", help="widths as fractions of hidden size")
ap.add_argument("--site-stride", type=int, default=4, help="single-site sweeps visit every Nth site")
ap.add_argument("--mlp-hidden", type=int, default=1024)
ap.add_argument("--mlp-steps", type=int, default=400)
ap.add_argument("--skip-mlp", action="store_true")
args = ap.parse_args()

S = session_from(args)
r = S.runner
meta = S.fit_meta()
d, n_sites = meta["d"], meta["n_sites"]
ks = [int(round(f * d)) for f in map(float, args.ks.split(","))]
single_sites = sorted(set(range(0, n_sites, args.site_stride)) | {n_sites - 1})
T1 = meta["T"] - 1
n_seq = meta["n_tokens"] // T1
# Every 10th sequence is held out; the fit split is ordered by domain, so a tail split would hold out one domain.
seq_id = torch.arange(n_seq * T1, device=S.device) // T1
val_rows = (seq_id % 10 == 9).nonzero().squeeze(1)
fit_rows = (seq_id % 10 != 9).nonzero().squeeze(1)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ---- PCA per site ----
pca = {}
for s in range(n_sites):
    X = S.fit_states(s)
    mu = X.mean(0)
    Xc = X - mu
    evals, evecs = torch.linalg.eigh(Xc.T @ Xc / len(Xc))
    order = evals.argsort(descending=True)
    pca[s] = dict(mu=mu, V=evecs[:, order], ev=evals[order].clamp_min(0))
log("pca done")

spectrum = {s: {k: (p["ev"][:k].sum() / p["ev"].sum()).item() for k in ks} for s, p in pca.items()}

# Shared basis: average of trace-normalized per-site covariances, so late (large-norm) sites don't dominate.
C = sum((p["V"] * (p["ev"] / p["ev"].sum())) @ p["V"].T for p in pca.values()) / n_sites
ev_sh, V_sh = torch.linalg.eigh(C)
V_sh = V_sh[:, ev_sh.argsort(descending=True)]


def projector(mu, Vk):
    return lambda h: mu + ((h - mu) @ Vk) @ Vk.T


def proj(s, k, shared=False):
    V = V_sh if shared else pca[s]["V"]
    return projector(pca[s]["mu"], V[:, :k])


def local_err(s, f):
    """Relative error of the next sub-layer's update when its input is replaced by f(h)."""
    if s == n_sites - 1:
        return float("nan")
    i, half = divmod(s, 2)
    num = den = 0.0
    for b in S.eval_batches():
        h = S.state(b, s)
        step = (lambda x: r.mixer(i, x, b["ctx"])) if half == 0 else (lambda x: r.mlp(i, x))
        with torch.no_grad():
            u, u2 = step(h)[:, 1:], step(f(h))[:, 1:]
        num += (u2 - u).pow(2).sum().item()
        den += u.pow(2).sum().item()
    return (num / den) ** 0.5


results = {"model": args.model, "ks": ks, "site_names": meta["site_names"], "spectrum": spectrum,
           "single": {}, "all_sites": {}, "all_shared": {}, "mlp_single": {}, "mlp_all": {}, "mlp_val_r2": {}}
out_path = S.results / "width.json"

# ---- M1 ----
S.eval_batches()
results["noise_floor"] = S.kl(0)
log(f"eval cache built; unedited rerun KL {results['noise_floor']['all']['mean']:.2e} (storage noise floor)")
for k in ks:
    results["all_sites"][k] = S.kl(0, edit={s: proj(s, k) for s in range(n_sites)})
    results["all_shared"][k] = S.kl(0, edit={s: proj(s, k, shared=True) for s in range(n_sites)})
    log(f"all-sites k={k}: KL {results['all_sites'][k]['all']['mean']:.4f}  shared {results['all_shared'][k]['all']['mean']:.4f}")
save_json(results, out_path)
for s in single_sites:
    results["single"][s] = {}
    for k in ks:
        f = proj(s, k)
        results["single"][s][k] = dict(kl=S.kl(s, edit={s: f}), local=local_err(s, f))
    log(f"single {meta['site_names'][s]}: " + "  ".join(f"k={k}:{results['single'][s][k]['kl']['all']['mean']:.4f}" for k in ks))
    save_json(results, out_path)

# ---- M2: MLP decoder from the kept coordinates to the discarded part ----
class Decoder(nn.Module):
    def __init__(self, k, H, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, H), nn.GELU(), nn.Linear(H, d))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, z):
        return self.net(z)


def train_decoder(s, k):
    p = pca[s]
    Vk = p["V"][:, :k]
    Xc = S.fit_states(s) - p["mu"]
    Z = Xc @ Vk
    R = Xc - Z @ Vk.T
    zs = Z[fit_rows].std(0) + 1e-6
    rs = R[fit_rows].pow(2).mean().sqrt() + 1e-6
    torch.manual_seed(0)
    dec = Decoder(k, args.mlp_hidden, d).to(S.device)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.mlp_steps)
    with torch.enable_grad():
        for _ in range(args.mlp_steps):
            idx = fit_rows[torch.randint(len(fit_rows), (1024,), device=S.device)]
            loss = (dec(Z[idx] / zs) - R[idx] / rs).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        pred = dec(Z[val_rows] / zs) * rs
        r2 = 1 - (pred - R[val_rows]).pow(2).sum() / R[val_rows].pow(2).sum()
    mu = p["mu"]

    if r2 <= 0:
        return projector(mu, Vk), r2.item()

    def f(h):
        z = (h - mu) @ Vk
        return mu + z @ Vk.T + dec(z / zs) * rs

    return f, r2.item()


if not args.skip_mlp:
    for k in ks:
        fs = {}
        results["mlp_val_r2"][k] = {}
        for s in range(n_sites):
            fs[s], results["mlp_val_r2"][k][s] = train_decoder(s, k)
        log(f"decoders k={k}: median val R2 of the cut part {torch.tensor(list(results['mlp_val_r2'][k].values())).median():.3f}")
        results["mlp_all"][k] = S.kl(0, edit=fs)
        log(f"mlp all-sites k={k}: KL {results['mlp_all'][k]['all']['mean']:.4f}")
        for s in single_sites:
            results["mlp_single"].setdefault(s, {})[k] = dict(kl=S.kl(s, edit={s: fs[s]}), local=local_err(s, fs[s]))
        log(f"mlp single k={k}: " + "  ".join(f"{s}:{results['mlp_single'][s][k]['kl']['all']['mean']:.4f}" for s in single_sites))
        save_json(results, out_path)

save_json(results, out_path)
log("done")
