# M3: own-state test. Can each mixing layer's update be predicted from the token's own state, without context? (2026-10-17)
import argparse
import json
import time

import torch
import torch.nn as nn

from common import add_session_args, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap, bs=4)
ap.add_argument("--mlp-hidden", type=int, default=2048)
ap.add_argument("--mlp-steps", type=int, default=600)
ap.add_argument("--ridge", type=float, default=1e-2)
ap.add_argument("--greedy-candidates", type=int, default=3)
ap.add_argument("--greedy-budget", type=float, default=0.5, help="stop greedy substitution once the score exceeds this (nats)")
ap.add_argument("--greedy-score", choices=["mean", "mean+copy_p99"], default="mean",
                help="what greedy minimizes; mean+copy_p99 also charges the worst 1%% of copy-task answer tokens")
ap.add_argument("--greedy-only", action="store_true", help="reuse per-layer KL from results/boundary.json; refit predictors only")
ap.add_argument("--layers", default="", help="comma list to restrict to (smoke tests); default all")
args = ap.parse_args()

S = session_from(args)
r = S.runner
meta = S.fit_meta()
d, L = meta["d"], r.L
T1 = meta["T"] - 1
n_seq = meta["n_tokens"] // T1
# Every 10th sequence is held out; the fit split is ordered by domain, so a tail split would hold out one domain.
seq_id = torch.arange(n_seq * T1, device=S.device) // T1
val_rows = (seq_id % 10 == 9).nonzero().squeeze(1)
fit_rows = (seq_id % 10 != 9).nonzero().squeeze(1)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


class Predictor(nn.Module):
    """Ridge map plus a zero-initialized MLP correction on standardized input; the MLP is kept only if it helps on held-out data."""

    def __init__(self, W, b, xm, xs, H):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.lin.weight.data, self.lin.bias.data = W.T.contiguous(), b
        self.xm, self.xs = xm, xs
        self.mlp = nn.Sequential(nn.Linear(d, H), nn.GELU(), nn.Linear(H, d))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        self.use_mlp = True

    def forward(self, x, use_mlp=None):
        y = self.lin(x)
        if self.use_mlp if use_mlp is None else use_mlp:
            y = y + self.mlp((x - self.xm) / self.xs)
        return y


def r2(pred, y):
    return (1 - (pred - y).pow(2).sum() / (y - y.mean(0)).pow(2).sum()).item()


def fit_layer(i):
    norm = r.lm.layers[i].input_layernorm
    with torch.no_grad():
        X = norm(S.fit_states(2 * i))
    U = S.fit_states(2 * i + 1) - S.fit_states(2 * i)
    Xf, Uf, Xv, Uv = X[fit_rows], U[fit_rows], X[val_rows], U[val_rows]
    xm, um = Xf.mean(0), Uf.mean(0)
    A = (Xf - xm).T @ (Xf - xm)
    A += args.ridge * A.diagonal().mean() * torch.eye(d, device=A.device)
    W = torch.linalg.solve(A, (Xf - xm).T @ (Uf - um))
    b = um - xm @ W
    torch.manual_seed(0)
    P = Predictor(W, b, xm, Xf.std(0) + 1e-6, args.mlp_hidden).to(S.device)
    us = Uf.pow(2).mean().sqrt()
    opt = torch.optim.Adam(P.mlp.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.mlp_steps)
    with torch.enable_grad():
        for _ in range(args.mlp_steps):
            idx = torch.randint(len(Xf), (1024,), device=S.device)
            loss = ((P(Xf[idx]) - Uf[idx]) / us).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        fit = {"r2_linear": r2(P(Xv, False), Uv), "r2_mlp": r2(P(Xv, True), Uv),
               "update_rel_norm": (Uv.norm(dim=1) / S.fit_states(2 * i)[val_rows].norm(dim=1)).mean().item()}
    P.use_mlp = fit["r2_mlp"] > fit["r2_linear"]
    fit["used_mlp"] = P.use_mlp
    return P, um, fit


out_path = S.results / "boundary.json"
if args.greedy_only:
    results = json.loads(out_path.read_text())
    results["layers"] = {int(i): v for i, v in results["layers"].items()}
else:
    results = {"model": args.model, "layer_types": r.types, "layers": {}}
greedy_key = "greedy" if args.greedy_score == "mean" else "greedy_tail"
results[greedy_key] = []
S.eval_batches()
log("eval cache built")

layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(L))
preds = {}
for i in layers:
    P, um, fit = fit_layer(i)
    preds[i] = P
    if args.greedy_only:
        log(f"L{i} predictor refit (R2 {fit['r2_mlp'] if fit['used_mlp'] else fit['r2_linear']:.3f})")
        continue
    norm = r.lm.layers[i].input_layernorm
    sub = (lambda P, norm: lambda h, u: P(norm(h)))(P, norm)
    res = dict(fit, kind=r.types[i],
               kl_predict=S.kl(2 * i, mixer_edit={i: sub}),
               kl_drop=S.kl(2 * i, mixer_edit={i: lambda h, u: torch.zeros_like(u)}),
               kl_mean=S.kl(2 * i, mixer_edit={i: (lambda um: lambda h, u: um.expand_as(u))(um)}))
    results["layers"][i] = res
    log(f"L{i} {r.types[i][:6]}  R2 lin {fit['r2_linear']:.3f} mlp {fit['r2_mlp']:.3f}  "
        f"KL predict {res['kl_predict']['all']['mean']:.4f} drop {res['kl_drop']['all']['mean']:.4f}  "
        f"copy p99 predict {res['kl_predict'].get('copy', {}).get('p99', float('nan')):.3f}")
    save_json(results, out_path)


# Greedy: keep substituting the layer whose substitution costs least given the ones already substituted.
def sub_fn(i):
    norm = r.lm.layers[i].input_layernorm
    return lambda h, u: preds[i](norm(h))


def score(rep):
    s = rep["all"]["mean"]
    if args.greedy_score == "mean+copy_p99" and "copy" in rep:
        s += rep["copy"]["p99"]
    return s


chosen, remaining = [], list(layers)
while remaining:
    remaining.sort(key=lambda i: score(results["layers"][i]["kl_predict"]))
    best = None
    for i in remaining[:args.greedy_candidates]:
        rep = S.kl(0, mixer_edit={j: sub_fn(j) for j in chosen + [i]})
        if best is None or score(rep) < score(best[1]):
            best = (i, rep)
    i, rep = best
    chosen.append(i)
    remaining.remove(i)
    results[greedy_key].append({"layer": i, "kind": r.types[i], "n_substituted": len(chosen), "kl": rep, "score": score(rep)})
    log(f"{greedy_key} +L{i} ({r.types[i][:6]}): {len(chosen)} substituted, KL {rep['all']['mean']:.4f}  "
        f"copy p99 {rep.get('copy', {}).get('p99', float('nan')):.3f}")
    save_json(results, out_path)
    if score(rep) > args.greedy_budget:
        break

log("done")
