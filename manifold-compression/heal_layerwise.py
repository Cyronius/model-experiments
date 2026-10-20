# Heal a width-k compressed residual stream one half-layer at a time, then measure all-sites KL (2026-10-18).
import argparse
import time

import torch
import torch.nn as nn

from common import add_session_args, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap)
ap.add_argument("--k", type=int, default=512)
ap.add_argument("--warp", action="store_true", help="learned encoder + decoder at every site, trained jointly with the blocks")
ap.add_argument("--hidden", type=int, default=2048, help="warp hidden width")
ap.add_argument("--warp-steps", type=int, default=3000)
ap.add_argument("--steps", type=int, default=300, help="healing steps per half-layer")
ap.add_argument("--batch", type=int, default=8, help="sequences per healing step")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--warp-lr", type=float, default=1e-4)
ap.add_argument("--layers", type=int, default=0, help="heal only the first N layers (0 = all); the rest stay original")
ap.add_argument("--tag", default="")
args = ap.parse_args()

S = session_from(args)
r, dev = S.runner, S.device
L, T, d, k = r.L, S.T, r.cfg.hidden_size, args.k
fit = S.data["fit"]
IDS = torch.stack([torch.tensor(x[1]) for x in fit])
n = len(fit)
print(f"{n} fit sequences, T={T}, d={d}, k={k}, warp={args.warp}", flush=True)
S.eval_batches()
for p in S.model.parameters():
    p.requires_grad_(False)


def mlp(i, o, hidden):
    net = nn.Sequential(nn.Linear(i, hidden), nn.GELU(), nn.Linear(hidden, o))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class Compressor(nn.Module):
    """PCA projection to k dims, plus (optionally) a learned encoder/decoder pair; works on [..., d]."""

    def __init__(self, X):
        super().__init__()
        mu = X.mean(0)
        Xc = X - mu
        ev, V = torch.linalg.eigh(Xc.T @ Xc / len(Xc))
        Vk = V[:, ev.argsort(descending=True)[:k]]
        Z = Xc @ Vk
        R = Xc - Z @ Vk.T
        self.register_buffer("mu", mu)
        self.register_buffer("Vk", Vk)
        self.register_buffer("zs", Z.std(0))
        self.register_buffer("rs", R.pow(2).mean().sqrt())
        self.register_buffer("xs", Xc.pow(2).mean().sqrt())
        self.enc = self.dec = None
        if args.warp:
            self.enc, self.dec = mlp(d, k, args.hidden), mlp(k, d, args.hidden)
            self.to(dev)
            self.pretrain(Xc, Z, R)

    def pretrain(self, Xc, Z, R):
        torch.manual_seed(0)
        opt = torch.optim.AdamW(list(self.enc.parameters()) + list(self.dec.parameters()), lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.warp_steps)
        for _ in range(args.warp_steps):
            idx = torch.randint(len(Xc), (1024,), device=Xc.device)
            z = Z[idx] / self.zs + self.enc(Xc[idx] / self.xs)
            loss = (self.dec(z) - R[idx] / self.rs).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    def forward(self, h):
        x = h - self.mu
        base = (x @ self.Vk) @ self.Vk.T
        if self.enc is None:
            return self.mu + base
        z = (x @ self.Vk) / self.zs + self.enc(x / self.xs)
        return self.mu + base + self.dec(z) * self.rs


def tokens(stream):
    return stream[:, 1:].reshape(-1, d).float()


def sub_update(s, h, ctx):
    i, half = divmod(s, 2)
    return r.mixer(i, h, ctx) if half == 0 else r.mlp(i, h)


def sub_params(s):
    layer = r.lm.layers[s // 2]
    mods = [layer.input_layernorm, layer.linear_attn if r.types[s // 2] == "linear_attention" else layer.self_attn] if s % 2 == 0 \
        else [layer.post_attention_layernorm, layer.mlp]
    return [p for m in mods for p in m.parameters()]


def splice(h_clean, h_new):
    return torch.cat([h_clean[:, :1], h_new[:, 1:]], 1)


@torch.no_grad()
def embed_all():
    return torch.cat([r.prepare(IDS[i:i + 8])[0].half() for i in range(0, n, 8)])


@torch.no_grad()
def teacher_step(s, h):
    out = []
    for i in range(0, n, 8):
        hb = h[i:i + 8].float()
        _, ctx = r.prepare(IDS[i:i + 8])
        out.append((hb + sub_update(s, hb, ctx)).half())
    return torch.cat(out)


t0 = time.time()
n_heal = (args.layers or L) * 2
print("pass 1: fit the compressors on the teacher stream", flush=True)
comps = {}
h_t = embed_all()
comps[0] = Compressor(tokens(h_t)).to(dev)
for s in range(2 * L):
    h_t = teacher_step(s, h_t)
    comps[s + 1] = Compressor(tokens(h_t)).to(dev)
    if s % 8 == 0:
        print(f"  site {s + 1} fitted  {time.time() - t0:.0f}s", flush=True)
for c in comps.values():
    c.eval()
before = S.kl(0, edit={s: c for s, c in comps.items()})
print(f"before healing: KL {before['all']['mean']:.4f}  copy {before['copy']['mean']:.4f}", flush=True)

out = {"args": vars(args), "before": before, "curve": [], "mse": []}
print("pass 2: heal", flush=True)
h_t = embed_all()
with torch.no_grad():
    h_s = torch.cat([splice(h_t[i:i + 8].float(), comps[0](h_t[i:i + 8].float())).half() for i in range(0, n, 8)])
for s in range(n_heal):
    target = teacher_step(s, h_t)
    params = sub_params(s)
    for p in params:
        p.requires_grad_(True)
    groups = [{"params": params, "lr": args.lr}]
    comp = comps[s + 1]
    if args.warp:
        wp = list(comp.enc.parameters()) + list(comp.dec.parameters())
        for p in wp:
            p.requires_grad_(True)
        groups.append({"params": wp, "lr": args.warp_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    var = tokens(target[: min(n, 64)]).var(0).mean()
    first = last = None
    for step in range(args.steps):
        idx = torch.randperm(n)[:args.batch]
        hb, tb = h_s[idx].float(), target[idx].float()
        _, ctx = r.prepare(IDS[idx])
        pred = splice(hb, comp(hb + sub_update(s, hb, ctx)))
        loss = (pred[:, 1:] - tb[:, 1:]).pow(2).mean() / var
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    for p in params:
        p.requires_grad_(False)
    if args.warp:
        for p in wp:
            p.requires_grad_(False)
    with torch.no_grad():
        nxt = []
        for i in range(0, n, 8):
            hb = h_s[i:i + 8].float()
            _, ctx = r.prepare(IDS[i:i + 8])
            nxt.append(splice(hb, comp(hb + sub_update(s, hb, ctx))).half())
        h_s = torch.cat(nxt)
    h_t = target
    out["mse"].append({"site": s + 1, "first": first, "last": last})
    msg = f"half-layer {s}: loss {first:.3f} -> {last:.3f}  {time.time() - t0:.0f}s"
    if s % 2 == 1:
        kl = S.kl(0, edit={j: comps[j] for j in range(s + 2)})
        out["curve"].append({"layer": s // 2, "kl": kl["all"]["mean"], "copy": kl["copy"]["mean"]})
        msg += f"  KL after layer {s // 2}: {kl['all']['mean']:.4f}"
    print(msg, flush=True)
    save_json(out, S.results / f"heal{args.tag}.json")

after = S.kl(0, edit={s: c for s, c in comps.items()})
out["after"] = after
print("ALL SITES after healing: KL " + f"{after['all']['mean']:.4f}  " + "  ".join(f"{dom} {after[dom]['mean']:.4f}" for dom in after if dom != "all"), flush=True)
print("before healing:          KL " + f"{before['all']['mean']:.4f}  " + "  ".join(f"{dom} {before[dom]['mean']:.4f}" for dom in before if dom != "all"), flush=True)
save_json(out, S.results / f"heal{args.tag}.json")
