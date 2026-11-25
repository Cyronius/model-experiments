"""Train dense / loop / oploop on Big Muff-family pedal circuits (gen_analog.py data).

Each node token predicts its voltage at 16 phases of the last input period. Loss: Huber on volts,
weighted by the halting distribution, plus a cost per expected step (node tokens only).

  python train_analog.py --kind oploop --K 8 --name a-oploop-k8
Results: results/analog/<name>.json and a line in results/analog/results.jsonl.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from gen_analog import LABELS, ROLES, TYPES
from model import Net, halt_step, halting_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "npu-kernel-coder", "train"))
import memguard  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="oploop", choices=["dense", "oploop", "moe"])
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--name", required=True)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--T", type=int, default=24)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--beta", type=float, default=0.0005, help="cost per expected step (loss is in volts)")
ap.add_argument("--beta-start", type=float, default=0.4)
ap.add_argument("--kl", type=float, default=0.0, help="weight of KL(halting || geometric prior), on from step 0")
ap.add_argument("--prior", type=float, default=0.2)
ap.add_argument("--load", default=None, help="state dict to start from (with --steps 0: evaluate only)")
ap.add_argument("--aux", type=float, default=0.01)
ap.add_argument("--tau", default="2.0,0.5")
ap.add_argument("--freeze-frac", type=float, default=0.5)
ap.add_argument("--fourier", type=int, default=6)
ap.add_argument("--mse", type=float, default=0.0, help="weight of an extra MSE term")
ap.add_argument("--huber", type=float, default=0.05, help="Huber transition, volts")
ap.add_argument("--asinh", type=float, default=0.0, help="if >0: predict asinh(target/S), volts = S*sinh(out)")
ap.add_argument("--edge-frac", type=float, default=0.0, help="fraction of each batch drawn from rows with a big jump")
ap.add_argument("--edge-thr", type=float, default=0.1, help="volts: a row is an edge row if any node moves more than this")
ap.add_argument("--eval-every", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "analog"))
args = ap.parse_args()

memguard.start(12)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = os.environ.get("DEV", "cuda")
here = os.path.dirname(os.path.abspath(__file__))


def load(name):
    d = np.load(os.path.join(here, "cache", os.environ.get("DATA_DIR", "analog"), f"{name}.npz"))
    return {("ctype" if k == "type" else k): torch.from_numpy(d[k]) for k in d.files}  # "type" clashes with nn.Module


train, test, ood = load(os.environ.get("TRAIN_SET", "train")), load(os.environ.get("TEST_SET", "test")), load("ood")
NTR = len(train["ctype"])
print(f"train {NTR} circuits", flush=True)
EDGE = torch.nonzero(train["target"].abs().flatten(1).amax(1) > args.edge_thr)[:, 0].numpy() if args.edge_frac else None
if EDGE is not None:
    print(f"edge rows: {len(EDGE)} ({len(EDGE) / NTR:.1%})", flush=True)


def enc(t):  # target volts -> training space
    return torch.asinh(t / args.asinh) if args.asinh else t


def dec(y):  # model output -> volts
    return args.asinh * torch.sinh(y.clamp(-12, 12)) if args.asinh else y

vocab = {"ctype": len(TYPES), "t1": LABELS + 1, "t2": LABELS + 1, "t3": LABELS + 1}
net = Net(args.kind, vocab, d=args.d, hidden=args.hidden, K=args.K, T=args.T, n_out=train["target"].shape[-1], cont_dim=3, fourier=args.fourier).to(dev)
if args.load:
    net.load_state_dict(torch.load(args.load, map_location=dev))
n_params = sum(p.numel() for p in net.parameters())
print(f"{args.name}: {n_params / 1e6:.2f}M params", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
tau0, tau1 = map(float, args.tau.split(","))
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda" and not os.environ.get("NOAMP"))


def get(data, idx):
    return {k: v[idx].to(dev, non_blocking=True) for k, v in data.items()}


def lr_at(s):
    w = 500
    return args.lr * (s / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / (args.steps - w))))


@torch.no_grad()
def evaluate(data, T, n=None):
    net.eval()
    N = len(data["ctype"]) if n is None else min(n, len(data["ctype"]))
    modes = ["readout", "freeze"] + (["shuffle"] if args.kind in ("oploop", "moe") and args.K > 1 else [])
    acc = {m: {"rmse": [], "hs": [], "tgt": []} for m in modes + ["last"]}
    meta = {"depth": [], "role": []}
    out_rel = {m: [] for m in modes + ["last"]}
    for s in range(0, N, 256):
        c = get(data, slice(s, min(N, s + 256)))
        node = c["node"]
        meta["depth"].append(c["depth"][node].cpu())
        meta["role"].append(c["role"][node].cpu())
        is_out = (c["role"] == ROLES.index("out"))[node]
        for mode in modes:
            with torch.autocast(**amp):
                o = net(c, T=T, freeze=mode == "freeze", route="shuffle" if mode == "shuffle" else None)
            hs = halt_step(o["lam"])
            pred = dec(o["logits"].float())  # [T,B,N,16]
            preds = {mode: pred[-1] if mode == "freeze" else pred.gather(0, hs[None, ..., None].expand(1, *pred.shape[1:]))[0]}
            if mode == "readout":
                preds["last"] = pred[-1]
            for m, p in preds.items():
                err = p - c["target"]
                rmse = err.pow(2).mean(-1).sqrt()[node]
                acc[m]["rmse"].append(rmse.cpu())
                acc[m]["tgt"].append(c["target"].pow(2).mean(-1).sqrt()[node].cpu())
                acc[m]["hs"].append((hs[node] + 1).float().cpu() if m != "last" else torch.full_like(rmse.cpu(), T))
                tgt = c["target"][node][is_out]
                ac = tgt - tgt.mean(-1, keepdim=True)
                pe = p[node][is_out]
                pac = pe - pe.mean(-1, keepdim=True)
                out_rel[m].append(((pac - ac).pow(2).mean(-1).sqrt() / ac.pow(2).mean(-1).sqrt().clamp_min(1e-3)).cpu())
    depth = torch.cat(meta["depth"]).numpy()
    role = torch.cat(meta["role"]).numpy()
    r = {"T": T}
    for m in acc:
        rm = torch.cat(acc[m]["rmse"]).numpy()
        hs = torch.cat(acc[m]["hs"]).numpy()
        orl = torch.cat(out_rel[m]).numpy()
        tg = torch.cat(acc[m]["tgt"]).numpy()
        big = tg > 0.1  # nodes that move more than 100 mV this sample (zero prediction scores 0 / rel 1)
        r[m] = {"rmse_mean": float(rm.mean()), "rmse_median": float(np.median(rm)),
                "within_50mV": float((rm < 0.05).mean()), "within_20mV": float((rm < 0.02).mean()),
                "within_5mV": float((rm < 0.005).mean()), "within_1mV": float((rm < 0.001).mean()),
                "out_ac_rel_err_median": float(np.median(orl)),
                "edge_w50": float((rm[big] < 0.05).mean()) if big.any() else None,
                "edge_rel_median": float(np.median(rm[big] / tg[big])) if big.any() else None,
                "mean_steps": float(hs.mean()),
                "by_depth": {int(dd): {"rmse": float(rm[depth == dd].mean()), "w50": float((rm[depth == dd] < 0.05).mean()),
                                       "steps": float(hs[depth == dd].mean()), "n": int((depth == dd).sum())}
                             for dd in np.unique(depth)},
                "by_role": {ROLES[int(rr)]: {"rmse": float(rm[role == rr].mean()), "w50": float((rm[role == rr] < 0.05).mean()),
                                             "steps": float(hs[role == rr].mean())} for rr in np.unique(role)}}
    hs = torch.cat(acc["readout"]["hs"]).numpy()
    r["corr_steps_depth"] = float(np.corrcoef(hs, depth)[0, 1]) if hs.std() > 0 else None
    net.train()
    return r


os.makedirs(args.out, exist_ok=True)
log = []
t0 = time.time()
net.train()
for step in range(1, args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    frac = step / args.steps
    tau = tau0 * (tau1 / tau0) ** frac
    beta = args.beta * min(1.0, max(0.0, (frac - args.beta_start) / 0.2))
    idx = rng.integers(0, NTR, args.bs)
    if EDGE is not None:
        ne = int(args.bs * args.edge_frac)
        idx[:ne] = EDGE[rng.integers(0, len(EDGE), ne)]
    c = get(train, torch.from_numpy(idx))
    with torch.autocast(**amp):
        o = net(c, tau=tau, freeze=bool(rng.random() < args.freeze_frac))
    node = c["node"].float()
    T = o["logits"].shape[0]
    tgt = enc(c["target"])
    err = F.huber_loss(o["logits"].float(), tgt[None].expand_as(o["logits"]), delta=args.huber,
                       reduction="none").mean(-1)  # [T,B,N]
    if args.mse:
        err = err + args.mse * (o["logits"].float() - tgt[None]).pow(2).mean(-1)
    p = halting_dist(o["lam"])
    task = ((p * err).sum(0) * node).sum() / node.sum()
    exp_steps = ((p * torch.arange(1, T + 1, device=dev)[:, None, None]).sum(0) * node).sum() / node.sum()
    loss = task + beta * exp_steps
    if args.kl:  # PonderNet prior: keeps the halting head from saturating at "never halt"
        q = args.prior * (1 - args.prior) ** torch.arange(T, device=dev, dtype=torch.float32)
        q = (q / q.sum())[:, None, None]
        kl = (p * (p.clamp_min(1e-8).log() - q.log())).sum(0)
        loss = loss + args.kl * (kl * node).sum() / node.sum()
    if "aux" in o:
        loss = loss + args.aux * o["aux"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        print(f"step {step} task {task.item():.5f} E[steps] {exp_steps.item():.2f} beta {beta:.5f} "
              f"tau {tau:.2f} {time.time() - t0:.0f}s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        r = evaluate(test, args.T, n=1024)
        print(json.dumps({"step": step, "rmse": r["readout"]["rmse_mean"], "w50": r["readout"]["within_50mV"],
                          "w50_freeze": r["freeze"]["within_50mV"], "steps": r["readout"]["mean_steps"],
                          "edge_w50": r["readout"]["edge_w50"], "edge_rel": r["readout"]["edge_rel_median"], "w1": r["readout"]["within_1mV"],
                          "corr": r["corr_steps_depth"], "out_rel": r["readout"]["out_ac_rel_err_median"],
                          "w50_shuffle": r.get("shuffle", {}).get("within_50mV")}), flush=True)
        log.append({"step": step, "rmse": r["readout"]["rmse_mean"], "w50": r["readout"]["within_50mV"]})

final = {"args": vars(args), "params": n_params, "train_seconds": time.time() - t0,
         "test": evaluate(test, args.T), "ood": evaluate(ood, int(args.T * 1.5)), "log": log}
torch.save(net.state_dict(), os.path.join(args.out, f"{args.name}.pt"))
json.dump(final, open(os.path.join(args.out, f"{args.name}.json"), "w"), indent=1)
te, oo = final["test"], final["ood"]
summ = {"name": args.name, "kind": args.kind, "K": args.K, "d": args.d, "hidden": args.hidden, "params": n_params,
        "rmse": te["readout"]["rmse_mean"], "w50": te["readout"]["within_50mV"], "w20": te["readout"]["within_20mV"], "w5": te["readout"]["within_5mV"], "w1": te["readout"]["within_1mV"],
        "edge_w50": te["readout"]["edge_w50"], "edge_rel": te["readout"]["edge_rel_median"],
        "w50_freeze": te["freeze"]["within_50mV"], "w50_shuffle": te.get("shuffle", {}).get("within_50mV"),
        "out_rel": te["readout"]["out_ac_rel_err_median"], "steps": te["readout"]["mean_steps"],
        "corr": te["corr_steps_depth"],
        "ood_w50": oo["readout"]["within_50mV"], "ood_w50_last": oo["last"]["within_50mV"],
        "ood_out_rel": oo["readout"]["out_ac_rel_err_median"], "ood_steps": oo["readout"]["mean_steps"]}
with open(os.path.join(args.out, "results.jsonl"), "a") as f:
    f.write(json.dumps(summ) + "\n")
print(json.dumps(summ), flush=True)
