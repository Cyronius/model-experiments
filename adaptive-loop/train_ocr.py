"""Train dense / loop / oploop on OCR cleanup (gen_ocr.py data).

Each input character predicts two output slots (character or EMPTY); concatenated, they give the
clean text. Loss: cross-entropy per slot, weighted by the halting distribution, plus a cost per
expected step.

  python train_ocr.py --kind oploop --K 8 --name o-oploop-k8
Results: results/ocr/<name>.json and a line in results/ocr/results.jsonl.
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

if os.environ.get("OCR_DATA", "ocr") == "ocr2":  # v2: fineweb-edu, line breaks (gen_ocr2.py)
    from gen_ocr2 import CHARS, EMPTY, L, decode
else:
    from gen_ocr import CHARS, EMPTY, L, decode
from model import Net, halt_step, halting_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "npu-kernel-coder", "train"))
import memguard  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="oploop", choices=["dense", "oploop", "moe"])
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--name", required=True)
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--T", type=int, default=16)
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--beta", type=float, default=0.002, help="cost per expected step")
ap.add_argument("--beta-start", type=float, default=0.4)
ap.add_argument("--kl", type=float, default=0.0, help="weight of KL(halting || geometric prior), on from step 0")
ap.add_argument("--prior", type=float, default=0.2, help="per-step halting prob of the geometric prior")
ap.add_argument("--aux", type=float, default=0.01)
ap.add_argument("--tau", default="2.0,0.5")
ap.add_argument("--freeze-frac", type=float, default=0.5)
ap.add_argument("--eval-every", type=int, default=3000)
ap.add_argument("--load", default=None, help="state dict to start from (with --steps 0: evaluate only)")
ap.add_argument("--freeze-z", action="store_true", help="freeze halted tokens' attention input too (cacheable K/V)")
ap.add_argument("--phase-steps", type=int, default=0, help="save full state and exit (code 3) every N steps; rerun with the same args to resume")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "ocr"))
args = ap.parse_args()

memguard.start(12)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = os.environ.get("DEV", "cuda")
here = os.path.dirname(os.path.abspath(__file__))
V = len(CHARS) + 1  # chars + EMPTY (inputs use the same id for PAD)


def load(name):
    d = np.load(os.path.join(here, "cache", os.environ.get("OCR_DATA", "ocr"), f"{name}.npz"))
    return {k: torch.from_numpy(d[k].astype(np.int64) if d[k].dtype == np.int16 else d[k]) for k in d.files}


train, test, heavy = load(os.environ.get("TRAIN_SET", "train")), load("test"), load("heavy")
NTR = len(train["x"])
vocab = {"ch": V, "pos": L}
net = Net(args.kind, vocab, d=args.d, hidden=args.hidden, K=args.K, T=args.T, n_out=2 * V, freeze_z=args.freeze_z).to(dev)
if args.load:
    net.load_state_dict(torch.load(args.load, map_location=dev))
n_params = sum(p.numel() for p in net.parameters())
print(f"{args.name}: {n_params / 1e6:.2f}M params, train {NTR}", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
tau0, tau1 = map(float, args.tau.split(","))
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")
POS = torch.arange(L, device=dev)


def get(data, idx):
    b = {k: v[idx].to(dev, non_blocking=True) for k, v in data.items()}
    b["ch"] = b.pop("x")
    b["pos"] = POS.expand(b["ch"].shape[0], -1)
    return b


def lr_at(s):
    w = 500
    return args.lr * (s / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / (args.steps - w))))


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(data, T, n=None, cer_n=0):
    net.eval()
    N = len(data["x"]) if n is None else min(n, len(data["x"]))
    modes = ["readout", "freeze"] + (["shuffle"] if args.kind in ("oploop", "moe") and args.K > 1 else [])
    st = {m: {"ok": [], "hs": []} for m in modes + ["last"]}
    dmg_all, cer = [], {"input": [0, 0], **{m: [0, 0] for m in ["readout", "freeze"]}}
    for s in range(0, N, 256):
        c = get(data, slice(s, min(N, s + 256)))
        live = ~c["pad"]
        dmg_all.append(c["dmg"][live].cpu())
        for mode in modes:
            with torch.autocast(**amp):
                o = net(c, T=T, freeze=mode == "freeze", route="shuffle" if mode == "shuffle" else None)
            hs = halt_step(o["lam"])
            pred = o["logits"].float().view(*o["logits"].shape[:-1], 2, V).argmax(-1)  # [T,B,L,2]
            preds = {mode: pred[-1] if mode == "freeze" else
                     pred.gather(0, hs[None, ..., None].expand(1, *pred.shape[1:]))[0]}
            if mode == "readout":
                preds["last"] = pred[-1]
            for m, p in preds.items():
                ok = (p == c["y"]).all(-1)[live]
                st[m]["ok"].append(ok.cpu())
                st[m]["hs"].append((hs[live] + 1).float().cpu() if m != "last" else torch.full(ok.shape, float(T)))
                if m in ("readout", "freeze") and s < cer_n:
                    P, Y, X, PD = p.cpu().numpy(), c["y"].cpu().numpy(), c["ch"].cpu().numpy(), c["pad"].cpu().numpy()
                    for i in range(len(P)):
                        k = int((~PD[i]).sum())
                        clean = decode(Y[i, :k])
                        cer[m][0] += lev(decode(P[i, :k]), clean)
                        cer[m][1] += len(clean)
                        if m == "readout":
                            cer["input"][0] += lev("".join(CHARS[x] for x in X[i, :k]), clean)
                            cer["input"][1] += len(clean)
    dmg = torch.cat(dmg_all).numpy()
    r = {"T": T}
    for m in st:
        ok = torch.cat(st[m]["ok"]).numpy()
        hs = torch.cat(st[m]["hs"]).numpy()
        r[m] = {"acc_clean": float(ok[~dmg].mean()), "acc_damaged": float(ok[dmg].mean()),
                "steps_clean": float(hs[~dmg].mean()), "steps_damaged": float(hs[dmg].mean()),
                "mean_steps": float(hs.mean())}
    if cer_n:
        r["cer"] = {m: v[0] / max(v[1], 1) for m, v in cer.items()}
    net.train()
    return r


os.makedirs(args.out, exist_ok=True)
log = []
t0 = time.time()
start = 1
state_path = os.path.join(args.out, f"{args.name}.state.pt")
if args.phase_steps and os.path.exists(state_path):
    stt = torch.load(state_path, map_location=dev, weights_only=False)
    net.load_state_dict(stt["net"]); opt.load_state_dict(stt["opt"])
    rng.bit_generator.state = stt["rng"]; torch.set_rng_state(stt["torch_rng"].cpu())
    log, start, t0 = stt["log"], stt["step"] + 1, time.time() - stt["elapsed"]
    print(f"resumed at step {start}", flush=True)
net.train()
for step in range(start, args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    frac = step / args.steps
    tau = tau0 * (tau1 / tau0) ** frac
    beta = args.beta * min(1.0, max(0.0, (frac - args.beta_start) / 0.2))
    c = get(train, torch.from_numpy(rng.integers(0, NTR, args.bs)))
    with torch.autocast(**amp):
        o = net(c, tau=tau, freeze=bool(rng.random() < args.freeze_frac))
    live = (~c["pad"]).float()
    T = o["logits"].shape[0]
    lg = o["logits"].float().view(T, *c["y"].shape, V)
    ce = F.cross_entropy(lg.reshape(-1, V), c["y"][None].expand(T, *c["y"].shape).reshape(-1),
                         reduction="none").view(T, *c["y"].shape).mean(-1)  # [T,B,L]
    p = halting_dist(o["lam"])
    task = ((p * ce).sum(0) * live).sum() / live.sum()
    exp_steps = ((p * torch.arange(1, T + 1, device=dev)[:, None, None]).sum(0) * live).sum() / live.sum()
    loss = task + beta * exp_steps
    if args.kl:  # PonderNet prior: keeps the halting head from saturating at "never halt"
        q = args.prior * (1 - args.prior) ** torch.arange(T, device=dev, dtype=torch.float32)
        q = (q / q.sum())[:, None, None]
        kl = (p * (p.clamp_min(1e-8).log() - q.log())).sum(0)
        loss = loss + args.kl * (kl * live).sum() / live.sum()
    if "aux" in o:
        loss = loss + args.aux * o["aux"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        print(f"step {step} task {task.item():.4f} E[steps] {exp_steps.item():.2f} beta {beta:.4f} "
              f"tau {tau:.2f} {time.time() - t0:.0f}s", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        r = evaluate(test, args.T, n=1024)
        print(json.dumps({"step": step, **{k: round(v, 4) for k, v in r["readout"].items()},
                          "shuffle_damaged": r.get("shuffle", {}).get("acc_damaged")}), flush=True)
        log.append({"step": step, **r["readout"]})
    if args.phase_steps and step % args.phase_steps == 0 and step < args.steps:
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "log": log, "step": step, "elapsed": time.time() - t0}, state_path)
        print(f"phase done at step {step}", flush=True)
        sys.exit(3)

final = {"args": vars(args), "params": n_params, "train_seconds": time.time() - t0,
         "test": evaluate(test, args.T, cer_n=1024), "heavy": evaluate(heavy, int(args.T * 1.5), cer_n=1024),
         "log": log}
torch.save(net.state_dict(), os.path.join(args.out, f"{args.name}.pt"))
json.dump(final, open(os.path.join(args.out, f"{args.name}.json"), "w"), indent=1)
te, hv = final["test"], final["heavy"]
summ = {"name": args.name, "kind": args.kind, "K": args.K, "d": args.d, "hidden": args.hidden, "params": n_params,
        "cer_in": te["cer"]["input"], "cer": te["cer"]["readout"], "cer_freeze": te["cer"]["freeze"],
        "acc_clean": te["readout"]["acc_clean"], "acc_damaged": te["readout"]["acc_damaged"],
        "shuffle_damaged": te.get("shuffle", {}).get("acc_damaged"),
        "steps_clean": te["readout"]["steps_clean"], "steps_damaged": te["readout"]["steps_damaged"],
        "heavy_cer_in": hv["cer"]["input"], "heavy_cer": hv["cer"]["readout"],
        "heavy_acc_damaged": hv["readout"]["acc_damaged"]}
with open(os.path.join(args.out, "results.jsonl"), "a") as f:
    f.write(json.dumps(summ) + "\n")
print(json.dumps(summ), flush=True)
