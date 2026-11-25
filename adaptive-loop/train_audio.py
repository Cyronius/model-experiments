"""Real-time pedal model: a small GRU runs per audio sample; for the family model an op-loop encoder
reads the circuit once and conditions the GRU (its embedding is a constant extra GRU input).

  python train_audio.py --mode muff --name au-muff-h32          # one fixed Big Muff
  python train_audio.py --mode family --name au-family-k8       # circuit-conditioned
Loss: error-to-signal ratio (ESR) with first-order pre-emphasis, as in neural amp/pedal modelling.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_analog import LABELS, TYPES
from model import Net

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "npu-kernel-coder", "train"))
import memguard  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, help="muff, family, or fixedN (N fixed family circuits)")
ap.add_argument("--name", required=True)
ap.add_argument("--hidden", type=int, default=32, help="GRU size (per-sample cost)")
ap.add_argument("--cond", type=int, default=32, help="circuit embedding size (family)")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--T", type=int, default=12, help="encoder steps")
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--seg", type=int, default=4800, help="training segment length in samples")
ap.add_argument("--warm", type=int, default=256, help="unscored warm-up samples at the start of each training segment")
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--load-encoder", default=None, help="DC-task checkpoint to start the encoder from")
ap.add_argument("--enc-lr", type=float, default=None, help="encoder learning rate (default: --lr)")
ap.add_argument("--no-cond", action="store_true", help="family data, but the GRU gets no circuit embedding")
ap.add_argument("--cond-mode", choices=["concat", "hyper"], default="hyper",
                help="concat: embedding is an extra GRU input; hyper: encoder writes the GRU input weights, gate biases, h0, readout")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "audio"))
args = ap.parse_args()

memguard.start(12)
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = os.environ.get("DEV", "cuda")
here = os.path.dirname(os.path.abspath(__file__))
YS = 4.0  # output volts are divided by this


def load(name):
    d = np.load(os.path.join(here, "cache", "audio", f"{args.mode}-{name}.npz"))
    return {("ctype" if k == "type" else k): torch.from_numpy(d[k]) for k in d.files}


class PedalNet(nn.Module):
    def __init__(self):
        super().__init__()
        cin = 1
        if args.mode != "muff":
            vocab = {"ctype": len(TYPES), "t1": LABELS + 1, "t2": LABELS + 1, "t3": LABELS + 1}
            self.enc = Net("oploop", vocab, d=128, hidden=256, K=args.K, T=args.T, n_out=1, cont_dim=3, fourier=6)
            self.proj = nn.Linear(2 * 128, args.cond)
            cin += args.cond
        H = args.hidden
        self.hyper = args.mode != "muff" and args.cond_mode == "hyper"
        if self.hyper:
            # per circuit: input weights (3H), gate biases (3H), readout (H + 1), initial state (H)
            self.n_hyper = 3 * H + 3 * H + H + 1 + H
            self.base = nn.Parameter(torch.cat([torch.randn(3 * H) * 0.5, torch.zeros(3 * H),
                                                torch.randn(H) * 0.01, torch.zeros(1 + H)]))  # small readout: start near silence
            self.proj = nn.Linear(2 * 128, self.n_hyper)
            nn.init.normal_(self.proj.weight, std=0.01)
            nn.init.zeros_(self.proj.bias)
            self.gru = nn.GRU(3 * H, H, batch_first=True)  # input = precomputed gate inputs; W_ih fixed to identity
            with torch.no_grad():
                self.gru.weight_ih_l0.copy_(torch.eye(3 * H))
                self.gru.bias_ih_l0.zero_()
            self.gru.weight_ih_l0.requires_grad_(False)
            self.gru.bias_ih_l0.requires_grad_(False)
        else:
            self.gru = nn.GRU(cin, H, batch_first=True)
        self.lin = nn.Linear(H, 1)

    def encode(self, c):
        """Circuit embedding: mean of final encoder states plus the output node's state."""
        o = self.enc(c, T=args.T, freeze=False)
        h = o["h"]  # [B,N,d] final state
        live = (~c["pad"]).float()[..., None]
        mean = (h * live).sum(1) / live.sum(1)
        outn = (h * c["is_out"].float()[..., None]).sum(1)
        z = torch.cat([mean, outn], -1)
        if self.hyper:
            return self.base + self.proj(z), o
        return torch.tanh(self.proj(z)), o

    def forward(self, x, c=None, h0=None):
        inp = x[..., None]
        o = None
        if self.hyper:
            H = args.hidden
            p, o = self.encode(c)
            w_in, b, w_out, b_out, hi = p.split([3 * H, 3 * H, H, 1, H], -1)
            gates = x[..., None] * w_in[:, None] + b[:, None]  # [B,S,3H], per-circuit input weights and biases
            y, hT = self.gru(gates, hi[None].contiguous() if h0 is None else h0)
            return (y * w_out[:, None]).sum(-1) + b_out, hT, o
        if args.mode != "muff":
            emb, o = self.encode(c)
            if args.no_cond:
                emb = torch.zeros_like(emb)
            inp = torch.cat([inp, emb[:, None].expand(-1, x.shape[1], -1)], -1)
        y, hT = self.gru(inp, h0)
        return x + self.lin(y)[..., 0], hT, o  # residual: dry signal + learned difference


def esr(pred, tgt, pre=0.85):
    p = pred[:, 1:] - pre * pred[:, :-1]
    t = tgt[:, 1:] - pre * tgt[:, :-1]
    return ((p - t) ** 2).sum(1) / ((t ** 2).sum(1) + 1e-5)


train, test = load("train"), load("test")
net = PedalNet().to(dev)
if args.load_encoder and args.mode != "muff":
    sd = torch.load(args.load_encoder, map_location=dev)
    missing = net.enc.load_state_dict({k: v for k, v in sd.items() if not k.startswith("head.")}, strict=False)
    print("encoder loaded; missing:", missing.missing_keys, flush=True)
n_params = sum(p.numel() for p in net.parameters())
n_core = sum(p.numel() for p in list(net.gru.parameters()) + list(net.lin.parameters()))
print(f"{args.name}: {n_params / 1e6:.3f}M params, per-sample core {n_core} params, train {len(train['x'])} clips", flush=True)
enc_params = [p for n, p in net.named_parameters() if n.startswith(("enc.", "proj."))]
core_params = [p for n, p in net.named_parameters() if not n.startswith(("enc.", "proj."))]
groups = [{"params": core_params, "base": args.lr}]
if enc_params:
    groups.append({"params": enc_params, "base": args.enc_lr or args.lr})
opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)


def batch(data, idx):
    b = {k: v[idx].to(dev) for k, v in data.items()}
    return b["x"], b["y"] / YS, (b if args.mode != "muff" else None)


def lr_at(s):
    w = 300
    return args.lr * (s / w if s < w else 0.5 * (1 + math.cos(math.pi * (s - w) / (args.steps - w))))


@torch.no_grad()
def evaluate(data, n=None):
    net.eval()
    N = len(data["x"]) if n is None else min(n, len(data["x"]))
    e, e_raw = [], []
    for s in range(0, N, 128):
        x, y, c = batch(data, slice(s, min(N, s + 128)))
        p, _, _ = net(x, c)
        e.append(esr(p, y).cpu())
        e_raw.append((((p - y) ** 2).sum(1) / ((y ** 2).sum(1) + 1e-5)).cpu())
    net.train()
    e, e_raw = torch.cat(e).numpy(), torch.cat(e_raw).numpy()
    return {"esr_mean": float(e.mean()), "esr_median": float(np.median(e)), "esr_raw_median": float(np.median(e_raw)),
            "frac_esr_below_0.05": float((e < 0.05).mean())}


@torch.no_grad()
def cpu_speed(block=256, seconds=2.0):
    """Real-time factor of the per-sample core on 1 CPU thread, processing audio in blocks."""
    torch.set_num_threads(1)
    gru, lin = net.gru.cpu().eval(), net.lin.cpu().eval()
    cin = gru.input_size
    S = int(48000 * seconds)
    x = torch.randn(1, S, cin) * 0.1
    h = None
    t0 = time.perf_counter()
    for i in range(0, S, block):
        y, h = gru(x[:, i:i + block], h)
        lin(y)
    dt = time.perf_counter() - t0
    net.gru.to(dev), net.lin.to(dev)
    return seconds / dt


os.makedirs(args.out, exist_ok=True)
t0 = time.time()
NTR = len(train["x"])
for step in range(1, args.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step) * g["base"] / args.lr
    x, y, c = batch(train, torch.from_numpy(rng.integers(0, NTR, args.bs)))
    if args.seg < x.shape[1]:
        o = int(rng.integers(0, x.shape[1] - args.seg))
        x, y = x[:, o:o + args.seg], y[:, o:o + args.seg]
    p, _, _ = net(x, c)
    w = args.warm if args.seg < train["x"].shape[1] else 0
    pp, yy = p[:, w:], y[:, w:]
    pe, te = pp[:, 1:] - 0.85 * pp[:, :-1], yy[:, 1:] - 0.85 * yy[:, :-1]
    # per-clip ESR with a level floor (10 mV RMS at the output): quiet circuits count as much as loud
    # ones, and near-silent clips are scored on absolute error instead of exploding
    floor = (0.01 / YS) ** 2
    loss = (((pe - te) ** 2).mean(1) / ((te ** 2).mean(1) + floor)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for g in opt.param_groups:  # clip each group on its own: encoder gradients must not shrink the core's
        torch.nn.utils.clip_grad_norm_(g["params"], 1.0)
    opt.step()
    if step % 250 == 0:
        print(f"step {step} esr {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    if step % 1500 == 0 or step == args.steps:
        print(json.dumps({"step": step, **evaluate(test, n=512)}), flush=True)

res = {"name": args.name, "mode": args.mode, "params": n_params, "core_params": n_core, "hidden": args.hidden,
       **evaluate(test), "realtime_x_1thread": cpu_speed(), "train_seconds": time.time() - t0}
torch.save(net.state_dict(), os.path.join(args.out, f"{args.name}.pt"))
with open(os.path.join(args.out, "results.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
print(json.dumps(res), flush=True)
