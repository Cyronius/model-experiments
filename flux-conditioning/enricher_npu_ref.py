"""Enricher v2 as the NPU op list (folded weights, padded heads, bias tricks, CFG via euler ops), in numpy, vs PyTorch (2026-11-21)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

import enricher_v2 as ev

sys.path.insert(0, r"C:\code\phlegm\open-kernels\open_kernels\designs\dit_gemm")
import pack  # noqa: E402

A = Path(__file__).resolve().parents[1] / "artifacts" / "flux-conditioning"
D, HEADS, HD, PHD = 512, 8, 64, 128          # model width, heads, model head dim, NPU head dim
QKV_N, FF_N = 3 * HEADS * PHD, 3072          # NPU GEMM widths (N % 1024)


class Ops:
    """The NPU primitives. npu=False is exact fp32 algebra; npu=True rounds like the kernels."""

    def __init__(self, npu: bool):
        self.npu = npu
        self.eps = 1e-6 if npu else 1e-5

    def bf(self, x):
        return x.astype(bfloat16).astype(np.float32) if self.npu else x.astype(np.float32)

    def bfp(self, x):
        """bfp16ebs8 rounding along the last axis (blocks of 8), the GEMM's operand format."""
        if not self.npu:
            return x
        return pack.bfp16ebs8_decode(pack.bfp16ebs8_encode(x.reshape(-1))).reshape(x.shape).astype(np.float32)

    def gemm(self, a, w):                     # w: [K, N]
        return self.bf(self.bfp(a) @ self.bfp(w.T).T)

    def ln(self, x, w, b):
        m = x.mean(-1, keepdims=True)
        v = (x * x).mean(-1, keepdims=True) - m * m if self.npu else x.var(-1, keepdims=True)
        t = self.bf((x - m) / np.sqrt(np.maximum(v, 0) + self.eps))
        return self.bf(t * self.bf(w) + self.bf(b))

    def bias(self, x, b):
        return self.bf(x + self.bf(b))

    def relu(self, x):
        return np.maximum(x, 0)

    def euler(self, a, b, dt):
        return self.bf(a + np.float32(dt) * b)

    def fa(self, qkv, start, n):
        """dit_fa on rows [start, start+n): 8 heads of 128, scale 1/sqrt(128); rows outside are not keys."""
        q, k, v = (qkv[:, i * HEADS * PHD:(i + 1) * HEADS * PHD].reshape(-1, HEADS, PHD) for i in range(3))
        kk, vv = k[start:start + n], v[start:start + n]
        s = np.einsum("qhd,khd->hqk", q, kk) / math.sqrt(PHD)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return self.bf(np.einsum("hqk,khd->qhd", p, vv).reshape(len(qkv), -1))


def fold_layer(sd, p):
    """One TransformerEncoderLayer as NPU weights: q prescaled sqrt(2) and heads padded 64 -> 128;
    v's pad column 64 of head 0 is 1 (from the bias op), so attention output column 64 is a ones
    column that carries out_proj's bias; FFN1 column 2048 is relu(0 + 1) = 1 for linear2's bias."""
    W, b = sd[p + "self_attn.in_proj_weight"], sd[p + "self_attn.in_proj_bias"]
    qkv, qkv_b = np.zeros((D, QKV_N), np.float32), np.zeros(QKV_N, np.float32)
    for i, scale in enumerate((math.sqrt(PHD / HD), 1.0, 1.0)):
        for h in range(HEADS):
            c = i * HEADS * PHD + h * PHD
            r = i * D + h * HD
            qkv[:, c:c + HD] = W[r:r + HD].T * scale
            qkv_b[c:c + HD] = b[r:r + HD] * scale
    qkv_b[2 * HEADS * PHD + HD] = 1.0
    out = np.zeros((HEADS * PHD, 1024), np.float32)
    Wo = sd[p + "self_attn.out_proj.weight"]
    for h in range(HEADS):
        out[h * PHD:h * PHD + HD, :D] = Wo[:, h * HD:(h + 1) * HD].T
    out[HD, :D] = sd[p + "self_attn.out_proj.bias"]
    ff1, ff1_b = np.zeros((D, FF_N), np.float32), np.zeros(FF_N, np.float32)
    ff1[:, :4 * D] = sd[p + "linear1.weight"].T
    ff1_b[:4 * D] = sd[p + "linear1.bias"]
    ff1_b[4 * D] = 1.0
    ff2 = np.zeros((4 * D + 512, 1024), np.float32)
    ff2[:4 * D, :D] = sd[p + "linear2.weight"].T
    ff2[4 * D, :D] = sd[p + "linear2.bias"]
    return dict(ln1=(sd[p + "norm1.weight"], sd[p + "norm1.bias"]), ln2=(sd[p + "norm2.weight"], sd[p + "norm2.bias"]),
                qkv=qkv, qkv_b=qkv_b, out=out, ff1=ff1, ff1_b=ff1_b, ff2=ff2)


def fold(model):
    sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
    emb = np.zeros((512, 1024), np.float32)    # A = [x | ctx_in] (256 + 256 columns)
    emb[:256, :D], emb[256:, :D] = sd["inp.weight"].T, sd["ctx.weight"].T
    out = np.zeros((D, 1024), np.float32)
    out[:, :256] = sd["out.weight"].T
    len_out = np.zeros((D, 1024), np.float32)
    len_out[:, :ev.N_LEN] = sd["len_out.weight"].T
    return dict(sd=sd, emb=emb, out=out, out_b=sd["out.bias"], len_out=len_out, len_out_b=sd["len_out.bias"],
                body=[fold_layer(sd, f"body.layers.{l}.") for l in range(8)],
                len_body=[fold_layer(sd, f"len_body.layers.{l}.") for l in range(2)],
                norm=(sd["norm.weight"], sd["norm.bias"]), len_norm=(sd["len_norm.weight"], sd["len_norm.bias"]))


def time_rows(model, steps):
    """The time MLP of the fixed step times: a constant table, computed once at load."""
    with torch.no_grad():
        t = torch.tensor([1 - i / steps for i in range(steps)])
        return model.time(ev.time_embed(t)).float().numpy()


def encoder(o: Ops, layers, X, seqs, final):
    """Residual stream X [rows, 512]; seqs: (start, count) key ranges for dit_fa; final: (w, b) of the last LN.
    Per layer: QKV GEMM, bias, fa per sequence, out GEMM, res+LN, FFN1 GEMM, bias, relu, FFN2 GEMM, res+LN."""
    Y = o.ln(X, *layers[0]["ln1"])
    for i, L in enumerate(layers):
        qkv = o.bias(o.gemm(Y, L["qkv"]), L["qkv_b"])
        att = np.zeros((len(X), HEADS * PHD), np.float32)
        for start, n in seqs:
            att[start:start + n] = o.fa(qkv[start:start + n], 0, n)
        X = o.bf(X + o.gemm(att, L["out"])[:, :D])
        Y = o.ln(X, *L["ln2"])
        h = o.relu(o.bias(o.gemm(Y, L["ff1"]), L["ff1_b"]))
        X = o.bf(X + o.gemm(h[:, :4 * D + 512], L["ff2"])[:, :D])
        Y = o.ln(X, *(layers[i + 1]["ln1"] if i + 1 < len(layers) else final))
    return Y


def npu_enrich(o: Ops, F, trows, ctx, n, noise, steps, g):
    """Rows: 64 ctx slots (valid ones right-aligned), 32 cond window, 32 free window."""
    sd, nc = F["sd"], len(ctx)
    c0 = 64 - nc
    A_in = np.zeros((128, 512), np.float32)
    A_in[c0:64, 256:] = ctx
    x = noise.copy()
    rev = np.clip(n - 1 - np.arange(ev.WIN), 0, None)
    win_const = sd["inp.bias"] + sd["pos"] + sd["rpos"][rev] + sd["kind"][0]
    for i in range(steps):
        A_in[64:96, :256] = A_in[96:128, :256] = x
        const = np.zeros((128, D), np.float32)                      # host-written per prompt
        const[c0:64] = sd["ctx.bias"] + sd["ctx_pos"][:nc] + sd["kind"][1]
        const[64:96] = const[96:128] = win_const + trows[i]
        X = o.bf(o.bf(const) + o.gemm(o.bf(A_in), F["emb"])[:, :D])
        Y = encoder(o, F["body"], X, [(c0, nc + n), (96, n)], F["norm"])
        P = o.bias(o.gemm(Y, F["out"])[:, :256], F["out_b"])
        t = 1 - i / steps
        d1 = o.euler(P[64:96], P[96:128], -1.0)                      # cond - free
        G = o.euler(P[96:128], d1, g)                               # free + g (cond - free)
        E = o.euler(G, x, -1.0)
        x = o.euler(x, E, 1.0 / (t * steps))
    return x[:n]


def npu_length(o: Ops, F, ctx):
    sd, nc = F["sd"], len(ctx)
    A_in = np.zeros((65, 512), np.float32)
    A_in[1:1 + nc, 256:] = ctx
    const = np.zeros((65, D), np.float32)
    const[0] = sd["kind"][2]
    const[1:1 + nc] = sd["ctx.bias"] + sd["ctx_pos"][:nc] + sd["kind"][1]
    emb = o.gemm(o.bf(A_in), F["emb"])[:, :D]
    X = o.bf(o.bf(const) + emb)
    Y = encoder(o, F["len_body"], X, [(0, nc + 1)], F["len_norm"])
    return o.bias(o.gemm(Y[:1], F["len_out"])[:, :ev.N_LEN], F["len_out_b"])[0]


@torch.no_grad()
def torch_enrich(model, ctx, n, noise, steps, g):
    c = torch.zeros(1, ev.CTX, 256)
    c[0, :len(ctx)] = torch.from_numpy(ctx)
    cm = torch.arange(ev.CTX)[None] < len(ctx)
    wm = torch.arange(ev.WIN)[None] < n
    x = torch.from_numpy(noise)[None] * wm[..., None]
    for i in range(steps):
        t = torch.full((1,), 1 - i / steps)
        out = model(x, t, c, cm, wm)
        free = model(x, t, torch.zeros_like(c), torch.zeros_like(cm), wm)
        out = free + g * (out - free)
        x = (x - (x - out) / t[:, None, None] / steps) * wm[..., None]
    return x[0, :n].numpy(), model.length_logits(c, cm)[0].numpy()


def cos(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="test bases")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=A / "enricher-npu" / "ref.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    model = ev.load_model(A / "enricher-v2" / "run2-cfg" / "cfg.pt", "cpu")[0].float()
    data = ev.Data(A / "enricher-v2" / "data", "cpu")
    F, trows = fold(model), time_rows(model, args.steps)
    groups = [i for i, g in enumerate(data.groups) if g["split"] == "test"][:: 97][:args.n]
    rng = np.random.default_rng(0)
    rows = []
    for gi in groups:
        ctx, cm, _, _, n_true = data.gather(torch.tensor([data.groups[gi]["items"][0]]))
        nc = int(cm.sum())
        ctx = ctx[0, :nc].numpy()
        n = int(n_true[0])
        noise = np.zeros((ev.WIN, 256), np.float32)
        noise[:n] = rng.standard_normal((n, 256)).astype(np.float32)
        ref, ref_len = torch_enrich(model, ctx, n, noise, args.steps, args.guidance)
        r = {"group": gi, "nc": nc, "n": n}
        for name, o in (("exact", Ops(False)), ("npu", Ops(True))):
            got = npu_enrich(o, F, trows, ctx, n, noise, args.steps, args.guidance)
            lg = npu_length(o, F, ctx)
            raw_ref, raw_got = data.to_raw(torch.from_numpy(ref)).numpy(), data.to_raw(torch.from_numpy(got)).numpy()
            r[name] = {"pca_max_abs": float(np.abs(got - ref).max()), "row_cos_min": float(cos(got, ref).min()),
                       "raw_cos_min": float(cos(raw_got, raw_ref).min()), "len_logit_max_abs": float(np.abs(lg - ref_len).max()),
                       "len_argmax_same": bool(lg.argmax() == ref_len.argmax())}
        rows.append(r)
        print(json.dumps(r), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
