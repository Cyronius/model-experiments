"""Inference that actually skips halted tokens, and the OCR cleanup command.

`run` is the freeze-mode forward of an oploop Net for one sequence, computing only tokens that are
still active: their queries, attention outputs, router, one expert each, head and halting. Halted
tokens keep cached keys/values. Exact for models trained with freeze_z (their halted tokens' attention
input is frozen); for other models it matches freeze mode up to the step-embedding drift of halted keys.

  python infer.py ocr results/ocr/o2-oploop-k8.pt < damaged.txt > clean.txt
  python infer.py check results/ocr/o2-oploop-k8.pt     # equality vs Net.forward + CPU timing
"""
import argparse
import sys
import time

import torch
import torch.nn.functional as F

from model import Net


@torch.no_grad()
def run(net, inp, T=None, rule="cdf"):
    """inp: dict of [1, N] tensors without padding. Returns (outputs [N, n_out], steps [N])."""
    T = T or net.T
    e = net.embed(inp)[0]
    N, d = e.shape
    attn, H = net.attn[0], net.attn[0].h
    dh = d // H
    h = torch.zeros_like(e)
    out = torch.zeros(N, net.head.out_features)
    steps = torch.zeros(N, dtype=torch.long)
    Kc = torch.zeros(H, N, dh)
    Vc = torch.zeros(H, N, dh)
    surv = torch.ones(N)
    idx = torch.arange(N)
    for t in range(T):
        z = h[idx] + e[idx] + net.step.weight[min(t, net.t_emb - 1)]
        q, k, v = attn.qkv(net.n1[0](z)).view(-1, 3, H, dh).permute(1, 2, 0, 3)
        Kc[:, idx], Vc[:, idx] = k, v
        a = attn.o(F.scaled_dot_product_attention(q, Kc, Vc).transpose(0, 1).reshape(-1, d))
        y = net.n2[0](z + a)
        if net.K == 1:
            m = net.experts.one(0, y)
        else:
            choice = net.router(y).argmax(-1)
            m = torch.empty_like(y)
            for kk in choice.unique().tolist():
                sel = choice == kk
                m[sel] = net.experts.one(kk, y[sel])
        hn = h[idx] + a + m
        hf = net.nf(hn)
        out[idx] = net.head(hf)
        lam = torch.sigmoid(net.halt(hf).squeeze(-1))
        h[idx] = hn
        steps[idx] = t + 1
        if rule == "cdf":
            surv[idx] = surv[idx] * (1 - lam)
            stop = (1 - surv[idx]) >= 0.5
        else:
            stop = lam > 0.5
        idx = idx[~stop]
        if len(idx) == 0:
            break
    return out, steps


def ocr_vocab(sd):
    """v1 models (gen_ocr) have 96 input ids; v2 (gen_ocr2) add "\n" and have 97."""
    import gen_ocr
    import gen_ocr2
    return gen_ocr2 if sd["emb.ch.weight"].shape[0] == len(gen_ocr2.CHARS) + 1 else gen_ocr


def load_ocr(path, **kw):
    sd = torch.load(path, map_location="cpu")
    g = ocr_vocab(sd)
    CHARS, L = g.CHARS, g.L
    K = sd["router.weight"].shape[0] if "router.weight" in sd else 1
    d, hidden = sd["experts.w1"].shape[1], sd["experts.w1"].shape[2]
    net = Net("oploop", {"ch": len(CHARS) + 1, "pos": L}, d=d, hidden=hidden, K=K, T=16, n_out=2 * (len(CHARS) + 1), **kw)
    net.load_state_dict(sd)
    net.vocab_mod = g
    return net.eval()


def windows(text, maxlen=110):
    """Split at spaces into pieces of at most maxlen characters (the model saw ~100-char windows)."""
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + maxlen)
        if j < len(text):
            k = max(text.rfind(" ", i + 1, j), text.rfind("\n", i + 1, j))
            j = k + 1 if k > i else j
        out.append(text[i:j])
        i = j
    return out


def ocr_clean(net, text, T=16):
    g = net.vocab_mod
    CHARS, CI, EMPTY = g.CHARS, g.CI, g.EMPTY
    V = len(CHARS) + 1
    res, steps = [], []
    # v2 models read single line breaks themselves (wraps, hyphenated wraps); blank lines split paragraphs
    blocks = text.split("\n\n") if "\n" in CI else text.split("\n")
    for line in blocks:
        parts = []
        for w in windows(line):
            # training windows never end in whitespace, so cut it off the window and put it back after
            core = w.rstrip(" \n")
            tail = " " if len(core) < len(w) else ""
            if not core:
                parts.append(tail)
                continue
            w = core
            ids = torch.tensor([[CI.get(c, CI["?"]) for c in w]])
            o, s = run(net, {"ch": ids, "pos": torch.arange(len(w))[None]}, T=T)
            slots = o.view(-1, 2, V).argmax(-1).reshape(-1).tolist()
            parts.append("".join(CHARS[c] for c in slots if c < EMPTY) + tail)
            steps.append(s)
        res.append("".join(parts))
    s = torch.cat(steps) if steps else torch.zeros(0)
    return ("\n\n" if "\n" in CI else "\n").join(res), s


def check(path):
    from gen_ocr import CHARS, CI
    torch.manual_seed(0)
    net = load_ocr(path, freeze_z=True)
    import numpy as np
    d = np.load("cache/ocr/test.npz")
    torch.set_num_threads(1)
    worst, n_tok, t_sparse, t_full = 0.0, 0, 0.0, 0.0
    from model import HALT_RULE
    for i in range(64):
        k = int((~d["pad"][i]).sum())
        inp = {"ch": torch.from_numpy(d["x"][i:i + 1, :k].astype(np.int64)), "pos": torch.arange(k)[None]}
        t0 = time.perf_counter()
        o, s = run(net, inp, T=16, rule=HALT_RULE)
        t_sparse += time.perf_counter() - t0
        t0 = time.perf_counter()
        ref = net(inp, T=16, freeze=True)["logits"][-1, 0]
        t_full += time.perf_counter() - t0
        worst = max(worst, (o - ref).abs().max().item())
        n_tok += k
    print(f"max |sparse - forward| = {worst:.2e} (halting rule {HALT_RULE})")
    print(f"1 thread, batch 1: sparse {n_tok / t_sparse:,.0f} chars/s, full-compute forward {n_tok / t_full:,.0f} chars/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ocr", "check"])
    ap.add_argument("ckpt")
    a = ap.parse_args()
    if a.cmd == "check":
        check(a.ckpt)
    else:
        net = load_ocr(a.ckpt)
        torch.set_num_threads(4)
        clean, s = ocr_clean(net, sys.stdin.read())
        sys.stdout.write(clean)
        print(f"[{len(s)} chars, mean steps {s.float().mean():.2f}]", file=sys.stderr)
