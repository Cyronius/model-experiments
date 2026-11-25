"""Looped transformer with a per-step operation choice and per-token halting.

kind = "dense"  : T distinct blocks (attention + MLP), early exit.
kind = "oploop" : shared attention, then a router picks 1 of K expert MLPs per token each step.
                  K = 1 is a plain looped block ("loop").

Every step: z = h + e + step_emb(t) (input re-injection); a = attn(norm(z)); m = op(norm(z + a));
h = h + a + m. After each step a shared head predicts the output and a halting probability.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, key_mask=None):
        B, N, d = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        return self.o(y.transpose(1, 2).reshape(B, N, d))


class Experts(nn.Module):
    """K MLPs with stacked weights."""

    def __init__(self, K, d, hidden):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(K, d, hidden) * d ** -0.5)
        self.w2 = nn.Parameter(torch.randn(K, hidden, d) * hidden ** -0.5 * 0.5)

    def one(self, k, x):
        return F.gelu(x @ self.w1[k]) @ self.w2[k]

    def mix(self, g, x):
        """sum_k g[..., k] * expert_k(x), all experts in one batched matmul. x [B,N,d], g [B,N,K]."""
        hid = F.gelu(torch.einsum("bnd,kdh->bnkh", x, self.w1)) * g[..., None]
        return torch.einsum("bnkh,khd->bnd", hid, self.w2)


class Net(nn.Module):
    def __init__(self, kind, vocab, d=128, heads=4, hidden=256, K=8, T=20, n_out=2, t_emb=32, cont_dim=0, fourier=0, freeze_z=False):
        super().__init__()
        # freeze_z: in freeze mode a halted token's attention input z is frozen too (not only h), so its
        # keys/values are constant and inference can cache them and skip the token entirely (infer.py)
        self.freeze_z = freeze_z
        self.fourier = fourier  # continuous inputs also enter as sin/cos(2^k pi x), k < fourier
        self.cont = nn.Linear(cont_dim * (1 + 2 * fourier), d) if cont_dim else None
        self.kind, self.K, self.T, self.t_emb = kind, K, T, t_emb
        self.emb = nn.ModuleDict({name: nn.Embedding(n, d) for name, n in vocab.items()})
        self.step = nn.Embedding(t_emb, d)
        nb = T if kind in ("dense", "moe") else 1
        self.attn = nn.ModuleList(Attn(d, heads) for _ in range(nb))
        self.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(nb))
        self.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(nb))
        if kind == "dense":
            self.experts = Experts(T, d, hidden)  # expert t is layer t's MLP
        elif kind == "moe":  # standard MoE: T distinct layers, each with its own K experts and router
            self.experts = nn.ModuleList(Experts(K, d, hidden) for _ in range(T))
            self.router = nn.ModuleList(nn.Linear(d, K) for _ in range(T))
            for r in self.router:
                nn.init.zeros_(r.weight)
        else:
            self.experts = Experts(K, d, hidden)
            self.router = nn.Linear(d, K)
            nn.init.zeros_(self.router.weight)
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out)
        self.halt = nn.Linear(d, 1)
        nn.init.constant_(self.halt.bias, -2.0)

    def embed(self, inp):
        e = sum(self.emb[k](inp[k]) for k in self.emb)
        if self.cont is None:
            return e
        x = inp["cont"].float()
        if self.fourier:
            f = x[..., None] * (math.pi * 2.0 ** torch.arange(self.fourier, device=x.device))
            x = torch.cat([x, f.sin().flatten(-2), f.cos().flatten(-2)], -1)
        return e + self.cont(x)

    def forward(self, inp, T=None, tau=1.0, freeze=False, halt_thresh=0.5, route=None):
        """route: None, "shuffle" or "random" (eval-only routing ablations).
        Returns per-step logits [T,B,N,n_out], halting lambdas [T,B,N], router info."""
        T = T or self.T
        e = self.embed(inp)
        h = torch.zeros_like(e)
        key_mask = ~inp["pad"][:, None, None, :] if "pad" in inp else None  # True = attend
        logits, lams, choices, aux = [], [], [], []
        active = torch.ones(e.shape[:2], dtype=torch.bool, device=e.device)
        surv = torch.ones(e.shape[:2], device=e.device)
        for t in range(T):
            b = min(t, len(self.attn) - 1)
            z = h + e + self.step(torch.tensor(min(t, self.t_emb - 1), device=e.device))
            if freeze and self.freeze_z:
                z = z if t == 0 else torch.where(active[..., None], z, z_prev)
                z_prev = z
            a = self.attn[b](self.n1[b](z), key_mask)
            y = self.n2[b](z + a)
            if self.kind == "dense":
                m = self.experts.one(min(t, self.T - 1), y)
            elif self.K == 1:
                m = self.experts.one(0, y)
            else:
                ex, rt = (self.experts[b], self.router[b]) if self.kind == "moe" else (self.experts, self.router)
                r = rt(y).float()
                p = r.softmax(-1)
                if self.training:
                    g = F.gumbel_softmax(r, tau=tau, hard=True)  # forward: one-hot, backward: soft
                    m = ex.mix(g.to(y.dtype), y)
                    idx = g.argmax(-1)
                else:
                    idx = r.argmax(-1)
                    if route == "shuffle":  # same per-step expert mix, but not tied to the token
                        flat = idx.flatten()
                        idx = flat[torch.randperm(flat.numel(), device=flat.device)].view_as(idx)
                    elif route == "random":
                        idx = torch.randint_like(idx, self.K)
                    m = torch.zeros_like(y)
                    for k in range(self.K):
                        sel = idx == k
                        if sel.any():
                            m[sel] = ex.one(k, y[sel]).to(m.dtype)
                f = F.one_hot(idx, self.K).float().mean((0, 1))
                aux.append(self.K * (f * p.mean((0, 1))).sum())
                choices.append(idx)
            h_new = h + a + m
            hf = self.nf(h_new)
            with torch.autocast("cuda", enabled=False):  # fp32 head: bf16 resolves only ~30 mV at 4-8 V
                lg = self.head(hf.float())
            lam = torch.sigmoid(self.halt(hf).float().squeeze(-1))
            if freeze:  # halted tokens keep their state and outputs
                h = torch.where(active[..., None], h_new, h)
                if t > 0:
                    lg = torch.where(active[..., None], lg, logits[-1])
                surv = surv * (1 - lam)
                active = active & ((1 - surv < halt_thresh) if HALT_RULE == "cdf" else (lam <= halt_thresh))
            else:
                h = h_new
            logits.append(lg)
            lams.append(lam)
        out = {"logits": torch.stack(logits), "lam": torch.stack(lams), "h": h}
        if choices:
            out["choice"] = torch.stack(choices)
            out["aux"] = torch.stack(aux).mean()
        return out


def halting_dist(lam):
    """PonderNet: p_t = lam_t * prod_{s<t}(1 - lam_s), with the last step taking the remainder."""
    lam = lam.clone()
    lam[-1] = 1.0
    surv = torch.cumprod(torch.cat([torch.ones_like(lam[:1]), 1 - lam[:-1]]), 0)
    return lam * surv


HALT_RULE = os.environ.get("HALT_RULE", "lambda")  # "lambda": lambda_t > 0.5; "cdf": P(halted by t) >= 0.5


def halt_step(lam, thresh=0.5):
    """First step that meets the halting rule (last step if none). Returns [B,N] step indices."""
    over = (1 - torch.cumprod(1 - lam, 0) >= thresh) if HALT_RULE == "cdf" else lam > thresh
    over[-1] = True
    return over.float().argmax(0)
