"""Transformer trunk plus the three MLP variants: dense, MoW, MoE.

Both routed variants are written so that a soft (all-choices) forward costs the
same as one dense MLP forward, which is what makes the soft-to-hard schedule
affordable here.

  MoW: tiers are contiguous prefixes, so tier t's output is the running sum of
  per-block projections p_0..p_t. Computing all four blocks once costs exactly
  one full projection, and any mixture over tiers is then a per-token reweight
  of those blocks: y = sum_b p_b * (sum_{t>=b} pi_t * scale_t).

  MoE: total expert hidden equals d_ff, so evaluating every expert densely also
  costs one full projection. Top-k and adaptive-k are reweights of the same
  per-expert outputs.

FLOPs are never inferred from wall clock -- they are accounted analytically from
the tier/expert choices, which is what the frontier figure reports.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, PAD, mlp_flops_per_token, attn_flops_per_token


@dataclass
class Ctx:
    """Per-forward control. `tiers` pins a hard choice; otherwise the router runs."""
    tiers: Optional[Sequence[int]] = None   # per-layer tier index, MoW fixed path
    temp: float = 1.0                       # router softmax temperature
    hard: bool = False                      # straight-through hard selection
    collect: bool = True                    # gather routing stats for diagnostics


@dataclass
class LayerInfo:
    """Per-layer routing diagnostics. Tensors are [B, T] or [B, T, C]."""
    flops_frac: Optional[torch.Tensor] = None   # differentiable, per token
    probs: Optional[torch.Tensor] = None        # routing distribution
    choice: Optional[torch.Tensor] = None       # hard argmax per token
    k_probs: Optional[torch.Tensor] = None      # adaptive-k only
    k_choice: Optional[torch.Tensor] = None
    balance_loss: Optional[torch.Tensor] = None
    fixed_frac: float = 1.0                     # non-routed FLOPs fraction


# ---------------------------------------------------------------- router

class Router(nn.Module):
    """Linear head on the (layer-normed) MLP input, producing choice logits."""

    def __init__(self, d_model: int, n_choices: int):
        super().__init__()
        self.lin = nn.Linear(d_model, n_choices)
        nn.init.normal_(self.lin.weight, std=0.01)   # start near-uniform
        nn.init.zeros_(self.lin.bias)

    def forward(self, u: torch.Tensor, temp: float, hard: bool):
        logits = self.lin(u) / max(temp, 1e-4)
        probs = F.softmax(logits, dim=-1)
        choice = probs.argmax(dim=-1)
        if hard:
            onehot = F.one_hot(choice, probs.shape[-1]).to(probs.dtype)
            probs = onehot + probs - probs.detach()   # straight-through
        return probs, choice


def _entropy(probs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    h = -(probs.clamp_min(1e-9).log() * probs).sum(-1)
    return (h * valid).sum() / valid.sum().clamp_min(1)


# ---------------------------------------------------------------- MLPs

class DenseMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.dense_hidden or cfg.d_ff
        self.hidden = h
        self.fc = nn.Linear(cfg.d_model, h)
        self.proj = nn.Linear(h, cfg.d_model)
        self.act = nn.GELU()
        self.frac = h / cfg.d_ff

    def forward(self, u, valid, ctx: Ctx):
        return self.proj(self.act(self.fc(u))), LayerInfo(fixed_frac=self.frac)


class MoWMLP(nn.Module):
    """Nested variable-width MLP. Tier t uses hidden neurons [0, tiers[t])."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        tiers = tuple(cfg.tiers)
        assert list(tiers) == sorted(tiers) and tiers[-1] == cfg.d_ff, \
            "tiers must be increasing prefixes ending at d_ff"
        self.tiers = tiers
        self.d_ff = cfg.d_ff
        self.blocks = list(zip((0,) + tiers[:-1], tiers))
        self.fc = nn.Linear(cfg.d_model, cfg.d_ff)
        self.proj = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
        # One scalar per tier. Without it the 25% path contributes to the residual
        # stream at a different magnitude than the 100% path and never catches up.
        self.scale = nn.Parameter(torch.ones(len(tiers))) if cfg.tier_scales else None
        self.router = Router(cfg.d_model, len(tiers)) if cfg.router else None

    def _scales(self, device):
        if self.scale is None:
            return torch.ones(len(self.tiers), device=device)
        return self.scale

    def forward_tier(self, u, tier_idx: int):
        """Hard path: genuinely slice the weights, so cost tracks the tier."""
        h = self.tiers[tier_idx]
        a = self.act(F.linear(u, self.fc.weight[:h], self.fc.bias[:h]))
        y = F.linear(a, self.proj.weight[:, :h])
        return y * self._scales(u.device)[tier_idx] + self.proj.bias

    def forward_mixed(self, u, probs):
        """Soft/straight-through path over all tiers at one dense MLP's cost."""
        a = self.act(self.fc(u))
        parts = [F.linear(a[..., s:e], self.proj.weight[:, s:e]) for s, e in self.blocks]
        # w[..., b] = sum_{t >= b} probs[..., t] * scale[t]
        w = (probs * self._scales(u.device)).flip(-1).cumsum(-1).flip(-1)
        y = sum(p * w[..., i: i + 1] for i, p in enumerate(parts))
        return y + self.proj.bias

    def forward(self, u, valid, ctx: Ctx, layer_idx: int = 0):
        fracs = torch.tensor([t / self.d_ff for t in self.tiers], device=u.device)
        if self.router is None or (ctx.tiers is not None):
            t = 0 if ctx.tiers is None else int(ctx.tiers[layer_idx])
            return self.forward_tier(u, t), LayerInfo(fixed_frac=self.tiers[t] / self.d_ff)
        probs, choice = self.router(u, ctx.temp, ctx.hard)
        y = self.forward_mixed(u, probs)
        return y, LayerInfo(flops_frac=(probs * fracs).sum(-1), probs=probs, choice=choice)


class MoEMLP(nn.Module):
    """N dense experts with a per-token router. n_experts * expert_hidden == d_ff,
    so a dense evaluation of all experts costs one reference MLP forward."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.E, self.he, self.d_ff = cfg.n_experts, cfg.expert_hidden, cfg.d_ff
        self.mode, self.top_k = cfg.moe_mode, cfg.top_k
        self.k_choices = tuple(cfg.k_choices)
        self.balance_coef = cfg.balance_coef
        self.fc = nn.Linear(cfg.d_model, self.E * self.he)
        self.proj_w = nn.Parameter(torch.empty(self.E, self.he, cfg.d_model))
        self.proj_b = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.proj_w, std=0.02)
        self.act = nn.GELU()
        self.router = Router(cfg.d_model, self.E)
        self.k_router = Router(cfg.d_model, len(self.k_choices)) if self.mode == "adaptive" else None

    @staticmethod
    def _topk_gate(probs, k):
        """Renormalized top-k gate, zeros elsewhere. Gradient flows via probs."""
        if k >= probs.shape[-1]:
            return probs
        thresh = probs.topk(k, dim=-1).values[..., -1:]
        g = probs * (probs >= thresh).to(probs.dtype)
        return g / g.sum(-1, keepdim=True).clamp_min(1e-9)

    def _balance(self, probs, gate, valid):
        """Switch-style load balance: E * sum_e f_e * P_e, minimized when uniform."""
        v = valid.unsqueeze(-1).to(probs.dtype)
        n = v.sum().clamp_min(1)
        sel = (gate > 0).to(probs.dtype) * v
        f = sel.sum((0, 1))
        f = f / f.sum().clamp_min(1e-9)
        P = (probs * v).sum((0, 1)) / n
        return self.E * (f * P).sum()

    def forward(self, u, valid, ctx: Ctx, layer_idx: int = 0):
        B, T, _ = u.shape
        probs, choice = self.router(u, ctx.temp, ctx.hard)

        info = LayerInfo(probs=probs, choice=choice)
        if self.mode == "adaptive":
            k_probs, k_choice = self.k_router(u, ctx.temp, ctx.hard)
            gate = sum(k_probs[..., i: i + 1] * self._topk_gate(probs, k)
                       for i, k in enumerate(self.k_choices))
            kf = torch.tensor([k * self.he / self.d_ff for k in self.k_choices], device=u.device)
            info.flops_frac = (k_probs * kf).sum(-1)
            info.k_probs, info.k_choice = k_probs, k_choice
        else:
            gate = self._topk_gate(probs, self.top_k)
            frac = self.top_k * self.he / self.d_ff
            info.flops_frac = torch.full((B, T), frac, device=u.device)
            info.fixed_frac = frac

        a = self.act(self.fc(u)).view(B, T, self.E, self.he)
        per_expert = torch.einsum("bteh,ehd->bted", a, self.proj_w)
        y = (per_expert * gate.unsqueeze(-1)).sum(2) + self.proj_b

        if self.balance_coef > 0:
            info.balance_loss = self.balance_coef * self._balance(probs, gate, valid)
        return y, info


def build_mlp(cfg: ModelConfig) -> nn.Module:
    return {"dense": DenseMLP, "mow": MoWMLP, "moe": MoEMLP}[cfg.mlp](cfg)


# ---------------------------------------------------------------- trunk

def rope_tables(head_dim: int, block_size: int, base: float = 10000.0):
    """(cos, sin) of shape (block_size, head_dim), each angle repeated for the pair
    it rotates. Buffers, not parameters -- there is nothing here to learn."""
    half = head_dim // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float32) / half)
    ang = torch.outer(torch.arange(block_size, dtype=torch.float32), inv)
    return torch.cat([ang.cos()] * 2, dim=-1), torch.cat([ang.sin()] * 2, dim=-1)


def _rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class CausalSelfAttention(nn.Module):
    """Position enters either through the token embedding (learned absolute, the
    original) or here through rotary embeddings.

    Rotary matters for arithmetic specifically. With absolute positions, the circuit
    that produces answer digit i has to be learned separately for every i, because
    nothing ties position i to position i+1. That makes each digit an independent
    lottery, and runs get stuck with one digit at chance. Rotary makes attention
    depend on the offset between query and key, so one learned "read the operand
    digit k places back" circuit serves every digit at once.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.nh, self.hd = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.rope = cfg.pos_encoding == "rope"
        if self.rope:
            assert self.hd % 2 == 0, "rotary needs an even head dimension"
            cos, sin = rope_tables(self.hd, cfg.block_size)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def _apply_rope(self, t, T):
        cos, sin = self.cos[:T].to(t.dtype), self.sin[:T].to(t.dtype)
        return t * cos + _rotate_half(t) * sin

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = lambda t: t.view(B, T, self.nh, self.hd).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        if self.rope:
            q, k = self._apply_rope(q, T), self._apply_rope(k, T)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = build_mlp(cfg)

    def forward(self, x, valid, ctx: Ctx):
        x = x + self.attn(self.ln1(x))
        u = self.ln2(x)                      # the router reads this
        if isinstance(self.mlp, DenseMLP):
            y, info = self.mlp(u, valid, ctx)
        else:
            y, info = self.mlp(u, valid, ctx, self.layer_idx)
        return x + y, info


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # rotary carries position inside attention instead, so the absolute table
        # would be a second, redundant position signal
        self.pos = None if cfg.pos_encoding == "rope" else \
            nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)
        for n, p in self.named_parameters():   # scaled residual init
            if n.endswith("proj.weight") or n.endswith("out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, ctx: Optional[Ctx] = None):
        ctx = ctx or Ctx()
        B, T = idx.shape
        valid = idx != PAD
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=idx.device))
        x = self.drop(x)
        infos: List[LayerInfo] = []
        for blk in self.blocks:
            x, info = blk(x, valid, ctx)
            infos.append(info)
        return self.head(self.ln_f(x)), infos

    # -------------------------------------------------- losses & accounting

    def routing_losses(self, infos, valid, lambda_flops: float):
        """-> (flops_penalty, balance_penalty, mean_flops_fraction)."""
        v = valid.to(torch.float32)
        n = v.sum().clamp_min(1)
        fr, bal = [], []
        for info in infos:
            if info.flops_frac is not None:
                fr.append((info.flops_frac * v).sum() / n)
            else:
                fr.append(torch.tensor(info.fixed_frac, device=v.device))
            if info.balance_loss is not None:
                bal.append(info.balance_loss)
        mean_frac = torch.stack(fr).mean()
        balance = torch.stack(bal).sum() if bal else torch.zeros((), device=v.device)
        return lambda_flops * mean_frac, balance, mean_frac

    def flops_per_token(self, mean_mlp_frac: float) -> dict:
        c = self.cfg
        mlp = c.n_layer * mlp_flops_per_token(c.d_model, c.d_ff) * mean_mlp_frac
        attn = c.n_layer * attn_flops_per_token(c.d_model, c.block_size)
        head = 2.0 * c.d_model * c.vocab_size
        return {"mlp": mlp, "attn": attn, "head": head, "total": mlp + attn + head,
                "mlp_frac": mean_mlp_frac}

    def n_params(self, trainable_only: bool = False) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------- self-test

def _selftest():
    from .config import DataConfig
    from . import data as D
    import random

    torch.manual_seed(0)
    dcfg = DataConfig()

    # 1. slice path and block path must agree exactly on a one-hot distribution
    cfg = ModelConfig(mlp="mow", router=True)
    mow = MoWMLP(cfg)
    with torch.no_grad():
        mow.scale.copy_(torch.tensor([0.7, 1.3, 0.9, 1.1]))
        u = torch.randn(3, 5, cfg.d_model)
        for t in range(len(cfg.tiers)):
            oh = F.one_hot(torch.full((3, 5), t), len(cfg.tiers)).float()
            a, b = mow.forward_tier(u, t), mow.forward_mixed(u, oh)
            err = (a - b).abs().max().item()
            assert err < 1e-4, (t, err)
    print("MoW: slice path == block path for every tier (max err < 1e-4)")

    # 2. straight-through gives a hard forward with a live gradient
    mow.zero_grad()
    u = torch.randn(2, 4, cfg.d_model, requires_grad=True)
    probs, _ = mow.router(u, temp=1.0, hard=True)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 4))
    assert ((probs > 0.999) | (probs < 1e-6)).all(), "hard forward should be one-hot"
    mow.forward_mixed(u, probs).sum().backward()
    assert mow.router.lin.weight.grad.abs().sum() > 0, "no gradient through the router"
    print("MoW: straight-through is one-hot forward with non-zero router gradient")

    # 3. MoE dense-expert evaluation costs one reference MLP; gates behave
    for mode, kw in [("topk", dict(top_k=1)), ("topk", dict(top_k=2)),
                     ("adaptive", dict(k_choices=(1, 2, 3, 4)))]:
        c = ModelConfig(mlp="moe", moe_mode=mode, **kw)
        moe = MoEMLP(c)
        u = torch.randn(2, 6, c.d_model)
        valid = torch.ones(2, 6, dtype=torch.bool)
        y, info = moe(u, valid, Ctx())
        assert y.shape == u.shape
        assert info.flops_frac.shape == (2, 6)
        assert info.balance_loss is not None and info.balance_loss.item() > 0
    print("MoE: top-1 / top-2 / adaptive-k all forward with live balance loss")

    # 4. analytic FLOPs fractions land where they should
    m = Transformer(ModelConfig(mlp="mow", router=True))
    for t, want in zip(range(4), [0.25, 0.5, 0.75, 1.0]):
        got = m.flops_per_token(m.cfg.tiers[t] / m.cfg.d_ff)["mlp_frac"]
        assert abs(got - want) < 1e-9
    fine = ModelConfig(mlp="moe", n_experts=16, expert_hidden=32,
                       moe_mode="adaptive", k_choices=(1, 2, 4, 8, 16))
    assert [k * fine.expert_hidden / fine.d_ff for k in fine.k_choices] == \
           [0.0625, 0.125, 0.25, 0.5, 1.0]
    print("FLOPs: MoW tiers -> 25/50/75/100%, fine-grained k -> 6.25..100%")

    # 5. parameter counts: MoW and MoE are matched to the dense reference
    counts = {}
    for name, c in [("dense", ModelConfig(mlp="dense")),
                    ("mow", ModelConfig(mlp="mow", router=True)),
                    ("moe-4x128", ModelConfig(mlp="moe", n_experts=4, expert_hidden=128)),
                    ("moe-16x32", ModelConfig(mlp="moe", n_experts=16, expert_hidden=32,
                                              moe_mode="adaptive", k_choices=(1, 2, 4, 8, 16)))]:
        counts[name] = Transformer(c).n_params()
    print("params:", {k: f"{v/1e6:.3f}M" for k, v in counts.items()})
    assert abs(counts["mow"] - counts["dense"]) / counts["dense"] < 0.01
    assert abs(counts["moe-4x128"] - counts["dense"]) / counts["dense"] < 0.02

    # 6. overfit gate: 64 examples to ~0 loss at full width
    model = Transformer(ModelConfig(mlp="mow"))
    batch = D.sample_batch(random.Random(0), 64, dcfg)
    x = torch.from_numpy(batch.x); y = torch.from_numpy(batch.y)
    msk = torch.from_numpy(batch.mask)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ctx = Ctx(tiers=[3] * model.cfg.n_layer)
    for step in range(401):
        logits, _ = model(x, ctx)
        ls = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
        loss = (ls * msk.view(-1)).sum() / msk.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  overfit step {step:3d}: loss {loss.item():.4f}")
    assert loss.item() < 0.02, f"overfit gate failed: {loss.item():.4f}"
    print(f"overfit gate: 64 examples -> loss {loss.item():.5f} (< 0.02)")
    print("\nall model self-tests passed")


if __name__ == "__main__":
    _selftest()
