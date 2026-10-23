# Klein-style single-stream DiT for 32x32 RGBA sprites, shared by train.py and prof_train.py; 2026-10-17.
import math
import torch, torch.nn as nn, torch.nn.functional as F

HEAD = 128
NCAT, NFP = 8, 3
# prof_train.py flips these to measure them; 'off' settings exist only for timing.
FAST = {'norm': True, 'rope': 'stack', 'qknorm': True, 'joint_qk': False, 'addcmul': False}


class RMSNorm(nn.Module):
    def __init__(s, d):
        super().__init__(); s.w = nn.Parameter(torch.ones(d))

    def forward(s, x):
        if FAST['norm']:
            return F.rms_norm(x, (x.shape[-1],), s.w, 1e-6)
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype) * s.w


def rope_table(side):
    # Half of each head encodes the row, half the column (2D RoPE, as in klein).
    f = 1.0 / (10000 ** (torch.arange(0, HEAD // 4) / (HEAD // 4)))
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij')
    ang = torch.cat([ys.reshape(-1, 1) * f, xs.reshape(-1, 1) * f], 1)
    return torch.cos(ang), torch.sin(ang)


def rope(x, cos, sin):
    if FAST['rope'] == 'off':
        return x
    if FAST['rope'] == 'half':
        # Rotate-half pairing (dim i with i+64): fewer passes over q/k than the interleaved form.
        c, s_ = torch.cat([cos, cos], -1).to(x.dtype), torch.cat([sin, sin], -1).to(x.dtype)
        x1, x2 = x.chunk(2, -1)
        return x * c + torch.cat([-x2, x1], -1) * s_
    if FAST['rope'] == 'stack':
        a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)
    a, b = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2).to(x.dtype)


class Block(nn.Module):
    def __init__(s, h):
        super().__init__()
        s.h, s.nh, s.mlp = h, h // HEAD, 3 * h
        s.norm = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        s.mod = nn.Linear(h, 3 * h)
        s.lin1 = nn.Linear(h, 3 * h + 2 * s.mlp)
        s.qn, s.kn = RMSNorm(HEAD), RMSNorm(HEAD)
        s.lin2 = nn.Linear(h + s.mlp, h)
        nn.init.zeros_(s.mod.weight); nn.init.zeros_(s.mod.bias)

    def forward(s, x, c, cos, sin):
        B, T, _ = x.shape
        sh, sc, g = s.mod(F.silu(c)).unsqueeze(1).chunk(3, -1)
        xm = torch.addcmul(sh, s.norm(x), 1 + sc) if FAST['addcmul'] else s.norm(x) * (1 + sc) + sh
        qkv, a, b = s.lin1(xm).split([3 * s.h, s.mlp, s.mlp], -1)
        qkv = qkv.view(B, T, 3, s.nh, HEAD).permute(2, 0, 3, 1, 4)
        v = qkv[2]
        if FAST['joint_qk']:
            qk = qkv[:2]
            if FAST['qknorm']:
                qk = F.rms_norm(qk, (HEAD,), eps=1e-6) * torch.stack([s.qn.w, s.kn.w]).view(2, 1, 1, 1, HEAD).to(qk.dtype)
            q, k = rope(qk, cos, sin).unbind(0)
        else:
            q, k = qkv[0], qkv[1]
            if FAST['qknorm']:
                q, k = s.qn(q), s.kn(k)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, s.h)
        out = s.lin2(torch.cat([o, F.silu(a) * b], -1))
        return torch.addcmul(x, g, out) if FAST['addcmul'] else x + g * out


class DiT(nn.Module):
    def __init__(s, h, depth, patch, vocab=0):
        super().__init__()
        s.p, s.side = patch, 32 // patch
        s.inp = nn.Linear(4 * patch * patch, h)
        s.t1, s.t2 = nn.Linear(256, h), nn.Linear(h, h)
        s.cat = nn.Embedding(NCAT + 1, h)     # last index = no label
        s.fp = nn.Embedding(NFP + 1, h)
        if vocab:   # caption words: learned embeddings, mean-pooled, 0 = padding/none
            s.wemb = nn.Embedding(vocab, h, padding_idx=0)
            s.wproj = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        s.blocks = nn.ModuleList([Block(h) for _ in range(depth)])
        s.fnorm = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        s.fmod = nn.Linear(h, 2 * h)
        s.outp = nn.Linear(h, 4 * patch * patch)
        for m in (s.fmod, s.outp):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        cos, sin = rope_table(s.side)
        s.register_buffer('cos', cos, persistent=False); s.register_buffer('sin', sin, persistent=False)

    def forward(s, x, t, cat, fp, txt=None):
        B, p, n = x.shape[0], s.p, s.side
        x = x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(B, n * n, 4 * p * p)
        half = torch.exp(-math.log(10000) * torch.arange(128, device=x.device) / 128)
        te = torch.cat([torch.cos(t[:, None] * 1000 * half), torch.sin(t[:, None] * 1000 * half)], -1)
        c = s.t2(F.silu(s.t1(te))) + s.cat(cat) + s.fp(fp)
        if txt is not None:
            m = (txt > 0).to(c.dtype)
            e = (s.wemb(txt) * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            c = c + s.wproj(e) * (m.sum(1, keepdim=True) > 0)
        x = s.inp(x)
        for blk in s.blocks:
            x = blk(x, c, s.cos, s.sin)
        sh, sc = s.fmod(F.silu(c)).unsqueeze(1).chunk(2, -1)
        x = s.outp(s.fnorm(x) * (1 + sc) + sh)
        return x.view(B, n, n, 4, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, 4, 32, 32)
