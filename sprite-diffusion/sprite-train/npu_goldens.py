# Per-op fp32 goldens of one sprite DiT evaluation (block 0 in detail, whole-model output), for the NPU port; 2026-10-19.
import argparse
import numpy as np, torch, torch.nn.functional as F
import dit

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--cat', type=int, default=3)
ap.add_argument('--n', type=int, default=4, help='sprites; CFG doubles the batch')
ap.add_argument('--t', type=float, default=0.75)
args = ap.parse_args()

s = torch.load(args.ckpt, map_location='cpu'); a = s['args']
m = dit.DiT(a['hidden'], a['depth'], a['patch']).eval(); m.load_state_dict(s['ema'])
g = torch.Generator().manual_seed(1)
x = torch.randn(args.n, 4, 32, 32, generator=g)
xx = torch.cat([x, x])
cat = torch.cat([torch.full((args.n,), args.cat), torch.full((args.n,), dit.NCAT)])
fp = torch.full((2 * args.n,), dit.NFP)
t = torch.full((2 * args.n,), args.t)
G = {}
blk = m.blocks[0]


def hook_block(mod, inp, out):
    xin, c, cos, sin = inp
    B, T, _ = xin.shape
    sh, sc, gt = mod.mod(F.silu(c)).chunk(3, -1)
    xm = mod.norm(xin) * (1 + sc[:, None]) + sh[:, None]
    qkv, ga, up = mod.lin1(xm).split([3 * mod.h, mod.mlp, mod.mlp], -1)
    q, k, v = qkv.view(B, T, 3, mod.nh, dit.HEAD).permute(2, 0, 3, 1, 4)
    qr, kr = dit.rope(mod.qn(q), cos, sin), dit.rope(mod.kn(k), cos, sin)
    o = F.scaled_dot_product_attention(qr, kr, v).transpose(1, 2).reshape(B, T, mod.h)
    lin2_in = torch.cat([o, F.silu(ga) * up], -1)
    G.update(b0_x=xin, b0_shift=sh, b0_scale=sc, b0_gate=gt, b0_xm=xm, b0_lin1=mod.lin1(xm), b0_q=qr.transpose(1, 2).reshape(B, T, -1),
             b0_k=kr.transpose(1, 2).reshape(B, T, -1), b0_v=v.transpose(1, 2).reshape(B, T, -1), b0_attn=o, b0_lin2_in=lin2_in,
             b0_lin2=mod.lin2(lin2_in), b0_out=out)


def hook_inp(mod, inp, out):
    G.update(patches=inp[0], inp_out=out)


def hook_last(mod, inp, out):
    G.update(last_out=out)


blk.register_forward_hook(hook_block)
m.inp.register_forward_hook(hook_inp)
m.blocks[-1].register_forward_hook(hook_last)
with torch.no_grad():
    v = m(xx, t, cat, fp)
    half = torch.exp(-np.log(10000) * torch.arange(128) / 128)
    te = torch.cat([torch.cos(t[:, None] * 1000 * half), torch.sin(t[:, None] * 1000 * half)], -1)
    c = m.t2(F.silu(m.t1(te))) + m.cat(cat) + m.fp(fp)
    fsh, fsc = m.fmod(F.silu(c)).chunk(2, -1)
np.savez(args.out, noise=x.numpy(), cat=cat.numpy(), t=args.t, patch=a['patch'], cond=c.numpy(), final_shift=fsh.numpy(), final_scale=fsc.numpy(),
         v=v.numpy(), cos=m.cos.numpy(), sin=m.sin.numpy(), **{k: val.numpy() for k, val in G.items()})
print('wrote', args.out, {k: tuple(val.shape) for k, val in G.items()})
