# Sprite DiT checkpoint -> fp32 NPU layouts (padded GEMM weights, modulation tables per step and category); igpu-venv; 2026-10-19.
import argparse, json
import numpy as np, torch, torch.nn.functional as F
import dit

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--steps', default='8,16')
args = ap.parse_args()

s = torch.load(args.ckpt, map_location='cpu'); a = s['args']
m = dit.DiT(a['hidden'], a['depth'], a['patch']).eval(); m.load_state_dict(s['ema'])
H, P = a['hidden'], 4 * a['patch'] ** 2
assert H == 384, 'layouts below assume hidden 384 (3 heads, padded to 4)'
W = {}

# inp: A = [patch values | 1 | 0...] read as K 512; row P carries the bias.
w = np.zeros((512, 1024), np.float32)
w[:P, :H] = m.inp.weight.detach().numpy().T
w[P, :H] = m.inp.bias.detach().numpy()
W['inp'] = w

LIN1_COLS = [(0, 0, 384), (512, 384, 384), (1024, 768, 384), (1536, 1152, 1152), (2688, 2304, 1152)]   # q, k, v, gate, up
for i, b in enumerate(m.blocks):
    w = np.zeros((512, 4096), np.float32)
    wt, bias = b.lin1.weight.detach().numpy().T, b.lin1.bias.detach().numpy()
    for dst, src, n in LIN1_COLS:
        w[:H, dst:dst + n] = wt[:, src:src + n]
        w[H, dst:dst + n] = bias[src:src + n]      # XM column 384 is a constant 1
    W[f'lin1_{i}'] = w
    w = np.zeros((1536, 1024), np.float32)
    w[:, :H] = b.lin2.weight.detach().numpy().T
    W[f'lin2_{i}'] = w
    qk = np.zeros(384, np.float32)
    qk[:128], qk[128:256] = b.qn.w.detach().numpy(), b.kn.w.detach().numpy()
    W[f'qkw_{i}'] = qk
w = np.zeros((512, 1024), np.float32)
w[:H, :P] = m.outp.weight.detach().numpy().T
w[H, :P] = m.outp.bias.detach().numpy()
W['outp'] = w

# RoPE table: per position p < side, cos[32] then sin[32] of p * f_j (dit.rope_table's frequencies).
f = 1.0 / (10000 ** (np.arange(0, 32) / 32))
side = 32 // a['patch']
ang = np.arange(side)[:, None] * f[None]
W['rope'] = np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)

# Modulation per (steps, step index, category incl. none): ln0 [shift, scale]; res_i [gate, shift', scale', gate * lin2 bias].
meta = {'patch': a['patch'], 'hidden': H, 'depth': a['depth'], 'P': P, 'side': side, 'steps': []}
cats = torch.arange(dit.NCAT + 1)
fp = torch.full_like(cats, dit.NFP)
with torch.no_grad():
    for n in [int(x) for x in args.steps.split(',')]:
        ts = torch.linspace(1, 0, n + 1)
        ln0 = np.zeros((n, len(cats), 2, H), np.float32)
        res = np.zeros((n, len(cats), a['depth'], 4, H), np.float32)
        for k in range(n):
            t = ts[k].expand(len(cats))
            half = torch.exp(-np.log(10000) * torch.arange(128) / 128)
            te = torch.cat([torch.cos(t[:, None] * 1000 * half), torch.sin(t[:, None] * 1000 * half)], -1)
            c = F.silu(m.t2(F.silu(m.t1(te))) + m.cat(cats) + m.fp(fp))
            mods = [b.mod(c).chunk(3, -1) for b in m.blocks]    # (shift, scale, gate) per block
            fsh, fsc = m.fmod(c).chunk(2, -1)
            ln0[k, :, 0], ln0[k, :, 1] = mods[0][0].numpy(), mods[0][1].numpy()
            for i, b in enumerate(m.blocks):
                g = mods[i][2]
                nsh, nsc = (mods[i + 1][0], mods[i + 1][1]) if i + 1 < len(m.blocks) else (fsh, fsc)
                res[k, :, i] = torch.stack([g, nsh, nsc, g * b.lin2.bias]).permute(1, 0, 2).numpy()
        W[f'ln0_s{n}'], W[f'res_s{n}'], W[f'ts_s{n}'] = ln0, res, ts.numpy()
        meta['steps'].append(n)
np.savez(args.out, meta=json.dumps(meta), **W)
print('wrote', args.out, meta)
