# Times train.py's step with each speed-up switched on in turn, plus bench.py's bare blocks as the ceiling; 2026-10-17.
import argparse, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import dit

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='C:/dev/sprite-data/trainset.npz')
ap.add_argument('--patch', type=int, default=4)
ap.add_argument('--hidden', type=int, default=384)
ap.add_argument('--depth', type=int, default=10)
ap.add_argument('--steps', type=int, default=40)
args = ap.parse_args()
dev = 'cuda'
d = np.load(args.data)
X_host = torch.from_numpy(d['sprites'][:50000]).permute(0, 3, 1, 2).contiguous()
C_host = torch.from_numpy(np.clip(d['cat'][:50000], 0, None).astype(np.int64))
FP_host = torch.from_numpy(d['fp'][:50000].astype(np.int64))


def run(name, sync_each, data_gpu, foreach_ema, fused, fast, batch, attn_math=False, **flags):
    dit.FAST.update(norm=fast, rope='stack', qknorm=True, joint_qk=False, addcmul=False)
    dit.FAST.update(flags)
    torch.manual_seed(0)
    model = dit.DiT(args.hidden, args.depth, args.patch).to(dev)
    ema = dit.DiT(args.hidden, args.depth, args.patch).to(dev).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=fused or None)
    X, C, FP = (X_host.to(dev), C_host.to(dev), FP_host.to(dev)) if data_gpu else (X_host, C_host, FP_host)
    mp, ep = list(model.parameters()), list(ema.parameters())
    acc = torch.zeros((), device=dev)

    def step():
        nonlocal acc
        idx = torch.randint(len(X), (batch,), device=X.device)
        x = X[idx].to(dev, non_blocking=True).float() / 127.5 - 1
        cat, fp = C[idx].to(dev), FP[idx].to(dev)
        t = torch.sigmoid(torch.randn(batch, device=dev))
        n = torch.randn_like(x)
        xt = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * n
        with torch.autocast(dev, dtype=torch.bfloat16):
            loss = F.mse_loss(model(xt, t, cat, fp).float(), n - x)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(mp, 1.0, foreach=True); opt.step()
        with torch.no_grad():
            if foreach_ema:
                torch._foreach_lerp_(ep, mp, 0.001)
            else:
                for pe, pm in zip(ep, mp):
                    pe.lerp_(pm, 0.001)
        if sync_each:
            loss.item()
        else:
            acc += loss.detach()

    ctx = torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]) if attn_math else torch.enable_grad()
    with ctx:
        for _ in range(5):
            step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
    sps = args.steps * batch / (time.time() - t0)
    print(f'{name:48s} {sps:6.0f} samples/s', flush=True)
    del model, ema, opt; torch.cuda.empty_cache()
    return sps


def ceiling(batch):
    class Blk(nn.Module):   # bench.py's bare block
        def __init__(s, h):
            super().__init__(); s.n1 = nn.LayerNorm(h); s.qkv = nn.Linear(h, 3 * h); s.o = nn.Linear(h, h)
            s.n2 = nn.LayerNorm(h); s.f1 = nn.Linear(h, 6 * h); s.f2 = nn.Linear(3 * h, h); s.H = h // 128

        def forward(s, x):
            B, T, C = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.H, 128).permute(2, 0, 3, 1, 4)
            x = x + s.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, C))
            a, b = s.f1(s.n2(x)).chunk(2, -1); return x + s.f2(F.silu(a) * b)
    m = nn.Sequential(*[Blk(args.hidden) for _ in range(args.depth)]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-4)
    x = torch.randn(batch, (32 // args.patch) ** 2, args.hidden, device=dev)

    def step():
        with torch.autocast(dev, dtype=torch.bfloat16):
            l = m(x).float().pow(2).mean()
        l.backward(); opt.step(); opt.zero_grad()
    for _ in range(5):
        step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    print(f'{"ceiling: bench.py bare blocks":48s} {args.steps * batch / (time.time() - t0):6.0f} samples/s', flush=True)


print(f'patch {args.patch}, {(32 // args.patch) ** 2} tokens, h{args.hidden} x {args.depth}', flush=True)
ceiling(256)
cur = dict(sync_each=False, data_gpu=True, foreach_ema=True, fused=True, fast=True, batch=256)
run('A current train.py settings', **cur)
run('B RoPE off (timing only)', **cur, rope='off')
run('C QK norm off (timing only)', **cur, qknorm=False)
run('D RoPE + QK norm off (timing only)', **cur, rope='off', qknorm=False)
run('E rotate-half RoPE', **cur, rope='half')
run('F rotate-half RoPE + joint q/k', **cur, rope='half', joint_qk=True)
run('G = F + addcmul', **cur, rope='half', joint_qk=True, addcmul=True)
run('A again (drift check)', **cur)
ceiling(256)
