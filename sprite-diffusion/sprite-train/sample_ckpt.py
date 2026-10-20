# Samples a checkpoint under several settings (same noise) into one contact sheet per setting; igpu-venv or cuda; 2026-10-19.
import argparse, math, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw
import dit

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--per-cat', type=int, default=12)
ap.add_argument('--device', default='cuda')
args = ap.parse_args()
dev = args.device
s = torch.load(args.ckpt, map_location=dev)
a = s['args']
NAMES = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature']
models = {}
for k in ('ema', 'model'):
    m = dit.DiT(a['hidden'], a['depth'], a['patch']).to(dev).eval()
    m.load_state_dict(s[k]); models[k] = m


@torch.no_grad()
def sample(model, cat, steps, cfg, seed=0):
    g = torch.Generator(dev).manual_seed(seed)
    x = torch.randn(len(cat), 4, 32, 32, device=dev, generator=g)
    fp = torch.full_like(cat, dit.NFP); nc, nf = torch.full_like(cat, dit.NCAT), torch.full_like(cat, dit.NFP)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    for i in range(steps):
        t = ts[i].expand(len(cat))
        with torch.autocast(dev, dtype=torch.bfloat16):
            v = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cat, nc]), torch.cat([fp, nf])).float()
        vc, vu = v.chunk(2)
        x = x + (ts[i + 1] - ts[i]) * (vu + cfg * (vc - vu))
    return x


def sheet(x, path, title):
    arr = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    arr[..., 3] = np.where(arr[..., 3] >= 128, 255, 0)
    cols = args.per_cat
    im = Image.new('RGB', (cols * 66, len(NAMES) * 66 + 14), (118, 118, 118))
    ImageDraw.Draw(im).text((2, 1), title, fill=(255, 255, 0))
    for i, p in enumerate(arr):
        q = Image.fromarray(p).resize((64, 64), Image.NEAREST)
        im.paste(q, ((i % cols) * 66 + 1, (i // cols) * 66 + 15), q)
    im.save(path)


cat = torch.arange(len(NAMES), device=dev).repeat_interleave(args.per_cat)
os.makedirs(args.out, exist_ok=True)
for name, which, steps, cfg in [('A_ema_16_cfg3', 'ema', 16, 3.0), ('B_ema_50_cfg3', 'ema', 50, 3.0), ('C_ema_50_cfg1.5', 'ema', 50, 1.5),
                                ('D_ema_50_cfg5', 'ema', 50, 5.0), ('E_raw_50_cfg3', 'model', 50, 3.0)]:
    sheet(sample(models[which], cat, steps, cfg), f'{args.out}/{name}.png', f'{name}  (rows: {", ".join(NAMES)})')
    print('wrote', name, flush=True)
