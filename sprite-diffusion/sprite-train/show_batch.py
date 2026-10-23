# A fresh labelled contact sheet from a category-only checkpoint (new seed each call); 2026-10-19.
import sys, random
import numpy as np, torch
from PIL import Image, ImageDraw
import dit

ck, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev = 'cuda'
s = torch.load(ck, map_location=dev); a = s['args']
m = dit.DiT(a['hidden'], a['depth'], a['patch']).to(dev).eval(); m.load_state_dict(s['ema'])
NAMES = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature', 'none']
random.seed(seed)
cats = [c for c in range(8) for _ in range(6)] + [8, 8]
random.shuffle(cats)
cat = torch.tensor(cats, device=dev)
g = torch.Generator(dev).manual_seed(seed)
x = torch.randn(len(cat), 4, 32, 32, device=dev, generator=g)
nc, fp = torch.full_like(cat, dit.NCAT), torch.full_like(cat, dit.NFP)
ts = torch.linspace(1, 0, 51, device=dev)
with torch.no_grad():
    for i in range(50):
        t = ts[i].expand(len(cat))
        with torch.autocast(dev, dtype=torch.bfloat16):
            v = m(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cat, nc]), torch.cat([fp, fp])).float()
        vc, vu = v.chunk(2)
        x = x + (ts[i + 1] - ts[i]) * (vu + 3.0 * (vc - vu))
arr = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy(); arr[..., 3] = np.where(arr[..., 3] >= 128, 255, 0)
cols, cell, S = 10, 104, 96
im = Image.new('RGB', (cols * cell, 5 * (cell + 12)), (118, 118, 118)); d = ImageDraw.Draw(im)
for i, p in enumerate(arr):
    cx, cy = (i % cols) * cell + 4, (i // cols) * (cell + 12) + 2
    q = Image.fromarray(p).resize((S, S), Image.NEAREST); im.paste(q, (cx, cy), q)
    d.text((cx, cy + S + 1), NAMES[cats[i]], fill=(255, 255, 255))
im.save(out); print('wrote', out)
if len(sys.argv) > 4:
    np.savez(sys.argv[4], x=x.float().cpu().numpy(), cat=np.array(cats))
