# Samples the caption-conditioned checkpoint from hand-written prompts, 12 seeds each, into one sheet; 2026-10-19.
import json, os, re, sys
import numpy as np, torch
from PIL import Image, ImageDraw
import dit

ck, vocab_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
dev = 'cuda'
s = torch.load(ck, map_location=dev); a = s['args']
vocab = json.load(open(vocab_path))
m = dit.DiT(a['hidden'], a['depth'], a['patch'], len(vocab) + 1).to(dev).eval(); m.load_state_dict(s['ema'])
NAMES = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature']
PROMPTS = ['weapon, sword', 'weapon, axe', 'weapon, bow', 'food, apple, red', 'food, bread', 'food, cake', 'item, potion, red', 'item, key, gold',
           'item, book', 'creature, zombie, green', 'creature, knight, gray', 'creature, dragon', 'nature, tree, green', 'structure, door, brown',
           'effect, fire, orange', 'terrain, grass, green']


def tokens(p):
    ws = [w for w in re.split(r'[^a-z0-9]+', p.lower()) if w in vocab][:12]
    return [vocab[w] for w in ws] + [0] * (12 - len(ws)), [w for w in re.split(r'[^a-z0-9]+', p.lower()) if w and w not in vocab]


@torch.no_grad()
def sample(cat, txt, steps=50, cfg=3.0, seed=0):
    g = torch.Generator(dev).manual_seed(seed)
    x = torch.randn(len(cat), 4, 32, 32, device=dev, generator=g)
    nc, nf, fp = torch.full_like(cat, dit.NCAT), torch.full_like(cat, dit.NFP), torch.full_like(cat, dit.NFP)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    for i in range(steps):
        t = ts[i].expand(len(cat))
        with torch.autocast(dev, dtype=torch.bfloat16):
            v = m(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cat, nc]), torch.cat([fp, nf]), torch.cat([txt, torch.zeros_like(txt)])).float()
        vc, vu = v.chunk(2)
        x = x + (ts[i + 1] - ts[i]) * (vu + cfg * (vc - vu))
    return x


cols, S = 12, 64
im = Image.new('RGB', (cols * (S + 2) + 150, len(PROMPTS) * (S + 2)), (118, 118, 118)); d = ImageDraw.Draw(im)
for r, p in enumerate(PROMPTS):
    ids, unknown = tokens(p)
    cat = torch.full((cols,), NAMES.index(p.split(',')[0]), device=dev)
    x = sample(cat, torch.tensor([ids] * cols, device=dev))
    arr = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy(); arr[..., 3] = np.where(arr[..., 3] >= 128, 255, 0)
    d.text((2, r * (S + 2) + 24), p + (f'  [unknown: {unknown}]' if unknown else ''), fill=(255, 255, 0))
    for j, q in enumerate(arr):
        t = Image.fromarray(q).resize((S, S), Image.NEAREST); im.paste(t, (150 + j * (S + 2), r * (S + 2) + 1), t)
im.save(out); print('wrote', out)
