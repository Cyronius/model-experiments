# Raw vs snapped sheet from a show_batch.py .npz (CPU only); 2026-10-19.
import sys
import numpy as np
from PIL import Image, ImageDraw
from snap import snap2

z = np.load(sys.argv[1]); out = sys.argv[2]
NAMES = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature', 'none']
raw = ((np.clip(z['x'], -1, 1) + 1) * 127.5).round().astype(np.uint8).transpose(0, 2, 3, 1)
raw[..., 3] = np.where(raw[..., 3] >= 128, 255, 0)
sn = np.stack([snap2(x) for x in z['x']])
cols, S = 10, 96; cell = 2 * S + 12
im = Image.new('RGB', (cols * cell, 5 * (S + 14)), (118, 118, 118)); d = ImageDraw.Draw(im)
ncol = lambda p: len(np.unique(p[p[..., 3] > 0][:, :3], axis=0))
for i in range(len(raw)):
    cx, cy = (i % cols) * cell + 4, (i // cols) * (S + 14) + 2
    for j, p in enumerate((raw[i], sn[i])):
        q = Image.fromarray(p).resize((S, S), Image.NEAREST); im.paste(q, (cx + j * (S + 2), cy), q)
    d.text((cx, cy + S + 1), f"{NAMES[z['cat'][i]]} {ncol(raw[i])}->{ncol(sn[i])}", fill=(255, 255, 255))
im.save(out)
print('median colours raw', np.median([ncol(p) for p in raw]), 'snapped', np.median([ncol(p) for p in sn]))
