# Drops text that the sprite cutter kept: labels drawn on the sheets (weapon names, captions,
# single letters). Two passes, both automatic:
#   1. SigLIP zero-shot: a sprite whose best "text" prompt beats its best object prompt by
#      more than MARGIN is text.
#   2. Per sheet, the colours of pass-1 text form that sheet's font palette; a cut-out that
#      SigLIP leans towards text (margin > 0) and whose colours all lie inside it is a letter
#      in the same font. (Palette alone also caught NES sprites: their 3 colours often fit.)
# Writes DATA/selftrain/clean-<sprites file>.jsonl (id, keep, reason, text margin) and a
# contact sheet of drops. Run under C:\dev\igpu-venv; resumable (embeddings are cached).
#   selftrain_clean.py [sprites2.jsonl]
import json, os, random, sys, numpy as np
from collections import Counter, defaultdict
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA
import siglipkit

D = f'{DATA}/selftrain'
SRC = sys.argv[1] if len(sys.argv) > 1 else 'sprites2.jsonl'
MARGIN = 0.015
rs = [r for r in map(json.loads, open(f'{D}/{SRC}')) if r.get('id')]
E = siglipkit.embed_cached([r['id'] for r in rs], [f"{D}/render/{r['id']}.png" for r in rs],
                           f'{D}/emb-{siglipkit.TAG}/tsr')
text_p = ['text', 'letters', 'a word written in a font', 'a single letter of the alphabet', 'a number digit']
obj_p = ['a person', 'a monster', 'an item', 'a weapon', 'food', 'a magic effect', 'a tree', 'a wall',
         'a creature', 'a sword', 'a potion']
Tt = siglipkit.embed_text([f'pixel art sprite of {p}' for p in text_p])
To = siglipkit.embed_text([f'pixel art sprite of {p}' for p in obj_p])
margin = (E @ Tt.T).max(1) - (E @ To.T).max(1)


def colours(r):
    a = np.array(Image.open(f"{D}/sprites/{r['id']}.png"))
    px = a[a[..., 3] > 0][:, :3]
    return {tuple(c) for c in np.unique(px, axis=0)}


palette = defaultdict(set)
for i, r in enumerate(rs):
    if margin[i] > MARGIN:
        palette[r['asset']] |= colours(r)
out = []
for i, r in enumerate(rs):
    if margin[i] > MARGIN:
        keep, why = False, 'text'
    elif margin[i] > 0 and r['asset'] in palette and colours(r) <= palette[r['asset']]:
        keep, why = False, 'font palette'
    else:
        keep, why = True, ''
    out.append(dict(id=r['id'], keep=keep, reason=why, text_margin=round(float(margin[i]), 4)))
with open(f'{D}/clean-{SRC}', 'w') as fh:
    for o in out:
        fh.write(json.dumps(o) + '\n')
seeds = {r['id']: r['seed'] for r in rs}
dropped = Counter((seeds[o['id']], o['reason']) for o in out if not o['keep'])
for c in ['creature', 'item', 'weapon', 'food', 'effect']:
    n = sum(seeds[o['id']] == c for o in out)
    print(f"{c:9s} {n:6d} sprites, dropped as text {dropped[(c, 'text')]:5d}, font palette {dropped[(c, 'font palette')]:5d}")
random.seed(1)
rows = [('dropped: text', [o for o in out if o['reason'] == 'text']),
        ('dropped: font palette', [o for o in out if o['reason'] == 'font palette']),
        ('kept: weapon seeds', [o for o in out if o['keep'] and seeds[o['id']] == 'weapon']),
        ('kept: item seeds', [o for o in out if o['keep'] and seeds[o['id']] == 'item'])]
sheet = Image.new('RGB', (16 * 66, len(rows) * 80), (40, 40, 40)); d = ImageDraw.Draw(sheet)
for k, (name, os_) in enumerate(rows):
    for j, o in enumerate(random.sample(os_, min(16, len(os_)))):
        sheet.paste(Image.open(f"{D}/render/{o['id']}.png").resize((64, 64), Image.NEAREST), (j * 66 + 1, k * 80 + 14))
    d.text((2, k * 80 + 1), name, fill=(0, 255, 255))
sheet.save(f'{D}/clean_check.png')
