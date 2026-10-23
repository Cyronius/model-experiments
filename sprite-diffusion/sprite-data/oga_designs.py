# Distinct-design counts and threshold contact sheets for the labelled OGA shards (plan 1e); 2026-10-17.
import glob, json, os, random, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA, render

O = f'{DATA}/oga'
CUT = f'{O}/cut'
THRESH = [0.90, 0.93, 0.95, 0.97]


def swap_key(a):
    # Same alpha mask and same brightness ordering = a palette swap of one design.
    m = a[..., 3] > 0
    lum = (a[..., :3].astype(np.int32) @ np.array([299, 587, 114])) // 1000
    ranks = np.unique(lum[m], return_inverse=True)[1]
    return m.tobytes() + ranks.astype(np.uint8).tobytes()


S, R, E, L = [], [], [], []
for lab in sorted(glob.glob(f'{CUT}/*.labels.jsonl')):
    shard = lab[:-len('.labels.jsonl')]
    emb = glob.glob(shard + '.emb-*.npz')
    if not emb:
        continue
    S.append(np.load(shard + '.npz')['sprites'])
    R += [json.loads(l) for l in open(shard + '.jsonl', encoding='utf8')]
    L += [json.loads(l) for l in open(lab, encoding='utf8')]
    E.append(np.load(emb[0])['E'].astype(np.float32))
S, E = np.concatenate(S), np.concatenate(E).astype(np.float16)   # float16 here, float32 per pack below
n = len(R)
swap_rep = {}
swap_groups = np.array([swap_rep.setdefault(swap_key(a), i) for i, a in enumerate(S)])

by_pack = defaultdict(list)
for i, r in enumerate(R):
    by_pack[r['pack']].append(i)
design = {t: np.arange(n) for t in THRESH}
for idx in by_pack.values():
    Ef = E[idx].astype(np.float32)
    sims = Ef @ Ef.T
    for t in THRESH:
        reps = []
        for k in range(len(idx)):
            hit = next((r for r in reps if sims[k, r] >= t), None) if reps else None
            if hit is None:
                reps.append(k)
            else:
                design[t][idx[k]] = idx[hit]

lines = [f'# OGA distinct designs (partial: {n} labelled sprites, {len(by_pack)} packs)', '',
         f'- palette-swap groups: {len(set(swap_groups))} ({100 * len(set(swap_groups)) / n:.0f}% of sprites)']
for t in THRESH:
    lines.append(f'- near-duplicate merge within pack at cosine >= {t}: {len(set(design[t]))} designs '
                 f'({100 * len(set(design[t])) / n:.0f}%)')
lines += ['', '| category (label) | sprites | designs @0.90 | @0.93 | @0.95 | @0.97 |', '|---|---:|---:|---:|---:|---:|']
labels = [l['label'] for l in L]
for c in sorted({x for x in labels if x}, key=lambda c: -labels.count(c)) + [None]:
    ix = [i for i in range(n) if labels[i] == c]
    lines.append(f'| {c or "unlabelled"} | {len(ix)} | ' + ' | '.join(str(len({design[t][i] for i in ix})) for t in THRESH) + ' |')
open(f'{O}/report-designs.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# Pairs straddling each threshold, to pick one by eye.
random.seed(0)
bins = [(0.88, 0.90), (0.90, 0.93), (0.93, 0.95), (0.95, 0.97), (0.97, 1.01)]
found = {b: [] for b in bins}
packs = list(by_pack.values())
random.shuffle(packs)
for idx in packs:
    if len(idx) < 2:
        continue
    sub = random.sample(idx, min(40, len(idx)))
    sims = E[sub].astype(np.float32) @ E[sub].astype(np.float32).T
    for a in range(len(sub)):
        for b in range(a + 1, len(sub)):
            for lo, hi in bins:
                if lo <= sims[a, b] < hi and len(found[(lo, hi)]) < 12:
                    found[(lo, hi)].append((sub[a], sub[b]))
    if all(len(v) >= 12 for v in found.values()):
        break
sheet = Image.new('RGB', (12 * 106, len(bins) * 66), (40, 40, 40))
d = ImageDraw.Draw(sheet)
for k, b in enumerate(bins):
    for j, (x, y) in enumerate(found[b]):
        sheet.paste(render(S[x]).resize((48, 48), Image.NEAREST), (j * 106 + 2, k * 66 + 14))
        sheet.paste(render(S[y]).resize((48, 48), Image.NEAREST), (j * 106 + 52, k * 66 + 14))
    d.text((2, k * 66 + 1), f'cosine {b[0]:.2f}-{min(b[1], 1):.2f}', fill=(0, 255, 255))
sheet.save(f'{O}/check-design-pairs.png')

# Sprites by text margin, to set the text threshold.
tm = np.array([l['text_margin'] for l in L])
tbins = [(-1, -0.02), (-0.02, 0), (0, 0.015), (0.015, 0.03), (0.03, 0.05), (0.05, 1)]
sheet = Image.new('RGB', (20 * 50, len(tbins) * 62), (40, 40, 40))
d = ImageDraw.Draw(sheet)
for k, (lo, hi) in enumerate(tbins):
    ix = np.nonzero((tm >= lo) & (tm < hi))[0]
    for j, i in enumerate(random.sample(list(ix), min(20, len(ix)))):
        sheet.paste(render(S[i]).resize((48, 48), Image.NEAREST), (j * 50 + 1, k * 62 + 12))
    d.text((2, k * 62), f'text margin {lo}..{hi}: {len(ix)} sprites ({100 * len(ix) / n:.1f}%)', fill=(0, 255, 255))
sheet.save(f'{O}/check-text-margin.png')
