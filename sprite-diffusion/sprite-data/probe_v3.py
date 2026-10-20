# Linear probe trained on pack+so400m agreement labels, calibrated on the true-labelled packs, labels the rest; igpu-venv; 2026-10-18.
import glob, json, os, random, sys
from collections import Counter
import numpy as np, torch
from PIL import Image, ImageDraw
os.environ.setdefault('SIGLIP', 'google/siglip2-so400m-patch16-256')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA, render
import siglipkit
from harvest_oga import LABELS

D, B, CUT = f'{DATA}/selftrain', f'{DATA}/bakeoff', f'{DATA}/oga/cut'
CLASSES = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature']
FOOD_GROUPS = ('dawnlike:Potion', 'dt2:flask', 'ninja:items-Potion')
PRECISION, CAP = 0.85, 25000
dev = siglipkit.dev

packs = [json.loads(l) for l in open(f'{B}/corpus.jsonl', encoding='utf8')]
key = np.array(['food' if r['group'] in FOOD_GROUPS else {'character': 'creature', 'monster': 'creature', 'tile': 'terrain'}.get(r['category'], r['category'])
                for r in packs])
Ep = siglipkit.embed_cached([r['id'] for r in packs], [f"{B}/render/{r['id']}.png" for r in packs], f'{D}/emb-{siglipkit.TAG}/packs')

shards, E, rows = [], [], []
for npz in sorted(glob.glob(f'{CUT}/*.npz')):
    if '.emb-' in npz:
        continue
    s = npz[:-4]
    emb, lab = glob.glob(s + '.emb-*.npz'), s + '.labels.jsonl'
    if not emb or not os.path.exists(lab):
        continue
    if not os.path.exists(s + '.labels.v2.jsonl'):
        os.replace(lab, s + '.labels.v2.jsonl')           # keep the relabel_v2 output
    L = [json.loads(l) for l in open(s + '.labels.v2.jsonl', encoding='utf8')]
    shards.append((s, len(L))); E.append(np.load(emb[0])['E']); rows += L
E = np.concatenate(E)
print(f'{len(rows)} sprites, {len(shards)} shards', flush=True)

tr = [i for i, r in enumerate(rows) if r['label_src'] == 'agree' and r['label'] in CLASSES]
rng = random.Random(0)
by = {c: [i for i in tr if rows[i]['label'] == c] for c in CLASSES}
tr = [i for c in CLASSES for i in rng.sample(by[c], min(CAP, len(by[c])))]
Xt = torch.tensor(E[tr].astype(np.float32), device=dev)
yt = torch.tensor([CLASSES.index(rows[i]['label']) for i in tr], device=dev)
W = torch.zeros(Xt.shape[1], len(CLASSES), device=dev, requires_grad=True)
b = torch.zeros(len(CLASSES), device=dev, requires_grad=True)
opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=1e-4)
cw = torch.bincount(yt, minlength=len(CLASSES)).float().clamp(min=1); cw = (cw.sum() / cw) / len(CLASSES)
for _ in range(300):
    opt.zero_grad()
    loss = (torch.nn.functional.cross_entropy(Xt @ W * 30 + b, yt, reduction='none') * cw[yt]).sum() / cw[yt].sum()
    loss.backward(); opt.step()


def probs(X):
    with torch.no_grad():
        return torch.softmax(torch.tensor(X.astype(np.float32), device=dev) @ W * 30 + b, 1).cpu().numpy()


Pp = probs(Ep)
pred, conf = np.array(CLASSES)[Pp.argmax(1)], Pp.max(1)
acc = {c: (pred[key == c] == c).mean() for c in CLASSES}
lines = [f'# probe_v3: linear probe on {len(tr)} agreement labels, tested on {len(packs)} true-labelled pack sprites', '',
         f'Plain per-class accuracy (no threshold): ' + ', '.join(f'{c} {100 * acc[c]:.0f}%' for c in CLASSES),
         f'Balanced: {100 * np.mean(list(acc.values())):.0f}%', '',
         '| category | min prob for >=85% precision | pack sprites kept | precision | recall |', '|---|---:|---:|---:|---:|']
thr = {}
for c in CLASSES:
    for t in np.arange(0.40, 1.0, 0.01):
        sel = (pred == c) & (conf >= t)
        if sel.sum() >= 30 and (key[sel] == c).mean() >= PRECISION:
            thr[c] = float(t)
            lines.append(f'| {c} | {t:.2f} | {int(sel.sum())} | {(key[sel] == c).mean():.2f} | {sel.sum() * (key[sel] == c).mean() / (key == c).sum():.2f} |')
            break
    else:
        lines.append(f'| {c} | none reaches 85% | | | |')
print('\n'.join(lines), flush=True)

P = np.concatenate([probs(E[i:i + 50000]) for i in range(0, len(E), 50000)])
ppred, pconf = np.array(CLASSES)[P.argmax(1)], P.max(1)
n, k, new = Counter(), 0, []
for s, cnt in shards:
    with open(s + '.labels.jsonl', 'w', encoding='utf8') as fh:
        for j in range(cnt):
            r = dict(rows[k + j]); i = k + j
            if r['label'] is None and r['text_margin'] < 0 and r['seed'] != 'ui':
                c = ppred[i]
                if r['seed'] == c and pconf[i] >= 0.5:
                    r['label'], r['label_src'] = c, 'probe-agree'
                elif c in thr and pconf[i] >= thr[c]:
                    r['label'], r['label_src'] = c, 'probe'
                    new.append((s, j, c))
            n[(r['label'], r['label_src'])] += 1
            fh.write(json.dumps(r) + '\n')
    k += cnt
lines += ['', '| category | agree | family | so400m only | probe + seed agree | probe only | total |', '|---|---:|---:|---:|---:|---:|---:|']
for c in CLASSES:
    v = [n[(c, s)] for s in ('agree', 'family', 'so400m', 'probe-agree', 'probe')]
    lines.append(f'| {c} | ' + ' | '.join(map(str, v)) + f' | {sum(v)} |')
open(f'{DATA}/oga/report-probe-v3.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines[-11:]), flush=True)

random.seed(1)
cols = 20
sheet = Image.new('RGB', (cols * 50, len(CLASSES) * 62), (40, 40, 40))
d = ImageDraw.Draw(sheet)
cache = {}
for r_i, c in enumerate(CLASSES):
    pick = random.sample([x for x in new if x[2] == c], min(cols, sum(1 for x in new if x[2] == c)))
    for j, (s, idx, _) in enumerate(pick):
        S = cache.setdefault(s, np.load(s + '.npz')['sprites'])
        sheet.paste(render(S[idx]).resize((48, 48), Image.NEAREST), (j * 50 + 1, r_i * 62 + 12))
    d.text((2, r_i * 62), f'probe-only {c}', fill=(0, 255, 255))
sheet.save(f'{DATA}/oga/check-probe-only.png')
