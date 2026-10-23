# Relabels all shards from stored embeddings: food = food and drink, plus a so400m-only label where calibrated precise on pack keys; igpu-venv; 2026-10-18.
import glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np
os.environ.setdefault('SIGLIP', 'google/siglip2-so400m-patch16-256')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA
import siglipkit
from harvest_oga import seed as pack_seed, LABELS

D, B = f'{DATA}/selftrain', f'{DATA}/bakeoff'
CUT = f'{DATA}/oga/cut'
MERGE = siglipkit.MERGE
FAMILY = {'item', 'weapon', 'food'}
FOOD_GROUPS = ('dawnlike:Potion', 'dt2:flask', 'ninja:items-Potion')
PRECISION = 0.85
cats = list(siglipkit.CAT_PROMPTS)
T = np.stack([siglipkit.embed_text([f'pixel art sprite of {p}' for p in siglipkit.CAT_PROMPTS[c]]).mean(0) for c in cats])
T /= np.linalg.norm(T, axis=1, keepdims=True)


def predict(E):
    S = E @ T.T
    top = S.argsort(1)[:, ::-1]
    zs = np.array([MERGE.get(cats[i], cats[i]) for i in top[:, 0]])
    return zs, S[np.arange(len(S)), top[:, 0]] - S[np.arange(len(S)), top[:, 1]]


packs = [json.loads(l) for l in open(f'{B}/corpus.jsonl', encoding='utf8')]
key = np.array(['food' if r['group'] in FOOD_GROUPS else {'character': 'creature', 'monster': 'creature', 'tile': 'terrain'}.get(r['category'], r['category'])
                for r in packs])
Ep = siglipkit.embed_cached([r['id'] for r in packs], [f"{B}/render/{r['id']}.png" for r in packs], f'{D}/emb-{siglipkit.TAG}/packs')
zp, mp = predict(Ep)
lines = ['# relabel_v2 calibration on pack keys (so400m zero-shot, food = food and drink)', '',
         '| category | min margin for >=85% precision | pack sprites kept | precision | recall |', '|---|---:|---:|---:|---:|']
thr = {}
for c in sorted(set(key)):
    for m in np.arange(0.0, 0.06, 0.0025):
        sel = (zp == c) & (mp >= m)
        if sel.sum() >= 30 and (key[sel] == c).mean() >= PRECISION:
            thr[c] = float(m)
            lines.append(f'| {c} | {m:.4f} | {int(sel.sum())} | {(key[sel] == c).mean():.2f} | {sel.sum() * (key[sel] == c).mean() / (key == c).sum():.2f} |')
            break
    else:
        lines.append(f'| {c} | none reaches 85% | | | |')
print('\n'.join(lines), flush=True)
open(f'{DATA}/oga/report-calibration.md', 'w').write('\n'.join(lines) + '\n')

n, moved = Counter(), Counter()
for npz in sorted(glob.glob(f'{CUT}/*.npz')):
    if '.emb-' in npz:
        continue
    shard = npz[:-4]
    emb = glob.glob(shard + '.emb-*.npz')
    old = shard + '.labels.jsonl'
    if not emb or not os.path.exists(old):
        continue
    if not os.path.exists(shard + '.labels.v1.jsonl'):
        os.replace(old, shard + '.labels.v1.jsonl')       # keep the agreement-only labels
    prev = {r['id']: r for r in map(json.loads, open(shard + '.labels.v1.jsonl', encoding='utf8'))}
    recs = [json.loads(l) for l in open(shard + '.jsonl', encoding='utf8')]
    zs, mg = predict(np.load(emb[0])['E'].astype(np.float32))
    with open(old + '.tmp', 'w', encoding='utf8') as fh:
        for r, z, m in zip(recs, zs, mg):
            seed = 'food' if r['archive'] == 'itch' else pack_seed({'title': r['title'], 'tags': r['tags']}, r['member'] + ' ' + r['path'])
            tm = prev[r['id']]['text_margin']
            label, src = None, None
            if seed in LABELS and seed == z:
                label, src = seed, 'agree'
            elif z in thr and m >= thr[z] and z in LABELS:
                if seed in FAMILY and z in FAMILY:
                    label, src = z, 'family'
                elif seed is None and tm < 0:
                    label, src = z, 'so400m'
            n[(label, src)] += 1
            fh.write(json.dumps(dict(id=r['id'], seed=seed, zs=str(z), margin=round(float(m), 4), text_margin=tm, label=label, label_src=src)) + '\n')
    os.replace(old + '.tmp', old)
tot = Counter()
for (label, src), v in n.items():
    tot[(label, src)] += v
out = ['', '| category | agree | family | so400m only | total |', '|---|---:|---:|---:|---:|']
for c in sorted(LABELS):
    a, f, s = (tot[(c, k)] for k in ('agree', 'family', 'so400m'))
    out.append(f'| {c} | {a} | {f} | {s} | {a + f + s} |')
open(f'{DATA}/oga/report-relabel-v2.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
