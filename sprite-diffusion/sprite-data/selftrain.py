# Self-training experiment: does adding seed-labelled TSR sprites (sheet section/title
# keywords) to the SigLIP category probe close the gap to the 85% gate?
#   A  packs only, leave-one-pack-out              (the bake-off baseline)
#   B  packs + TSR seeds, leave-one-pack-out       (the question)
#   C  TSR seeds only, tested on every pack        (pure transfer from ripped art)
#   D  TSR held-out games: packs + other games' seeds -> agreement with held-out seeds
# A-C score the bake-off's whole pack sprites (true keys, character+monster = creature).
# Embeddings are cached in chunks under DATA/selftrain/emb-<model>/, so pausing (stop the
# process) and re-running the same command picks up where it stopped.
# Run with C:\dev\igpu-venv. SIGLIP=google/siglip2-so400m-patch14-384 for the slow model;
# SPRITES=<file> picks the cut (default sprites2.jsonl, filtered by clean-<file> if present).
import json, os, sys, numpy as np, torch
from collections import Counter, defaultdict
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA
import siglipkit

B, D = f'{DATA}/bakeoff', f'{DATA}/selftrain'
TAG = siglipkit.TAG
SPRITES = os.environ.get('SPRITES', 'sprites2.jsonl')   # the re-cut with coloured boxes stripped
MERGE = {'character': 'creature', 'monster': 'creature'}
CLASSES = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature']
dev = siglipkit.dev


def train_lr(X, y, w_sample=None, epochs=300):
    Xt = torch.tensor(X, device=dev); yt = torch.tensor([CLASSES.index(c) for c in y], device=dev)
    W = torch.zeros(X.shape[1], len(CLASSES), device=dev, requires_grad=True)
    b = torch.zeros(len(CLASSES), device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=1e-3)
    cw = torch.bincount(yt, minlength=len(CLASSES)).float().clamp(min=1); cw = (cw.sum() / cw) / len(CLASSES)
    ws = torch.ones(len(y), device=dev) if w_sample is None else torch.tensor(w_sample, device=dev, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(Xt @ W * 30 + b, yt, reduction='none') * cw[yt] * ws).sum() / (cw[yt] * ws).sum()
        loss.backward(); opt.step()
    return lambda Z: np.array(CLASSES)[(torch.tensor(Z, device=dev) @ W * 30 + b).argmax(1).cpu().numpy()]


# --- data ---------------------------------------------------------------------------------
packs = [json.loads(l) for l in open(f'{B}/corpus.jsonl')]
for r in packs:
    r['cat'] = MERGE.get(r['category'], r['category'])
Ep = siglipkit.embed_cached([r['id'] for r in packs], [f"{B}/render/{r['id']}.png" for r in packs], f'{D}/emb-{TAG}/packs')


def whole(r):   # same tile rule as bakeoff_score.py
    a = np.array(Image.open(f"{B}/corpus/{r['id']}.png"))[..., 3]
    ys, xs = np.nonzero(a)
    cell = r['source'] == 'dawnlike' or r['group'].startswith(('ninja:Tileset', 'dt2:wall', 'dt2:floor'))
    return not (cell and (ys.max() - ys.min() + 1 >= 16 or xs.max() - xs.min() + 1 >= 16))


test = [k for k, r in enumerate(packs) if r['eval'] and whole(r)]
SCORED = [c for c in CLASSES if sum(packs[k]['cat'] == c for k in test) >= 20]
tsr = [r for r in map(json.loads, open(f'{D}/{SPRITES}')) if r.get('id') and r.get('seed')]
if os.path.exists(f'{D}/clean-{SPRITES}'):                 # drop text the cutter kept (selftrain_clean.py)
    keep = {o['id'] for o in map(json.loads, open(f'{D}/clean-{SPRITES}')) if o['keep']}
    tsr = [r for r in tsr if r['id'] in keep]
Et = siglipkit.embed_cached([r['id'] for r in tsr], [f"{D}/render/{r['id']}.png" for r in tsr], f'{D}/emb-{TAG}/tsr')
games = sorted({r['game'] + '|' + r['platform'] for r in tsr})
# Each game counts the same in training, however many sprites it has.
per_game = Counter(r['game'] + '|' + r['platform'] for r in tsr)
w_tsr = np.array([1000.0 / per_game[r['game'] + '|' + r['platform']] for r in tsr])
w_tsr *= len(tsr) / w_tsr.sum()


def balanced(pred, idx):
    return 100 * np.mean([np.mean([pred[j] == packs[k]['cat'] for j, k in enumerate(idx) if packs[k]['cat'] == c])
                          for c in SCORED if any(packs[k]['cat'] == c for k in idx)])


lines = [f'# Self-training ({TAG}, {SPRITES}{" + clean" if os.path.exists(f"{D}/clean-{SPRITES}") else ""})', '',
         f'TSR seeded sprites: {len(tsr)} from {len(games)} games; seeds {dict(Counter(r["seed"] for r in tsr))}.',
         f'Pack test set: {len(test)} whole sprites; balanced accuracy over {", ".join(SCORED)}.', '',
         '| probe training data | balanced accuracy on pack keys |', '|---|---:|']
for label, use_tsr, use_packs in [('A: packs only (leave-one-pack-out)', False, True),
                                  ('B: packs + TSR seeds (leave-one-pack-out)', True, True),
                                  ('C: TSR seeds only', True, False)]:
    pred = {}
    for src in sorted({packs[k]['source'] for k in test}):
        te = [k for k in test if packs[k]['source'] == src]
        Xs, ys, ws = [], [], []
        if use_packs:
            tr = [k for k, r in enumerate(packs) if r['source'] != src]
            Xs.append(Ep[tr]); ys += [packs[k]['cat'] for k in tr]; ws.append(np.ones(len(tr)))
        if use_tsr:
            Xs.append(Et); ys += [r['seed'] for r in tsr]; ws.append(w_tsr)
        f = train_lr(np.concatenate(Xs), ys, np.concatenate(ws))
        for k, p in zip(te, f(Ep[te])):
            pred[k] = p
    idx = [k for k in test if k in pred]
    lines.append(f'| {label} | {balanced([pred[k] for k in idx], idx):.0f}% |')
    per = {c: 100 * np.mean([pred[k] == c for k in idx if packs[k]['cat'] == c]) for c in SCORED}
    lines.append(f'|   per class | ' + ', '.join(f'{c} {v:.0f}%' for c, v in per.items()) + ' |')

# D: held-out games, 5 folds by game.
rng = np.random.default_rng(0); fold = {g: k for g, k in zip(games, rng.permutation(len(games)) % 5)}
gk = [fold[r['game'] + '|' + r['platform']] for r in tsr]
agree = defaultdict(list)
for k in range(5):
    tr = [i for i in range(len(tsr)) if gk[i] != k]; te = [i for i in range(len(tsr)) if gk[i] == k]
    f = train_lr(np.concatenate([Ep, Et[tr]]), [r['cat'] for r in packs] + [tsr[i]['seed'] for i in tr],
                 np.concatenate([np.ones(len(packs)), w_tsr[tr]]))
    for i, p in zip(te, f(Et[te])):
        agree[tsr[i]['seed']].append(p == tsr[i]['seed'])
lines += ['', 'D: held-out games (5-fold by game), probe agreement with the seeds of games it never saw: ' +
          ', '.join(f'{c} {100 * np.mean(v):.0f}% (n={len(v)})' for c, v in sorted(agree.items())) +
          f'; balanced {np.mean([100 * np.mean(v) for v in agree.values()]):.0f}%.']
open(f'{D}/report-selftrain-{TAG}-{SPRITES.split(".")[0]}.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
