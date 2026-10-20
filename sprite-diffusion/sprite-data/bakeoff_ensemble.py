# Does combining the bake-off labellers beat the best one alone? On the whole-sprite eval
# set, with character and monster merged into "creature", it scores:
#   singles, hard majority votes (ties go to the strongest member), soft fusion of the
#   SigLIP probabilities, agreement filters (accuracy vs how many sprites survive), and the
#   oracle (any member right) as the ceiling.
# Run with C:\dev\igpu-venv after bakeoff_siglip.py (both models) and bakeoff_vlm.py.
import itertools, json, os, sys, numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA

B = f'{DATA}/bakeoff'
CATS = ['character', 'monster', 'item', 'weapon', 'food', 'terrain', 'structure', 'nature', 'effect']
MERGE = {'character': 'creature', 'monster': 'creature', 'boss': 'creature', 'animal': 'creature'}
M = lambda c: MERGE.get((c or '').lower(), (c or '').lower())
CAT_PROMPTS = {
    'character': ['a person', 'a human character', 'a hero', 'a villager'],
    'monster': ['a monster', 'a creature', 'an animal', 'an enemy'],
    'item': ['an item', 'a potion', 'a key', 'a coin', 'a piece of armor', 'a tool'],
    'weapon': ['a weapon', 'a sword', 'an axe', 'a bow', 'a spear'],
    'food': ['food', 'a fruit', 'a piece of meat', 'a vegetable'],
    'terrain': ['a ground tile', 'a floor tile', 'grass', 'water'],
    'structure': ['a wall', 'a door', 'part of a building', 'a fence', 'furniture'],
    'nature': ['a tree', 'a bush', 'a plant', 'a rock'],
    'effect': ['a magic effect', 'an explosion', 'fire', 'a slash effect'],
}
CAT_DESC = {'character': 'a person or humanoid hero', 'monster': 'a monster, creature or animal',
            'item': 'an item such as a potion, key, coin or armor', 'weapon': 'a weapon such as a sword, axe or bow',
            'food': 'food', 'terrain': 'a ground or floor tile', 'structure': 'a wall, door or building part',
            'nature': 'a tree, bush, plant or rock', 'effect': 'a magic effect, explosion or slash'}
corpus = [json.loads(l) for l in open(f'{B}/corpus.jsonl')]
ev = [r for r in corpus if r['eval']]
pos_all = {r['id']: i for i, r in enumerate(corpus)}
pos_ev = {r['id']: i for i, r in enumerate(ev)}


def whole(r):   # same tile rule as bakeoff_score.py
    a = np.array(Image.open(f"{B}/corpus/{r['id']}.png"))[..., 3]
    ys, xs = np.nonzero(a)
    cell = r['source'] == 'dawnlike' or r['group'].startswith(('ninja:Tileset', 'dt2:wall', 'dt2:floor'))
    return not (cell and (ys.max() - ys.min() + 1 >= 16 or xs.max() - xs.min() + 1 >= 16))


keys = {r['id']: M(r['category']) for r in ev if whole(r)}
ids = list(keys)
SCORED = [c for c in sorted(set(keys.values())) if sum(v == c for v in keys.values()) >= 20]


def text_emb(mid, texts):
    model = AutoModel.from_pretrained(mid, dtype=torch.float16).to('cuda').eval(); proc = AutoProcessor.from_pretrained(mid)
    x = proc(text=texts, padding='max_length', max_length=64, truncation=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        o = model.get_text_features(**x)
    o = o if torch.is_tensor(o) else o.pooler_output
    return torch.nn.functional.normalize(o.float(), dim=-1).cpu().numpy()


def zs_probs(mid, E):
    """Zero-shot class probabilities (9 classes) from cached image embeddings."""
    flat = [p for c in CATS for p in CAT_PROMPTS[c]]
    T = text_emb(mid, [f'pixel art sprite of {p}' for p in flat])
    k = 0; Tc = []
    for c in CATS:
        n = len(CAT_PROMPTS[c]); Tc.append(T[k:k + n].mean(0)); k += n
    Tc = np.stack(Tc); Tc /= np.linalg.norm(Tc, axis=1, keepdims=True)
    S = 100 * (E @ Tc.T)
    P = np.exp(S - S.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)


def merged(P):
    """9-class probabilities -> merged-class probabilities, as {class: column}."""
    out = {}
    for j, c in enumerate(CATS):
        out[M(c)] = out.get(M(c), 0) + P[:, j]
    return out


Eb = np.load(f'{B}/siglip_img-siglip2-base-patch16-256.npy')          # whole corpus
Es = np.load(f'{B}/siglip_img-siglip2-so400m-patch14-384.npy')        # eval sprites only
Pb = merged(zs_probs('google/siglip2-base-patch16-256', Eb[[pos_all[i] for i in ids]]))
Ps = merged(zs_probs('google/siglip2-so400m-patch14-384', Es[[pos_ev[i] for i in ids]]))

# Probe probabilities, leave-one-source-out on the base embeddings (as in bakeoff_siglip.py).
Pp9 = np.zeros((len(ids), len(CATS)))
cat_ok = [i for i, r in enumerate(corpus) if r['category'] in CATS]
for src in sorted({r['source'] for r in corpus}):
    tr = [i for i in cat_ok if corpus[i]['source'] != src]
    te = [k for k, i in enumerate(ids) if corpus[pos_all[i]]['source'] == src]
    if not te:
        continue
    Xt = torch.tensor(Eb[tr], device='cuda'); yt = torch.tensor([CATS.index(corpus[i]['category']) for i in tr], device='cuda')
    W = torch.zeros(Eb.shape[1], len(CATS), device='cuda', requires_grad=True); b0 = torch.zeros(len(CATS), device='cuda', requires_grad=True)
    opt = torch.optim.Adam([W, b0], lr=1e-2, weight_decay=1e-3)
    w = torch.bincount(yt, minlength=len(CATS)).float().clamp(min=1); w = (w.sum() / w) / len(CATS)
    for _ in range(300):
        opt.zero_grad(); torch.nn.functional.cross_entropy(Xt @ W * 30 + b0, yt, weight=w).backward(); opt.step()
    with torch.no_grad():
        Z = torch.tensor(Eb[[pos_all[ids[k]] for k in te]], device='cuda')
        Pp9[te] = torch.softmax(Z @ W * 30 + b0, 1).cpu().numpy()
Pp = merged(Pp9)


def argmax(P):
    cls = list(P); A = np.stack([P[c] for c in cls], 1)
    return {i: cls[j] for i, j in zip(ids, A.argmax(1))}


def load(name):
    p = f'{B}/out/{name}.jsonl'
    return {d['id']: d for d in map(json.loads, open(p))}


hard = {'so400m': argmax(Ps), 'base': argmax(Pb), 'probe': argmax(Pp),
        'qwen3vl': {i: M(v['pred'].get('category')) for i, v in load('qwen3vl-it_4b-eval').items() if i in keys}}
fl = load('florence-2-large-eval')
D = text_emb('google/siglip2-base-patch16-256', [f'pixel art sprite of {CAT_DESC[c]}' for c in CATS])
C = text_emb('google/siglip2-base-patch16-256', [f'pixel art sprite of {fl[i]["caption"]}' for i in ids])
hard['florence'] = {i: M(CATS[j]) for i, j in zip(ids, (C @ D.T).argmax(1))}
gem = {i: M(v['pred'].get('category')) for i, v in load('gemma3_4b-eval').items() if i in keys}


def score(pred, subset=None):
    sub = [i for i in (subset or ids) if i in pred]
    per = [np.mean([pred[i] == keys[i] for i in sub if keys[i] == c]) for c in SCORED if any(keys[i] == c for i in sub)]
    return 100 * np.mean(per), 100 * np.mean([pred[i] == keys[i] for i in sub]), len(sub)


def vote(members, order):
    out = {}
    for i in ids:
        vs = [hard[m][i] for m in members if i in hard[m]]
        cnt = {v: vs.count(v) for v in vs}; top = max(cnt.values())
        tied = [v for v in cnt if cnt[v] == top]
        out[i] = next(hard[m][i] for m in order if hard[m].get(i) in tied)
    return out


def soft(*Ps_, w=None):
    w = w or [1] * len(Ps_)
    return argmax({c: sum(wk * P[c] for wk, P in zip(w, Ps_)) for c in Ps_[0]})


rank = ['so400m', 'base', 'probe', 'qwen3vl', 'florence']
L = [f'Whole sprites: {len(ids)}; balanced accuracy over {", ".join(SCORED)} (character+monster = creature).', '',
     '| combination | balanced | plain |', '|---|---:|---:|']
row = lambda n, p: L.append(f'| {n} | {score(p)[0]:.0f}% | {score(p)[1]:.0f}% |')
for m in rank:
    row(m + ' alone', hard[m])
for k in (3, 5):
    for combo in itertools.combinations(rank, k):
        if 'so400m' in combo:
            row('vote: ' + ' + '.join(combo), vote(combo, rank))
row('soft: so400m + base', soft(Ps, Pb))
row('soft: so400m + probe', soft(Ps, Pp))
row('soft: so400m + base + probe', soft(Ps, Pb, Pp))
oracle = {i: keys[i] if any(hard[m].get(i) == keys[i] for m in rank) else hard['so400m'][i] for i in ids}
row('oracle (any of the five right)', oracle)
L += ['', '**Agreement as a filter** (keep a sprite only when the members agree):', '',
      '| members | sprites kept | balanced accuracy on kept |', '|---|---:|---:|']
for combo in [('so400m', 'base'), ('so400m', 'probe'), ('so400m', 'qwen3vl'), ('so400m', 'base', 'probe'),
              ('so400m', 'base', 'qwen3vl'), ('so400m', 'base', 'probe', 'qwen3vl')]:
    agree = {i: hard['so400m'][i] for i in ids if len({hard[m][i] for m in combo}) == 1}
    b_, p_, n = score(agree)
    L.append(f'| {" = ".join(combo)} | {n} ({100 * n / len(ids):.0f}%) | {b_:.0f}% |')
gs = list(gem)
gem_vote = {}
for i in gs:
    vs = [hard['so400m'][i], hard['qwen3vl'][i], gem[i]]
    gem_vote[i] = max(set(vs), key=lambda v: (vs.count(v), v == hard['so400m'][i]))
L += ['', f'gemma3 subset ({len(gs)} sprites): so400m alone {score(hard["so400m"], gs)[0]:.0f}%, gemma3 alone '
      f'{score(gem, gs)[0]:.0f}%, vote so400m+qwen3vl+gemma3 {score(gem_vote, gs)[0]:.0f}%.']
open(f'{B}/report-ensemble.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
