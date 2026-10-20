# Bake-off entrant: SigLIP 2 image embeddings, scored two ways on the automatic keys.
#   zero-shot - cosine similarity to text prompts per class
#   probe     - logistic regression on the embeddings, trained only on automatic labels:
#               category: leave-one-source-out (train on the other packs, test on the held-out
#               pack's eval sprites), so the score includes a change of art style;
#               facing: grouped 5-fold by character/monster identity, with horizontally
#               flipped copies added as extra training data (left <-> right).
# Run with C:\dev\igpu-venv. Caches embeddings in DATA/bakeoff/siglip_*.npy and writes
# DATA/bakeoff/out/siglip-eval.jsonl (zero-shot and probe predictions per eval sprite).
import json, time, os, sys, numpy as np, torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA

MID = os.environ.get('SIGLIP', 'google/siglip2-base-patch16-256')   # so400m-patch14-384: 13x slower here
B = f'{DATA}/bakeoff'
dev = 'cuda'
recs = [json.loads(l) for l in open(f'{B}/corpus.jsonl')]
EVAL_ONLY = os.environ.get('EVAL_ONLY') == '1'   # zero-shot on the eval sprites only (for the slow so400m)
if EVAL_ONLY:
    recs = [r for r in recs if r['eval']]
model = AutoModel.from_pretrained(MID, dtype=torch.float16).to(dev).eval()
proc = AutoProcessor.from_pretrained(MID)


def feat(o):   # transformers 5 returns an output object; 4.x returned the tensor
    return o if torch.is_tensor(o) else o.pooler_output


def embed_images(paths, flip=False, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert('RGB') for p in paths[i:i + bs]]
        if flip:
            ims = [ImageOps.mirror(im) for im in ims]
        x = proc(images=ims, return_tensors='pt').to(dev)
        with torch.no_grad():
            e = feat(model.get_image_features(pixel_values=x['pixel_values'].half()))
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def embed_text(texts):
    x = proc(text=texts, padding='max_length', max_length=64, return_tensors='pt').to(dev)
    with torch.no_grad():
        e = feat(model.get_text_features(**x))
    return torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()


paths = [f"{B}/render/{r['id']}.png" for r in recs]
tag = MID.split('/')[-1]
cache = f'{B}/siglip_img-{tag}.npy'
if os.path.exists(cache) and len(np.load(cache)) == len(recs):
    E = np.load(cache)
else:
    t0 = time.time(); E = embed_images(paths); dt = time.time() - t0
    np.save(cache, E)
    print(f'embedded {len(recs)} in {dt:.0f}s = {1000 * dt / len(recs):.1f} ms/sprite', flush=True)
fac_idx = [i for i, r in enumerate(recs) if r['facing'] in ('front', 'back', 'left', 'right')]
if EVAL_ONLY:
    fac_idx = []
fcache = f'{B}/siglip_img_flip-{tag}.npy'
if not fac_idx:
    EF = None
elif os.path.exists(fcache) and len(np.load(fcache)) == len(fac_idx):
    EF = np.load(fcache)
else:
    EF = embed_images([paths[i] for i in fac_idx], flip=True); np.save(fcache, EF)

CATS = ['character', 'monster', 'item', 'weapon', 'food', 'terrain', 'structure', 'nature', 'effect']
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
FAC = ['front', 'back', 'left', 'right']
FAC_PROMPTS = {'front': ['facing the viewer', 'seen from the front'],
               'back': ['facing away from the viewer', 'seen from behind'],
               'left': ['facing left', 'seen from the side, looking left'],
               'right': ['facing right', 'seen from the side, looking right']}


def zero_shot(prompts, classes):
    T = np.stack([embed_text([f'pixel art sprite of {p}' for p in prompts[c]]).mean(0) for c in classes])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return np.array(classes)[(E @ T.T).argmax(1)]


zs_cat = zero_shot(CAT_PROMPTS, CATS)
zs_fac = zero_shot({k: [f'a character {v}' for v in vs] for k, vs in FAC_PROMPTS.items()}, FAC)


def train_lr(X, y, classes, epochs=300, wd=1e-3):
    Xt = torch.tensor(X, device=dev); yt = torch.tensor([classes.index(c) for c in y], device=dev)
    W = torch.zeros(X.shape[1], len(classes), device=dev, requires_grad=True)
    b = torch.zeros(len(classes), device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2, weight_decay=wd)
    w = torch.bincount(yt, minlength=len(classes)).float().clamp(min=1); w = (w.sum() / w) / len(classes)
    for _ in range(epochs):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Xt @ W * 30 + b, yt, weight=w); loss.backward(); opt.step()
    return lambda Z: np.array(classes)[(torch.tensor(Z, device=dev) @ W * 30 + b).argmax(1).cpu().numpy()]


# Category probe, leave-one-source-out.
pr_cat, pr_fac = {}, {}
cat_ok = [i for i, r in enumerate(recs) if r['category'] in CATS]
for src in ([] if EVAL_ONLY else sorted({r['source'] for r in recs})):
    tr = [i for i in cat_ok if recs[i]['source'] != src]
    te = [i for i in cat_ok if recs[i]['source'] == src and recs[i]['eval']]
    if not te:
        continue
    f = train_lr(E[tr], [recs[i]['category'] for i in tr], CATS)
    for i, p in zip(te, f(E[te])):
        pr_cat[i] = p

# Facing probe, grouped 5-fold; flipped copies swap left/right.
swap = {'left': 'right', 'right': 'left', 'front': 'front', 'back': 'back'}
groups = sorted({recs[i]['group'] for i in fac_idx})
rng = np.random.default_rng(0); fold = {g: k for g, k in zip(groups, rng.permutation(len(groups)) % 5)}
fpos = {i: j for j, i in enumerate(fac_idx)}
side_idx = [i for i, r in enumerate(recs) if r['facing'] == 'side' and r['eval']]
for k in range(5 if fac_idx else 0):
    tr = [i for i in fac_idx if fold[recs[i]['group']] != k]
    te = [i for i in fac_idx if fold[recs[i]['group']] == k and recs[i]['eval']]
    X = np.concatenate([E[tr], EF[[fpos[i] for i in tr]]])
    y = [recs[i]['facing'] for i in tr] + [swap[recs[i]['facing']] for i in tr]
    f = train_lr(X, y, FAC)
    for i, p in zip(te, f(E[te])):
        pr_fac[i] = p
    if k == 0 and side_idx:          # course 'side' sprites: none are in the facing training set
        for i, p in zip(side_idx, f(E[side_idx])):
            pr_fac[i] = p

os.makedirs(f'{B}/out', exist_ok=True)
out = f'{B}/out/siglip_{tag}-eval.jsonl' if EVAL_ONLY else f'{B}/out/siglip-eval.jsonl'
with open(out, 'w') as fh:
    for i, r in enumerate(recs):
        if r['eval']:
            fh.write(json.dumps(dict(id=r['id'], zs_category=zs_cat[i], zs_facing=zs_fac[i],
                                     probe_category=pr_cat.get(i), probe_facing=pr_fac.get(i))) + '\n')
print('wrote', out)

# TSR: a category probe trained on every pack sprite, plus zero-shot.
if not EVAL_ONLY and os.path.exists(f'{B}/tsr.jsonl'):
    tsr = [json.loads(l) for l in open(f'{B}/tsr.jsonl')]
    ET = embed_images([f"{B}/render/{r['id']}.png" for r in tsr])
    f = train_lr(E[cat_ok], [recs[i]['category'] for i in cat_ok], CATS)
    T_ = np.stack([embed_text([f'pixel art sprite of {p}' for p in CAT_PROMPTS[c]]).mean(0) for c in CATS])
    T_ /= np.linalg.norm(T_, axis=1, keepdims=True)
    zs = np.array(CATS)[(ET @ T_.T).argmax(1)]
    with open(f'{B}/out/siglip-tsr.jsonl', 'w') as fh:
        for r, p, z in zip(tsr, f(ET), zs):
            fh.write(json.dumps(dict(id=r['id'], probe_category=p, zs_category=z)) + '\n')
    print('wrote', f'{B}/out/siglip-tsr.jsonl')
