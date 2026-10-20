# Scores every bake-off entrant whose output exists in DATA/bakeoff/out/ against the
# automatic keys, and the TSR set by agreement. Run with C:\dev\igpu-venv (it embeds
# subjects and captions with SigLIP 2's text tower). Writes DATA/bakeoff/report.md.
import json, os, sys, collections, numpy as np, torch
from transformers import AutoModel, AutoProcessor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA

B = f'{DATA}/bakeoff'
MID = os.environ.get('SIGLIP', 'google/siglip2-base-patch16-256')   # so400m-patch14-384: 13x slower here
model = AutoModel.from_pretrained(MID, dtype=torch.float16).to('cuda').eval()
proc = AutoProcessor.from_pretrained(MID)


def feat(o):   # transformers 5 returns an output object; 4.x returned the tensor
    return o if torch.is_tensor(o) else o.pooler_output
CATS = ['character', 'monster', 'item', 'weapon', 'food', 'terrain', 'structure', 'nature', 'effect']


def T(texts):
    out = []
    for i in range(0, len(texts), 64):
        x = proc(text=[f'pixel art sprite of {t}' for t in texts[i:i + 64]], padding='max_length',
                 max_length=64, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out.append(torch.nn.functional.normalize(feat(model.get_text_features(**x)).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def load(name):
    p = f'{B}/out/{name}.jsonl'
    return {d['id']: d for d in map(json.loads, open(p))} if os.path.exists(p) else None


def norm_cat(c):
    c = (c or '').strip().lower()
    return {'boss': 'monster', 'animal': 'monster', 'creature': 'monster', 'tile': 'terrain'}.get(c, c)


keys = {r['id']: r for r in map(json.loads, open(f'{B}/corpus.jsonl')) if r['eval']}
# Grid-sheet cells (DawnLike, Ninja tilesets, DT2 floor/wall frames) are often slices of a larger
# object or scene. A cell whose sprite spans its full width or height touches the cell edges,
# so it is scored as a tile, apart from the whole sprites the labellers are really tested on.
from PIL import Image
for i, r in keys.items():
    a = np.array(Image.open(f'{B}/corpus/{i}.png'))[..., 3]
    ys, xs = np.nonzero(a)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    cell = r['source'] == 'dawnlike' or r['group'].startswith(('ninja:Tileset', 'dt2:wall', 'dt2:floor'))
    r['tile'] = bool(cell and (h >= 16 or w >= 16))
SPLIT = os.environ.get('SPLIT', 'sprites')          # sprites | tiles | all
keys = {i: r for i, r in keys.items() if SPLIT == 'all' or r['tile'] == (SPLIT == 'tiles')}
CAT_DESC = {'character': 'a person or humanoid hero', 'monster': 'a monster, creature or animal',
            'item': 'an item such as a potion, key, coin or armor', 'weapon': 'a weapon such as a sword, axe or bow',
            'food': 'food', 'terrain': 'a ground or floor tile', 'structure': 'a wall, door or building part',
            'nature': 'a tree, bush, plant or rock', 'effect': 'a magic effect, explosion or slash'}
CT = T([CAT_DESC[c] for c in CATS])
vocab = sorted({r['subject'] for r in keys.values() if r['subject']})
VT = T(vocab)


def text_to_cat(texts):
    return [CATS[i] for i in (T(texts) @ CT.T).argmax(1)]


def subject_ranks(ids, texts):
    S = T(texts) @ VT.T
    order = np.argsort(-S, axis=1)
    return [int(np.where(order[k] == vocab.index(keys[i]['subject']))[0][0]) for k, i in enumerate(ids)]


# Balanced accuracy only over categories with enough eval sprites in this split.
SCORED = [c for c in CATS if sum(r['category'] == c for r in keys.values()) >= 20]
entrants = {}   # name -> dict(category={id:..}, facing=.., pose=.., subject_text=.., sec=..)
for name in ['qwen3vl-it_4b-eval', 'gemma3_4b-eval']:
    d = load(name)
    if d:
        entrants[name.replace('-eval', '')] = dict(
            category={i: norm_cat(v['pred'].get('category')) for i, v in d.items()},
            facing={i: v['pred'].get('facing') for i, v in d.items()},
            pose={i: v['pred'].get('pose') for i, v in d.items()},
            subject_text={i: v['pred'].get('subject') or v['pred'].get('caption') or '' for i, v in d.items()},
            sec=float(np.median([v['sec'] for v in d.values()])), n=len(d))
d = load('siglip-eval')
if d:
    entrants['siglip zero-shot'] = dict(category={i: v['zs_category'] for i, v in d.items()},
                                        facing={i: v['zs_facing'] for i, v in d.items()})
    entrants['siglip probe'] = dict(category={i: v['probe_category'] for i, v in d.items() if v['probe_category']},
                                    facing={i: v['probe_facing'] for i, v in d.items() if v['probe_facing']})
for res in ['patch14-384', 'patch16-256']:
    d = load(f'siglip_siglip2-so400m-{res}-eval')
    if d:
        entrants[f'siglip-so400m-{res} zero-shot'] = dict(category={i: v['zs_category'] for i, v in d.items()},
                                                          facing={i: v['zs_facing'] for i, v in d.items()})
d = load('florence-2-large-eval')
if d:
    ids = list(d)
    entrants['florence-2-large'] = dict(category=dict(zip(ids, text_to_cat([d[i]['caption'] for i in ids]))),
                                        subject_text={i: d[i]['caption'] for i in ids},
                                        sec=float(np.median([v['sec'] for v in d.values()])), n=len(d))

lines = [f'# Labeller bake-off ({SPLIT})', '',
         f'Eval set: {len(keys)} pack sprites with automatic keys. Category accuracy is the mean of the '
         f'per-category accuracies over {", ".join(SCORED)} (the categories with >= 20 sprites here). Facing: front/back/left/right, where a course "side" key '
         'accepts left or right. Subject: rank of the true subject among '
         f'{len(vocab)} subject names by SigLIP text similarity to the labeller\'s subject (or caption).', '',
         '| entrant | category (balanced) | course | dawnlike | ninja | dt2 | facing | pose | subject top-1 | top-5 | s/sprite |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
report = {}
for name, e in entrants.items():
    cat = e.get('category', {})
    per_cat = [np.mean([cat.get(i) == r['category'] for i, r in keys.items() if r['category'] == c and i in cat])
               for c in SCORED if any(r['category'] == c and i in cat for i, r in keys.items())]
    per_src = {s: np.mean([cat.get(i) == r['category'] for i, r in keys.items() if r['source'] == s and i in cat])
               if any(r['source'] == s and i in cat for i, r in keys.items()) else float('nan')
               for s in ['course', 'dawnlike', 'ninja', 'dt2']}
    fac = e.get('facing', {})
    fk = [(i, r['facing']) for i, r in keys.items() if r['facing'] and i in fac]
    fac_acc = np.mean([(fac[i] in ('left', 'right')) if k == 'side' else fac[i] == k for i, k in fk]) if fk else float('nan')
    pose = e.get('pose', {})
    pk = [(i, r['pose']) for i, r in keys.items() if r['pose'] and i in pose]
    pose_acc = np.mean([pose[i] == k for i, k in pk]) if pk else float('nan')
    st = e.get('subject_text', {})
    sk = [i for i, r in keys.items() if r['subject'] and i in st]
    ranks = subject_ranks(sk, [st[i] for i in sk]) if sk else []
    top1 = np.mean([r == 0 for r in ranks]) if ranks else float('nan')
    top5 = np.mean([r < 5 for r in ranks]) if ranks else float('nan')
    sec = e.get('sec', float('nan'))
    report[name] = dict(category=float(np.mean(per_cat)), per_source=per_src, facing=fac_acc, pose=pose_acc,
                        subject_top1=top1, subject_top5=top5, sec=sec)
    f = lambda v: '–' if v != v else f'{100 * v:.0f}%'
    lines.append(f'| {name} | {f(np.mean(per_cat))} | ' + ' | '.join(f(per_src[s]) for s in per_src) +
                 f' | {f(fac_acc)} | {f(pose_acc)} | {f(top1)} | {f(top5)} | ' + ('–' if sec != sec else f'{sec:.2f}') + ' |')
    if cat:
        conf = collections.Counter((r['category'], cat[i]) for i, r in keys.items() if i in cat and cat[i] != r['category'])
        report[name]['top_confusions'] = [f'{a}->{b} x{n}' for (a, b), n in conf.most_common(6)]

# TSR agreement: seed vs SigLIP probe vs VLM (hints)
tsr = {r['id']: r for r in map(json.loads, open(f'{B}/tsr.jsonl'))} if os.path.exists(f'{B}/tsr.jsonl') else {}
sig_t = load('siglip-tsr')
lines += ['', '## TSR agreement (no answer key)', '']
for vname in ['qwen3vl-it_4b-tsr-hints', 'gemma3_4b-tsr-hints']:
    v = load(vname)
    if not (v and sig_t):
        continue
    def seedcat(s):
        return {'boss': 'monster', 'tile': 'terrain|structure'}.get(s, s)
    def eq(a, b):
        return a is not None and b is not None and (a == b or (('|' in a) and b in a.split('|')) or (('|' in b) and a in b.split('|')))
    rows = []
    for i, r in tsr.items():
        if i in v and i in sig_t:
            rows.append((seedcat(r['seed']), sig_t[i]['probe_category'], norm_cat(v[i]['pred'].get('category'))))
    seeded = [x for x in rows if x[0] and x[0] != 'ui']
    lines.append(f'**{vname}** ({len(rows)} sprites, {len(seeded)} with a seed): '
                 f'seed=VLM {100 * np.mean([eq(s, m) for s, p, m in seeded]):.0f}%, '
                 f'seed=probe {100 * np.mean([eq(s, p) for s, p, m in seeded]):.0f}%, '
                 f'probe=VLM {100 * np.mean([eq(p, m) for s, p, m in rows]):.0f}%, '
                 f'2-of-3 majority exists {100 * np.mean([eq(s, p) or eq(s, m) or eq(p, m) for s, p, m in seeded]):.0f}%, '
                 f'all three agree {100 * np.mean([eq(s, p) and eq(p, m) for s, p, m in seeded]):.0f}%.')
    dis = collections.Counter((s, m) for s, p, m in seeded if not eq(s, m))
    lines.append(f'Top seed/VLM disagreements: ' + ', '.join(f'{a}->{b} x{n}' for (a, b), n in dis.most_common(6)))
lines += ['', '## Top category confusions (key -> prediction)', '']
for name, rr in report.items():
    if rr.get('top_confusions'):
        lines.append(f'- **{name}**: ' + ', '.join(rr['top_confusions']))
open(f'{B}/report-{SPLIT}.md', 'w').write('\n'.join(lines) + '\n')
json.dump(report, open(f'{B}/report-{SPLIT}.json', 'w'), indent=1, default=float)
print('\n'.join(lines))
