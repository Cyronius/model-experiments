# Assembles the training set (packs + labelled OGA shards + labelled TSR sprites) into one npz; 2026-10-17.
import glob, json, os, sys
from collections import Counter
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sprite-data'))
from spritekit import DATA, trim, footprint, on_canvas
import captions

CATS = ['creature', 'item', 'weapon', 'food', 'effect', 'terrain', 'structure', 'nature']
FOOTPRINTS = ['16x16', '16x32', '32x32']
TEXT_MAX = 0.015          # set by eye on oga/check-text-margin.png (plan 1e)
MERGE = {'character': 'creature', 'monster': 'creature', 'tile': 'terrain'}
FOOD_GROUPS = ('dawnlike:Potion', 'dt2:flask', 'ninja:items-Potion')   # food includes drinks and potions
OUT = f'{DATA}/trainset.npz'

sprites, cat, fp, group, source, caption = [], [], [], [], [], []


def add(a, c, g, src, subject='', style='', tags=()):
    a = trim(a)
    if a is None or footprint(a) is None:
        return
    sprites.append(on_canvas(a)); fp.append(FOOTPRINTS.index(footprint(a)))
    cat.append(CATS.index(c) if c in CATS else -1); group.append(g); source.append(src)
    caption.append(captions.build(c if c in CATS else None, subject, style, tags, captions.colour_words(on_canvas(a))))


for r in map(json.loads, open(f'{DATA}/bakeoff/corpus.jsonl', encoding='utf8')):
    add(np.array(Image.open(f"{DATA}/bakeoff/corpus/{r['id']}.png").convert('RGBA')),
        'food' if r['group'] in FOOD_GROUPS else MERGE.get(r['category'], r['category']), r['group'], 'pack',
        subject=' '.join(captions.tokens(r['group'].split(':', 1)[-1].replace('-', ' '))), style=r['source'])

for lab in sorted(glob.glob(f'{DATA}/oga/cut/*.labels.jsonl')):
    shard = lab[:-len('.labels.jsonl')]
    S = np.load(shard + '.npz')['sprites']
    meta = [json.loads(l) for l in open(shard + '.jsonl', encoding='utf8')]
    for a, m, l in zip(S, meta, map(json.loads, open(lab, encoding='utf8'))):
        if l['text_margin'] > TEXT_MAX or m['seed'] == 'ui':
            continue
        add(a, l['label'], m['pack'], 'oga', subject=captions.subject_from_path(m['path']), tags=captions.tags_from(m['tags']))

tsr_labels = f'{DATA}/selftrain/labels-sprites2.jsonl'
if os.path.exists(tsr_labels):
    meta = {r['id']: r for r in map(json.loads, open(f'{DATA}/selftrain/sprites2.jsonl', encoding='utf8')) if r.get('id')}
    for l in map(json.loads, open(tsr_labels, encoding='utf8')):
        m = meta[l['id']]
        add(np.array(Image.open(f"{DATA}/selftrain/sprites/{l['id']}.png").convert('RGBA')), l['label'],
            m['game'] + '|' + m['platform'], 'tsr', subject=' '.join(captions.tokens(m['title'] or '')),
            style=' '.join(x for x in [captions.PLATFORM.get(m['platform']), captions.ROLE_WORD.get(m.get('role') or '')] if x))

groups = {g: i for i, g in enumerate(sorted(set(group)))}
np.savez_compressed(OUT, sprites=np.stack(sprites), cat=np.array(cat, np.int8), fp=np.array(fp, np.int8),
                    group=np.array([groups[g] for g in group], np.int32), source=np.array(source),
                    caption=np.array(caption), cats=np.array(CATS), footprints=np.array(FOOTPRINTS))
cc = Counter(cat)
print(f'{len(sprites)} sprites in {len(groups)} groups -> {OUT}')
print('sources', dict(Counter(source)))
print('labels', {(CATS[k] if k >= 0 else 'none'): v for k, v in sorted(cc.items())})
