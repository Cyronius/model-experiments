# A label only where seed and so400m agree: two independent sources (plan 1d/1e); igpu-venv.
import json, os, sys, numpy as np
from collections import Counter
os.environ.setdefault('SIGLIP', 'google/siglip2-so400m-patch16-256')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spritekit import DATA
import siglipkit

D = f'{DATA}/selftrain'
SRC = sys.argv[1] if len(sys.argv) > 1 else 'sprites2.jsonl'
CAT_PROMPTS, MERGE = siglipkit.CAT_PROMPTS, siglipkit.MERGE

rs = [r for r in map(json.loads, open(f'{D}/{SRC}')) if r.get('id')]
clean = f'{D}/clean-{SRC}'
if os.path.exists(clean):
    keep = {o['id'] for o in map(json.loads, open(clean)) if o['keep']}
    rs = [r for r in rs if r['id'] in keep]
print(f'{len(rs)} sprites to label with {siglipkit.TAG}', flush=True)
E = siglipkit.embed_cached([r['id'] for r in rs], [f"{D}/render/{r['id']}.png" for r in rs],
                           f'{D}/emb-{siglipkit.TAG}/tsr')
cats = list(CAT_PROMPTS)
T = np.stack([siglipkit.embed_text([f'pixel art sprite of {p}' for p in CAT_PROMPTS[c]]).mean(0) for c in cats])
T /= np.linalg.norm(T, axis=1, keepdims=True)
S = E @ T.T
top = S.argsort(1)[:, ::-1]
out, n = f'{D}/labels-{SRC}', Counter()
with open(out, 'w') as fh:
    for i, r in enumerate(rs):
        zs = MERGE.get(cats[top[i, 0]], cats[top[i, 0]])
        label = r.get('seed') if r.get('seed') == zs else None
        n[(r.get('seed'), zs == r.get('seed'))] += 1
        fh.write(json.dumps(dict(id=r['id'], seed=r.get('seed'), role=r.get('role'), zs=zs,
                                 margin=round(float(S[i, top[i, 0]] - S[i, top[i, 1]]), 4), label=label)) + '\n')
lines = [f'# TSR labels ({siglipkit.TAG}, {SRC})', '', '| seed | sprites | seed = so400m (labelled) |', '|---|---:|---:|']
for s in sorted({k for k, _ in n}, key=str):
    tot = n[(s, True)] + n[(s, False)]
    lines.append(f'| {s} | {tot} | {n[(s, True)]} ({100 * n[(s, True)] / tot:.0f}%) |')
open(f'{D}/report-labels-{SRC.split(".")[0]}.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
