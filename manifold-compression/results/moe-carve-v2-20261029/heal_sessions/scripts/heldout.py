"""Did session 2 learn anything that carries to unseen problems? Teacher-forced fit of T3M and of
a session copy (argv[1], default session 2's decayed_75.pt) to 200 kept answers of shard 1, which no session has trained on:
the stored top-20 KL and the cross-entropy of the teacher's tokens, by source and thinking mode.
Run from manifold-compression."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import torch

import moe_carve as mc
import slim_heal as sh
import slim_batch as sb

mc.DEVICE = 'cuda'
import heal_geometry as hg
import heal_probe as hp
import heal_sessions as hs

N = 200
COPY = sys.argv[1] if len(sys.argv) > 1 else 'decayed_75.pt'
rules = json.loads(Path('heal_sessions_g2_rules.json').read_text())
geometry = json.loads(Path(rules['base']['geometry_rules']).read_text())
folder = hg.install(geometry, rules['base']['run'])
sh.RULES = folder / 'rules.json'
heal_rules, docs = sh.setup()
model, slim, vocab, _ = hp.load_healed(heal_rules, docs)
model.eval()
full, short, kept = sb.load_tokenizers()
tokens = hs.StudentTokens(full, short, kept, len(vocab.index))
pool = hs.Pool(Path(rules['pool']), rules['select'], vocab.eot, tokens)
rows, chosen, teacher = pool.shard(pool.shards.index('shard_001'))
picked = chosen[::max(1, len(chosen) // N)][:N]
items = []
for key in picked:
    r = rows[key]
    prompt = tokens.prompt(r['prompt_ids'])
    answer, a_ids, a_lp = tokens.answer(r['answer_ids'], teacher[f'ids_{key}'][f'ids_{key}'],
                                        teacher[f'logprobs_{key}'][f'logprobs_{key}'].astype(np.float32))
    if len(prompt) + len(answer) + 1 > 4096:
        continue
    items.append((f"{r['source']}_{'on' if r['thinking'] else 'off'}", prompt, answer, a_ids, a_lp))
print(len(items), 'answers', flush=True)

arms = {'t3m': torch.load(folder / 'state.pt', weights_only=False, map_location='cpu'),
        'copy': torch.load(Path(rules['folder']) / COPY, weights_only=False, map_location='cpu')}
out = {}
for name, state in arms.items():
    slim.load({'lora': state['lora'], 'routers': state['routers']})
    slim.on, slim.stash, slim.recall = True, None, None
    t0 = time.time()
    by = defaultdict(lambda: [0., 0., 0])
    per = []
    with torch.no_grad():
        for group, prompt, answer, a_ids, a_lp in items:
            ids = torch.tensor([prompt + answer + [vocab.eot]], device=mc.DEVICE)
            h = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, :-1]
            first, n = len(prompt) - 1, len(answer)
            h, target = h[first:first + n], ids[0, 1:][first:first + n]
            ti = torch.from_numpy(a_ids).to(mc.DEVICE)
            tl = torch.from_numpy(a_lp).to(mc.DEVICE)
            kl = ce = 0.
            for c in range(0, n, 1024):
                k, e = hs.stored_loss(h[c:c + 1024], target[c:c + 1024], ti[c:c + 1024], tl[c:c + 1024],
                                      vocab.head, vocab.index)
                kl, ce = kl + float(k), ce + float(e)
            for g in (group, 'all'):
                by[g][0] += kl
                by[g][1] += ce
                by[g][2] += n
            per.append([kl / n, ce / n, n])
    slim.on = False
    out[name] = {'by': {g: {'kl': v[0] / v[2], 'ce': v[1] / v[2], 'tokens': v[2]} for g, v in sorted(by.items())},
                 'per_answer': per, 'seconds': time.time() - t0}
    print(name, json.dumps({g: round(v['kl'], 4) for g, v in out[name]['by'].items()}), f'{time.time() - t0:.0f} s', flush=True)
    Path(rules['folder'], 'heldout_shard1.json' if COPY == 'decayed_75.pt' else f'heldout_shard1_{COPY[:-3]}.json').write_text(json.dumps(out, indent=1))
a = np.array([p[0] for p in out['t3m']['per_answer']])
b = np.array([p[0] for p in out['copy']['per_answer']])
print(f'per-answer KL, copy - t3m: mean {np.mean(b - a):+.4f}, copy lower on {int((b < a).sum())}/{len(a)}', flush=True)
