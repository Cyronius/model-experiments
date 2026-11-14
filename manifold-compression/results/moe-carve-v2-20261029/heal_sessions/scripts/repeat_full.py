"""Is the full eval nondeterministic, and only for trained copies? The 116 eval docs five times in
one process, alternating session 2's copy (decayed_75) and T3M. Run from manifold-compression."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch

import moe_carve as mc
import slim_heal as sh
import slim_batch as sb

mc.DEVICE = 'cuda'
import heal_geometry as hg
import heal_probe as hp

rules = json.loads(Path('heal_sessions_g2_rules.json').read_text())
geometry = json.loads(Path(rules['base']['geometry_rules']).read_text())
folder = hg.install(geometry, rules['base']['run'])
sh.RULES = folder / 'rules.json'
heal_rules, docs = sh.setup()
model, slim, vocab, _ = hp.load_healed(heal_rules, docs)
evals = list(sb.docs_of(docs, 'eval'))
arms = {'t3m': torch.load(folder / 'state.pt', weights_only=False, map_location='cpu'),
        'copy75': torch.load(Path(rules['folder']) / 'decayed_75.pt', weights_only=False, map_location='cpu')}
out = []
for repeat, name in enumerate(['copy75', 't3m', 'copy75', 'copy75', 't3m']):
    state = arms[name]
    slim.load({'lora': state['lora'], 'routers': state['routers']})
    t0 = time.time()
    rows = sh.evaluate(model, slim, vocab, evals)
    kl = [round(r['kl']['student'], 3) for r in rows]
    cp = [k for k, r in zip(kl, rows) if r['id'].startswith('copy')]
    row = {'repeat': repeat, 'arm': name, 'mean': sum(kl) / len(kl), 'copy_mean': sum(cp) / len(cp),
           'copy_max': max(cp), 'kl': kl}
    out.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != 'kl'}), f'{time.time() - t0:.0f} s', flush=True)
    Path(rules['folder'], 'repeat_full.json').write_text(json.dumps(out, indent=1))
