"""Teacher vectors for the confirmation tasks through the shared API cache (no scoring here).
Touché: all subsample docs under the 504 contract + 48 queries (generic web-search instruction).
NQ: only documents longer than the 504 contract (cut text); the rest use the robro612 vectors."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, SHARED_WORK
import shared
R = SHARED_WORK / 'reserved'
OUT = WORK / 'stage5'; OUT.mkdir(parents=True, exist_ok=True)
GENERIC = 'Given a web search query, retrieve relevant passages that answer the query'
def jl(p): return [json.loads(l) for l in open(p)]
docs = jl(R / 'touche-docs.jsonl')
cut = [shared.cut_to_contract(d['text'], 504) for d in docs]
est = sum(len(t) for t in cut) / 4.2
print('touche docs', len(cut), 'approx tokens', int(est), 'approx $', est * 1e-8, flush=True)
if shared.spend('A')['dollars'] + est * 1e-8 > 4.0:
    sys.exit('would approach the cap; stop')
np.save(OUT / 'touche_T_docs_504.npy', shared.embed(cut, plan='A', progress=True))
qs = jl(R / 'touche-queries.jsonl')
np.save(OUT / 'touche_T_queries.npy', shared.embed([shared.query_input(q['text'], GENERIC) for q in qs], plan='A'))
nqd = jl(R / 'nq-docs.jsonl')
long_idx = [i for i, d in enumerate(nqd) if shared.n_content_tokens(d['text']) > 504]
print('nq docs over 504:', len(long_idx), flush=True)
if long_idx:
    v = shared.embed([shared.cut_to_contract(nqd[i]['text'], 504) for i in long_idx], plan='A')
    np.save(OUT / 'nq_T_docs_long504.npy', v); np.save(OUT / 'nq_long_idx.npy', np.array(long_idx))
print(json.dumps(shared.spend('A')))
