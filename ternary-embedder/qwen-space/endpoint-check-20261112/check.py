"""One-off check of the hosted Qwen3-Embedding-8B endpoint against the local
bf16 cache. Sends only public training text (MS MARCO / NLI / Quora rows that
are already cached locally). Never prints the key."""
import hashlib, json, sys, urllib.error, urllib.request
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT = Path('/home/cyrus/code/model-experiments/ternary-embedder')
R2 = ROOT / 'training-v2/work/round2'
KEY = (Path.home() / '.config/deepinfra/api_key').read_text().strip()
URL = 'https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Embedding-8B'
INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'
OUT = Path(sys.argv[1])
spent = {'calls': 0, 'tokens': 0, 'cost': 0.0}


def call(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return None, {'http_error': e.code, 'body': e.read().decode('utf-8', 'replace')[:400].replace(KEY, '<key>')}
    emb = np.asarray(body.pop('embeddings'), dtype=np.float64)
    spent['calls'] += 1
    status = body.get('inference_status') or {}
    spent['tokens'] += int(body.get('input_tokens') or status.get('tokens_input') or 0)
    spent['cost'] += float(status.get('cost') or 0.0)
    return emb, body


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cos(a, b):
    return (unit(a) * unit(b)).sum(-1)


def stats(c):
    return {'min': round(float(c.min()), 5), 'median': round(float(np.median(c)), 5), 'max': round(float(c.max()), 5)}


data = R2 / 'balanced-refined.jsonl'
manifest = json.loads((R2 / 'teacher-cache/manifest.json').read_text())
want = manifest['prompted_metadata']['data_sha256']
if hashlib.sha256(data.read_bytes()).hexdigest() != want:
    sys.exit('data file does not match the cache manifest')
rows = [json.loads(l) for l in data.read_text().splitlines()]
snap = next((ROOT / 'cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots').iterdir())
tk = Tokenizer.from_file(str(snap / 'tokenizer.json'))


def ntok(t):
    return len(tk.encode(t).ids)


# Short texts only, so the cache's 128-token cut cannot be the cause of a difference.
p_idx, seen = [], set()
for i, r in enumerate(rows):
    t = r['positive']
    if t in seen or len(t) < 150 or not 40 <= ntok(t) <= 100:
        continue
    seen.add(t); p_idx.append(i)
    if len(p_idx) == 24:
        break
q_idx, seen = [], set()
for i, r in enumerate(rows):
    if r['query'] in seen or ntok(r['query']) > 40:
        continue
    seen.add(r['query']); q_idx.append(i)
    if len(q_idx) == 16:
        break
passages = [rows[i]['positive'] for i in p_idx]
queries = [rows[i]['query'] for i in q_idx]
P = np.load(R2 / 'teacher-cache/passages.npy', mmap_mode='r')
cached_p = np.asarray(P[p_idx, 0], dtype=np.float64)
cached_qi = np.asarray(np.load(R2 / 'teacher-cache/prompted-queries.npy', mmap_mode='r')[q_idx], dtype=np.float64)
cached_qp = np.asarray(np.load(R2 / 'teacher-cache/plain-queries.npy', mmap_mode='r')[q_idx], dtype=np.float64)

res = {'endpoint': URL, 'passages': len(passages), 'queries': len(queries)}
a, meta = call({'inputs': passages})
if a is None:
    res['error'] = meta
    OUT.write_text(json.dumps(res, indent=2) + '\n'); print(json.dumps(res, indent=2)); sys.exit(1)
res['response_fields_besides_embeddings'] = meta
res['shape'] = list(a.shape)
norms = np.linalg.norm(a, axis=-1)
res['default_norm'] = {'min': round(float(norms.min()), 4), 'max': round(float(norms.max()), 4)}
res['passage_cos_vs_local_bf16_4096'] = stats(cos(a, cached_p))
res['passage_cos_vs_local_bf16_prefix1024'] = stats(cos(a[:, :1024], cached_p[:, :1024]))

a2, _ = call({'inputs': passages})
res['repeat_call_cos'] = stats(cos(a, a2)) if a2 is not None else _
res['repeat_call_max_abs_diff'] = float(np.abs(a - a2).max()) if a2 is not None else None

b, m = call({'inputs': passages, 'normalize': True})
if b is None:
    res['normalize_true'] = m
else:
    nb = np.linalg.norm(b, axis=-1)
    res['normalize_true'] = {'norm_min': round(float(nb.min()), 4), 'norm_max': round(float(nb.max()), 4),
                             'cos_vs_default': stats(cos(a, b))}

c, m = call({'inputs': passages, 'dimensions': 1024})
if c is None:
    res['dimensions_1024'] = m
else:
    nc = np.linalg.norm(c, axis=-1)
    res['dimensions_1024'] = {'shape': list(c.shape), 'norm_min': round(float(nc.min()), 4), 'norm_max': round(float(nc.max()), 4)}
    if c.shape[1] == 1024:
        res['dimensions_1024']['cos_vs_prefix_of_4096'] = stats(cos(c, a[:, :1024]))
        res['dimensions_1024']['max_abs_diff_vs_raw_prefix'] = float(np.abs(c - a[:, :1024]).max())

d, m = call({'inputs': [f'Instruct: {INSTRUCTION}\nQuery:{q}' for q in queries]})
res['query_inline_template_vs_local_prompted'] = stats(cos(d, cached_qi)) if d is not None else m
e, m = call({'inputs': queries, 'custom_instruction': INSTRUCTION})
if e is None:
    res['query_custom_instruction'] = m
else:
    res['query_custom_instruction_vs_local_prompted'] = stats(cos(e, cached_qi))
    res['query_custom_instruction_vs_local_plain'] = stats(cos(e, cached_qp))
    if d is not None:
        res['query_custom_instruction_vs_inline'] = stats(cos(e, d))
f, m = call({'inputs': queries})
if f is None:
    res['query_bare'] = m
else:
    res['query_bare_vs_local_plain'] = stats(cos(f, cached_qp))
    res['query_bare_vs_local_prompted'] = stats(cos(f, cached_qi))
res['spent'] = spent
OUT.write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps(res, indent=2))
