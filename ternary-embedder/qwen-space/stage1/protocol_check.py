"""Stage 1 steps 1-2: which free sources live in the hosted model's space, and
which query protocol to use. Writes training-v2/work/shared-qwen8b/protocol-check.json.

Run with DEEPINFRA_API_KEY in the environment. Sends only public dataset text.
"""
import glob, hashlib, io, json, re, sys, zipfile
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJECT, SHARED_WORK, dump, unit
import shared
from manifest import record, done

N = 200
rng = np.random.default_rng(0)
OUT = SHARED_WORK / 'protocol-check.json'
res = json.loads(OUT.read_text()) if OUT.exists() else {}
res.update({'date': '2026-11-12', 'endpoint': shared.URL, 'provider': shared.PROVIDER, 'model': shared.MODEL,
            'served_version_label': 'excl-v1 (DeepInfra model page; not a Hugging Face revision)',
            'request': 'inputs only; no dimensions, no normalize, custom_instruction empty unless stated',
            'thresholds': {'accept': 'median >= 0.99 and p5 >= 0.97', 'first_fit_only': '0.95 <= median < 0.99',
                           'reject': 'median < 0.95'}})
res.setdefault('sources', {})


def stats(api, ref):
    out = {}
    for k in (4096, 1024):
        c = (unit(api[:, :k]) * unit(ref[:, :k])).sum(1)
        out[f'cos{k}'] = {'median': round(float(np.median(c)), 5), 'p5': round(float(np.percentile(c, 5)), 5),
                          'min': round(float(c.min()), 5)}
    m, p5 = out['cos4096']['median'], out['cos4096']['p5']
    m1, p51 = out['cos1024']['median'], out['cos1024']['p5']
    if min(m, m1) >= 0.99 and min(p5, p51) >= 0.97:
        out['verdict'] = 'accepted'
    elif min(m, m1) >= 0.95:
        out['verdict'] = 'first-fit only'
    else:
        out['verdict'] = 'rejected'
    out['n'] = len(api)
    return out


def save():
    dump(OUT, res)


def jaswanth():
    files = sorted(glob.glob(str(SHARED_WORK / 'jaswanth/data/*.parquet')))
    picks = rng.choice(len(files), 20, replace=False)
    texts, vecs = [], []
    for fi in picks:
        t = pq.read_table(files[fi])
        rows = rng.choice(t.num_rows, 10, replace=False)
        sub = t.take(rows)
        codes = np.array(sub.column('embedding').to_pylist(), np.float32)
        amax = np.array(sub.column('abs_max_val').to_pylist(), np.float32)
        texts += sub.column('text').to_pylist(); vecs.append(codes * amax[:, None] / 127)
    ref = np.concatenate(vecs)
    api = shared.embed(texts, plan='A').astype(np.float32)
    return stats(api, ref) | {'sample': '20 random files x 10 random rows, seed 0, texts exactly as stored'}


def ryanmarcus():
    from huggingface_hub import HfFileSystem
    REV = 'e39015f788e59a66a03538d3945bac57f976bd61'
    f = HfFileSystem().open(f'datasets/RyanMarcus/qwen-8B-embedding-distill@{REV}/qwen-8b-distill-data.parquet',
                            'rb', block_size=1 << 22)
    p = pq.ParquetFile(f)
    # skip the first 130 row groups (they duplicate Jaswanth); two random later groups, 100 rows each
    groups = sorted(rng.choice(np.arange(130, p.metadata.num_row_groups), 2, replace=False).tolist())
    texts, vecs = [], []
    for g in groups:
        rg = p.read_row_group(g)
        rows = rng.choice(rg.num_rows, 100, replace=False)
        texts += [rg.column('text')[int(i)].as_py() for i in rows]
        emb = rg.column('embedding').to_numpy(zero_copy_only=False)
        vecs.append(np.stack([emb[int(i)] for i in rows]).astype(np.float32))
    ref = np.concatenate(vecs)
    api = shared.embed(texts, plan='A').astype(np.float32)
    return stats(api, ref) | {'sample': f'row groups {groups}, 100 random rows each, seed 0', 'revision': REV,
                              'texts_with_query_prefix': sum(t.startswith('query: ') for t in texts)}


def nq():
    from huggingface_hub import HfFileSystem, hf_hub_download
    from tokenizers import Tokenizer
    REV = '985ee9aebdbb6aa21035b2b93d35f6634d243456'
    REPO = 'robro612/nq_qwen3_embedding_8b'
    d = SHARED_WORK / 'reserved/nq'
    d.mkdir(parents=True, exist_ok=True)
    ids_path = hf_hub_download(REPO, 'doc_ids.npy', repo_type='dataset', revision=REV, local_dir=d)
    doc_ids = np.load(ids_path, allow_pickle=False)
    pick = np.sort(rng.choice(len(doc_ids), N, replace=False))
    f = HfFileSystem().open(f'datasets/{REPO}@{REV}/documents.npy', 'rb', block_size=1 << 16)
    ver = np.lib.format.read_magic(f)
    shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
    off = f.tell()
    assert shape == (len(doc_ids), 4096) and dtype == np.dtype('<f2') and not fortran
    ref = []
    for i in pick:
        f.seek(off + int(i) * 4096 * 2)
        ref.append(np.frombuffer(f.read(8192), '<f2').astype(np.float32))
    ref = np.stack(ref)
    want = {str(doc_ids[i]) for i in pick}
    texts = {}
    with zipfile.ZipFile(SHARED_WORK / 'reserved/nq.zip') as z:
        with z.open('nq/corpus.jsonl') as fh:
            for line in io.TextIOWrapper(fh, encoding='utf-8'):
                r = json.loads(line)
                if r['_id'] in want:
                    texts[r['_id']] = (r.get('title', '') + ' ' + r['text']).strip()
    qtok = Tokenizer.from_file(str(next((PROJECT / 'cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots').iterdir())
                                    / 'tokenizer.json'))
    inputs, cut = [], 0
    for i in pick:
        t = texts[str(doc_ids[i])]
        e = qtok.encode(t, add_special_tokens=False)
        if len(e.ids) > 511:  # 512 incl. the end-of-text token
            t = t[:e.offsets[511][0]]; cut += 1
        inputs.append(t)
    api = shared.embed(inputs, plan='A').astype(np.float32)
    return stats(api, ref) | {'sample': f'{N} random documents (seed 0) by row of doc_ids.npy', 'revision': REV,
                              'documents_cut_to_512_qwen_tokens': cut}


def round2():
    R2 = PROJECT / 'training-v2/work/round2'
    data = R2 / 'balanced-refined.jsonl'
    man = json.loads((R2 / 'teacher-cache/manifest.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == man['prompted_metadata']['data_sha256']
    rows = [json.loads(l) for l in data.read_text().splitlines()]
    from tokenizers import Tokenizer
    qtok = Tokenizer.from_file(str(next((PROJECT / 'cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots').iterdir())
                                    / 'tokenizer.json'))
    P = np.load(R2 / 'teacher-cache/passages.npy', mmap_mode='r')
    # passages (positive, column 0) under 128 Qwen tokens; first occurrence of each text
    order = rng.permutation(len(rows))
    p_idx, seen = [], set()
    for i in order:
        t = rows[i]['positive']
        if t in seen or len(qtok.encode(t).ids) >= 127:
            continue
        seen.add(t); p_idx.append(int(i))
        if len(p_idx) == N:
            break
    ptexts = [rows[i]['positive'] for i in p_idx]
    pref = np.asarray(P[p_idx, 0], np.float32)
    papi = shared.embed(ptexts, plan='A').astype(np.float32)
    out = stats(papi, pref) | {'sample': f'{N} positive passages under 127 Qwen tokens, seed 0',
                               'passages_shape': list(P.shape), 'texts_sha256': man['prompted_metadata']['data_sha256']}
    # queries: step 2
    instr = man['prompted_metadata']['instruction']
    q_idx, seen = [], set()
    for i in order:
        q = rows[i]['query']
        if q in seen or len(qtok.encode(shared.query_input(q, instr)).ids) >= 127:
            continue
        seen.add(q); q_idx.append(int(i))
        if len(q_idx) == N:
            break
    qs = [rows[i]['query'] for i in q_idx]
    qref = np.asarray(np.load(R2 / 'teacher-cache/prompted-queries.npy', mmap_mode='r')[q_idx], np.float32)
    full = shared.embed([shared.query_input(q, instr) for q in qs], plan='A').astype(np.float32)
    field = shared.embed(qs, instruction=instr, plan='A').astype(np.float32)
    qp = {'instruction': instr, 'n': N,
          'full_string_vs_prompted_cache': stats(full, qref),
          'custom_instruction_vs_prompted_cache': stats(field, qref)}
    a = qp['full_string_vs_prompted_cache']['cos4096']['median']
    b = qp['custom_instruction_vs_prompted_cache']['cos4096']['median']
    qp['chosen'] = 'full string Instruct: {instruction}\\nQuery:{query}, custom_instruction empty' if a >= b or max(a, b) < 0.99 \
        else 'custom_instruction field'
    return out, qp


if __name__ == '__main__':
    which = sys.argv[1:] or ['jaswanth', 'ryanmarcus', 'nq', 'round2']
    for w in which:
        if w == 'round2':
            s, qp = round2()
            res['sources']['round2-teacher-cache'] = s
            res['query_protocol'] = qp
        else:
            res['sources'][w] = globals()[w]()
        res['spend_plan_A_after'] = shared.spend('A')
        save()
        print(w, json.dumps(res['sources'].get(w) or res['sources']['round2-teacher-cache']), flush=True)
    print(json.dumps(res, indent=2))
