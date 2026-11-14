"""Stage 5 step 1: open the reserved confirmation tasks once, for the frozen final model only.

    python confirm.py RELEASE_FOLDER

Writes qwen-space/stage5/activation.json first (frozen hashes), then result.json.
TS uses the released C runtime (the frozen artifact). SS384 uses the unchanged Q8 student
(windowed at the 504 contract). TT uses the public NQ vectors (API vectors for the 206 NQ documents
longer than the contract) and API vectors for Touché.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / 'stage1'))
from common import WORK, SHARED_WORK, dump, evaluate, features, load_student, paired, prefix, sha256, student_vectors
import shared

R = SHARED_WORK / 'reserved'
W5 = WORK / 'stage5'


def jl(p):
    return [json.loads(l) for l in open(p)]


def task(tag):
    docs = jl(R / f'{tag}-docs.jsonl')
    qs = jl(R / f'{tag}-queries.jsonl')
    qrels = {}
    for line in open(R / f'{tag}-qrels.tsv'):
        q, c, g = line.rstrip('\n').split('\t')
        qrels.setdefault(q, {})[c] = int(g)
    keep = [q for q in qs if any(g > 0 for g in qrels.get(q['_id'], {}).values())]
    order = sorted(range(len(docs)), key=lambda i: docs[i]['_id'])
    keepq = sorted(range(len(keep)), key=lambda i: keep[i]['_id'])
    return {'name': tag, 'cids': [docs[i]['_id'] for i in order], 'texts': [docs[i]['text'] for i in order],
            'qids': [keep[i]['_id'] for i in keepq], 'queries': [keep[i]['text'] for i in keepq],
            'qrels': {q: qrels[q] for q in (keep[i]['_id'] for i in keepq)},
            '_doc_order': order, '_q_ids_all': [q['_id'] for q in qs], 'dropped_queries_without_positive': len(qs) - len(keep)}


def teacher(tag, d):
    if tag == 'touche':
        D = np.load(W5 / 'touche_T_docs_504.npy')[d['_doc_order']]
        allq = np.load(W5 / 'touche_T_queries.npy')
        pos = {q: i for i, q in enumerate(d['_q_ids_all'])}
        Q = allq[[pos[q] for q in d['qids']]]
        return D, Q
    nq = R / 'nq'
    sub = np.load(nq / 'subsample-documents.npy')
    sid = [str(x) for x in np.load(nq / 'subsample-doc-ids.npy')]
    raw_docs = jl(R / 'nq-docs.jsonl')
    long_idx = np.load(W5 / 'nq_long_idx.npy'); long_v = np.load(W5 / 'nq_T_docs_long504.npy')
    long_map = {raw_docs[i]['_id']: v for i, v in zip(long_idx, long_v)}
    pos = {c: i for i, c in enumerate(sid)}
    D = np.stack([long_map[c] if c in long_map else sub[pos[c]] for c in d['cids']]).astype(np.float16)
    Qall = np.load(nq / 'queries.npy')[:, 0]
    qid = [str(x) for x in np.load(nq / 'queries_ids.npy')]
    qp = {q: i for i, q in enumerate(qid)}
    return D, Qall[[qp[q] for q in d['qids']]]


def bm25(d):
    import bm25s, Stemmer
    st = Stemmer.Stemmer('english')
    docs = [shared.cut_to_contract(t, 504) for t in d['texts']]
    r = bm25s.BM25(); r.index(bm25s.tokenize(docs, stopwords='en', stemmer=st, show_progress=False), show_progress=False)
    ids, sc = r.retrieve(bm25s.tokenize(d['queries'], stopwords='en', stemmer=st, show_progress=False), k=1000,
                         show_progress=False)
    S = np.full((len(d['queries']), len(docs)), -1e9, np.float32)
    for i in range(len(S)):
        S[i, ids[i]] = sc[i]
    return S


def mixed(d, q, R1024, P1024, calib, variant):
    s_ref = q @ R1024.T; s_prov = q @ P1024.T
    if variant == 'cal':
        s_prov = calib['a'] * s_prov + calib['b']
    _, rows_ref = evaluate(s_ref, d)
    lookup = {c: i for i, c in enumerate(d['cids'])}
    out = {}
    for p in (0.01, 0.10, 0.50):
        acc = {'ndcg10': [], 'recall_ratio': [], 'top10_share': []}
        for seed in range(5):
            prov = np.zeros(len(d['cids']), bool)
            prov[np.random.default_rng(seed).choice(len(prov), int(round(p * len(prov))), replace=False)] = True
            m, rows = evaluate(np.where(prov[None], s_prov, s_ref), d)
            hm = hr = 0; share = []
            for r, rr in zip(rows, rows_ref):
                rel = [lookup[c] for c, g in d['qrels'][r['query_id']].items() if g > 0 and c in lookup and prov[lookup[c]]]
                tm = {lookup[c] for c in r['top10']}; tr = {lookup[c] for c in rr['top10']}
                hm += sum(i in tm for i in rel); hr += sum(i in tr for i in rel)
                share.append(sum(prov[i] for i in tm) / 10)
            acc['ndcg10'].append(m['ndcg10']); acc['recall_ratio'].append(hm / hr if hr else float('nan'))
            acc['top10_share'].append(float(np.mean(share)))
        out[f'p{int(p * 100)}'] = {k: {'mean': float(np.nanmean(v)), 'per_seed': v} for k, v in acc.items()}
    return out


def main():
    rel = Path(sys.argv[1])
    sys.path.insert(0, str(rel))
    from tde import Embedder
    torch.set_num_threads(12)
    cfg = json.loads((rel / 'config.json').read_text())
    act = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'release': str(rel),
           'model_tde_sha256': sha256(rel / 'model.tde'), 'library_sha256': sha256(rel / 'libtde.so'),
           'head_sha256': cfg.get('head_file_sha256'),
           'nq_ids_sha256': sha256(R / 'nq-subsample-ids.txt'), 'touche_ids_sha256': sha256(R / 'touche-subsample-ids.txt'),
           'note': 'Frozen before computing any score on the reserved tasks. No change to the model after this.'}
    act_path = HERE / 'activation.json'
    if act_path.exists():
        old = json.loads(act_path.read_text())
        assert old['model_tde_sha256'] == act['model_tde_sha256'], 'model changed after activation'
    else:
        dump(act_path, act)
    variant = cfg['calibration']['variant']; calib = cfg['calibration']
    emb = Embedder(rel, workers=8, dim=4096)
    q8 = load_student('q8')
    res = {'activation': act, 'tasks': {}}
    for tag in ('nq', 'touche'):
        d = task(tag)
        T_D, T_Q = teacher(tag, d)
        cut = [shared.cut_to_contract(t, 504) for t in d['texts']]
        t0 = time.time(); P = emb.encode(cut); t_rt = time.time() - t0
        r = {'docs': len(cut), 'queries': len(d['queries']), 'dropped_queries_without_positive': d['dropped_queries_without_positive'],
             'runtime_encode_seconds_8_workers_not_a_benchmark': t_rt}
        rows = {}
        for k in (1024, 4096):
            q = prefix(T_Q, k)
            for name, D in (('TT', T_D), ('TS', P)):
                m, rw = evaluate(q @ prefix(D, k).T, d)
                r[f'{name}_{k}'] = {'ndcg10': m['ndcg10'], 'recall10': m['recall10']}; rows[f'{name}_{k}'] = rw
        F = features(q8, cut, 'windowed')
        Fq = features(q8, d['queries'], 'single')
        m, rows['SS384'] = evaluate(student_vectors(q8, Fq) @ student_vectors(q8, F).T, d)
        r['SS384_q8_504'] = {'ndcg10': m['ndcg10'], 'recall10': m['recall10']}
        m, _ = evaluate(bm25(d), d)
        r['BM25_504'] = {'ndcg10': m['ndcg10'], 'recall10': m['recall10']}
        r['paired_TS_minus_SS384_1024'] = paired(rows['SS384'], rows['TS_1024'])
        cs = (prefix(P, 1024) * prefix(T_D, 1024)).sum(1)
        r['cos_prov_ref_1024'] = {'mean': float(cs.mean()), 'p5': float(np.percentile(cs, 5))}
        r['mixed_index_1024'] = mixed(d, prefix(T_Q, 1024), prefix(T_D, 1024), prefix(P, 1024), calib, variant)
        r['mixed_index_variant'] = variant
        res['tasks'][tag] = r
        dump(HERE / 'result.json', res)
        print(tag, json.dumps({k: v for k, v in r.items() if k != 'mixed_index_1024'}), flush=True)
    res['spend_plan_A_after'] = shared.spend('A')
    dump(HERE / 'result.json', res)


if __name__ == '__main__':
    main()
