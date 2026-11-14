"""Stage 1 steps 3-5: dev-task teacher vectors through the API, then TT, SS384 and BM25
baselines at the 128 and 504 contracts. Also caches the student features that
Stage 2 reuses (training-v2/work/qwen-space/dev/)."""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import WORK, dump, evaluate, features, load_student, prefix, student_vectors, Timer
import shared

TASKS = ['SciFact', 'NFCorpus', 'ArguAna']
CONTRACTS = {'128': 126, '504': 504}
DEV = WORK / 'dev'
DEV.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).parent


def cached(path, fn):
    path = Path(path)
    if path.exists():
        return np.load(path)
    x = fn()
    np.save(path, x)
    return x


def docs(d, c):
    return [shared.cut_to_contract(t, CONTRACTS[c]) for t in d['texts']]


def teacher(d):
    """Teacher vectors (float16, as returned) for both contracts and the prompted queries."""
    name = d['name']
    out = {}
    for c in CONTRACTS:
        out[c] = cached(DEV / f'{name}_T_docs_{c}.npy', lambda: shared.embed(docs(d, c), plan='A', progress=True))
    out['q'] = cached(DEV / f'{name}_T_queries.npy',
                      lambda: shared.embed([shared.query_input(q, d['instruction']) for q in d['queries']],
                                           plan='A', progress=True))
    return out


def student_feats(d, precision):
    name = d['name']
    m = None
    out = {}
    for c, mode in [('128', 'single'), ('504', 'windowed')]:
        p = DEV / f'{name}_F_{precision}_{c}_{mode}.npy'
        if not p.exists():
            m = m or load_student(precision)
            with Timer(f'{name} {precision} {c} {mode}'):
                np.save(p, features(m, docs(d, c), mode))
        out[c] = np.load(p)
    q = DEV / f'{name}_F_{precision}_queries.npy'
    if not q.exists():
        m = m or load_student(precision)
        np.save(q, features(m, d['queries'], 'single'))
    out['q'] = np.load(q)
    return out


def bm25_scores(d, c):
    import bm25s, Stemmer
    st = Stemmer.Stemmer('english')
    corpus = bm25s.tokenize(docs(d, c), stopwords='en', stemmer=st, show_progress=False)
    r = bm25s.BM25()
    r.index(corpus, show_progress=False)
    q = bm25s.tokenize(d['queries'], stopwords='en', stemmer=st, show_progress=False)
    # full score matrix
    S = np.zeros((len(d['queries']), len(d['texts'])), np.float32)
    ids, sc = r.retrieve(q, k=min(1000, len(d['texts'])), show_progress=False)
    for i in range(len(S)):
        S[i, ids[i]] = sc[i]
    return S


def main():
    torch.set_num_threads(8)
    res = {'tasks': {}, 'contracts': {'128': '126 content tokens, single window',
                                      '504': '504 content tokens; student windowed 4 x 126'},
           'query_protocol': 'Instruct: {instruction}\\nQuery:{query}', 'instructions': shared.DEV_PROMPTS}
    rows = {}
    for name in TASKS:
        d = shared.dev_task(name)
        T = teacher(d)
        r = res['tasks'][name] = {'docs': len(d['texts']), 'queries': len(d['queries'])}
        lens = np.array([shared.n_content_tokens(t) for t in d['texts']])
        r['doc_content_tokens'] = {'median': float(np.median(lens)), 'p90': float(np.percentile(lens, 90)),
                                   'share_over_126': float((lens > 126).mean()),
                                   'share_over_504': float((lens > 504).mean())}
        for c in CONTRACTS:
            for k in (512, 1024, 2048, 4096):
                m, rw = evaluate(prefix(T['q'], k) @ prefix(T[c], k).T, d)
                r[f'TT_{c}_{k}'] = {'ndcg10': m['ndcg10'], 'recall10': m['recall10']}
                rows[f'{name}|TT_{c}_{k}'] = rw
            m, rw = evaluate(bm25_scores(d, c), d)
            r[f'BM25_{c}'] = {'ndcg10': m['ndcg10'], 'recall10': m['recall10']}
            rows[f'{name}|BM25_{c}'] = rw
        for precision in ('q8', 'float'):
            F = student_feats(d, precision)
            model = load_student(precision)
            qv = student_vectors(model, F['q'])
            for c in CONTRACTS:
                m, rw = evaluate(qv @ student_vectors(model, F[c]).T, d)
                r[f'SS384_{precision}_{c}'] = {'ndcg10': m['ndcg10'], 'recall10': m['recall10']}
                rows[f'{name}|SS384_{precision}_{c}'] = rw
        # sanity: Q8 with the historical pipeline (truncate the raw text at 128 tokens)
        model = load_student('q8')
        raw = features(model, d['texts'], 'single')
        qv = student_vectors(model, features(model, d['queries'], 'single'))
        m, _ = evaluate(qv @ student_vectors(model, raw).T, d)
        r['SS384_q8_raw_truncate128'] = {'ndcg10': m['ndcg10'], 'recall10': m['recall10']}
        print(name, json.dumps(r), flush=True)
        dump(OUT / 'result.json', res)
    core = lambda key: float(np.mean([res['tasks'][t][key]['ndcg10'] for t in ('SciFact', 'NFCorpus')]))
    res['core_mean_ndcg10'] = {k: core(k) for k in res['tasks']['SciFact'] if isinstance(res['tasks']['SciFact'][k], dict)
                               and 'ndcg10' in res['tasks']['SciFact'][k]}
    res['spend_plan_A_after'] = shared.spend('A')
    dump(OUT / 'result.json', res)
    (WORK / 'stage1-rows.json').write_text(json.dumps(rows))
    print(json.dumps(res['core_mean_ndcg10'], indent=1))


if __name__ == '__main__':
    main()
