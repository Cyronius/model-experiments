"""Review probe (no training, no API calls): is the G2 failure a score-scale
problem, and does a vector-side correction of the released model's output fix it?

Corrections are fit ONLY on the Jaswanth validation split (public, never trained
on). The dev tasks are used to read the effect, as in Stage 2. Reserved tasks
are not touched."""
import glob, json, re, sys, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PROJECT = Path('/home/cyrus/code/model-experiments/ternary-embedder')
sys.path.insert(0, str(PROJECT / 'qwen-space'))
from common import WORK, SHARED_WORK, prefix, unit, evaluate  # noqa: E402
import shared  # noqa: E402

REL = '/home/cyrus/models/doc-embed-3l-qwen8b'
sys.path.insert(0, REL)
from tde import Embedder  # noqa: E402

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
K = 1024
TASKS = ['SciFact', 'NFCorpus', 'ArguAna']; CORE = TASKS[:2]
emb = Embedder(REL, workers=6)


def cached(name, fn):
    p = OUT / f'{name}.npy'
    if not p.exists():
        t = time.perf_counter(); np.save(p, fn()); print(f'[{name}] {time.perf_counter() - t:.0f}s', flush=True)
    return np.load(p)


# ---------------------------------------------------------------- fit data: Jaswanth validation split
def key(p):
    m = re.search(r'train_part_([\d_]+)-(\d+)-of', p)
    return tuple(int(x) for x in m[1].split('_')), int(m[2])


def val_pairs():
    idx = np.load(WORK / 'jaswanth-index.npz')
    val = np.load(WORK / 'stage2/data.npz')['val']
    assert idx['val'][val].all()
    files = sorted(glob.glob(str(SHARED_WORK / 'jaswanth/data/*.parquet')), key=key)
    assert [Path(f).name for f in files] == list(idx['files'])
    texts, vecs = [], []
    for fi in np.unique(idx['fileidx'][val]):
        rows = idx['row'][val[idx['fileidx'][val] == fi]]
        t = pq.read_table(files[fi]).take(rows)
        codes = np.array(t.column('embedding').to_pylist(), np.float32)
        amax = np.array(t.column('abs_max_val').to_pylist(), np.float32)
        vecs.append(prefix(codes * amax[:, None] / 127, K)); texts += t.column('text').to_pylist()
    return texts, np.concatenate(vecs)


if not (OUT / 'val_T.npy').exists():
    vt, vT = val_pairs()
    np.save(OUT / 'val_T.npy', vT); (OUT / 'val_texts.json').write_text(json.dumps(vt))
vT = np.load(OUT / 'val_T.npy'); vt = json.loads((OUT / 'val_texts.json').read_text())
vP = cached('val_P', lambda: emb.encode(vt, dim=K))
print('validation pairs', vP.shape, 'mean cos prov/ref', round(float((vP * vT).sum(1).mean()), 4), flush=True)

# ---------------------------------------------------------------- corrections (fit on validation only)
mu_t, mu_p = vT.mean(0), vP.mean(0)
Ct = np.cov((vT - mu_t).T); Cp = np.cov((vP - mu_p).T)
wt, Ut = np.linalg.eigh(Ct)
sd_t = np.sqrt(np.maximum(wt, 1e-12))
sd_p = np.sqrt(np.maximum(np.einsum('ij,jk,ki->i', Ut.T, Cp, Ut), 1e-12))
print('mean-vector norm: refined %.4f  provisional %.4f' % (np.linalg.norm(mu_t), np.linalg.norm(mu_p)))
print('total variance:   refined %.4f  provisional %.4f' % (wt.sum(), np.trace(Cp)))
order = np.argsort(-wt)
for lo, hi in ((0, 16), (16, 64), (64, 256), (256, 1024)):
    sel = order[lo:hi]
    print('  teacher PCA components %4d-%4d: provisional/refined std ratio %.3f' % (lo, hi, float((sd_p[sel] / sd_t[sel]).mean())))


def diag(gamma):
    g = (sd_t / sd_p) ** gamma
    return lambda P: unit(mu_t + ((P - mu_p) @ Ut * g) @ Ut.T)


def coral(shrink=0.1):
    """Whitening-colouring: full covariance matching, with shrinkage toward the identity scale."""
    wp, Up = np.linalg.eigh(Cp)
    wp = (1 - shrink) * wp + shrink * wp.mean()
    A = Up / np.sqrt(wp) @ Up.T
    B = Ut * np.sqrt(np.maximum(wt, 0)) @ Ut.T
    M = A @ B
    return lambda P: unit(mu_t + (P - mu_p) @ M)


FIX = {'released (no correction)': lambda P: P,
       'mean matched only': diag(0.0),
       'per-component spread, half': diag(0.5),
       'per-component spread, full': diag(1.0),
       'full covariance matched': coral()}

# ---------------------------------------------------------------- dev tasks
DATA = {n: shared.dev_task(n) for n in TASKS}
TQ = {n: prefix(np.load(WORK / f'dev/{n}_T_queries.npy'), K) for n in TASKS}
TD = {n: prefix(np.load(WORK / f'dev/{n}_T_docs_504.npy'), K) for n in TASKS}
PD = {n: cached(f'{n}_P', lambda n=n: emb.encode(DATA[n]['texts'], dim=K)) for n in TASKS}


def mixed10(n, V, seeds=5, p=0.10):
    d = DATA[n]; q = TQ[n]
    s_ref = q @ TD[n].T; s_prov = q @ V.T
    _, rows_ref = evaluate(s_ref, d)
    lookup = {c: i for i, c in enumerate(d['cids'])}
    nd, rr_, sh = [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        prov = np.zeros(len(d['cids']), bool)
        prov[rng.choice(len(prov), int(round(p * len(prov))), replace=False)] = True
        m, rows = evaluate(np.where(prov[None, :], s_prov, s_ref), d)
        hit_m = hit_r = 0; share = []
        for r, rr in zip(rows, rows_ref):
            rel = [lookup[c] for c, g in d['qrels'][r['query_id']].items() if g > 0 and c in lookup and prov[lookup[c]]]
            tm = {lookup[c] for c in r['top10']}; tr = {lookup[c] for c in rr['top10']}
            hit_m += sum(i in tm for i in rel); hit_r += sum(i in tr for i in rel)
            share.append(sum(prov[i] for i in tm) / 10)
        nd.append(m['ndcg10']); rr_.append(hit_m / hit_r if hit_r else np.nan); sh.append(np.mean(share))
    return float(np.mean(nd)), float(np.nanmean(rr_)), float(np.mean(sh))


def score_stats(n, V):
    d = DATA[n]; q = TQ[n]
    lookup = {c: i for i, c in enumerate(d['cids'])}; qi = {x: i for i, x in enumerate(d['qids'])}
    S = q @ V.T
    rel = [S[qi[k], lookup[c]] for k, v in d['qrels'].items() if k in qi for c, g in v.items() if g > 0 and c in lookup]
    return float(S.mean()), float(np.quantile(S, 0.999)), float(np.mean(rel))


res = {}
print('\nscores of hosted queries against documents (mean / 99.9th pct of all pairs / mean of relevant pairs):')
for n in TASKS:
    print(f'  {n:9s} refined     %.3f / %.3f / %.3f' % score_stats(n, TD[n]))
    print(f'  {n:9s} provisional %.3f / %.3f / %.3f' % score_stats(n, PD[n]))
print()
for name, fx in FIX.items():
    r = {}
    for n in TASKS:
        V = fx(PD[n])
        m, _ = evaluate(TQ[n] @ V.T, DATA[n])
        nd10, rr, sh = mixed10(n, V)
        st = score_stats(n, V)
        r[n] = {'TS': m['ndcg10'], 'mixed_ndcg': nd10, 'recall_ratio': rr, 'top10_share': sh,
                'cos': float((V * TD[n]).sum(1).mean()), 'q999': st[1], 'rel_mean': st[2]}
    core = {k: float(np.mean([r[n][k] for n in CORE])) for k in ('TS', 'mixed_ndcg', 'recall_ratio', 'top10_share', 'cos')}
    res[name] = {'tasks': r, 'core': core}
    c = core
    print(f"{name:28s} core: TS {100*c['TS']:.2f}  cos {c['cos']:.3f}  mixed p=10%: nDCG {100*c['mixed_ndcg']:.2f}  "
          f"recall ratio {c['recall_ratio']:.3f}  top-10 share {100*c['top10_share']:.1f}%   "
          f"| ArguAna TS {100*r['ArguAna']['TS']:.2f} rr {r['ArguAna']['recall_ratio']:.3f} share {100*r['ArguAna']['top10_share']:.1f}%",
          flush=True)
(OUT / 'result.json').write_text(json.dumps(res, indent=1) + '\n')
