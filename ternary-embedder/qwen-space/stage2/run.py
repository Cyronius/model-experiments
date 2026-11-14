"""Stage 2: head-only Qwen-space test. Fits arms H1-H4 (+ data curve), evaluates
TS on the dev tasks, mixed-index test, calibration, gates. Writes
qwen-space/stage2/result.json and the selected head to training-v2/work/qwen-space/stage2/.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import PROJECT, WORK, dump, evaluate, features, load_student, paired, prefix, unit, Timer
import shared
from heads import Ridge, Standardize, prefix_loss_np, train_mlp

W2 = WORK / 'stage2'
DEV = WORK / 'dev'
TASKS = ['SciFact', 'NFCorpus', 'ArguAna']
CORE = ['SciFact', 'NFCorpus']
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
RES = HERE / 'result.json'
res = json.loads(RES.read_text()) if RES.exists() else {}


def save():
    dump(RES, res)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------ data
def load_split(precision, mode, split):
    return np.load(W2 / f'F_{precision}_{mode}_{split}.npy')[KEEP[split]]


def final_leak_mask():
    """Re-filter against the final (DONE) shared leakage list: True = keep."""
    L = PROJECT / 'training-v2/work/shared-qwen8b/leakage'
    assert (PROJECT / 'training-v2/work/shared-qwen8b/leakage-jaswanth/DONE').exists(), \
        'leakage-jaswanth not DONE'
    lines = (L / 'jaswanth-removed.tsv').read_text().splitlines()
    col = lines[0].split('\t').index('sha256')
    leak = {l.split('\t')[col] for l in lines[1:] if l}
    import hashlib
    keep = {}
    for split in ('train', 'val'):
        texts = json.loads((W2 / f'texts_{split}.json').read_text())
        keep[split] = np.array([hashlib.sha256(t.encode('utf-8')).hexdigest() not in leak for t in texts])
    return keep


KEEP = final_leak_mask()
res['leakage_refilter_dropped'] = {k: int((~v).sum()) for k, v in KEEP.items()}
_ytr = np.load(W2 / 'targets_train.npy', mmap_mode='r')
Ytr = _ytr if KEEP['train'].all() else np.asarray(_ytr)[KEEP['train']]
Yv = np.load(W2 / 'targets_val.npy').astype(np.float32)[KEEP['val']]


def dev_feats(name, precision, mode):
    """Dev-doc features at both contracts for a training feature mode (128 contract = single window)."""
    f128 = np.load(DEV / f'{name}_F_{precision}_128_single.npy')
    if mode == 'windowed':
        f504 = np.load(DEV / f'{name}_F_{precision}_504_windowed.npy')
    else:
        p = DEV / f'{name}_F_{precision}_504_{mode}.npy'
        if not p.exists():
            d = shared.dev_task(name)
            m = load_student(precision, native512=True)
            np.save(p, features(m, [shared.cut_to_contract(t, 504) for t in d['texts']], mode))
        f504 = np.load(p)
    return {'128': f128, '504': f504, 'q': np.load(DEV / f'{name}_F_{precision}_queries.npy')}


DATA = {n: shared.dev_task(n) for n in TASKS}
TEACH = {n: {'128': np.load(DEV / f'{n}_T_docs_128.npy'), '504': np.load(DEV / f'{n}_T_docs_504.npy'),
             'q': np.load(DEV / f'{n}_T_queries.npy')} for n in TASKS}
S1ROWS = json.loads((WORK / 'stage1-rows.json').read_text())
S1 = json.loads((PROJECT / 'qwen-space/stage1/result.json').read_text())


# ------------------------------------------------------------------ evaluation
def eval_head(head, precision, mode, rows_out=None, tag=None):
    out = {}
    for n in TASKS:
        F = dev_feats(n, precision, mode)
        for c in ('128', '504'):
            P = head.raw(F[c])
            for k in (1024, 4096):
                m, rw = evaluate(prefix(TEACH[n]['q'], k) @ prefix(P, k).T, DATA[n])
                out[f'{n}|TS_{c}_{k}'] = {'ndcg10': m['ndcg10'], 'recall10': m['recall10']}
                if rows_out is not None:
                    rows_out[f'{n}|TS_{c}_{k}'] = rw
            cs = (unit(P) * unit(TEACH[n][c].astype(np.float32))).sum(1)
            cs1 = (prefix(P, 1024) * prefix(TEACH[n][c], 1024)).sum(1)
            out[f'{n}|cos_prov_ref_{c}'] = {'mean4096': float(cs.mean()), 'p5_4096': float(np.percentile(cs, 5)),
                                           'mean1024': float(cs1.mean()), 'p5_1024': float(np.percentile(cs1, 5))}
    for c in ('128', '504'):
        for k in (1024, 4096):
            out[f'core|TS_{c}_{k}'] = float(np.mean([out[f'{n}|TS_{c}_{k}']['ndcg10'] for n in CORE]))
    return out


# ------------------------------------------------------------------ arms
def fit_ridge(X, Y, Xv, n=None):
    std = Standardize(X if n is None else X[:n])
    A, B, cnt = None, None, 0
    lim = len(X) if n is None else n
    for i in range(0, lim, 50000):
        a, b, c = Ridge.gram(X[i:min(i + 50000, lim)], Y[i:min(i + 50000, lim)], std)
        A = a if A is None else A + a; B = b if B is None else B + b; cnt += c
    best = None
    curve = []
    for lam in LAMBDAS:
        r = Ridge(); r.std = std
        reg = lam * cnt * np.eye(A.shape[0]); reg[-1, -1] = 0
        r.W = np.linalg.solve(A + reg, B).astype(np.float32)
        vl = prefix_loss_np(r.raw(Xv), Yv)
        curve.append({'lambda': lam, 'val_loss': vl})
        if best is None or vl < best[0]:
            best = (vl, lam, r)
    return best[2], {'lambda': best[1], 'val_loss': best[0], 'lambda_curve': curve}


def margin_data(precision='q8'):
    R2 = PROJECT / 'training-v2/work/round2/teacher-cache'
    val = np.load(W2 / 'r2_val.npy')
    P = np.load(R2 / 'passages.npy', mmap_mode='r')
    Q = np.load(R2 / 'prompted-queries.npy', mmap_mode='r')
    tr = np.where(~val)[0]
    Qt = np.asarray(Q[tr], np.float32)
    tp = np.asarray(P[tr, 0], np.float32); tn = np.asarray(P[tr, 1], np.float32)
    return {'Fpos': np.load(W2 / 'r2_F_pos.npy')[tr], 'Fneg': np.load(W2 / 'r2_F_neg.npy')[tr], 'Q': Qt,
            'tpos': (Qt * tp).sum(1), 'tneg': (Qt * tn).sum(1)}


def run_arm(arm, precision, mode, n=None):
    X = load_split(precision, mode, 'train')
    Xv = load_split(precision, mode, 'val')
    if n is not None:
        X = X[:n]
    Y = Ytr[:len(X)]
    if arm == 'H1':
        head, info = fit_ridge(X, Y, Xv)
    else:
        Yf = np.asarray(Y, np.float16)
        head, hist = train_mlp(X, Yf, Xv, Yv, epochs=6, margin=margin_data() if arm == 'H3' else None,
                               log=lambda h: log(arm, mode, h))
        info = {'val_loss': hist[-1]['val_loss'], 'history': hist}
    return head, info


def save_head(head, path, meta):
    st = head.state()
    np.savez(path, **{k: np.asarray(v) for k, v in st.items() if k != 'kind'}, kind=st['kind'],
             meta=json.dumps(meta))


# ------------------------------------------------------------------ mixed index + calibration
def calibration(head, precision, mode):
    """Affine a*s+b on provisional scores, fit on the round-2 cache validation split
    (training-source validation; never dev tasks)."""
    R2 = PROJECT / 'training-v2/work/round2/teacher-cache'
    val = np.where(np.load(W2 / 'r2_val.npy'))[0]
    P = np.load(R2 / 'passages.npy', mmap_mode='r')
    Q = prefix(np.asarray(np.load(R2 / 'prompted-queries.npy', mmap_mode='r')[val], np.float32), 1024)
    Fp = np.load(W2 / 'r2_F_pos.npy')[val]; Fn = np.load(W2 / 'r2_F_neg.npy')[val]
    prov = prefix(head.raw(np.concatenate([Fp, Fn])), 1024)
    ref = prefix(np.concatenate([np.asarray(P[val, 0], np.float32), np.asarray(P[val, 1], np.float32)]), 1024)
    sp = Q @ prov.T; sr = Q @ ref.T  # all queries x all val passages (own pos/neg + others)
    a, b = np.polyfit(sp.ravel(), sr.ravel(), 1)
    return {'a': float(a), 'b': float(b), 'pairs': int(sp.size), 'queries': int(len(val)),
            'note': 'least squares of refined score on provisional score, 1024 prefix, round-2 cache validation split'}


def mixed(n, P504, calib, ps=(0.01, 0.10, 0.50), seeds=5):
    d = DATA[n]
    q = prefix(TEACH[n]['q'], 1024)
    R = prefix(TEACH[n]['504'], 1024); V = prefix(P504, 1024)
    s_ref = q @ R.T; s_prov = q @ V.T
    _, rows_ref = evaluate(s_ref, d)
    lookup = {c: i for i, c in enumerate(d['cids'])}
    out = {}
    for p in ps:
        for cal in (False, True):
            acc = {'ndcg10': [], 'recall_ratio': [], 'top10_share': []}
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                prov = np.zeros(len(d['cids']), bool)
                prov[rng.choice(len(prov), int(round(p * len(prov))), replace=False)] = True
                sp = calib['a'] * s_prov + calib['b'] if cal else s_prov
                S = np.where(prov[None, :], sp, s_ref)
                m, rows = evaluate(S, d)
                hit_m = hit_r = 0; share = []
                for qi, (r, rr) in enumerate(zip(rows, rows_ref)):
                    rel = [lookup[c] for c, g in d['qrels'][r['query_id']].items() if g > 0 and c in lookup and prov[lookup[c]]]
                    tm = {lookup[c] for c in r['top10']}; tr = {lookup[c] for c in rr['top10']}
                    hit_m += sum(i in tm for i in rel); hit_r += sum(i in tr for i in rel)
                    share.append(sum(prov[i] for i in tm) / 10)
                acc['ndcg10'].append(m['ndcg10'])
                acc['recall_ratio'].append(hit_m / hit_r if hit_r else float('nan'))
                acc['top10_share'].append(float(np.mean(share)))
            out[f'p{int(p * 100)}_{"cal" if cal else "raw"}'] = {k: {'mean': float(np.nanmean(v)), 'per_seed': v}
                                                                  for k, v in acc.items()}
    return out


# ------------------------------------------------------------------ main
def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
    res.setdefault('arms', {})
    heads = {}
    for arm in ('H1', 'H2', 'H3'):
        for mode in ('windowed', 'native'):
            key = f'{arm}_q8_{mode}'
            log('fit', key)
            head, info = run_arm(arm, 'q8', mode)
            heads[key] = head
            save_head(head, W2 / f'head_{key}.npz', {'arm': arm, 'mode': mode, 'precision': 'q8', 'n': 300000})
            ev = eval_head(head, 'q8', mode)
            res['arms'][key] = {'fit': info, 'eval': ev}
            log(key, 'val', info['val_loss'], 'core TS504@1024', ev['core|TS_504_1024'])
            save()
    # H4 float-feature control (ridge, windowed)
    head, info = run_arm('H1', 'float', 'windowed')
    res['arms']['H4_float_windowed'] = {'fit': info, 'eval': eval_head(head, 'float', 'windowed')}
    save()
    # per arm: the better reading mode by validation loss
    chosen = {}
    for arm in ('H1', 'H2', 'H3'):
        w, nt = res['arms'][f'{arm}_q8_windowed'], res['arms'][f'{arm}_q8_native']
        chosen[arm] = 'windowed' if w['fit']['val_loss'] <= nt['fit']['val_loss'] else 'native'
    res['mode_by_val_loss'] = chosen
    # final arm (rule fixed in STATUS.md before results): best core TS at 504 contract, 1024 dims
    final = max(chosen, key=lambda a: res['arms'][f'{a}_q8_{chosen[a]}']['eval']['core|TS_504_1024'])
    fmode = chosen[final]
    res['selected_arm'] = {'arm': final, 'mode': fmode,
                           'rule': 'best core-mean TS (504 contract, 1024 dims) among H1-H3, each in its better reading mode by validation loss'}
    save()
    # data curve
    res['data_curve'] = {}
    for n in (50_000, 100_000, 300_000):
        if n == 300_000:
            r = res['arms'][f'{final}_q8_{fmode}']
            res['data_curve'][str(n)] = {'val_loss': r['fit']['val_loss'], 'core_TS_504_1024': r['eval']['core|TS_504_1024']}
            continue
        head, info = run_arm(final, 'q8', fmode, n=n)
        ev = eval_head(head, 'q8', fmode)
        res['data_curve'][str(n)] = {'val_loss': info['val_loss'], 'core_TS_504_1024': ev['core|TS_504_1024']}
        save()
    # final head: the 300k fit
    finalize(heads[f'{final}_q8_{fmode}'], final, fmode, 300000)


def finalize(head, final, fmode, n_train, out_name='head_selected.npz'):
    rows = {}
    ev = eval_head(head, 'q8', fmode, rows_out=rows)
    res['final_eval'] = ev
    # paired TS - SS384 (q8): windowed SS384 at 504, single at 128
    res['paired_TS_minus_SS384'] = {}
    for n in TASKS:
        for c in ('128', '504'):
            for k in (1024, 4096):
                res['paired_TS_minus_SS384'][f'{n}|{c}_{k}'] = paired(S1ROWS[f'{n}|SS384_q8_{c}'], rows[f'{n}|TS_{c}_{k}'])
    calib = calibration(head, 'q8', fmode)
    res['calibration'] = calib
    res['mixed_index_504_1024'] = {}
    for n in TASKS:
        F = dev_feats(n, 'q8', fmode)
        res['mixed_index_504_1024'][n] = mixed(n, head.raw(F['504']), calib)
        save()
    # gates
    ss = float(np.mean([S1['tasks'][n]['SS384_q8_504']['ndcg10'] for n in CORE]))
    ts504 = ev['core|TS_504_1024']; ts128 = ev['core|TS_128_1024']
    bm = float(np.mean([S1['tasks'][n]['BM25_504']['ndcg10'] for n in CORE]))
    g2 = {k: float(np.mean([res['mixed_index_504_1024'][n]['p10_cal'][k]['mean'] for n in CORE]))
          for k in ('recall_ratio', 'top10_share')}
    g2raw = {k: float(np.mean([res['mixed_index_504_1024'][n]['p10_raw'][k]['mean'] for n in CORE]))
             for k in ('recall_ratio', 'top10_share')}
    # rule fixed before results: the calibrated variant if it passes, else the raw one
    best = 'cal' if (g2['recall_ratio'] >= 0.85 and g2['top10_share'] <= 0.15) else 'raw'
    g2use = g2 if best == 'cal' else g2raw
    res['gates'] = {
        'G1': {'TS_504_1024_core': ts504, 'SS384_q8_504_windowed_core': ss, 'diff': ts504 - ss,
               'pass': ts504 >= ss - 2.0, 'stop_rule_TS_lt_SS_minus_6': ts504 < ss - 6},
        'G2': {'p10': g2use, 'variant': best, 'raw': g2raw, 'calibrated': g2,
               'pass_a_recall_ratio_ge_0.85': g2use['recall_ratio'] >= 0.85,
               'pass_b_top10_share_le_0.15': g2use['top10_share'] <= 0.15},
        'G3': {'TS_504_1024_core': ts504, 'TS_128_1024_core': ts128, 'diff': ts504 - ts128,
               'pass': ts504 >= ts128 - 0.5},
        'BM25_504_core': bm, 'BM25_beats_TS_by_more_than_3': bm - ts504 > 3}
    res['gates']['G2']['pass'] = res['gates']['G2']['pass_a_recall_ratio_ge_0.85'] and res['gates']['G2']['pass_b_top10_share_le_0.15']
    save_head(head, W2 / out_name, {'arm': final, 'mode': fmode, 'precision': 'q8', 'n': n_train,
                                                'calibration': calib})
    res['selected_head'] = str(W2 / out_name)
    res['spend_plan_A_after'] = shared.spend('A')
    save()
    log(json.dumps(res['gates'], indent=1))


if __name__ == '__main__':
    main()
