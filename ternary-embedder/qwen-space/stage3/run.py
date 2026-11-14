"""Stage 3: which documents to refine. Simulates refining a fraction f of documents chosen by
each policy, on the dev tasks at the 504 contract and stored dimension 1024, using only vectors
already computed. Writes qwen-space/stage3/result.json."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / 'stage2'))
from common import WORK, dump, evaluate, prefix
import shared

TASKS = ['SciFact', 'NFCorpus', 'ArguAna']
CORE = ['SciFact', 'NFCorpus']
FRACS = [0, 0.05, 0.10, 0.25, 0.50, 1.0]
DEV = WORK / 'dev'


def load_head():
    from run_helpers import head_from_npz
    return head_from_npz(WORK / 'stage2/head_selected.npz')


def text_signals(texts):
    tk = shared._tokenizer()
    ntok, unk, split3 = [], [], []
    for e in tk.encode_batch(texts, add_special_tokens=False):
        ids = e.ids[:504]; w = e.word_ids[:504]
        ntok.append(len(ids))
        unk.append(sum(i == 100 for i in ids) / max(len(ids), 1))
        counts = {}
        for x in w:
            if x is not None:
                counts[x] = counts.get(x, 0) + 1
        split3.append(sum(c >= 3 for c in counts.values()) / max(len(counts), 1))
    return np.array(ntok, float), np.array(unk), np.array(split3)


def ndcg_subset(S, d, qidx):
    sub = dict(d, qids=[d['qids'][i] for i in qidx])
    return evaluate(S[qidx], sub)[0]['ndcg10']


def main():
    s2 = json.loads((HERE.parent / 'stage2/result.json').read_text())
    variant = s2['gates']['G2']['variant']
    calib = s2['calibration']
    mode = s2['selected_arm']['mode']
    head = load_head()
    res = {'stored_dimension': 1024, 'contract': '504', 'calibration_variant': variant, 'fractions': FRACS,
           'scored_queries': 'odd-position half of the queries (random permutation, seed 0); the even half is the '
                             'history for the query-driven policy. All policies are scored on the same half.',
           'tasks': {}}
    for n in TASKS:
        d = shared.dev_task(n)
        F = np.load(DEV / f'{n}_F_q8_504_{mode}.npy')
        raw = head.raw(F)
        conf = np.linalg.norm(raw, axis=1)
        prov = prefix(raw, 1024)
        ref = prefix(np.load(DEV / f'{n}_T_docs_504.npy'), 1024)
        q = prefix(np.load(DEV / f'{n}_T_queries.npy'), 1024)
        sp = q @ prov.T
        if variant == 'cal':
            sp = calib['a'] * sp + calib['b']
        sr = q @ ref.T
        perm = np.random.default_rng(0).permutation(len(q))
        hist, score_q = np.sort(perm[0::2]), np.sort(perm[1::2])
        ntok, unk, split3 = text_signals([shared.cut_to_contract(t, 504) for t in d['texts']])
        # query-driven: how often each doc enters the provisional-only top-20 of history queries
        top20 = np.argsort(-sp[hist], axis=1)[:, :20]
        hits = np.bincount(top20.ravel(), minlength=len(ref)).astype(float)
        oracle = 1 - (prov * ref).sum(1)
        rng = np.random.default_rng(1)
        tie = rng.random(len(ref)) * 1e-9
        orders = {'lowest_confidence': np.argsort(conf + tie), 'longest_first': np.argsort(-ntok - tie),
                  'unk_share': np.argsort(-unk - tie), 'split3_share': np.argsort(-split3 - tie),
                  'query_driven': np.argsort(-hits - tie), 'oracle': np.argsort(-oracle)}
        curves = {}
        for name, order in orders.items():
            c = []
            for f in FRACS:
                refined = np.zeros(len(ref), bool); refined[order[:int(round(f * len(ref)))]] = True
                c.append(ndcg_subset(np.where(refined[None], sr, sp), d, score_q))
            curves[name] = c
        rc = []
        for seed in range(5):
            r = np.random.default_rng(100 + seed).permutation(len(ref))
            c = []
            for f in FRACS:
                refined = np.zeros(len(ref), bool); refined[r[:int(round(f * len(ref)))]] = True
                c.append(ndcg_subset(np.where(refined[None], sr, sp), d, score_q))
            rc.append(c)
        curves['random'] = list(np.mean(rc, 0))
        res['tasks'][n] = {'curves': curves, 'docs': len(ref), 'scored_queries': len(score_q),
                           'conf_corr_with_cos': float(np.corrcoef(conf, 1 - oracle)[0, 1])}
        print(n, json.dumps({k: [round(x, 2) for x in v] for k, v in curves.items()}), flush=True)
    # summaries (core mean)
    pol = list(res['tasks']['SciFact']['curves'])
    core = {p: list(np.mean([res['tasks'][n]['curves'][p] for n in CORE], 0)) for p in pol}
    allm = {p: list(np.mean([res['tasks'][n]['curves'][p] for n in TASKS], 0)) for p in pol}
    x = np.array(FRACS)

    def auc(c):
        return float(np.sum((x[1:] - x[:-1]) * (np.array(c[1:]) + np.array(c[:-1])) / 2))

    def reach95(c):
        target = c[0] + 0.95 * (c[-1] - c[0])  # 95% of the gap closed
        abs_target = 0.95 * c[-1]               # 95% of the fully refined score
        out = {}
        for name, tgt in (('gap95', target), ('score95', abs_target)):
            v = None
            for i in range(len(x) - 1):
                if c[i] >= tgt:
                    v = float(x[i]); break
                if c[i] < tgt <= c[i + 1]:
                    v = float(x[i] + (tgt - c[i]) / (c[i + 1] - c[i]) * (x[i + 1] - x[i])); break
            out[name] = v if v is not None else 1.0
        return out

    res['core_mean_curves'] = core
    res['all3_mean_curves'] = allm
    res['auc_core'] = {p: auc(core[p]) for p in pol}
    res['reach_core'] = {p: reach95(core[p]) for p in pol}
    practical = [p for p in pol if p != 'oracle']
    best = max(practical, key=lambda p: res['auc_core'][p])
    res['best_practical_by_auc_core'] = best
    dump(HERE / 'result.json', res)
    print(json.dumps({k: res[k] for k in ('auc_core', 'reach_core', 'best_practical_by_auc_core')}, indent=1))


if __name__ == '__main__':
    main()
