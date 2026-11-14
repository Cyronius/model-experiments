"""Stage 2 data: filter Jaswanth (504 contract, leakage, validation), sample 300,000
training texts (seed 0), gather their teacher vectors and compute student features.

Outputs in training-v2/work/qwen-space/stage2/:
  data.npz        ids (global row), split, ntok; targets_{train,val}.npy (float16, unit)
  texts_{train,val}.json
  F_{precision}_{mode}_{split}.npy  (384-d pooled features)
  r2_*            round-2 cache triplet features for the margin arm
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PROJECT, WORK, SHARED_WORK, dump, features, load_student, Timer
import shared
from prep_index import FILES

OUT = WORK / 'stage2'
OUT.mkdir(parents=True, exist_ok=True)
N_TRAIN = 300_000


def leaked_hashes():
    d = SHARED_WORK / 'leakage'
    # The list may not be marked DONE yet; run.py re-filters against the final list before fitting.
    lines = (d / 'jaswanth-removed.tsv').read_text().splitlines()
    col = lines[0].split('\t').index('sha256')
    return {l.split('\t')[col] for l in lines[1:] if l}


def select():
    ix = np.load(WORK / 'jaswanth-index.npz')
    sha_hex = np.array([b.tobytes().hex() for b in ix['sha']])
    leak = leaked_hashes()
    leaked = np.isin(sha_hex, list(leak))
    fits = ix['ntok'] <= 504
    empty = ix['ntok'] == 0
    ok = fits & ~leaked & ~empty
    val = np.where(ok & ix['val'])[0]
    pool = np.where(ok & ~ix['val'])[0]
    rng = np.random.default_rng(0)
    train = rng.choice(pool, N_TRAIN, replace=False)  # order = data-curve order (first n)
    stats = {'rows': int(len(fits)), 'dropped_over_504': int((~fits).sum()),
             'dropped_leakage': int((leaked & fits).sum()), 'dropped_empty': int((empty & fits).sum()),
             'leak_hashes_loaded': len(leak), 'leak_list_done_at_selection': (SHARED_WORK / 'leakage/DONE').exists(), 'eligible': int(ok.sum()), 'validation': int(len(val)),
             'train_pool': int(len(pool)), 'train_sampled': N_TRAIN,
             'kept_ntok_percentiles': {str(p): float(np.percentile(ix['ntok'][ok], p)) for p in (5, 25, 50, 75, 95, 99)},
             'kept_share_over_126': float((ix['ntok'][ok] > 126).mean()),
             'all_ntok_percentiles': {str(p): float(np.percentile(ix['ntok'], p)) for p in (50, 95, 99)}}
    return ix, train, val, stats


def gather(ix, rows):
    by_file = {}
    for k, g in enumerate(rows):
        by_file.setdefault(int(ix['fileidx'][g]), []).append((int(ix['row'][g]), k))
    texts = [None] * len(rows)
    tg = np.zeros((len(rows), 4096), np.float16)
    for fi, items in sorted(by_file.items()):
        t = pq.read_table(FILES[fi])
        rr = np.array([r for r, _ in items]); kk = np.array([k for _, k in items])
        sub = t.take(rr)
        codes = np.array(sub.column('embedding').to_pylist(), np.float32)
        amax = np.array(sub.column('abs_max_val').to_pylist(), np.float32)
        v = codes * amax[:, None] / 127
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        tg[kk] = v.astype(np.float16)
        for k, s in zip(kk, sub.column('text').to_pylist()):
            texts[k] = s
        print('gather file', fi, flush=True)
    return texts, tg


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
    if not (OUT / 'data.npz').exists():
        ix, train, val, stats = select()
        print(json.dumps(stats, indent=1), flush=True)
        for name, rows in (('train', train), ('val', val)):
            texts, tg = gather(ix, rows)
            # sanity: every text fits the contract and is not validation / is validation
            assert all(shared.is_validation(t) == (name == 'val') for t in texts[:2000])
            np.save(OUT / f'targets_{name}.npy', tg)
            (OUT / f'texts_{name}.json').write_text(json.dumps(texts))
        np.savez(OUT / 'data.npz', train=train, val=val)
        dump(Path(__file__).parent / 'data-stats.json', stats)
    texts = {s: json.loads((OUT / f'texts_{s}.json').read_text()) for s in ('train', 'val')}
    for precision, mode in [('q8', 'windowed'), ('q8', 'native'), ('float', 'windowed')]:
        model = None
        for s in ('val', 'train'):
            p = OUT / f'F_{precision}_{mode}_{s}.npy'
            if p.exists():
                continue
            model = model or load_student(precision, native512=(mode == 'native'))
            with Timer(f'features {precision} {mode} {s}'):
                np.save(p, features(model, texts[s], mode))
    # round-2 triplets for the margin arm (student sees the first 126 tokens)
    R2 = PROJECT / 'training-v2/work/round2'
    if not (OUT / 'r2_F_neg.npy').exists():
        rows = [json.loads(l) for l in (R2 / 'balanced-refined.jsonl').read_text().splitlines()]
        model = load_student('q8')
        for col in ('pos', 'neg'):
            key = 'positive' if col == 'pos' else 'negative'
            with Timer(f'r2 {col}'):
                np.save(OUT / f'r2_F_{col}.npy', features(model, [r[key] for r in rows], 'single'))
        np.save(OUT / 'r2_val.npy', np.array([shared.is_validation(r['query']) for r in rows]))


if __name__ == '__main__':
    main()
