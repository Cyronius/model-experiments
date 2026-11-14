"""Stage 4 gates for the C runtime (run from the repository's .venv; torch is used only for the reference):
  1. token ids identical to the transformers tokenizer on every dev text
  2. on 1,000 mixed-length documents, min cosine to the Python reference >= 0.999 on each code path
  3. dev TS within 0.15 nDCG of the reference on each code path (504 contract, 1024 dims)
  4. the two code paths agree to min cosine 0.9999
Writes qwen-space/runtime/check.json.

    python check.py MODEL.tde [LIBRARY]
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from common import PROJECT, WORK, dump, evaluate, prefix, sha256
import shared
import pack

TASKS = ['SciFact', 'NFCorpus', 'ArguAna']
CORE = ['SciFact', 'NFCorpus']


def embedder(model, lib, force):
    os.environ['TDE_FORCE_AVX2'] = force
    from tde import Embedder
    return Embedder(model=model, tokenizer=PROJECT / 'tokenizer/tokenizer.json', library=lib, workers=4, dim=4096)


def main():
    model = Path(sys.argv[1]); lib = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'build/libtde.so'
    torch.set_num_threads(12)
    res = {'model': str(model), 'model_sha256': sha256(model), 'library_sha256': sha256(lib)}
    # 1. tokenizer parity
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(PROJECT / 'tokenizer', local_files_only=True)
    e = embedder(model, lib, '0')
    mism = total = 0
    for n in TASKS:
        d = shared.dev_task(n)
        texts = d['texts'] + d['queries']
        ours = [x.ids for x in e.tok.encode_batch(texts, add_special_tokens=False)]
        theirs = hf(texts, add_special_tokens=False)['input_ids']
        mism += sum(a != b for a, b in zip(ours, theirs)); total += len(texts)
    res['tokenizer'] = {'texts': total, 'mismatches': mism, 'pass': mism == 0}
    print(res['tokenizer'], flush=True)
    # 2/4. 1,000 mixed-length documents vs reference
    rng = np.random.default_rng(0)
    val = json.loads((WORK / 'stage2/texts_val.json').read_text())
    pool = [shared.cut_to_contract(t, 504) for t in shared.dev_task('NFCorpus')['texts']] + val
    lens = np.array([shared.n_content_tokens(t) for t in pool])
    pick = []
    for lo, hi in ((0, 16), (16, 64), (64, 128), (128, 256), (256, 400), (400, 505)):
        idx = np.where((lens >= lo) & (lens < hi))[0]
        pick += list(rng.choice(idx, min(167, len(idx)), replace=False))
    docs = [pool[i] for i in pick[:1000]]
    ref = pack.Reference(model)
    rv, rn, _ = ref.encode(docs)
    vecs = {}
    for force, name in (('0', 'vnni'), ('1', 'avx2')):
        e = embedder(model, lib, force)
        if name == 'vnni' and e.isa != 'avx512-vnni':
            res[name] = 'not available on this CPU'; continue
        v, c = e.encode(docs, return_confidence=True)
        cos = (v * rv).sum(1)
        res[f'{name}_vs_reference'] = {'n': len(docs), 'min_cos': float(cos.min()), 'median_cos': float(np.median(cos)),
                                       'conf_max_rel_err': float(np.abs(c / rn - 1).max()), 'pass': float(cos.min()) >= 0.999,
                                       'length_bins': [int(x) for x in np.histogram(lens[pick[:1000]], [0, 16, 64, 128, 256, 400, 505])[0]]}
        vecs[name] = v
        print(name, res[f'{name}_vs_reference'], flush=True)
    if len(vecs) == 2:
        c = (vecs['vnni'] * vecs['avx2']).sum(1)
        res['paths_agree'] = {'min_cos': float(c.min()), 'pass': float(c.min()) >= 0.9999}
    # 3. dev TS (504 contract, 1024 dims) runtime vs reference
    res['dev_TS'] = {}
    for n in TASKS:
        d = shared.dev_task(n)
        cut = [shared.cut_to_contract(t, 504) for t in d['texts']]
        q = prefix(np.load(WORK / f'dev/{n}_T_queries.npy'), 1024)
        rvv, _, _ = ref.encode(cut)
        row = {'reference': evaluate(q @ prefix(rvv, 1024).T, d)[0]['ndcg10']}
        for force, name in (('0', 'vnni'), ('1', 'avx2')):
            if name not in vecs:
                continue
            e = embedder(model, lib, force)
            row[name] = evaluate(q @ prefix(e.encode(cut), 1024).T, d)[0]['ndcg10']
        res['dev_TS'][n] = row
        print(n, row, flush=True)
    for name in vecs:
        diff = float(np.mean([res['dev_TS'][n][name] - res['dev_TS'][n]['reference'] for n in CORE]))
        res[f'dev_TS_core_diff_{name}'] = {'diff': diff, 'pass': abs(diff) <= 0.15,
                                          'per_task_max_abs': float(max(abs(res['dev_TS'][n][name] - res['dev_TS'][n]['reference']) for n in TASKS))}
    dump(HERE / 'check.json', res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
