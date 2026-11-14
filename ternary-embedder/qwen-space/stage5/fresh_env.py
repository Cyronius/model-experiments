"""Stage 5 step 4. Two halves:
  prepare (repo .venv):  python fresh_env.py prepare OUTDIR      -> docs.json (1,000 mixed docs + dev cut docs)
  run (fresh venv, -I):  python -I fresh_env.py run RELEASE OUTDIR [force_avx2]  -> vectors from the release package
  compare (repo .venv):  python fresh_env.py compare RELEASE OUTDIR  -> Stage 4 gates again, written to stage5/fresh-env.json
"""
import json, os, sys
from pathlib import Path


def prepare(out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from common import WORK
    import shared
    rng = np.random.default_rng(0)
    val = json.loads((WORK / 'stage2/texts_val.json').read_text())
    pool = [shared.cut_to_contract(t, 504) for t in shared.dev_task('NFCorpus')['texts']] + val
    lens = np.array([shared.n_content_tokens(t) for t in pool])
    pick = []
    for lo, hi in ((0, 16), (16, 64), (64, 128), (128, 256), (256, 400), (400, 505)):
        idx = np.where((lens >= lo) & (lens < hi))[0]
        pick += list(rng.choice(idx, min(167, len(idx)), replace=False))
    docs = [pool[i] for i in pick[:1000]]
    dev = {n: shared.dev_task(n)['texts'] for n in ('SciFact', 'NFCorpus', 'ArguAna')}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'docs.json').write_text(json.dumps({'mixed': docs, 'dev_raw': dev}))


def run(release, out, force):
    os.environ['TDE_FORCE_AVX2'] = force
    sys.path.insert(0, str(release))
    import numpy as np
    from tde import Embedder
    e = Embedder(release, workers=4, dim=4096)
    data = json.loads((Path(out) / 'docs.json').read_text())
    res = {'isa': e.isa, 'python': sys.version.split()[0], 'modules_torch': 'torch' in sys.modules,
           'modules_transformers': 'transformers' in sys.modules}
    try:
        import torch  # noqa
        res['torch_importable'] = True
    except ImportError:
        res['torch_importable'] = False
    tag = 'avx2' if force == '1' else 'default'
    np.save(Path(out) / f'mixed_{tag}.npy', e.encode(data['mixed']))
    # dev docs: raw text in; the package applies the contract itself (same cut as shared.cut_to_contract)
    for n, texts in data['dev_raw'].items():
        np.save(Path(out) / f'dev_{n}_{tag}.npy', e.encode(texts))
        if tag == 'default':
            json.dump([e.contract_text(t) for t in texts[:200]], open(Path(out) / f'contract_{n}.json', 'w'))
            json.dump(e.tokenize(texts), open(Path(out) / f'tokens_{n}.json', 'w'))
    res['modules_torch_after'] = 'torch' in sys.modules
    res['modules_transformers_after'] = 'transformers' in sys.modules
    (Path(out) / f'run_{tag}.json').write_text(json.dumps(res))
    print(json.dumps(res))


def compare(release, out):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
    import numpy as np, torch
    from common import WORK, dump, evaluate, prefix
    import shared, pack
    torch.set_num_threads(12)
    out = Path(out)
    data = json.loads((out / 'docs.json').read_text())
    ref = pack.Reference(Path(release) / 'model.tde')
    rv, _, _ = ref.encode(data['mixed'])
    res = {'runs': {t: json.loads((out / f'run_{t}.json').read_text()) for t in ('default', 'avx2')}}
    for t in ('default', 'avx2'):
        v = np.load(out / f'mixed_{t}.npy')
        c = (v * rv).sum(1)
        res[f'{t}_vs_reference'] = {'min_cos': float(c.min()), 'median_cos': float(np.median(c)), 'pass': float(c.min()) >= 0.999}
    c = (np.load(out / 'mixed_default.npy') * np.load(out / 'mixed_avx2.npy')).sum(1)
    res['paths_agree'] = {'min_cos': float(c.min()), 'pass': float(c.min()) >= 0.9999}
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[2] / 'tokenizer', local_files_only=True)
    res['dev_TS'] = {}; mism = 0; cmis = 0
    for n in ('SciFact', 'NFCorpus', 'ArguAna'):
        d = shared.dev_task(n)
        cut = [shared.cut_to_contract(t, 504) for t in d['texts']]
        toks = json.load(open(out / f'tokens_{n}.json'))
        theirs = hf(cut, add_special_tokens=False)['input_ids']
        mism += sum(a != b for a, b in zip(toks, theirs))
        cmis += sum(a != b for a, b in zip(json.load(open(out / f'contract_{n}.json')), cut[:200]))
        q = prefix(np.load(WORK / f'dev/{n}_T_queries.npy'), 1024)
        rvv, _, _ = ref.encode(cut)
        row = {'reference': evaluate(q @ prefix(rvv, 1024).T, d)[0]['ndcg10']}
        for t in ('default', 'avx2'):
            row[t] = evaluate(q @ prefix(np.load(out / f'dev_{n}_{t}.npy'), 1024).T, d)[0]['ndcg10']
        res['dev_TS'][n] = row
    res['token_ids_vs_transformers_on_contract_text'] = {'mismatches': mism, 'pass': mism == 0}
    res['contract_text_vs_shared_cut_first200'] = {'mismatches': cmis, 'pass': cmis == 0}
    for t in ('default', 'avx2'):
        diff = float(np.mean([res['dev_TS'][n][t] - res['dev_TS'][n]['reference'] for n in ('SciFact', 'NFCorpus')]))
        res[f'dev_TS_core_diff_{t}'] = {'diff': diff, 'pass': abs(diff) <= 0.15}
    dump(Path(__file__).resolve().parent / 'fresh-env.json', res)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prepare':
        prepare(sys.argv[2])
    elif cmd == 'run':
        run(Path(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else '0')
    else:
        compare(sys.argv[2], sys.argv[3])
