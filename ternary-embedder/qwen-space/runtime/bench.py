"""Stage 4 timings on this dev box (NOT server numbers). Run under `dibs run --measure`.
Tokenization included. Each configuration runs in its own subprocess pinned to fixed cores.

    python bench.py MODEL.tde LIBRARY OUT.json
"""
import json, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
LENGTHS = (16, 64, 128, 256, 504)
CORES = [2, 3, 4, 5]


def make_docs(n_per=64):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(PROJECT / 'tokenizer/tokenizer.json'))
    tk.no_truncation()
    src = [json.loads(l) for l in (PROJECT / 'comparisons/qwen06b/data/nfcorpus/corpus.jsonl').read_text().splitlines()]
    texts = [(r.get('title', '') + ' ' + r['text']).strip() for r in src]
    out = {}
    for L in LENGTHS:
        docs = []
        for t in texts:
            e = tk.encode(t, add_special_tokens=False)
            if len(e.ids) > L:
                docs.append(t[:e.offsets[L][0]])
            if len(docs) == n_per:
                break
        out[L] = docs
    return out


def child(model, lib, workers, force, docs_path):
    os.environ['TDE_FORCE_AVX2'] = force
    os.environ['RAYON_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.sched_setaffinity(0, set(CORES[:workers]))
    sys.path.insert(0, str(HERE))
    t0 = time.perf_counter()
    from tde import Embedder
    e = Embedder(model=model, tokenizer=PROJECT / 'tokenizer/tokenizer.json', library=lib, workers=workers)
    load_s = time.perf_counter() - t0
    docs = {int(k): v for k, v in json.loads(Path(docs_path).read_text()).items()}
    res = {'isa': e.isa, 'workers': workers, 'load_seconds': load_s, 'lengths': {}}
    for L, ds in docs.items():
        ntok = [len(x) for x in e.tokenize(ds)]
        r = {'content_tokens_median': float(np.median(ntok))}
        if workers == 1:
            for d in ds[:8]:
                e.encode([d])
            lat = []
            t_end = time.perf_counter() + 3.0
            i = 0
            while time.perf_counter() < t_end or len(lat) < 50:
                d = ds[i % len(ds)]; i += 1
                t = time.perf_counter(); e.encode([d]); lat.append(time.perf_counter() - t)
            lat = np.array(lat) * 1000
            r.update({'single_doc_median_ms': float(np.median(lat)), 'single_doc_p95_ms': float(np.percentile(lat, 95)),
                      'single_doc_n': len(lat)})
        batch = ds * max(1, 4 * workers // 1)
        e.encode(batch[:16])
        reps = []
        t_end = time.perf_counter() + 3.0
        while time.perf_counter() < t_end or len(reps) < 3:
            t = time.perf_counter(); e.encode(batch, batch=8); reps.append(time.perf_counter() - t)
        r['docs_per_second'] = float(len(batch) / np.median(reps))
        res['lengths'][str(L)] = r
    res['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(res))


def main():
    model, lib, out = sys.argv[1:4]
    docs_path = Path(out).with_suffix('.docs.json')
    docs_path.write_text(json.dumps({str(k): v for k, v in make_docs().items()}))
    results = {'note': 'Dev box (AMD Ryzen AI Max+ 395, Zen 5). Not server numbers.', 'cores': CORES, 'runs': []}
    for force in ('0', '1'):
        for workers in (1, 2, 4):
            p = subprocess.run([sys.executable, __file__, '--child', model, lib, str(workers), force, str(docs_path)],
                               capture_output=True, text=True)
            if p.returncode:
                print(p.stderr[-2000:]); sys.exit(1)
            r = json.loads(p.stdout.strip().splitlines()[-1])
            results['runs'].append(r)
            print(json.dumps(r), flush=True)
    Path(out).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == '--child':
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6])
    else:
        main()
