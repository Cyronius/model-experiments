"""trit_embed: the C++ host loop (TritNpu native=True) against the Python loop (native=False) on the same NPU
build, then speed of both. Run under `dibs run npu`. Usage: python check_native_host.py [design_dir]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
D = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/cyrus/code/phlegm-trit-embed/open_kernels/designs/trit_embed')
sys.path.insert(0, str(D))
from trit_npu import TritNpu, SETS
P = Path(__file__).resolve().parent.parent / 'comparisons/qwen06b/data'
torch.set_num_threads(2)
stsb = json.loads((P/'stsb-test.json').read_text())
sents = sorted({t for r in (stsb if isinstance(stsb, list) else stsb['rows']) for t in (r['sentence1'], r['sentence2'])})
docs = [(r.get('title', '') + ' ' + r['text']).strip() for r in map(json.loads, (P/'scifact/corpus.jsonl').read_text().splitlines())]
npu = TritNpu(Path.home()/'models/trit-embed-3l')
res = {'sets': SETS, 'sentences': len(sents), 'abstracts': len(docs)}
try:
    for name, texts in (('sentences', sents), ('abstracts', docs)):
        npu.native = False; a = npu.encode(texts)
        npu.native = True; b = npu.encode(texts)
        cos = (a * b).sum(1)
        res[f'{name}_native_vs_python'] = {'max_abs': float(np.abs(a - b).max()), 'min_cos': float(cos.min()),
                                           'identical_rows': int((a == b).all(1).sum())}
        print(name, res[f'{name}_native_vs_python'], flush=True)
    one = ['A man is playing a guitar on the street.']
    for native in (False, True):
        npu.native = native
        tag = 'native' if native else 'python'
        for name, texts in (('sentences', sents), ('abstracts', docs)):
            npu.encode(texts)
            ts = []
            for _ in range(3):
                t = time.perf_counter(); npu.encode(texts); ts.append(time.perf_counter() - t)
            res[f'{tag}_{name}_per_s'] = len(texts) / min(ts)
        for _ in range(50): npu.encode(one)
        s = []
        for _ in range(300):
            t = time.perf_counter(); npu.encode(one); s.append((time.perf_counter() - t) * 1000)
        res[f'{tag}_latency_ms'] = float(np.median(s))
        print(tag, {k: round(v, 3) for k, v in res.items() if k.startswith(tag)}, flush=True)
finally:
    npu.close()
print(json.dumps(res, indent=1))
