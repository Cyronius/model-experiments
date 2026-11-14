"""trit_embed with the C++ host loop: tokenization alone, encode_ids on pre-tokenized texts, full encode, and
how close the C++ embedding is to the model's own embed (read back from an input buffer). Run under `dibs run npu`."""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
D = Path('/home/cyrus/code/phlegm-trit-embed/open_kernels/designs/trit_embed')
sys.path.insert(0, str(D))
from trit_npu import TritNpu, SETS, TOKENS
P = Path(__file__).resolve().parent.parent / 'comparisons/qwen06b/data'
torch.set_num_threads(2)
docs = [(r.get('title', '') + ' ' + r['text']).strip() for r in map(json.loads, (P/'scifact/corpus.jsonl').read_text().splitlines())]
stsb = json.loads((P/'stsb-test.json').read_text())
sents = sorted({t for r in stsb for t in (r['sentence1'], r['sentence2'])})
npu = TritNpu(Path.home()/'models/trit-embed-3l')
res = {'sets': SETS}
def best(fn, k=3):
    fn(); ts = []
    for _ in range(k):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)
try:
    for name, texts in (('abstracts', docs), ('sentences', sents)):
        ids = npu.tokenize(texts)
        res[f'{name}_tokenize_s'] = best(lambda: npu.tokenize(texts))
        res[f'{name}_layout_s'] = best(lambda: npu.layout(ids))
        res[f'{name}_encode_ids_s'] = best(lambda: npu.encode_ids(ids))
        res[f'{name}_encode_s'] = best(lambda: npu.encode(texts))
        res[f'{name}_dispatches'] = npu.encode_ids(ids)[1]
    # C++ embedding vs model.embed: one full dispatch of abstract tokens, then read set 0's input buffer.
    ids = npu.tokenize(docs[:2])
    lay = npu.layout(ids)
    npu.run_native(lay, np.empty((2, 384), np.float32))
    _, runs, first, lengths, tid, pos, seg, _, _ = lay
    x = npu.buffers[npu.sets[(runs - 1) % SETS][1][1]].view(np.float32).reshape(32, 48, 8, 8).transpose(0, 2, 1, 3).reshape(256, 384)
    rows = np.flatnonzero(seg[(runs - 1) * TOKENS:])
    with torch.inference_mode():
        ref = npu.model.embed(torch.from_numpy(tid[(runs - 1) * TOKENS:][rows]), torch.from_numpy(pos[(runs - 1) * TOKENS:][rows])).numpy()
    d = np.abs(x[rows] - ref)
    res['embed_vs_model'] = {'rows': len(rows), 'max_abs': float(d.max()), 'identical_fraction': float((d == 0).mean())}
finally:
    npu.close()
print(json.dumps(res, indent=1))
