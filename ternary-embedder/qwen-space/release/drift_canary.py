"""Drift canary: re-embed 50 fixed public texts through the hosted teacher and compare with the
vectors stored at release time. The hosted model can change without notice; a drop in cosine
means refined vectors made after the change no longer match the space this embedder was fit to.

    export DEEPINFRA_API_KEY=...        # never pass the key on the command line
    python drift_canary.py [FOLDER]     # FOLDER holds canary.json and canary-vectors.npy

Cost: about 15,000 tokens (~$0.0002). Uses only the standard library and numpy.
Request: the full 4096-d vector (no `dimensions`), no `custom_instruction`, texts as stored.
"""
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np

URL = 'https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Embedding-8B'


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    meta = json.loads((folder / 'canary.json').read_text())
    ref = np.load(folder / 'canary-vectors.npy').astype(np.float32)
    key = os.environ.get('DEEPINFRA_API_KEY', '').strip()
    if not key:
        sys.exit('set DEEPINFRA_API_KEY')
    out = []
    for i in range(0, len(meta['texts']), 25):
        req = urllib.request.Request(URL, data=json.dumps({'inputs': meta['texts'][i:i + 25]}).encode(),
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as r:
            out.append(np.asarray(json.loads(r.read())['embeddings'], np.float32))
    new = np.concatenate(out)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    cos = (unit(new) * unit(ref)).sum(1)
    cos1024 = (unit(new[:, :1024]) * unit(ref[:, :1024])).sum(1)
    res = {'n': len(cos), 'stored_on': meta['date'], 'cos4096_min': float(cos.min()), 'cos4096_median': float(np.median(cos)),
           'cos1024_min': float(cos1024.min()), 'cos1024_median': float(np.median(cos1024))}
    res['verdict'] = ('unchanged' if res['cos4096_min'] >= 0.999 else
                      'small numeric change (check precision/provider)' if res['cos4096_median'] >= 0.99 else
                      'CHANGED: the hosted model no longer matches; refit or re-embed before mixing vectors')
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
