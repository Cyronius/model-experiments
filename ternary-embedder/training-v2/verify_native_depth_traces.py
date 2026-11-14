"""Check every native layer against CPU, including partial sequence buckets.

Run with the project Torch venv and dibs npu; no latency measurements or labels.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'native-stage'))
from fused_encoder import FusedEncoder, digest


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--full-bucket',action='store_true');a=p.parse_args()
    torch.set_num_threads(2)
    rows = []
    for depth in (2, 3):
        selection = json.loads((ROOT / 'native-depth' / f'frozen-{depth}.json').read_text())['models']['ternary']
        path = Path(selection['path'])
        assert digest(path) == selection['sha256']
        for cap in (32, 128):
            fixture = np.load(PROJECT / 'native-stage/fixtures' / f's{cap}.trace.npz')
            ids = torch.from_numpy(fixture['ids'])
            mask = torch.from_numpy(fixture['mask']).bool()
            engine = FusedEncoder(cap, checkpoint=path)
            try:
                batch=128//cap if a.full_bucket else 1
                for i in range(0,len(ids),batch):
                    q, m = ids[i:i+batch], mask[i:i+batch]
                    vector, native = engine(q, m, trace=True)
                    cpu, _, expected = engine.model(q.masked_fill(~m, 0), m)
                    assert len(native) == len(cpu) == depth + 1
                    vector_cosine = float(F.cosine_similarity(vector, expected).min())
                    assert vector_cosine > .995
                    layers = []
                    for stage, (actual, reference) in enumerate(zip(native, cpu)):
                        assert actual.shape == reference.shape
                        cosine = F.cosine_similarity(actual, reference, dim=-1)[m]
                        assert float(cosine.min()) > .99 and float(cosine.mean()) > .999
                        layers.append(dict(stage=stage, min_token_cosine=float(cosine.min()),
                                           mean_token_cosine=float(cosine.mean()),
                                           max_abs_error=float((actual-reference)[m].abs().max())))
                    rows.append(dict(depth=depth, cap=cap, batch=len(q), fixture=i,
                                     checkpoint_sha256=selection['sha256'], vector_cosine=vector_cosine, layers=layers))
            finally:
                engine.close()
    report = dict(scope='Historical fixed2/3layer checkpoints, all intermediate valid-token states. No latency or retrieval-quality claim.', runs=rows)
    filename='trace-verification-full.json' if a.full_bucket else 'trace-verification.json'
    (ROOT / 'native-depth'/filename).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cases=len(rows), min_vector_cosine=min(r['vector_cosine'] for r in rows),
                          min_token_cosine=min(l['min_token_cosine'] for r in rows for l in r['layers']))), flush=True)


if __name__ == '__main__':
    main()
