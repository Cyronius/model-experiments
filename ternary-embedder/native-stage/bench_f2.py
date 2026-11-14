"""Time raw NPU dispatches for any compiled encoder build directory (fused or fused2 variants)."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import statistics
import numpy as np
import torch
from encoder import ROOT, digest
from fused_encoder import weight_stream
from fused2_encoder import weight_stream2
from runtime import load_runtime
from model import Student

CHECKPOINT = ROOT.parent/'models/campaign-20261025/model-ternary.pt'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('builds', nargs='+', type=Path)
    p.add_argument('--runs', type=int, default=200)
    a = p.parse_args()
    model = Student.load(CHECKPOINT).eval()
    layers = model.dims['layers']
    streams = {186: weight_stream(model), 170: weight_stream2(model)}
    dll, _ = load_runtime()
    report = {}
    for build in a.builds:
        manifest = json.loads((build/'build.json').read_text())
        assert manifest['layers'] == layers
        weights = streams[manifest.get('packets', 186)]
        h = dll.tdn_open(os.fsencode(build/'final.xclbin'), os.fsencode(build/'insts.bin'))
        try:
            bufs = []
            for size in (weights.nbytes, 4096, 128*384*4, 32*8192, layers*32*8192):
                ptr = dll.tdn_buf(h, size)
                bufs.append(np.ctypeslib.as_array((ct.c_uint8*size).from_address(ptr)))
            bufs[0][:] = weights
            mask = bufs[1].view(np.int32)
            mask[:128] = 1
            bufs[2].view(np.float32)[:] = np.random.default_rng(0).standard_normal(128*384).astype(np.float32)
            for i in (0, 1, 2):
                dll.tdn_sync(h, i, 1, bufs[i].nbytes, 0)
            for _ in range(10):
                dll.tdn_run(h)
            ms = [dll.tdn_run(h) for _ in range(a.runs)]
            assert min(ms) > 0, ms[:5]
        finally:
            dll.tdn_close(h)
        row = {'median_ms': statistics.median(ms), 'p10_ms': float(np.percentile(ms, 10)),
               'p90_ms': float(np.percentile(ms, 90)), 'runs': a.runs,
               'tag': manifest.get('tag', build.name), 'xclbin_sha256': digest(build/'final.xclbin')}
        report[build.name] = row
        print(json.dumps({build.name: row}), flush=True)


if __name__ == '__main__':
    main()
