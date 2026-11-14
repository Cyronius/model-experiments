"""Split phlegm's trit_embed (current build) into NPU-only dispatch rate and host cost, then run the gpu-bench workloads.
Run under `dibs run npu`. Usage: python time_trit_r9.py [design_dir]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
D = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/cyrus/code/phlegm-trit-embed/open_kernels/designs/trit_embed')
sys.path.insert(0, str(D))
from trit_npu import TritNpu, TOKENS
torch.set_num_threads(2)
npu = TritNpu(Path.home()/'models/trit-embed-3l')
res = {'power_mode': None}
try:
    rng = np.random.default_rng(0)
    states = rng.standard_normal((TOKENS, 384)).astype(np.float32)
    seg = np.repeat(np.arange(1, 3, dtype=np.int32), TOKENS // 2)
    # NPU only: two sets in flight, fixed inputs already prepared.
    def npu_only(n):
        for k in range(n):
            s = k % 2
            if k >= 2: npu.wait(s)
            npu.submit(s, states, seg)
        npu.wait((n - 2) % 2); npu.wait((n - 1) % 2)
    npu_only(50)
    t = time.perf_counter(); npu_only(2000); res['npu_only_ms_per_dispatch'] = (time.perf_counter() - t) / 2000 * 1000
    # Host pieces per dispatch (256 tokens).
    ids = torch.from_numpy(rng.integers(1000, 20000, TOKENS)); pos = torch.from_numpy(np.tile(np.arange(128), 2))
    with torch.inference_mode():
        for _ in range(50): npu.model.embed(ids, pos)
        t = time.perf_counter()
        for _ in range(500): npu.model.embed(ids, pos)
    res['host_embed_ms_per_dispatch'] = (time.perf_counter() - t) / 500 * 1000
    t = time.perf_counter()
    for _ in range(200): npu.submit(0, states, seg); npu.wait(0)
    res['serial_submit_wait_ms'] = (time.perf_counter() - t) / 200 * 1000
    # gpu-bench workloads, tokenization included.
    TXT = {32: 'The cat sits beside the window while the rain falls.', 128: 'Scientific evidence supports careful testing of retrieval models. ' * 20}
    for cap, text in TXT.items():
        for batch in (1, 64):
            def call(): return npu.encode([text] * batch)
            t0 = time.perf_counter(); n = 0
            while n < 5 or time.perf_counter() - t0 < 1: call(); n += 1
            s = []
            for _ in range(100 if batch == 1 else 30):
                t = time.perf_counter(); call(); s.append((time.perf_counter() - t) * 1000)
            med = float(np.median(s)); res[f'cap{cap}-b{batch}'] = {'median_ms': med, 'emb_per_s': batch * 1000 / med}
finally:
    npu.close()
print(json.dumps(res, indent=1))
