"""Run the hardware-loop probe on the NPU and check every output word.

System python with XRT's pyxrt (/usr/lib64/python3.14/site-packages):
    python3 run.py [build_dir] [--runs N]

out[i] must equal in[i]*mul + i for i < count (default 3 and 1024). Words
from count on must NOT equal that value: the loop must have stopped. The
DDR output starts as 0xDEADBEEF, but the DMA copies the whole L1 tile, so
words the core skipped show stale L1 data, not the sentinel. If the
hardware loop body ran once (hipfire's report for its own encoder), only
the first couple of words would be right.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyxrt

N = 1024
SENTINEL = 0xDEADBEEF
TO_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build", nargs="?", type=Path, default=Path(__file__).parent / "build")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout-ms", type=int, default=10000)
    ap.add_argument("--mul", type=int, default=3, help="expected multiplier")
    ap.add_argument("--count", type=int, default=N, help="words the loop should write; the rest must stay untouched")
    a = ap.parse_args()

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(str(a.build / "final.xclbin"))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    k = pyxrt.kernel(ctx, "MLIR_AIE")

    words = np.fromfile(a.build / "insts.bin", dtype=np.uint32)
    instr = pyxrt.bo(dev, words.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    instr.write(words.tobytes(), 0)
    instr.sync(TO_DEV, words.nbytes, 0)

    b_in = pyxrt.bo(dev, 4 * N, pyxrt.bo.host_only, k.group_id(3))
    b_out = pyxrt.bo(dev, 4 * N, pyxrt.bo.host_only, k.group_id(4))
    h_in = np.frombuffer(b_in.map(), dtype=np.int32)
    h_out = np.frombuffer(b_out.map(), dtype=np.uint32)

    rng = np.random.default_rng(20261103)
    results = []
    for r in range(a.runs):
        x = rng.integers(-(1 << 20), 1 << 20, N, dtype=np.int32)
        h_in[:] = x
        b_in.sync(TO_DEV, 4 * N, 0)
        h_out[:] = SENTINEL
        b_out.sync(TO_DEV, 4 * N, 0)
        t0 = time.perf_counter()
        run = k(3, instr, int(words.size), b_in, b_out)
        state = run.wait(a.timeout_ms)
        ms = (time.perf_counter() - t0) * 1e3
        b_out.sync(FROM_DEV, 4 * N, 0)
        got = h_out.view(np.int32).copy()
        want = (x.astype(np.int64) * a.mul + np.arange(N)).astype(np.int32)
        # The core writes an L1 tile and the DMA copies all of it, so words the
        # loop skipped hold stale L1 data, not the DDR sentinel. Past `count`,
        # the check is that a word does NOT hold this run's full-loop value.
        tail = np.arange(N) >= a.count
        ok = np.where(tail, got != want, got == want)
        untouched = int((h_out == SENTINEL).sum())
        tail_equal_full_loop = int((got == want)[tail].sum())
        first_bad = int(np.argmin(ok)) if not ok.all() else None
        res = {
            "run": r,
            "state": str(state),
            "ms": round(ms, 3),
            "correct_words": int(ok.sum()),
            "untouched_words": untouched,
            "tail_words_equal_full_loop": tail_equal_full_loop,
            "first_bad": first_bad,
        }
        results.append(res)
        print(json.dumps(res))
        if "COMPLETED" not in str(state):
            break
    allok = all(r["correct_words"] == N for r in results)
    print("ALL_CORRECT" if allok else "MISMATCH")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
