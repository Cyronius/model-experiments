"""One NPU run for differential checking. System python with pyxrt.

    python3 npu_exec.py XCLBIN INSTS SIGNATURE.json SEED OUT.npz [--timeout-ms N]

SIGNATURE.json: [[elements, dtype], ...] for the runtime sequence's buffer
arguments (from the build's aie.mlir). Every buffer is filled from SEED:
floats with N(0, 0.5), integer bytes uniformly. So inputs and outputs start
identical in every run, and words nobody writes compare equal. After the run
every buffer is saved, so out-of-place writes show up too.
Exit 0 on ERT_CMD_STATE_COMPLETED, 2 on any other state (timeout, error).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pyxrt

TO_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
DTYPES = {"bf16": np.uint16, "f32": np.float32, "ui8": np.uint8, "i8": np.int8, "i16": np.int16, "i32": np.int32,
          "ui16": np.uint16, "ui32": np.uint32, "f16": np.float16}


def fill(rng, n: int, dt: str) -> np.ndarray:
    if dt == "bf16":
        f = rng.normal(0, 0.5, n).astype(np.float32)
        return (f.view(np.uint32) >> 16).astype(np.uint16)
    if dt in ("f32", "f16"):
        return rng.normal(0, 0.5, n).astype(DTYPES[dt])
    return rng.integers(0, 256, n * np.dtype(DTYPES[dt]).itemsize, dtype=np.uint8).view(DTYPES[dt])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("xclbin")
    ap.add_argument("insts")
    ap.add_argument("signature")
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--timeout-ms", type=int, default=20000)
    a = ap.parse_args()
    sig = json.load(open(a.signature))
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(a.xclbin)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    k = pyxrt.kernel(ctx, "MLIR_AIE")
    words = np.fromfile(a.insts, dtype=np.uint32)
    instr = pyxrt.bo(dev, words.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    instr.write(words.tobytes(), 0)
    instr.sync(TO_DEV, words.nbytes, 0)
    rng = np.random.default_rng(a.seed)
    bos, hosts, initial = [], [], []
    for i, (n, dt) in enumerate(sig):
        data = fill(rng, n, dt)
        initial.append(data.view(np.uint8).copy())
        bo = pyxrt.bo(dev, data.nbytes, pyxrt.bo.host_only, k.group_id(3 + i))
        h = np.frombuffer(bo.map(), dtype=data.dtype, count=data.size)
        h[:] = data
        bo.sync(TO_DEV, data.nbytes, 0)
        bos.append(bo)
        hosts.append(h)
    t0 = time.perf_counter()
    state = k(3, instr, int(words.size), *bos).wait(a.timeout_ms)
    ms = (time.perf_counter() - t0) * 1e3
    for bo, h in zip(bos, hosts):
        bo.sync(FROM_DEV, h.nbytes, 0)
    np.savez(a.out, *[h.copy() for h in hosts])
    ok = "COMPLETED" in str(state)
    changed = [int(np.count_nonzero(h.view(np.uint8) != i0)) for h, i0 in zip(hosts, initial)]
    print(json.dumps({"state": str(state), "ms": round(ms, 3), "changed_bytes": changed}))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
