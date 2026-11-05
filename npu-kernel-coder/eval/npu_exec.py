"""One NPU run for differential checking. System python with pyxrt.

    python3 npu_exec.py XCLBIN INSTS SIGNATURE.json SEED OUT.npz [--timeout-ms N]

SIGNATURE.json: [[elements, dtype], ...] for the runtime sequence's buffer
arguments (from the build's aie.mlir). Every buffer is filled from SEED:
floats with N(0, 0.5), integer bytes uniformly. So inputs and outputs start
identical in every run, and words nobody writes compare equal. After the run
every buffer is saved, so out-of-place writes show up too.
--guard N appends N bytes to every buffer, filled from a separate RNG
(SEED + 1000003, so the buffers themselves match runs without a guard), and
saves them as g_<i>: a program that writes past the end of a buffer (an
answer still sized for a larger A) changes its guard, which B never does.
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
    ap.add_argument("--guard", type=int, default=0, help="bytes of guard after every buffer")
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
    grng = np.random.default_rng(a.seed + 1000003)
    bos, hosts, guards, initial, ginitial = [], [], [], [], []
    for i, (n, dt) in enumerate(sig):
        data = fill(rng, n, dt)
        gdata = grng.integers(0, 256, a.guard, dtype=np.uint8)
        initial.append(data.view(np.uint8).copy())
        ginitial.append(gdata.copy())
        bo = pyxrt.bo(dev, data.nbytes + a.guard, pyxrt.bo.host_only, k.group_id(3 + i))
        mem = bo.map()
        h = np.frombuffer(mem, dtype=data.dtype, count=data.size)
        g = np.frombuffer(mem, dtype=np.uint8, count=a.guard, offset=data.nbytes)
        h[:] = data
        g[:] = gdata
        bo.sync(TO_DEV, data.nbytes + a.guard, 0)
        bos.append(bo)
        hosts.append(h)
        guards.append(g)
    t0 = time.perf_counter()
    state = k(3, instr, int(words.size), *bos).wait(a.timeout_ms)
    ms = (time.perf_counter() - t0) * 1e3
    for bo, h in zip(bos, hosts):
        bo.sync(FROM_DEV, h.nbytes + a.guard, 0)
    np.savez(a.out, *[h.copy() for h in hosts], **{f"g_{i}": g.copy() for i, g in enumerate(guards)})
    ok = "COMPLETED" in str(state)
    changed = [int(np.count_nonzero(h.view(np.uint8) != i0)) for h, i0 in zip(hosts, initial)]
    info = {"state": str(state), "ms": round(ms, 3), "changed_bytes": changed}
    if a.guard:
        info["guard_changed_bytes"] = [int(np.count_nonzero(g != g0)) for g, g0 in zip(guards, ginitial)]
    print(json.dumps(info))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
