"""Stream complete native dense projections through one core, one dispatch."""
import os
from pathlib import Path

import numpy as np
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction

ROOT = Path(__file__).resolve().parent
K, N = int(os.environ.get("STAGE_K", "384")), int(os.environ.get("STAGE_N", "384"))
GROUPS = K // 256 if K == 1536 else 1
assert (K, N) in ((384, 384), (384, 1536), (1536, 384))


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dense(a: In, w: In, out: Out):
    from aie.iron.kernels._common import _include_dirs
    from aie.utils import config
    includes = _include_dirs() + [str(Path(config.cxx_header_path()) / "aie_kernels"),
                                 str(Path(config.cxx_header_path()) / "aie_kernels/aie2p")]
    at = np.ndarray[(4 * K,), np.dtype[np.int8]]
    wt = np.ndarray[(8 * K,), np.dtype[np.uint8]]
    ot = np.ndarray[(128 * GROUPS,), np.dtype[np.int32]]
    allw = np.ndarray[(N * K // 4,), np.dtype[np.uint8]]
    allo = np.ndarray[(4 * N * GROUPS,), np.dtype[np.int32]]
    fn = ExternalFunction("projection", source_file=str(ROOT / "projection.cc"),
                          arg_types=[at, wt, ot], include_dirs=includes,
                          compile_flags=[f"-DSTAGE_K={K}"])
    ai, wi, oo = (ObjectFifo(t, name=name, depth=1) for t, name in ((at, "a"), (wt, "w"), (ot, "out")))

    def body(ax, wx, oy, compute):
        av = ax.acquire(1)
        for _ in range_(N // 32):
            wv, ov = wx.acquire(1), oy.acquire(1)
            compute(av, wv, ov)
            wx.release(1)
            oy.release(1)
        ax.release(1)

    worker = Worker(body, fn_args=[ai.cons(), wi.cons(), oo.prod(), fn], stack_size=0x1000)

    def sequence(av, wv, ov, ap, wp, oc):
        ap.fill(av)
        wp.fill(wv)
        oc.drain(ov, wait=True)

    rt = Runtime(sequence, [at, allw, allo, ai.prod(), wi.prod(), oo.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = dense
