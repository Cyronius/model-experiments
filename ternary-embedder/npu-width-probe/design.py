"""One-core native-width ternary arithmetic probe; no hidden K/N padding."""
from pathlib import Path
import os
import numpy as np
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).resolve().parent
K = int(os.environ.get("WIDTH_K", "384"))
REPEATS = int(os.environ.get("WIDTH_REPEATS", "1024"))
assert K in (384, 512, 1536, 2048)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def probe(a: In, w: In, out: Out):
    from aie.iron.kernels._common import _include_dirs
    from aie.utils import config
    include = _include_dirs()
    include += [str(Path(config.cxx_header_path()) / "aie_kernels"),
                str(Path(config.cxx_header_path()) / "aie_kernels/aie2p")]
    at = np.ndarray[(4 * K,), np.dtype[np.int8]]
    wt = np.ndarray[(8 * K,), np.dtype[np.uint8]]
    ot = np.ndarray[(128,), np.dtype[np.int32]]
    fn = ExternalFunction("width_probe", source_file=str(HERE / "probe.cc"),
                          arg_types=[at, wt, ot], include_dirs=include,
                          compile_flags=[f"-DWIDTH_K={K}", f"-DREPEATS={REPEATS}"])
    ai = ObjectFifo(at, name="a", depth=1)
    wi = ObjectFifo(wt, name="w", depth=1)
    oo = ObjectFifo(ot, name="out", depth=1)

    def body(ax, wx, oy, compute):
        av = ax.acquire(1)
        wv = wx.acquire(1)
        ov = oy.acquire(1)
        compute(av, wv, ov)
        ax.release(1)
        wx.release(1)
        oy.release(1)

    worker = Worker(body, fn_args=[ai.cons(), wi.cons(), oo.prod(), fn], stack_size=0x1000)

    def sequence(av, wv, ov, ap, wp, oc):
        ap.fill(av)
        wp.fill(wv)
        oc.drain(ov, wait=True)

    rt = Runtime(sequence, [at, wt, ot, ai.prod(), wi.prod(), oo.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = probe
SPECIALIZE = {}
