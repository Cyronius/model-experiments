"""Hardware-loop probe: one worker runs zol_1024 on one 1024-word tile.

Built with openflowlm-next/open_kernels/build_design.py (mlir-aie 1.4.2,
Peano). Structure copied from designs/rot13 there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
TILE = 1024  # must match zol_1024 in zol.cc


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def zol(src: In, dst: Out, *, tile: CompileTime[int] = TILE):
    ty = np.ndarray[(tile,), np.dtype[np.int32]]
    kernel = ExternalFunction(f"zol_{tile}", source_file=str(HERE / "zol.cc"), arg_types=[ty, ty])
    of_in = ObjectFifo(ty, name="zin", depth=2)
    of_out = ObjectFifo(ty, name="zout", depth=2)

    def core_body(rx, tx, fn):
        a = rx.acquire(1)
        b = tx.acquire(1)
        fn(a, b)
        rx.release(1)
        tx.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), kernel], stack_size=0xD00)

    def sequence(s, d, rx_prod, tx_cons):
        rx_prod.fill(s)
        tx_cons.drain(d, wait=True)

    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = zol
SPECIALIZE = {"tile": TILE}
