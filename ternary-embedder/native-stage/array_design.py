"""Broadcast one packed matrix to 32 four-token projection workers."""
import os
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

ROOT = Path(__file__).resolve().parent
K, N = int(os.environ['STAGE_K']), int(os.environ['STAGE_N'])
NC, NR = 8, 4
GROUPS = 6 if K == 1536 else 1


@iron.jit(aiecc_flags=['--dynamic-objFifos', '--alloc-scheme=basic-sequential'])
def dense(a: In, w: In, out: Out):
    from aie.iron.kernels._common import _include_dirs
    from aie.utils import config
    inc = _include_dirs() + [str(Path(config.cxx_header_path()) / 'aie_kernels'),
                            str(Path(config.cxx_header_path()) / 'aie_kernels/aie2p')]
    at = np.ndarray[(4*K,), np.dtype[np.int8]]
    ac = np.ndarray[(NR*4*K,), np.dtype[np.int8]]
    wt = np.ndarray[(8*K,), np.dtype[np.uint8]]
    ot = np.ndarray[(128*GROUPS,), np.dtype[np.int32]]
    oc = np.ndarray[(NR*128*GROUPS,), np.dtype[np.int32]]
    fn = ExternalFunction('projection', source_file=str(ROOT/'projection.cc'),
                          arg_types=[at, wt, ot], include_dirs=inc,
                          compile_flags=[f'-DSTAGE_K={K}'])
    weights = ObjectFifo(wt, name='weights', depth=1)
    aps, ocs, workers = [], [], []

    def body(ax, wx, oy, compute):
        av = ax.acquire(1)
        for _ in range_(N//32):
            wv, ov = wx.acquire(1), oy.acquire(1)
            compute(av, wv, ov)
            wx.release(1)
            oy.release(1)
        ax.release(1)

    for c in range(NC):
        ai = ObjectFifo(ac, name=f'ai{c}', depth=1)
        ao = ObjectFifo(oc, name=f'ao{c}', depth=1)
        ins = ai.cons().split([r*4*K for r in range(NR)], tile=Tile(c,1),
                             obj_types=[at]*NR, names=[f'a{c}_{r}' for r in range(NR)], depths=[1]*NR)
        outs = ao.prod().join([r*128*GROUPS for r in range(NR)], tile=Tile(c,1),
                             obj_types=[ot]*NR, names=[f'o{c}_{r}' for r in range(NR)], depths=[1]*NR)
        aps.append(ai.prod(tile=Tile(c,0)))
        ocs.append(ao.cons(tile=Tile(c,0)))
        for r in range(NR):
            workers.append(Worker(body, fn_args=[ins[r].cons(), weights.cons(), outs[r].prod(), fn],
                                  tile=Tile(c,2+r), stack_size=0x1000))

    na, no = NC*NR*4*K, NC*NR*4*N*GROUPS
    def sequence(av, wv, ov, ap, wp, output):
        tg_a, tg_o, tg_w = TaskGroup(), TaskGroup(), TaskGroup()
        for c in range(NC):
            ap[c].fill(av, tap=TensorAccessPattern((1,na), c*NR*4*K,
                         [1,1,NR,4*K], [0,0,4*K,1]), wait=True, group=tg_a)
            output[c].drain(ov, tap=TensorAccessPattern((1,no), c*NR*4*N*GROUPS,
                            [1,N//32,NR,128*GROUPS], [0,NR*128*GROUPS,128*GROUPS,1]),
                            wait=True, group=tg_o)
        wp.fill(wv, wait=True, group=tg_w)
        tg_o.finish()
        tg_a.finish()
        tg_w.finish()

    rt = Runtime(sequence, [np.ndarray[(na,), np.dtype[np.int8]],
                           np.ndarray[(N*K//4,), np.dtype[np.uint8]],
                           np.ndarray[(no,), np.dtype[np.int32]],
                           aps, weights.prod(tile=Tile(0,0)), ocs])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()

DESIGN = dense
