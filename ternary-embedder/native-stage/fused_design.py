"""Fixed-depth post-norm encoder, including attention and FFNs, one dispatch."""
import hashlib
import os
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, Out, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from ironutil import Pipeline, include_dirs

ROOT = Path(__file__).resolve().parent
CAP = int(os.environ.get('ENCODER_CAP','32'))
LAYERS = int(os.environ.get('ENCODER_LAYERS','2'))
assert CAP in (8,32,128)
assert LAYERS in (2,3)
NC,NR = 8,4
PACKETS = 186


@iron.jit(aiecc_flags=['--dynamic-objFifos','--alloc-scheme=basic-sequential'])
def encoder(wts: In, mask: In, x: In, kv: InOut, out: Out, *, cfg: CompileTime[int], depth: CompileTime[int], srchash: CompileTime[int]):
    E = np.ndarray[(4096,),np.dtype[np.uint8]]
    O = np.ndarray[(8192,),np.dtype[np.uint8]]
    OC = np.ndarray[(NR*8192,),np.dtype[np.uint8]]
    X = np.ndarray[(1536,),np.dtype[np.float32]]
    XC = np.ndarray[(NR*1536,),np.dtype[np.float32]]
    P = np.ndarray[(1024,),np.dtype[np.float32]]
    Q = np.ndarray[(1536,),np.dtype[bfloat16]]
    AQ = np.ndarray[(1664,),np.dtype[np.int8]]
    AQ2 = AQ
    M = np.ndarray[(128,),np.dtype[np.int32]]
    ML = np.ndarray[(32,),np.dtype[np.float32]]
    i32=np.int32
    def kernel(name,args):
        return ExternalFunction(name,source_file=str(ROOT/f'{name}.cc'),arg_types=args,
                                include_dirs=include_dirs()+[str(ROOT)])
    copy=kernel('ep_copy',[E,P])
    loadmask=kernel('ep_mask',[E,M])
    qx=kernel('ep_quant_x',[X,AQ])
    qq=kernel('ep_quant_q',[Q,AQ2,i32])
    qq384=qq
    linear=kernel('ep_linear',[E,P,AQ,X,Q,O,i32,i32,i32])
    linear256=linear
    ln=kernel('ep_norm',[X,E])
    att=kernel('ep_attend',[Q,E,P,ML,M,i32,i32,i32,i32])
    af=kernel('ep_attfinish',[P,Q,i32])
    xo=kernel('ep_out',[X,O])
    stream=ObjectFifo(E,name='broadcast',depth=1)
    inputs,outputs,workers=[],[],[]

    def make_body(core):
        def body(ax,bx,oy,q,par,aq,aq2,m,ml,cp,lm,qxf,qqf,qq384f,lin,lin256,norm,attn,attf,outf):
            xv=ax.acquire(1)
            e=bx.acquire(1);lm(e,m);bx.release(1)
            # Constant loop counts are preserved by dynamic FIFO lowering.
            for _ in range_(depth):
                qxf(xv,aq)
                # Q remains local; K and V leave in one 8192-byte core packet.
                dest=oy.acquire(1)
                for kind in range_(3):
                    e=bx.acquire(1);cp(e,par);bx.release(1)
                    for j in range_(12):
                        e=bx.acquire(1);lin(e,par,aq,xv,q,dest,j,kind,384);bx.release(1)
                oy.release(1)
                for h in range_(12):
                    for block in range_(4):
                        e=bx.acquire(1);attn(q,e,par,ml,m,h,block,core,CAP.bit_length()-1);bx.release(1)
                    attf(par,q,h)
                # O projection and the first post-residual LayerNorm.
                qq384f(q,aq,384)
                e=bx.acquire(1);cp(e,par);bx.release(1)
                # The dummy destination is the inactive output FIFO packet;
                # residual/GELU modes never access it.
                dest=oy.acquire(1)
                for j in range_(12):
                    e=bx.acquire(1);lin(e,par,aq,xv,q,dest,j,3,384);bx.release(1)
                e=bx.acquire(1);norm(xv,e);bx.release(1)
                qxf(xv,aq)
                for group in range_(6):
                    e=bx.acquire(1);cp(e,par);bx.release(1)
                    for j in range_(8):
                        e=bx.acquire(1);lin(e,par,aq,xv,q,dest,j,4,384);bx.release(1)
                    qqf(q,aq2,256)
                    e=bx.acquire(1);cp(e,par);bx.release(1)
                    for j in range_(12):
                        e=bx.acquire(1);lin256(e,par,aq2,xv,q,dest,j,3,256);bx.release(1)
                e=bx.acquire(1);norm(xv,e);bx.release(1)
                # This extra packet is consumed by the runtime as a layer trace.
                outf(xv,dest);oy.release(1)
            ax.release(1)
        return body

    for c in range(NC):
        fi=ObjectFifo(XC,name=f'xi{c}',depth=1)
        fo=ObjectFifo(OC,name=f'yo{c}',depth=1)
        ins=fi.cons().split([r*1536 for r in range(NR)],tile=Tile(c,1),obj_types=[X]*NR,
                            names=[f'x{c}_{r}' for r in range(NR)],depths=[1]*NR)
        outs=fo.prod().join([r*8192 for r in range(NR)],tile=Tile(c,1),obj_types=[O]*NR,
                            names=[f'y{c}_{r}' for r in range(NR)],depths=[1]*NR)
        inputs.append(fi.prod(tile=Tile(c,0)));outputs.append(fo.cons(tile=Tile(c,0)))
        for r in range(NR):
            tile=Tile(c,2+r)
            args=[ins[r].cons(),stream.cons(),outs[r].prod(),
                  Buffer(Q),Buffer(P),Buffer(AQ),Buffer(AQ2),Buffer(M),Buffer(ML),
                  copy,loadmask,qx,qq,qq384,linear,linear256,ln,att,af,xo]
            workers.append(Worker(make_body(c*NR+r),fn_args=args,tile=tile,stack_size=0x1400))

    WT=depth*PACKETS*4096
    KV=32*8192
    XO=depth*KV
    def linear_tap(total,offset,size):
        return TensorAccessPattern((1,total),offset,[1,1,1,size],[0,0,0,1])
    def sequence(w,m,x,scratch,o,wp,ins,outs):
        tx=TaskGroup()
        for c in range(NC):
            ins[c].fill(x,tap=linear_tap(32*1536,c*NR*1536,NR*1536),wait=True,group=tx)
        pipe=Pipeline(inflight=3)
        pipe.fill(wp,m,linear_tap(4096,0,4096))
        for layer in range(depth):
            base=layer*PACKETS*4096
            pipe.fill(wp,w,linear_tap(WT,base,39*4096))
            tk=TaskGroup()
            for c in range(NC):
                outs[c].drain(scratch,tap=linear_tap(KV,c*NR*8192,NR*8192),wait=True,group=tk)
            tk.finish()
            for h in range(12):
                for block in range(4):
                    tap=TensorAccessPattern((1,KV),block*8*8192+h*512,
                                            [1,8,1,512],[0,8192,0,1])
                    pipe.fill(wp,scratch,tap)
            pipe.fill(wp,w,linear_tap(WT,base+39*4096,147*4096))
            to=TaskGroup()
            for c in range(NC):
                outs[c].drain(o,tap=linear_tap(XO,layer*KV+c*NR*8192,NR*8192),wait=True,group=to)
            to.finish()
        pipe.finish();tx.finish()
    rt=Runtime(sequence,[np.ndarray[(WT,),np.dtype[np.uint8]],E,
                         np.ndarray[(32*1536,),np.dtype[np.float32]],
                         np.ndarray[(KV,),np.dtype[np.uint8]],np.ndarray[(XO,),np.dtype[np.uint8]],
                         stream.prod(tile=Tile(0,0)),inputs,outputs])
    return Program(iron.get_current_device(),rt,workers=workers).resolve_program()

DESIGN=encoder
SPECIALIZE={'cfg':CAP,'depth':LAYERS,'srchash':int(hashlib.sha256(b''.join(
    (ROOT/name).read_bytes() for name in ['fused.h','fused_design.py','vecmath.h','ironutil.py']+
    sorted(p.name for p in ROOT.glob('ep_*.cc')))).hexdigest()[:8],16)}
