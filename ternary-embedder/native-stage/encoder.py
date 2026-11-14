"""Complete hybrid encoder: all 12 ternary projections on NPU, other ops CPU.

Uses the archived tiny-planner's small XRT C ABI; persistent contexts and BOs.
No network, subprocess per batch, or effective float weight matrix on the NPU path.
"""
from __future__ import annotations
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from arithmetic import quantize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'tern-embed'))
from model import Student

from runtime import LIBRARY as DLL, load_runtime

CHECKPOINT = ROOT.parent/'tern-embed/pulls/grid1/runs/2a_384/model.pt'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Projection:
    def __init__(self, dll, k, n):
        self.dll, self.k, self.n = dll, k, n
        self.groups = 6 if k == 1536 else 1
        print(f'Opening NPU projection {k}x{n}',file=sys.stderr,flush=True)
        build = ROOT/f'build_array_k{k}_n{n}'
        self.manifest = json.loads((build/'build.json').read_text())
        for name, sha in self.manifest['artifact_sha256'].items():
            if digest(build/name) != sha:
                raise ValueError(f'Changed build artifact: {build/name}')
        for name, sha in self.manifest['source_sha256'].items():
            if digest(ROOT/name) != sha:
                raise ValueError(f'Rebuild changed source: {name}')
        self.handle = dll.tdn_open(os.fsencode(build/'final.xclbin'), os.fsencode(build/'insts.bin'))
        if not self.handle:
            raise RuntimeError(f'Cannot open XRT context for {k}x{n}')
        print(f'Opened NPU projection {k}x{n}',file=sys.stderr,flush=True)
        self.buffers = []
        for size, dtype in ((128*k, np.int8), (k*n//4, np.uint8),
                            (128*n*self.groups*4, np.int32)):
            ptr = dll.tdn_buf(self.handle, size)
            if not ptr:
                raise RuntimeError('XRT buffer allocation failed')
            raw = (ct.c_uint8*size).from_address(ptr)
            self.buffers.append(np.ctypeslib.as_array(raw).view(dtype))
        self.dispatch_ms = []
        self.phases = {'quantize_pack': 0., 'transfer_dispatch': 0., 'dequantize': 0.}

    def close(self):
        if self.handle:
            self.dll.tdn_close(self.handle)
            self.handle = None

    def __call__(self, x, packed_weights, scales, bias):
        t = time.perf_counter()
        shape = x.shape
        flat = x.detach().numpy().reshape(-1,self.k)
        if len(flat) > 128:
            raise ValueError('A dispatch accepts at most 128 token rows')
        aq, act_scale = quantize(flat)
        codes = np.zeros((128,self.k), dtype=np.int8)
        codes[:len(aq)] = aq
        self.buffers[0][:] = codes.reshape(32,4,self.k//16,16).transpose(0,2,1,3).reshape(-1)
        self.buffers[1][:] = packed_weights
        self.phases['quantize_pack'] += time.perf_counter()-t
        t = time.perf_counter()
        for i in (0,1):
            self.dll.tdn_sync(self.handle,i,1,self.buffers[i].nbytes,0)
        elapsed = self.dll.tdn_run(self.handle)
        if elapsed <= 0:
            raise RuntimeError(f'Incomplete NPU dispatch: {elapsed}')
        self.dispatch_ms.append(elapsed)
        self.dll.tdn_sync(self.handle,2,0,self.buffers[2].nbytes,0)
        self.phases['transfer_dispatch'] += time.perf_counter()-t
        t = time.perf_counter()
        products = self.buffers[2].reshape(8,self.n//32,4,self.groups,2,4,16)
        dot = products.transpose(0,2,5,3,1,4,6).reshape(128,self.groups,self.n)[:len(flat)].astype(np.float32)/4
        y = (dot*act_scale[...,None]).sum(1)*scales.reshape(1,-1)+bias.reshape(1,-1)
        self.phases['dequantize'] += time.perf_counter()-t
        return torch.from_numpy(y.reshape(*shape[:-1],self.n))


class Encoder:
    def __init__(self, checkpoint=CHECKPOINT):
        self.model = Student.load(checkpoint).eval()
        fixture = json.loads((ROOT/'fixtures/manifest.json').read_text())
        if fixture['checkpoint_sha256'] != digest(checkpoint):
            raise ValueError('Re-export fixtures for changed checkpoint')
        self.params = {}
        for key, meta in fixture['matrices'].items():
            path = ROOT/f'fixtures/{key}.w.bin'
            if digest(path) != meta['weight_sha256']:
                raise ValueError(f'Changed packed matrix {key}')
            p = np.load(ROOT/f'fixtures/{key}.params.npz')
            self.params[key] = (np.fromfile(path,np.uint8),p['scales'],p['bias'])
        dll, self.dll_dirs = load_runtime()
        self.projections = {}
        try:
            for k,n in ((384,384),(384,1536),(1536,384)):
                self.projections[k,n] = Projection(dll,k,n)
        except BaseException:
            self.close()
            raise

    def close(self):
        for proj in self.projections.values():
            proj.close()

    @torch.inference_mode()
    def __call__(self, ids, mask, trace=False):
        m, (b,t) = self.model, ids.shape
        if b*t > 128 or t > 128 or not mask.any(1).all():
            raise ValueError('Need nonempty sequences and at most 128 total token rows')
        x = m.proj(m.ln_e(m.word(ids)+m.pos.weight[:t][None]+m.tok_type.weight[0]))
        states = [x.clone()] if trace else []
        for _ in range(m.loops):
            for li, layer in enumerate(m.layers):
                def linear(name, value):
                    args = self.params[f'l{li}_{name}']
                    return self.projections[value.shape[-1],len(args[2])](value,*args)
                q,k,v = (linear(name,x).view(b,t,layer.heads,-1).transpose(1,2) for name in ('q','k','v'))
                scores = (q@k.transpose(-1,-2))/q.shape[-1]**.5
                att = scores.masked_fill(~mask[:,None,None,:],float('-inf')).softmax(-1)
                context = (att@v).transpose(1,2).reshape(b,t,384)
                x = layer.ln1(x+linear('o',context))
                x = layer.ln2(x+linear('fc2',F.gelu(linear('fc1',x),approximate=layer.gelu)))
                if trace:
                    states.append(x.clone())
        pooled = x[:,0] if m.pool == 'cls' else (x*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        vectors = F.normalize(m.out(pooled),dim=-1)
        if not torch.isfinite(vectors).all():
            raise ValueError('Nonfinite embeddings')
        return (vectors,states) if trace else vectors

    def profile(self, reset=False):
        report = {}
        for shape,p in self.projections.items():
            report[str(shape)] = {'dispatches':len(p.dispatch_ms), 'dispatch_sum_ms':sum(p.dispatch_ms),
                                 'host_phase_seconds':dict(p.phases)}
            if reset:
                p.dispatch_ms.clear()
                for key in p.phases:
                    p.phases[key] = 0.
        return report
