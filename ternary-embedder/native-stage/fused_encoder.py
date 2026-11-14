"""One persistent XRT dispatch for a fixed two- or three-layer encoder body."""
import ctypes as ct
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from arithmetic import pack_weights
from encoder import ROOT, DLL, CHECKPOINT, digest
from runtime import load_runtime
from model import Student,tern_codes
from query_head import install_query_head,query_role


def parameter_packet(bias,scale):
    p=np.zeros(1024,np.float32)
    p[:len(bias)]=np.asarray(bias).reshape(-1)
    p[512:512+len(scale)]=np.asarray(scale).reshape(-1)
    return p.view(np.uint8)


def weight_stream(model):
    packets=[]
    def linear(codes,scale,bias):
        packets.append(parameter_packet(bias,scale/4))
        for block in pack_weights(codes):
            p=np.zeros(4096,np.uint8)
            p[:block.size]=block.reshape(-1)
            packets.append(p)
    for layer in model.layers:
        start=len(packets)
        for name in ('q','k','v','o'):
            m=getattr(layer,name)
            c,s=tern_codes(m.weight)
            linear(c.numpy().astype(np.int8),s.numpy(),m.bias.detach().numpy())
            if name=='o':
                packets.append(parameter_packet(layer.ln1.weight.detach().numpy(),layer.ln1.bias.detach().numpy()))
        up,down=layer.fc1,layer.fc2
        uc,us=tern_codes(up.weight);dc,ds=tern_codes(down.weight)
        for g in range(6):
            linear(uc[g*256:(g+1)*256].numpy().astype(np.int8),us[g*256:(g+1)*256].numpy(),up.bias.detach().numpy()[g*256:(g+1)*256])
            bias=down.bias.detach().numpy() if g==0 else np.zeros(384,np.float32)
            linear(dc[:,g*256:(g+1)*256].numpy().astype(np.int8),ds.numpy(),bias)
        packets.append(parameter_packet(layer.ln2.weight.detach().numpy(),layer.ln2.bias.detach().numpy()))
        assert len(packets)-start==186
    return np.stack(packets).reshape(-1)


class FusedEncoder:
    def __init__(self,cap,checkpoint=CHECKPOINT,*,full_output=False):
        self.full_output=full_output
        self.cap=cap
        self.model=Student.load(checkpoint).eval()
        install_query_head(self.model,checkpoint)
        self.layer_count=self.model.dims['layers']
        required={'d':384,'emb':384,'ffn':1536,'heads':12,'loops':1,'eps':1e-12,'gelu':'tanh'}
        if not self.model.quantized or self.layer_count not in (2,3) or any(self.model.dims[k]!=v for k,v in required.items()):
            raise ValueError('Only fixed two- or three-layer native-384 checkpoints are supported')
        self.handle=None
        dll,self.dll_dirs=load_runtime(); self.dll=dll
        build=ROOT/(f'build_fused_cap{cap}' if self.layer_count==2 else f'build_fused3_cap{cap}')
        self.manifest=json.loads((build/'build.json').read_text())
        if self.manifest['layers']!=self.layer_count or self.manifest['cap']!=cap:
            raise ValueError('Compiled encoder shape differs from checkpoint/bucket')
        for name,sha in self.manifest['source_sha256'].items():
            if digest(ROOT/name)!=sha:
                raise ValueError(f'Rebuild changed source {name}')
        for name,sha in self.manifest['artifact_sha256'].items():
            if digest(build/name)!=sha:
                raise ValueError(f'Changed build artifact {name}')
        self.handle=dll.tdn_open(os.fsencode(build/'final.xclbin'),os.fsencode(build/'insts.bin'))
        if not self.handle:
            raise RuntimeError('Cannot open complete encoder XRT context')
        try:
            self.buffers=[]
            for size,dtype in ((self.layer_count*186*4096,np.uint8),(4096,np.uint8),(128*384*4,np.float32),
                               (32*8192,np.uint8),(self.layer_count*32*8192,np.uint8)):
                ptr=dll.tdn_buf(self.handle,size)
                if not ptr:
                    self.close()
                    raise RuntimeError('XRT buffer allocation failed')
                self.buffers.append(np.ctypeslib.as_array((ct.c_uint8*size).from_address(ptr)).view(dtype))
            self.buffers[0][:]=weight_stream(self.model)
            dll.tdn_sync(self.handle,0,1,self.buffers[0].nbytes,0)
            self.dispatch_ms=[]
            self.phases={'host_input':0.,'sync_dispatch':0.,'host_output':0.}
        except BaseException:
            self.close()
            raise

    @torch.inference_mode()
    def __call__(self,ids,mask,trace=False,query=False):
        m=self.model
        b,t=ids.shape
        if t!=self.cap or b*t>128 or not mask.any(1).all():
            raise ValueError('Input must match the compiled sequence bucket with nonempty real sequences')
        tick=time.perf_counter()
        if b<128//t:
            ni=torch.zeros((128//t,t),dtype=torch.long);nm=torch.zeros_like(ni,dtype=torch.bool)
            ni[:b]=ids;nm[:b]=mask;ids,mask=ni,nm
        # Block-floating attention packs eight keys together. Keep ignored
        # positions canonical so their exponent cannot depend on caller data.
        ids=ids.masked_fill(~mask,0)
        x=m.proj(m.ln_e(m.word(ids)+m.pos.weight[:t][None]+m.tok_type.weight[0]))
        self.buffers[1].fill(0)
        self.buffers[1].view(np.int32)[:128]=mask.numpy().reshape(-1)
        self.buffers[2][:]=x.numpy().reshape(32,4,48,8).transpose(0,2,1,3).reshape(-1)
        self.phases['host_input']+=time.perf_counter()-tick
        tick=time.perf_counter()
        for i in (1,2):
            self.dll.tdn_sync(self.handle,i,1,self.buffers[i].nbytes,0)
        ms=self.dll.tdn_run(self.handle)
        if ms<=0:
            raise RuntimeError(f'Incomplete full encoder dispatch: {ms}')
        self.dispatch_ms.append(ms)
        # Normal inference needs only the final layer. Keep full traces opt-in.
        layer_bytes=self.buffers[4].nbytes//self.layer_count
        self.dll.tdn_sync(self.handle,4,0,self.buffers[4].nbytes if trace or self.full_output else layer_bytes,
                          0 if trace or self.full_output else (self.layer_count-1)*layer_bytes)
        self.phases['sync_dispatch']+=time.perf_counter()-tick
        tick=time.perf_counter()
        raw=self.buffers[4].view(np.float32).reshape(self.layer_count,32,2048)
        selected=raw[:,:,:1536].copy() if self.full_output else (raw if trace else raw[-1:])
        layers=selected[:,:,:1536].reshape(len(selected),32,48,4,8).transpose(0,1,3,2,4).reshape(len(selected),128//t,t,384)[:,:b]
        final=torch.from_numpy(layers[-1].copy())
        mask=mask[:b]
        pooled=final[:,0] if m.pool=='cls' else (final*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
        with query_role(m,query):vector=F.normalize(m.out(pooled),dim=-1)
        self.phases['host_output']+=time.perf_counter()-tick
        if not torch.isfinite(vector).all():
            raise ValueError('Nonfinite fused encoder output')
        return (vector,[x[:b],*[torch.from_numpy(y.copy()) for y in layers[:-1]],final]) if trace else vector

    def profile(self,reset=False):
        p={'dispatches':len(self.dispatch_ms),'dispatch_sum_ms':sum(self.dispatch_ms),
           'dispatch_samples_ms':list(self.dispatch_ms),'host_phase_seconds':dict(self.phases)}
        if reset:
            self.dispatch_ms.clear()
            for k in self.phases:
                self.phases[k]=0.
        return p

    def close(self):
        if self.handle:
            self.dll.tdn_close(self.handle);self.handle=None
