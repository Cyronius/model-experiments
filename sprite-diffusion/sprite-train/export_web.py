"""Export inference-only EMA tensors and fixed-batch CFG ONNX; verify CPU parity."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
import dit

p=argparse.ArgumentParser(); p.add_argument('--ckpt',required=True);p.add_argument('--out',required=True);p.add_argument('--name',required=True);a=p.parse_args()
torch.set_num_threads(1)
# Equivalent RMSNorm expression supported by the portable ONNX operators.
def rms(self,x):
    return x * torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6) * self.w
dit.RMSNorm.forward=rms
s=torch.load(a.ckpt,map_location='cpu',weights_only=False);args=s['args']
m=dit.DiT(args['hidden'],args['depth'],args['patch']).eval();m.load_state_dict(s['ema'])
root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
# An explicit tensor-only allowlist: no args, optimizer, sampler state or training data.
save_file({k:v.detach().contiguous() for k,v in m.state_dict().items()},str(root/(a.name+'.safetensors')),metadata={'format':'pt','weights':'EMA'})
g=torch.Generator().manual_seed(142)
x=torch.randn((2,4,32,32),generator=g);t=torch.full((2,),0.5);cat=torch.tensor([3,dit.NCAT]);fp=torch.full((2,),dit.NFP,dtype=torch.int64)
path=root/(a.name+'.onnx')
torch.onnx.export(m,(x,t,cat,fp),str(path),input_names=['x','t','category','footprint'],output_names=['velocity'],opset_version=17,dynamo=False,external_data=False)
import onnx,onnxruntime as ort
onnx.checker.check_model(str(path))
opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
session=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
errors=[]
with torch.inference_mode():
 for category,time in [(0,.97),(3,.5),(7,.03)]:
  cat=torch.tensor([category,dit.NCAT]);t=torch.full((2,),time)
  ref=m(x,t,cat,fp).numpy();out=session.run(None,{'x':x.numpy(),'t':t.numpy(),'category':cat.numpy(),'footprint':fp.numpy()})[0]
  rel=float(np.linalg.norm(out-ref)/np.linalg.norm(ref));errors.append(rel)
  assert rel<1e-4,(a.name,rel)
 # End-to-end Euler parity, one category, fixed noise and 16 steps.
 original=torch.randn((1,4,32,32),generator=g);tx=original.clone();ox=original.numpy().copy();cat=torch.tensor([3,dit.NCAT])
 for i in range(16):
  t=torch.full((2,),1-i/16)
  tv=m(tx.repeat(2,1,1,1),t,cat,fp);tx=tx-(tv[1:]+3*(tv[:1]-tv[1:]))/16
  ov=session.run(None,{'x':np.repeat(ox,2,axis=0),'t':t.numpy(),'category':cat.numpy(),'footprint':fp.numpy()})[0];ox=ox-(ov[1:]+3*(ov[:1]-ov[1:]))/16
 final_rel=float(np.linalg.norm(ox-tx.numpy())/np.linalg.norm(tx.numpy()));assert final_rel<1e-3,final_rel
report={'name':a.name,'architecture':{'hidden':args['hidden'],'depth':args['depth'],'patch':args['patch'],'channels':4,'canvas':32},'weights':'EMA','cfg_batch':2,'dtype':'float32','opset':17,'velocity_relative_errors':errors,'euler_16_relative_error':final_rel,'files':{}}
for f in [path,root/(a.name+'.safetensors')]:report['files'][f.name]={'bytes':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
(root/(a.name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
