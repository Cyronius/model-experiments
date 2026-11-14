"""Grow a trained two-layer checkpoint to three layers with a training-only ramp.

The extra block clones the last trained block. During the first updates the
trainer interpolates its output from identity to the ordinary block. At alpha=1
exports use the unchanged three-layer Student format, with no inference gate.
"""
import argparse,copy,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'tern-embed'))
from model import Student,Layer
from pair_training_state import atomic_torch_save,digest

class GrowingLayer(Layer):
 alpha=0.
 def forward(self,x,mask):
  y,att=super().forward(x,mask)
  if self.alpha==1.:return y,att
  return x+self.alpha*(y-x),att

def install_growth(model):
 if model.dims['layers']!=3:raise ValueError('Depth growth requires a three-layer initialization')
 old=model.layers[-1];d=model.dims;quant=model.quantized
 new=GrowingLayer(d['d'],d['heads'],d['ffn'],d['eps'],d['gelu']);new.load_state_dict(old.state_dict());new=new.to(device=next(old.parameters()).device,dtype=next(old.parameters()).dtype)
 model.layers[-1]=new;model.set_quant(quant)

def set_growth(model,alpha):
 for layer in model.layers:
  if isinstance(layer,GrowingLayer):layer.alpha=float(alpha)

def grow(source,out,warmup=500):
 original=Student.load(source).eval()
 if original.dims['layers']!=2:raise ValueError('Expected the selected two-layer checkpoint')
 dims=dict(original.dims,layers=3);new=Student(**dims);state=new.state_dict();state.update(original.state_dict())
 for name,tensor in original.layers[-1].state_dict().items():state['layers.2.'+name]=tensor.clone()
 new.load_state_dict(state);new.set_quant(original.quantized)
 meta={'initialization_only':True,'depth_growth':{'source_sha256':digest(source),'cloned_layer':1,'new_layer':2,'warmup_steps':warmup,'training_alpha_starts_at':0.,'inference_alpha_after_training':1.}}
 atomic_torch_save(out,{'dims':new.dims,'state':new.state_dict(),'quant':new.quantized,'meta':meta})
 return {'path':str(out),'sha256':digest(out),'source_sha256':digest(source),'parameters':sum(p.numel() for p in new.parameters()),'warmup_steps':warmup}

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--warmup',type=int,default=500);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
 print(json.dumps(grow(a.source,a.out,a.warmup),indent=2))
if __name__=='__main__':main()
