"""Experimental trainable ternary row scales with ordinary ternary inference export.

The deployed matrices still have {-1,0,+1} codes and one BF16 scale per row.
Training-only log scales can adapt independently of master-weight absmean.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tern-embed'))
from model import TernLinear,tern_codes,quant_act,ste

class LearnedScaleLinear(TernLinear):
 def scale(self):
  raw=self.log_scale.exp().clamp(min=1e-5)
  return ste(raw,raw.to(torch.bfloat16).float())
 def forward(self,x):
  if not self.quant:raise ValueError('Learned row scales require ternary mode')
  codes,_=tern_codes(self.weight);scale=self.scale()
  # Identity master-weight STE plus a separate scale gradient; latter is zero in forward.
  effective=ste(self.weight,codes*scale)+codes*(scale-scale.detach())
  return F.linear(quant_act(x),effective,self.bias)

def install(model):
 if not model.quantized:raise ValueError('Do not reinterpret float weights as a trained ternary model')
 for layer in model.layers:
  for name in ('q','k','v','o','fc1','fc2'):
   old=getattr(layer,name)
   if isinstance(old,LearnedScaleLinear):continue
   devices=[old.weight.device.index] if old.weight.is_cuda else []
   # Constructor initialization is overwritten; do not perturb an auxiliary
   # head's shared random initialization in a matched control experiment.
   with torch.random.fork_rng(devices=devices):new=LearnedScaleLinear(old.in_features,old.out_features,bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype)
   new.load_state_dict(old.state_dict());new.quant=True
   _,scale=tern_codes(old.weight);new.log_scale=torch.nn.Parameter(scale.log())
   setattr(layer,name,new)
 return model

@torch.no_grad()
def inference_state(model):
 state={k:v for k,v in model.state_dict().items() if not k.endswith('.log_scale')}
 for prefix,layer in model.named_modules():
  if not isinstance(layer,LearnedScaleLinear):continue
  codes,_=tern_codes(layer.weight);fraction=codes.abs().mean(1,keepdim=True)
  # Absmean of this canonical master matrix encodes the independently trained scale.
  weight=codes*(layer.scale().detach()/fraction.clamp(min=1e-12))
  check_codes,check_scale=tern_codes(weight)
  if not torch.equal(check_codes,codes):raise ValueError('Ternary export changed codes')
  active=fraction[:,0]>0
  if not torch.equal(check_scale[active],layer.scale().detach()[active]):raise ValueError('Ternary export changed row scales')
  state[prefix+'.weight']=weight
 return state

def load_training(checkpoint,resume=None):
 from model import Student
 model=Student.load(checkpoint);install(model)
 if resume:
  saved=torch.load(resume,map_location='cpu',weights_only=False)
  if 'training_state' not in saved.get('resume',{}):raise ValueError('Learned-scale resume requires original master weights and log scales')
  model.load_state_dict(saved['resume']['training_state'])
 return model

def optimizer(model,lr,multiplier):
 named=list(model.named_parameters());normal=[v for k,v in named if not k.endswith('.log_scale')];scales=[v for k,v in named if k.endswith('.log_scale')]
 if not scales or multiplier<=0:raise ValueError('Expected learned row scales and positive LR multiplier')
 return torch.optim.AdamW([{'params':normal},{'params':scales,'lr_multiplier':multiplier,'weight_decay':0.}],lr=lr,weight_decay=.01)

def scales_only(model):
 model.requires_grad_(False);count=0
 for name,param in model.named_parameters():
  if name.endswith('.log_scale'):param.requires_grad_(True);count+=param.numel()
 if not count:raise ValueError('No learned row scales installed')
 return count
