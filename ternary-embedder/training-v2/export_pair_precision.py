"""Export the frozen float branch with packed W4/W8 dense matrices.

Accuracy/reference format only; no native low-bit kernel or speed claim.
Embedding tables, output projection, biases and norms remain float32.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'tern-embed'))
from model import Student,quant_act
from pair_training_state import digest,atomic_torch_save

class ReferenceLinear(torch.nn.Module):
 def __init__(self,weight,bias):
  super().__init__();self.register_buffer('weight',weight);self.register_buffer('bias',bias)
 def forward(self,x):return torch.nn.functional.linear(quant_act(x),self.weight,self.bias)

def pack_weight(weight,bits,scale_method='maxabs',group_size=64):
 if group_size not in (16,32,64) or weight.shape[1]%group_size:raise ValueError('Unsupported weight group')
 limit=2**(bits-1)-1;blocks=weight.detach().reshape(weight.shape[0],-1,group_size)
 scales=(blocks.abs().amax(-1,keepdim=True)/limit).clamp_min(1e-8).to(torch.bfloat16)
 if scale_method=='mse-grid':
  original=scales;error=torch.full_like(scales.float(),float('inf'))
  for fraction in torch.linspace(.5,1.,21):
   candidate=(original.float()*fraction).clamp_min(1e-8).to(torch.bfloat16)
   reconstructed=(blocks/candidate.float()).round().clamp(-limit,limit)*candidate.float()
   candidate_error=(reconstructed-blocks).square().mean(-1,keepdim=True)
   better=candidate_error<error;scales=torch.where(better,candidate,scales);error=torch.minimum(error,candidate_error)
 elif scale_method!='maxabs':raise ValueError('Unknown scale method')
 codes=(blocks/scales.float()).round().clamp(-limit,limit).to(torch.int8).flatten()
 if bits==4:
  u=codes.to(torch.int16).bitwise_and(15).to(torch.uint8);packed=u[::2]|(u[1::2]<<4)
 else:packed=codes.view(torch.uint8).clone()
 return {'shape':list(weight.shape),'bits':bits,'group':group_size,'scales':scales,'packed':packed}

def unpack_weight(value):
 if value['bits']==4:
  p=value['packed'];u=torch.stack((p&15,p>>4),dim=1).flatten().to(torch.int16);codes=torch.where(u>=8,u-16,u).float()
 else:codes=value['packed'].view(torch.int8).float()
 shape=value['shape'];return (codes.reshape(shape[0],-1,value['group'])*value['scales'].float()).reshape(shape)

def load_reference(path):
 blob=torch.load(path,map_location='cpu',weights_only=False);model=Student(**blob['dims']);state=dict(blob['float_state'])
 for name,value in blob['dense'].items():state[name]=unpack_weight(value)
 model.load_state_dict(state);model.eval()
 for layer in model.layers:
  for name in ('q','k','v','o','fc1','fc2'):
   old=getattr(layer,name);setattr(layer,name,ReferenceLinear(old.weight.detach(),old.bias.detach()))
 return model

def export(path,out,bits,scale_method='maxabs',group_size=64):
 source=Student.load(path).eval()
 if source.quantized:raise ValueError('Int4/int8 must derive from the independent float branch')
 state=source.state_dict();dense={}
 for i,layer in enumerate(source.layers):
  for n in ('q','k','v','o','fc1','fc2'):
   key=f'layers.{i}.{n}.weight';dense[key]=pack_weight(state.pop(key),bits,scale_method,group_size)
 blob={'format':'ternary-embedder-dense-lowbit-v1','dims':source.dims,'float_state':state,'dense':dense,'metadata':{'source_sha256':digest(path),'dense_weight_bits':bits,'activation_bits':8,'scale_dtype':'bfloat16','group_size':group_size,'weight_scale_method':scale_method,'scope':'Packed accuracy/reference artifact; native kernel performance unvalidated'}}
 projection=torch.load(path,map_location='cpu',weights_only=False).get('meta',{}).get('query_projection')
 if projection is not None:blob['metadata']['query_projection']=projection
 atomic_torch_save(out,blob);return {'path':str(out),'sha256':digest(out),'bytes':out.stat().st_size,'source_sha256':digest(path),'bits':bits}

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True);report=[]
 for bits in (4,8):report.append(export(a.checkpoint,a.out/f'model-w{bits}a8.pt',bits))
 (a.out/'precision-exports.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
