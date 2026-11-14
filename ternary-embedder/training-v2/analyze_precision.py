"""Local accuracy-only W4/W8 simulation from the independent float checkpoint.

Symmetric 64-weight groups, absmax scale rounded to bf16; embedding tables,
LayerNorms, biases and output projection stay float. Dense layers use the same
activation quantizer as ternary. Dequantized F.linear is not a speed benchmark.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'analysis'
sys.path.insert(0,str(PROJECT/'comparisons/qwen06b'))
from compare import data
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student,encode,evaluate
from model import quant_act,TernLinear
class FixedLowBit(torch.nn.Module):
 def __init__(self,layer,bits,activation):
  super().__init__();limit=2**(bits-1)-1;w=layer.weight.detach();blocks=w.reshape(w.shape[0],-1,64)
  scale=(blocks.abs().amax(-1,keepdim=True)/limit).clamp_min(1e-8).to(torch.bfloat16).float()
  code=(blocks/scale).round().clamp(-limit,limit);self.register_buffer('weight',(code*scale).reshape_as(w));self.register_buffer('bias',layer.bias.detach().clone());self.activation=activation
 def forward(self,x):return F.linear(quant_act(x) if self.activation else x,self.weight,self.bias)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--bits',type=int,choices=(0,4,8),required=True);p.add_argument('--float-activations',action='store_true');p.add_argument('--checkpoint',type=Path,default=PROJECT/'retrieval-diagnostics/float-control/runs/float150/model.pt');p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=True)
 path=a.checkpoint;name='float' if a.bits==0 else f'w{a.bits}a'+('float' if a.float_activations else '8')
 model=student(path,'float') if a.bits==0 else None
 if a.bits:
  from model import Student
  model=Student.load(path).eval()
  for layer in model.layers:
   for n in ('q','k','v','o','fc1','fc2'):setattr(layer,n,FixedLowBit(getattr(layer,n),a.bits,not a.float_activations))
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 result={'name':name,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bits':a.bits,'weight_group_size':64,'scale':'bf16 absmax','scope':'Accuracy-only simulation; no native low-bit runtime or speed measurement','tasks':{}}
 for task in ('SciFact','NFCorpus'):
  d=data(task);v=encode(model,tk,d['queries']+d['texts'],False,32);n=len(d['queries']);metrics,rows=evaluate(v[:n]@v[n:].T,d);result['tasks'][task]=metrics
  (a.out/f'precision-{name}-{task}-rows.json').write_text(json.dumps(rows)+'\n');print(name,task,metrics['ndcg10'],flush=True)
 pairs=json.loads((PROJECT/'comparisons/qwen06b/data/stsb-test.json').read_text());v=encode(model,tk,[r['sentence1'] for r in pairs]+[r['sentence2'] for r in pairs],False,32);n=len(pairs)
 result['stsbenchmark_spearman']=float(spearmanr([float(r['score']) for r in pairs],(v[:n]*v[n:]).sum(1)).statistic)*100
 result['mean']=float(np.mean([v['ndcg10'] for v in result['tasks'].values()]));(a.out/f'precision-{name}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
