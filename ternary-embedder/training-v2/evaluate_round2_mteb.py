"""Run the historical 12-task MTEB harness on the frozen winner and references."""
import argparse,importlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'tern-embed'))
from model import Student,TernLinear,tern_codes,ste,quant_act
from query_head import RoleOutput,install_query_head,query_role
from evaluate import TASKS
from transformers import AutoTokenizer

class FixedQuantizedLinear(torch.nn.Module):
 def __init__(self,layer):
  super().__init__();codes,scale=tern_codes(layer.weight)
  self.register_buffer('weight',ste(layer.weight,codes*scale).detach().clone())
  self.register_buffer('bias',layer.bias.detach().clone())
 def forward(self,x):return F.linear(quant_act(x),self.weight,self.bias)

class Encoder:
 def __init__(self,path,precision=None):
  if precision in ('q4','q8'):
   from export_pair_precision import load_reference
   self.model=load_reference(Path(path))
  else:self.model=Student.load(Path(path)).eval()
  self.tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);self.cache={}
  for layer in self.model.layers:
   for name in ('q','k','v','o','fc1','fc2'):
    linear=getattr(layer,name)
    if getattr(linear,'quant',False):setattr(layer,name,FixedQuantizedLinear(linear))
  install_query_head(self.model,path)
 @torch.inference_mode()
 def encode(self,sentences,batch_size=64,**kwargs):
  prompt=kwargs.get('prompt_type');is_query=(kwargs.get('is_query',False) or getattr(prompt,'value',prompt)=='query') and isinstance(self.model.out,RoleOutput)
  sentences=list(sentences);missing=list(dict.fromkeys(s for s in sentences if (is_query,s) not in self.cache))
  tokens=self.tokenizer(missing,truncation=True,max_length=128)['input_ids'] if missing else []
  order=sorted(range(len(missing)),key=lambda i:len(tokens[i]))
  for start in range(0,len(order),min(batch_size,64)):
   idx=order[start:start+min(batch_size,64)];enc=self.tokenizer.pad({'input_ids':[tokens[i] for i in idx]},padding=True,return_tensors='pt')
   with query_role(self.model,is_query):vectors=self.model(enc['input_ids'],enc['attention_mask'].bool())[2].numpy()
   for i,v in zip(idx,vectors):self.cache[is_query,missing[i]]=v.copy()
  return np.stack([self.cache[is_query,s] for s in sentences]) if sentences else np.empty((0,384),np.float32)
 def encode_queries(self,sentences,**kwargs):return self.encode(sentences,**dict(kwargs,prompt_type='query'))
 def encode_documents(self,sentences,**kwargs):return self.encode(sentences,**dict(kwargs,prompt_type='document'))

def main():
 import mteb
 from sentence_transformers import SentenceTransformer
 if os.environ.get('HF_DATASETS_OFFLINE')=='1':
  # datasets2.21 cannot infer the qrels default when corpus/queries/default
  # configs coexist offline. MTEB supplies names for corpus and queries only.
  retrieval=importlib.import_module('mteb.abstasks.AbsTaskRetrieval');original_load=retrieval.load_dataset
  def cached_retrieval_load(path,*args,**kwargs):
   if path in ('mteb/scifact','mteb/nfcorpus') and not args and kwargs.get('name') is None:kwargs['name']='default'
   return original_load(path,*args,**kwargs)
  retrieval.load_dataset=cached_retrieval_load
 p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['candidate']);p.add_argument('--tasks',nargs='+',default=list(TASKS));p.add_argument('--freeze',type=Path,default=ROOT/'round2/frozen-candidate.json');p.add_argument('--output',type=Path,default=ROOT/'round2/mteb');p.add_argument('--branch');a=p.parse_args()
 torch.set_num_threads(2);freeze=json.loads(a.freeze.read_text());freeze=freeze['models'][a.branch] if a.branch else freeze;checkpoint=Path(freeze['path'])
 import hashlib
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==freeze['sha256']
 refs={'candidate':checkpoint,'original':PROJECT/'tern-embed/pulls/grid1/runs/2a_384/model.pt',
       'bge_micro':PROJECT/'cache/hub/models--TaylorAI--bge-micro-v2/snapshots/3edf6d7de0faa426b09780416fe61009f26ae589',
       'minilm':PROJECT/'cache/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'}
 tasks=mteb.get_tasks(tasks=a.tasks,languages=['eng'])
 for name in a.models:
  model=Encoder(refs[name],freeze.get('precision') if name=='candidate' else 'ternary') if name in ('candidate','original') else SentenceTransformer(str(refs[name]),device='cpu')
  if name not in ('candidate','original'):model.max_seq_length=128
  mteb.MTEB(tasks=tasks).run(model,output_folder=str(a.output/name),eval_splits=['test'],encode_kwargs={'batch_size':64},verbosity=1)
if __name__=='__main__':main()
