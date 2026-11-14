"""Matched baseline gates; never used as training or checkpoint-selection data."""
import json,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'comparisons/qwen06b'))
from compare import data,evaluate

@torch.inference_mode()
def main():
 torch.set_num_threads(2);report={}
 for short,name in [('bge_micro','TaylorAI/bge-micro-v2'),('minilm','sentence-transformers/all-MiniLM-L6-v2')]:
  repo=PROJECT/'cache/hub'/('models--'+name.replace('/','--'))/'snapshots';path=next(repo.iterdir())
  model=AutoModel.from_pretrained(path,local_files_only=True).eval();tk=AutoTokenizer.from_pretrained(path,local_files_only=True)
  def encode(texts):
   v=[]
   for i in range(0,len(texts),32):
    x=tk(texts[i:i+32],padding=True,truncation=True,max_length=128,return_tensors='pt')
    h=model(**x).last_hidden_state;mask=x['attention_mask'][...,None]
    v.append(F.normalize((h*mask).sum(1)/mask.sum(1),dim=-1).numpy())
   return np.concatenate(v)
  row={'model':name,'revision':path.name,'max_length':128,'prompts':False,'tasks':{}}
  for task in ('SciFact','NFCorpus'):
   d=data(task);v=encode(d['queries']+d['texts']);nq=len(d['queries'])
   metrics,_=evaluate(v[:nq]@v[nq:].T,d);row['tasks'][task]=metrics
   print(short,task,metrics,flush=True)
  pairs=json.loads((PROJECT/'comparisons/qwen06b/data/stsb-test.json').read_text())
  v=encode([r['sentence1'] for r in pairs]+[r['sentence2'] for r in pairs]);n=len(pairs)
  row['stsbenchmark_spearman']=float(spearmanr([float(r['score']) for r in pairs],(v[:n]*v[n:]).sum(1)).statistic)*100
  row['mean_ndcg10']=float(np.mean([v['ndcg10'] for v in row['tasks'].values()]))
  report[short]=row;(ROOT/'baselines.json').write_text(json.dumps(report,indent=2)+'\n')
  print(short,json.dumps(row),flush=True)
if __name__=='__main__':main()
