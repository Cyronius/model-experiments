"""Evaluate a frozen round-2 winner against both targets, including reserved ArguAna."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'round2'
sys.path.insert(0,str(PROJECT/'comparisons/qwen06b'))
from compare import data
from round2_metrics import evaluate
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student,paired

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,default=OUT/'frozen-candidate.json');a=p.parse_args()
 frozen=json.loads(a.freeze.read_text());checkpoint=Path(frozen['path']);assert sha(checkpoint)==frozen['sha256']
 OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
 models=[('candidate',checkpoint),('original',PROJECT/'tern-embed/pulls/grid1/runs/2a_384/model.pt'),
         ('bge_micro','TaylorAI/bge-micro-v2'),('minilm','sentence-transformers/all-MiniLM-L6-v2')]
 results={'selection':frozen,'cap':128,'cpu_threads':2,'models':{}}
 all_rows={}
 for name,source in models:
  is_student=isinstance(source,Path)
  if is_student:
   m=student(source,'quant');tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
   identity=sha(source)
  else:
   repo=PROJECT/'cache/hub'/('models--'+source.replace('/','--'))/'snapshots';path=next(repo.iterdir())
   m=AutoModel.from_pretrained(path,local_files_only=True).eval();tk=AutoTokenizer.from_pretrained(path,local_files_only=True);identity=path.name
  def encode(texts):
   # Bucket by token length, preserving the original text order in the output.
   tokens=tk(texts,truncation=True,max_length=128)['input_ids'];order=np.argsort([len(t) for t in tokens],kind='stable')
   vectors=np.empty((len(texts),384),np.float32)
   for i in range(0,len(order),32):
    indexes=order[i:i+32];enc=tk.pad({'input_ids':[tokens[j] for j in indexes]},padding=True,return_tensors='pt')
    if is_student:v=m(enc['input_ids'],enc['attention_mask'].bool())[2]
    else:
     h=m(**enc).last_hidden_state;mask=enc['attention_mask'][...,None]
     v=F.normalize((h*mask).sum(1)/mask.sum(1),dim=-1)
    assert torch.isfinite(v).all();vectors[indexes]=v.numpy()
   return vectors
  row={'identity':identity,'tasks':{}};all_rows[name]={}
  for task in ('SciFact','NFCorpus','ArguAna'):
   d=data(task);texts=d['queries']+d['texts'];v=encode(texts);nq=len(d['queries'])
   metrics,rows=evaluate(v[:nq]@v[nq:].T,d);row['tasks'][task]=metrics;all_rows[name][task]=rows
   print(name,task,json.dumps(metrics),flush=True)
  pairs=json.loads((PROJECT/'comparisons/qwen06b/data/stsb-test.json').read_text())
  v=encode([r['sentence1'] for r in pairs]+[r['sentence2'] for r in pairs]);n=len(pairs)
  row['stsbenchmark_spearman']=float(spearmanr([float(r['score']) for r in pairs],(v[:n]*v[n:]).sum(1)).statistic)*100
  row['mean_core_retrieval']=float(np.mean([row['tasks'][task]['ndcg10'] for task in ('SciFact','NFCorpus')]))
  results['models'][name]=row;(OUT/'quality.json').write_text(json.dumps(results,indent=2)+'\n')
 candidate=results['models']['candidate'];results['gates']={};results['paired']={}
 for ref in ('original','bge_micro','minilm'):
  baseline=results['models'][ref]
  results['gates'][ref]={'each_core_retrieval':all(candidate['tasks'][t]['ndcg10']>baseline['tasks'][t]['ndcg10'] for t in ('SciFact','NFCorpus')),
                       'stsbenchmark':candidate['stsbenchmark_spearman']>baseline['stsbenchmark_spearman'],
                       'reserved_arguana':candidate['tasks']['ArguAna']['ndcg10']>baseline['tasks']['ArguAna']['ndcg10']}
  results['paired'][ref]={t:paired(all_rows[ref][t],all_rows['candidate'][t]) for t in ('SciFact','NFCorpus','ArguAna')}
 (OUT/'query-rows.json').write_text(json.dumps(all_rows)+'\n');(OUT/'quality.json').write_text(json.dumps(results,indent=2)+'\n')
 print('RESULTS',json.dumps(results),flush=True)
if __name__=='__main__':main()
