"""Compare reference models on the same mixed TRAIN-source holdout, not test tasks."""
import json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from rank_distill import PROJECT,ROOT,split_queries,sha
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student

@torch.inference_mode()
def main():
 torch.set_num_threads(2)
 path=ROOT/'work/round2/balanced-refined.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
 train,valid,held=split_queries(rows,np.random.default_rng(0));report={'data_sha256':sha(path),'train_rows':len(train),'validation_rows':len(valid),'models':{}}
 models=[('original',PROJECT/'tern-embed/pulls/grid1/runs/2a_384/model.pt'),('bge_micro','TaylorAI/bge-micro-v2'),('minilm','sentence-transformers/all-MiniLM-L6-v2')]
 for name,source in models:
  is_student=isinstance(source,Path)
  if is_student:m=student(source,'quant');tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
  else:
   repo=PROJECT/'cache/hub'/('models--'+source.replace('/','--'))/'snapshots';p=next(repo.iterdir())
   m=AutoModel.from_pretrained(p,local_files_only=True).eval();tk=AutoTokenizer.from_pretrained(p,local_files_only=True)
  ranks=[];by_source={}
  for start in range(0,len(valid),32):
   b=valid[start:start+32]
   if len(b)<2:continue
   docs=list(dict.fromkeys([rows[i]['positive'] for i in b]+[rows[i]['negative'] for i in b]))
   x=tk([rows[i]['query'] for i in b]+docs,padding=True,truncation=True,max_length=128,return_tensors='pt')
   if is_student:v=m(x['input_ids'],x['attention_mask'].bool())[2]
   else:
    h=m(**x).last_hidden_state;mask=x['attention_mask'][...,None];v=F.normalize((h*mask).sum(1)/mask.sum(1),dim=-1)
   scores=v[:len(b)]@v[len(b):].T;labels=torch.tensor([docs.index(rows[i]['positive']) for i in b])
   rr=(1+(scores>scores.gather(1,labels[:,None])).sum(1)).tolist();ranks.extend(rr)
   for i,r in zip(b,rr):by_source.setdefault(rows[i]['source'],[]).append(r)
  def metrics(r):return {'mrr':float(np.mean(1/np.array(r))),'recall1':float(np.mean(np.array(r)==1)),'queries':len(r)}
  report['models'][name]={**metrics(ranks),'by_source':{s:metrics(r) for s,r in by_source.items()}}
  (ROOT/'round2/validation-baselines.json').write_text(json.dumps(report,indent=2)+'\n')
  print(name,json.dumps(report['models'][name]),flush=True)
if __name__=='__main__':main()
