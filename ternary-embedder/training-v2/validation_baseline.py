"""Baseline MRR on the exact public-train query-group holdout used by the pilot."""
import json,sys
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student

@torch.inference_mode()
def main():
 torch.set_num_threads(2);path=ROOT/'work/recovered-targets/pilot.jsonl'
 rows=[json.loads(l) for l in path.read_text().splitlines()]
 groups=sorted(set(r['query'] for r in rows));np.random.default_rng(0).shuffle(groups)
 held=set(groups[:max(2,len(groups)//20)]);valid=[i for i,r in enumerate(rows) if r['query'] in held]
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);out={}
 for name,checkpoint,mode in [('float_control',PROJECT/'retrieval-diagnostics/float-control/runs/float150/model.pt','float'),
                              ('original_qat',PROJECT/'tern-embed/pulls/grid1/runs/2a_384/model.pt','quant')]:
  model=student(checkpoint,mode);ranks=[]
  for j in range(0,len(valid),32):
   b=valid[j:j+32];docs=list(dict.fromkeys([rows[i]['positive'] for i in b]+[rows[i]['negative'] for i in b]))
   enc=tk([rows[i]['query'] for i in b]+docs,padding=True,truncation=True,max_length=128,return_tensors='pt')
   vectors=model(enc['input_ids'],enc['attention_mask'].bool())[2];scores=vectors[:len(b)]@vectors[len(b):].T
   labels=torch.tensor([docs.index(rows[i]['positive']) for i in b]);positive=scores.gather(1,labels[:,None])
   ranks.extend((1+(scores>positive).sum(1)).tolist())
  out[name]={'mrr':float(np.mean(1/np.array(ranks))),'recall1':float(np.mean(np.array(ranks)==1)),'queries':len(ranks)}
 (ROOT/'validation-baselines.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
