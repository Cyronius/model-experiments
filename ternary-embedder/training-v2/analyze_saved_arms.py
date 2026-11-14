"""Post-hoc diagnostic, not independent acceptance or checkpoint promotion."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'analysis'
sys.path.insert(0,str(PROJECT/'comparisons/qwen06b'))
from compare import data
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student,encode,evaluate
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',type=int,default=0);a=p.parse_args();torch.set_num_threads(2)
 runs=ROOT/'runs/balanced-round2/ternary-embedder/training-v2/runs'
 paths=sorted(runs.glob('balanced-*/best.pt'))
 selected=json.loads((ROOT/'round2/frozen-candidate.json').read_text())['run']
 paths += [runs/selected/'model.pt',runs/'balanced-prompted-anchor1.0-positive.2/model.pt']
 paths=paths[a.group::2];tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 result={'scope':'Post-hoc development diagnosis; no promotion; test scores now observed','models':{}}
 for path in paths:
  name=path.parent.name+'/'+path.name;ck=torch.load(path,map_location='cpu',weights_only=False);m=student(path,'quant')
  row={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'step':ck['meta'].get('step',ck['meta'].get('steps')),'tasks':{}}
  for task in ('SciFact','NFCorpus'):
   d=data(task);v=encode(m,tk,d['queries']+d['texts'],False,32);n=len(d['queries']);metrics,_=evaluate(v[:n]@v[n:].T,d)
   row['tasks'][task]=metrics;print(name,task,metrics['ndcg10'],flush=True)
  row['mean']=float(np.mean([v['ndcg10'] for v in row['tasks'].values()]));result['models'][name]=row
  (OUT/f'arms-{a.group}.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
