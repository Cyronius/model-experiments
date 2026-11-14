"""Score the ternary CPU inference path against the matched Qwen results."""
import json,sys
import numpy as np,torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from compare import ROOT,PROJECT,data,evaluate
from diagnose import student,encode
from encoder import CHECKPOINT,digest

@torch.inference_mode()
def main():
 torch.set_num_threads(2)
 model=student(CHECKPOINT,'quant')
 tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 report={'checkpoint_sha256':digest(CHECKPOINT),'threads':2,'max_length':128,'device':'CPU','retrieval':{}}
 for name in ('SciFact','NFCorpus'):
  d=data(name);texts=d['queries']+d['texts']
  v=encode(model,tokenizer,texts,False,32)
  assert np.isfinite(v).all()
  nq=len(d['queries']);metrics,rows=evaluate(v[:nq]@v[nq:].T,d)
  report['retrieval'][name]=metrics
  print(name,json.dumps(metrics),flush=True)
 report['mean_ndcg10']=float(np.mean([v['ndcg10'] for v in report['retrieval'].values()]))
 pairs=json.loads((ROOT/'data/stsb-test.json').read_text())
 texts=[r['sentence1'] for r in pairs]+[r['sentence2'] for r in pairs]
 v=encode(model,tokenizer,texts,False,32);n=len(pairs)
 report['stsbenchmark_spearman']=float(spearmanr([float(r['score']) for r in pairs],np.sum(v[:n]*v[n:],axis=1)).statistic)*100
 (ROOT/'cpu-quality.json').write_text(json.dumps(report,indent=2)+'\n')
 print('CPU_QUALITY',json.dumps(report),flush=True)
if __name__=='__main__':main()
