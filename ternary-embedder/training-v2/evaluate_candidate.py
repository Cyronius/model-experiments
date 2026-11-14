"""One-time acceptance evaluation of a validation-selected frozen checkpoint."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'comparisons/qwen06b'))
from compare import data,evaluate
from diagnose import student,encode,paired

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--out',type=Path,default=ROOT/'candidate-results.json');a=p.parse_args()
 freeze=json.loads((ROOT/'frozen-candidate.json').read_text())
 if sha(a.checkpoint)!=freeze['sha256']:raise ValueError('Candidate was not frozen before evaluation')
 torch.set_num_threads(2);model=student(a.checkpoint,'quant')
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 result={'checkpoint_sha256':sha(a.checkpoint),'selection':freeze,'tasks':{},'device':'CPU','cap':128}
 baseline=json.loads((ROOT/'baselines.json').read_text())
 original=json.loads((PROJECT/'comparisons/qwen06b/cpu-quality.json').read_text())
 for name in ('SciFact','NFCorpus'):
  d=data(name);v=encode(model,tk,d['queries']+d['texts'],False,32);nq=len(d['queries'])
  assert np.isfinite(v).all()
  metrics,rows=evaluate(v[:nq]@v[nq:].T,d);result['tasks'][name]=metrics
  reference=json.loads((PROJECT/f'retrieval-diagnostics/outputs/{name}_per_query.json').read_text())['minilm']
  assert abs(np.mean([r['ndcg10'] for r in reference])*100-baseline['minilm']['tasks'][name]['ndcg10'])<1e-3
  result['tasks'][name]['paired_vs_minilm']=paired(reference,rows)
  a.out.with_name(name+'-candidate-rows.json').write_text(json.dumps(rows,indent=2)+'\n')
  print(name,metrics,flush=True)
 pairs=json.loads((PROJECT/'comparisons/qwen06b/data/stsb-test.json').read_text())
 v=encode(model,tk,[r['sentence1'] for r in pairs]+[r['sentence2'] for r in pairs],False,32);n=len(pairs)
 result['stsbenchmark_spearman']=float(spearmanr([float(r['score']) for r in pairs],(v[:n]*v[n:]).sum(1)).statistic)*100
 result['mean_ndcg10']=float(np.mean([r['ndcg10'] for r in result['tasks'].values()]))
 result['delta_vs_original']={'mean_retrieval':result['mean_ndcg10']-original['mean_ndcg10'],
                              'stsbenchmark':result['stsbenchmark_spearman']-original['stsbenchmark_spearman']}
 result['passes_each_reference']={name:all(result['tasks'][t]['ndcg10']>ref['tasks'][t]['ndcg10'] for t in ('SciFact','NFCorpus'))
                                     and result['stsbenchmark_spearman']>ref['stsbenchmark_spearman'] for name,ref in baseline.items()}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
