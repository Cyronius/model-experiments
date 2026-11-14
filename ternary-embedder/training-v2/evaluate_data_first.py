"""Local external-development scores for frozen data-first float/ternary branches."""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'data-first/results'
sys.path.insert(0,str(PROJECT/'comparisons/qwen06b'))
from compare import data
from round2_metrics import evaluate
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student,encode,paired
from pair_training_state import digest
from query_head import RoleOutput,install_query_head,query_role
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--freeze',type=Path,default=OUT/'frozen.json');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=True)
 frozen=json.loads(a.freeze.read_text());selection=frozen['models'][a.model];path=Path(selection['path']);assert digest(path)==selection['sha256']
 precision=selection.get('precision',a.model)
 if precision in ('q4','q8'):
  from export_pair_precision import load_reference
  m=load_reference(path)
 else:m=student(path,'quant' if precision=='ternary' else 'float')
 install_query_head(m,path)
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 result={'selection':selection,'token_cap':128,'tasks':{},'scope':'Observed external development tasks; reserved FiQA/SCIDOCS labels not used'};rows={}
 for task in ('SciFact','NFCorpus','ArguAna'):
  d=data(task);n=len(d['queries'])
  if isinstance(m.out,RoleOutput):
   with query_role(m,True):queries=encode(m,tk,d['queries'],False,32)
   with query_role(m,False):documents=encode(m,tk,d['texts'],False,32)
   v=np.concatenate((queries,documents))
  else:v=encode(m,tk,d['queries']+d['texts'],False,32)
  metrics,rr=evaluate(v[:n]@v[n:].T,d);result['tasks'][task]=metrics;rows[task]=rr;print(a.model,task,metrics['ndcg10'],flush=True)
  (a.output/(a.model+'-rows.json')).write_text(json.dumps(rows)+'\n')
 pairs=json.loads((PROJECT/'comparisons/qwen06b/data/stsb-test.json').read_text());v=encode(m,tk,[r['sentence1'] for r in pairs]+[r['sentence2'] for r in pairs],False,32);n=len(pairs)
 result['stsbenchmark_spearman']=float(spearmanr([float(r['score']) for r in pairs],(v[:n]*v[n:]).sum(1)).statistic)*100
 result['mean_core_retrieval']=float(np.mean([result['tasks'][t]['ndcg10'] for t in ('SciFact','NFCorpus')]))
 refpath=ROOT/'round2/quality.json';ref=json.loads(refpath.read_text());refrows=json.loads((ROOT/'round2/query-rows.json').read_text());result['reference_report_sha256']=digest(refpath);result['paired']={};result['gates']={}
 for name in ('original','bge_micro','minilm'):
  reference=ref['models'][name];result['paired'][name]={t:paired(refrows[name][t],rows[t]) for t in rows};result['gates'][name]={'each_core_retrieval':all(result['tasks'][t]['ndcg10']>reference['tasks'][t]['ndcg10'] for t in ('SciFact','NFCorpus')),'stsbenchmark':result['stsbenchmark_spearman']>reference['stsbenchmark_spearman']}
 (a.output/(a.model+'-quality.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
