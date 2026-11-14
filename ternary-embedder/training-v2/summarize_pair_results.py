"""Consolidate the frozen data-first precision family without selecting on tests."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data-first/results'
def interval(left,right):
 rng=np.random.default_rng(0);deltas=[];boots=[]
 for task in ('SciFact','NFCorpus'):
  a,b=left[task],right[task];assert [r['query_id'] for r in a]==[r['query_id'] for r in b]
  d=np.array([y['ndcg10']-x['ndcg10'] for x,y in zip(a,b)]);deltas.append(d.mean()*100);boots.append(d[rng.integers(0,len(d),(4000,len(d)))].mean(1)*100)
 return {'mean_delta_ndcg10':float(np.mean(deltas)),'paired_query_bootstrap_95_ci':np.percentile(np.mean(boots,axis=0),[2.5,97.5]).tolist(),'scope':'Query sampling only; not training-seed variation'}
def read(path):return json.loads(Path(path).read_text())
def main():
 baseline=read(ROOT/'round2/quality.json');oldrows=read(ROOT/'round2/query-rows.json');result={'scope':'Frozen checkpoint external development scores at 128 tokens; no broad MTEB superiority established','resources':read(OUT/'resources.json'),'frozen':read(OUT/'frozen.json'),'models':{},'references':baseline['models'],'reserved_tests_scored':False,'default_deployment_changed':False}
 for mode in ('float','ternary','q4','q8'):
  if mode in ('float','ternary'):
   score=read(OUT/(mode+'-quality.json'));row={'tasks':score['tasks'],'stsbenchmark_spearman':score['stsbenchmark_spearman'],'mean_core_retrieval':score['mean_core_retrieval']};newrows=read(OUT/(mode+'-rows.json'))
  else:
   name='w'+mode[1:]+'a8';score=read(OUT/('precision-'+name+'.json'));row={'tasks':score['tasks'],'stsbenchmark_spearman':score['stsbenchmark_spearman'],'mean_core_retrieval':score['mean'],'scope':score['scope']};newrows={t:read(OUT/f'precision-{name}-{t}-rows.json') for t in ('SciFact','NFCorpus')}
  if mode=='ternary':old=baseline['models']['original'];priorrows=oldrows['original'];prior_mean=old['mean_core_retrieval'];prior_sts=old['stsbenchmark_spearman']
  else:
   name={'float':'float','q4':'w4a8','q8':'w8a8'}[mode];old=read(ROOT/'analysis'/('precision-'+name+'.json'));prior_mean=old['mean'];prior_sts=old['stsbenchmark_spearman'];priorrows={t:read(ROOT/'analysis'/f'precision-{name}-{t}-rows.json') for t in ('SciFact','NFCorpus')}
  row['prior']={'mean_core_retrieval':prior_mean,'stsbenchmark_spearman':prior_sts};row['change']=interval(priorrows,newrows)
  row['relative_to_references']={name:interval(oldrows[name],newrows) for name in ('bge_micro','minilm')}
  row['beats_each_core_task']={name:all(row['tasks'][t]['ndcg10']>baseline['models'][name]['tasks'][t]['ndcg10'] for t in ('SciFact','NFCorpus')) for name in ('bge_micro','minilm')}
  result['models'][mode]=row
 result['target_met']=any(all(v['beats_each_core_task'].values()) for v in result['models'].values())
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print({k:(v['prior']['mean_core_retrieval'],v['mean_core_retrieval'],v['change']) for k,v in result['models'].items()})
if __name__=='__main__':main()
