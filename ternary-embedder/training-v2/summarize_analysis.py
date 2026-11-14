"""Consolidate local post-hoc diagnostics and precision accuracy simulations."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'analysis';sys.path.insert(0,str(ROOT.parent/'retrieval-diagnostics'))
from diagnose import paired

def mean_interval(left,right):
 rng=np.random.default_rng(0);samples=[];deltas=[]
 for task in ('SciFact','NFCorpus'):
  a=left[task];b=right[task];assert [r['query_id'] for r in a]==[r['query_id'] for r in b]
  d=np.array([y['ndcg10']-x['ndcg10'] for x,y in zip(a,b)]);deltas.append(d.mean()*100)
  samples.append(d[rng.integers(0,len(d),(4000,len(d)))].mean(1)*100)
 return {'mean_delta':float(np.mean(deltas)),'paired_query_bootstrap_95_ci':np.percentile(np.mean(samples,axis=0),[2.5,97.5]).tolist(),'scope':'Query resampling only; not training-seed uncertainty'}

def main():
 result={'scope':'Post-hoc development analysis, no checkpoint promotion; low-bit accuracy simulation only','precision':{},'arms':{}}
 orig=json.loads((ROOT/'round2/query-rows.json').read_text())['original']
 fr={t:json.loads((OUT/f'precision-float-{t}-rows.json').read_text()) for t in ('SciFact','NFCorpus')}
 for name in ('float','w4a8','w8a8'):
  x=json.loads((OUT/f'precision-{name}.json').read_text());rows={t:json.loads((OUT/f'precision-{name}-{t}-rows.json').read_text()) for t in ('SciFact','NFCorpus')}
  x['relative_to_ternary_original']=mean_interval(orig,rows);x['relative_to_float']=mean_interval(fr,rows);result['precision'][name]=x
 for path in OUT.glob('arms-*.json'):result['arms'].update(json.loads(path.read_text())['models'])
 result['full_holdout']=json.loads((OUT/'full-holdout.json').read_text());result['architecture']=json.loads((OUT/'architecture-costs.json').read_text())
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Precision:',{k:(v['mean'],v['relative_to_ternary_original'],v['relative_to_float']) for k,v in result['precision'].items()})
if __name__=='__main__':main()
