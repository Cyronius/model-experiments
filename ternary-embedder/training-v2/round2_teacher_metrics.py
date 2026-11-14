"""Check teacher rankings against public-train holdout positives; no test benchmarks."""
import json,numpy as np
from rank_distill import ROOT,split_queries
work=ROOT/'work/round2';rows=[json.loads(l) for l in (work/'balanced-refined.jsonl').read_text().splitlines()]
_,valid,_=split_queries(rows,np.random.default_rng(0));passages=np.load(work/'teacher-cache/passages.npy',mmap_mode='r');report={}
for name in ['prompted','plain']:
 query=np.load(work/f'teacher-cache/{name}-queries.npy',mmap_mode='r');ranks=[];by_source={}
 for start in range(0,len(valid),32):
  b=valid[start:start+32];docs={}
  for i in b:docs[rows[i]['positive']]=passages[i,0];docs[rows[i]['negative']]=passages[i,1]
  names=list(docs);scores=query[b]@np.stack(list(docs.values())).T
  labels=np.array([names.index(rows[i]['positive']) for i in b]);positive=scores[np.arange(len(b)),labels,None]
  rr=1+(scores>positive).sum(1);ranks.extend(rr.tolist())
  for i,r in zip(b,rr):by_source.setdefault(rows[i]['source'],[]).append(int(r))
 report[name]={'mrr':float(np.mean(1/np.array(ranks))),'by_source':{s:float(np.mean(1/np.array(r))) for s,r in by_source.items()}}
(ROOT/'round2/teacher-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
