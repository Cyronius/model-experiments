"""Freeze a winner using mixed TRAIN holdout ranking and unlabeled relation preservation.

No acceptance-task inputs or labels are read. The rule is defined before those
scores: per-source MRR >= baseline-0.01, then minimize cosine-relation drift
among candidates within 0.01 MRR of the best eligible mixed-holdout score.
"""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
from rank_distill import ROOT,PROJECT,split_queries,sha
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student,encode

@torch.inference_mode()
def main():
 torch.set_num_threads(2);out=ROOT/'round2';out.mkdir(exist_ok=True)
 target=out/'frozen-candidate.json'
 if target.exists():raise RuntimeError('Round-2 candidate is already frozen')
 data_path=ROOT/'work/round2/balanced-refined.jsonl'
 rows=[json.loads(l) for l in data_path.read_text().splitlines()]
 _,valid,_=split_queries(rows,np.random.default_rng(0));baseline=json.loads((out/'validation-baselines.json').read_text())
 assert baseline['data_sha256']==sha(data_path)
 base_metrics=baseline['models']['original'];rng=np.random.default_rng(17);sample=[]
 for source in sorted({r['source'] for r in rows}):
  texts=list(dict.fromkeys(t for i in valid if rows[i]['source']==source for t in (rows[i]['query'],rows[i]['positive'],rows[i]['negative'])))
  sample.extend(texts[j] for j in rng.choice(len(texts),size=min(256,len(texts)),replace=False))
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
 original_path=PROJECT/'tern-embed/pulls/grid1/runs/2a_384/model.pt'
 original=student(original_path,'quant');reference=encode(original,tk,sample,False,32);reference_sim=reference@reference.T
 choices=[]
 paths=list((ROOT/'runs/balanced-round2/ternary-embedder/training-v2/runs').glob('balanced-*/best.pt'))
 if len(paths)!=6:raise RuntimeError('Need all six completed round-2 arms')
 for path in paths:
  ck=torch.load(path,map_location='cpu',weights_only=False);meta=ck['meta'];metrics=meta['validation']
  assert ck['quant'] and ck['dims']==original.dims
  assert meta['config']['data_sha256']==sha(data_path)
  vec=encode(student(path,'quant'),tk,sample,False,32)
  drift=float(np.sqrt(np.mean(np.square(vec@vec.T-reference_sim))))
  eligible=all(metrics['by_source'][s]['mrr']>=base_metrics['by_source'][s]['mrr']-.01 for s in base_metrics['by_source'])
  choices.append({'run':path.parent.name,'path':str(path),'step':meta['step'],'mrr':metrics['mrr'],
                  'by_source':metrics['by_source'],'source_floor_passed':eligible,'cosine_relation_rms_drift':drift,'sha256':sha(path)})
 choices.append({'run':'original-null-control','path':str(original_path),'step':0,'mrr':base_metrics['mrr'],
                 'by_source':base_metrics['by_source'],'source_floor_passed':True,'cosine_relation_rms_drift':0.,'sha256':sha(original_path)})
 eligible=[c for c in choices if c['source_floor_passed']];best_mrr=max(c['mrr'] for c in eligible)
 band=[c for c in eligible if c['mrr']>=best_mrr-.01]
 selected=dict(min(band,key=lambda c:c['cosine_relation_rms_drift']))
 selected.update(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_before_test=True,
  rule='Per-source MRR no worse than original-0.01; within 0.01 of best eligible MRR, minimize held-text cosine-relation RMS drift',
  data_sha256=sha(data_path),geometry_texts=len(sample),geometry_sample_seed=17,
  geometry_sample_sha256=hashlib.sha256(json.dumps(sample).encode()).hexdigest(),all_candidates=choices)
 target.write_text(json.dumps(selected,indent=2)+'\n');print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
