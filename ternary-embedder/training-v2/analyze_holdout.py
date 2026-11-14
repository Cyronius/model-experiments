"""Re-score the training-source holdout against all its held documents, locally."""
import json,sys
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent;OUT=ROOT/'analysis'
sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
from diagnose import student,encode
from rank_distill import split_queries

def metrics(q,docs,gold,indices,sources,ownneg):
 scores=q@docs.T;result={};ranks=[];mass=[];ent=[];wrong=[]
 for i in range(len(q)):
  positives=gold[i];pmax=max(scores[i,j] for j in positives);ranks.append(1+int((scores[i]>pmax).sum()));wrong.append(float(scores[i,ownneg[i]]>=pmax))
  logits=scores[i]/.05;probs=np.exp(logits-logits.max());probs/=probs.sum();mass.append(float(probs[positives].sum()));ent.append(float(-(probs*np.log(probs+1e-30)).sum()))
 ranks=np.array(ranks)
 for source in sorted(set(sources)):
  idx=np.array([i for i,s in enumerate(sources) if s==source]);result[source]={'queries':len(idx),'mrr':float(np.mean(1/ranks[idx])),'recall1':float(np.mean(ranks[idx]==1)),'own_negative_ties_or_beats_positive':float(np.mean(np.array(wrong)[idx])),'positive_mass':float(np.mean(np.array(mass)[idx])),'softmax_effective_candidates':float(np.exp(np.mean(np.array(ent)[idx])))}
 return {'mrr':float(np.mean(1/ranks)),'by_source':result}

@torch.inference_mode()
def main():
 torch.set_num_threads(2);rows=[json.loads(l) for l in (ROOT/'work/round2/balanced-refined.jsonl').read_text().splitlines()];_,valid,_=split_queries(rows,np.random.default_rng(0))
 docs=list(dict.fromkeys([rows[i]['positive'] for i in valid]+[rows[i]['negative'] for i in valid]));lookup={d:i for i,d in enumerate(docs)};sources=[rows[i]['source'] for i in valid]
 known={}
 for i in valid:known.setdefault(rows[i]['query'],set()).add(lookup[rows[i]['positive']])
 gold=[sorted(known[rows[i]['query']]) for i in valid];ownneg=[lookup[rows[i]['negative']] for i in valid]
 first={}
 for i in valid:
  first[rows[i]['positive']]=(i,0);first[rows[i]['negative']]=(i,1)
 cache=ROOT/'work/round2/teacher-cache';td=np.load(cache/'passages.npy',mmap_mode='r')
 result={'queries':len(valid),'documents':len(docs),'scope':'All heldout documents; multiple known positives accepted; not external acceptance','models':{}}
 for name,qpath in [('teacher8_prompted','prompted-queries.npy'),('teacher8_plain','plain-queries.npy')]:
  q=np.load(cache/qpath,mmap_mode='r')[valid];d=np.stack([td[first[t][0],first[t][1]] for t in docs]);result['models'][name]=metrics(q,d,gold,valid,sources,ownneg);print(name,result['models'][name],flush=True)
  (OUT/'full-holdout.json').write_text(json.dumps(result,indent=2)+'\n')
 t=np.load(cache/'plain06.npy',mmap_mode='r');q=t[valid,0];d=np.stack([t[first[x][0],first[x][1]+1] for x in docs]);result['models']['teacher06_plain']=metrics(q,d,gold,valid,sources,ownneg)
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);runs=ROOT/'runs/balanced-round2/ternary-embedder/training-v2/runs'
 paths={'original':PROJECT/'tern-embed/pulls/grid1/runs/2a_384/model.pt','selected':runs/'balanced-prompted-anchor1.0/best.pt','gold_positive':runs/'balanced-prompted-anchor1.0-positive.2/best.pt'}
 texts=[rows[i]['query'] for i in valid]+docs;n=len(valid)
 for name,path in paths.items():
  vec=encode(student(path,'quant'),tk,texts,False,32);result['models'][name]=metrics(vec[:n],vec[n:],gold,valid,sources,ownneg);print(name,result['models'][name],flush=True)
  (OUT/'full-holdout.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
