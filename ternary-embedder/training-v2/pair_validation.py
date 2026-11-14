"""Full held-source document ranking, with all known positives accepted."""
import collections
import numpy as np,torch

class PairValidation:
 def __init__(self,rows,tokenizer,include_negatives=False):
  self.include_negatives=include_negatives
  self.by_source=collections.defaultdict(list)
  for row in rows:self.by_source[row['source']].append(row)
  keys=('query','positive','negative') if include_negatives else ('query','positive')
  self.texts=list(dict.fromkeys(r[k] for r in rows for k in keys if r.get(k)));self.lookup={t:i for i,t in enumerate(self.texts)}
  self.tokens=tokenizer(self.texts,truncation=True,max_length=128)['input_ids'];self.tokenizer=tokenizer
 @torch.inference_mode()
 def __call__(self,model,device,batch=128):
  was=model.training;model.eval();vectors=torch.empty((len(self.texts),model.dims['out_dim']),device=device)
  order=sorted(range(len(self.texts)),key=lambda i:len(self.tokens[i]))
  for off in range(0,len(order),batch):
   ii=order[off:off+batch];enc=self.tokenizer.pad({'input_ids':[self.tokens[i] for i in ii]},padding=True,return_tensors='pt').to(device)
   vectors[ii]=model(enc['input_ids'],enc['attention_mask'].bool())[2].float()
  result={}
  for source,rows in self.by_source.items():
   qdict={r['query_visible_key']:r['query'] for r in rows};ddict={r['positive_visible_key']:r['positive'] for r in rows}
   if self.include_negatives:
    for r in rows:
     if r.get('negative'):ddict.setdefault(r['negative_visible_key'],r['negative'])
   qkeys=list(qdict);dkeys=list(ddict);qi={k:i for i,k in enumerate(qkeys)};di={k:i for i,k in enumerate(dkeys)}
   q=vectors[[self.lookup[t] for t in qdict.values()]];d=vectors[[self.lookup[t] for t in ddict.values()]];scores=q@d.T
   mask=torch.zeros_like(scores,dtype=torch.bool)
   qindexes=torch.tensor([qi[r['query_visible_key']] for r in rows],device=device)
   dindexes=torch.tensor([di[r['positive_visible_key']] for r in rows],device=device)
   mask[qindexes,dindexes]=True
   best=scores.masked_fill(~mask,float('-inf')).amax(-1);ranks=1+(scores>best[:,None]).sum(-1);k=min(10,len(dkeys));top=scores.topk(k,dim=-1).indices;hit=mask.gather(1,top);discount=1/torch.log2(torch.arange(2,k+2,device=device,dtype=torch.float32));count=mask.sum(-1)
   ideal=(torch.arange(k,device=device)[None,:]<count[:,None]).float()@discount;ndcg=(hit.float()@discount)/ideal
   result[source]={'queries':len(qkeys),'documents':len(dkeys),'mrr':float((1/ranks.float()).mean()),'ndcg10':float(ndcg.mean())*100,'recall10':float((hit.sum(-1)/count).mean())}
  model.train(was);return {'by_source':result,'macro_ndcg10':float(np.mean([v['ndcg10'] for v in result.values()])), 'scope':'Each held source ranked against all held positives'+(' and audited explicit negatives' if self.include_negatives else '')+' from that source'}
