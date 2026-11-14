"""Human NLI supervision with a disposable classifier, inspired by SBERT.

The shared encoder still emits the same384-dimensional unit vectors. The
classification head is training-only and is never part of inference exports.
Reference: https://aclanthology.org/D19-1410.pdf , classification objective.
"""
import json
import torch
import torch.nn.functional as F
from pair_training_state import SourceSampler

def features(v,n,encoder_weight=1.,scale=10.):
 # Forward is unchanged; only gradients into the shared encoder are weighted.
 if encoder_weight==0:v=v.detach()
 elif encoder_weight!=1:v=v.detach()+encoder_weight*(v-v.detach())
 a,b=v[:n],v[n:];return torch.cat((a,b,(a-b).abs()),-1)*scale

class NliAuxiliary:
 def __init__(self,path,dev,tokenizer,dim,device,batch=64,seed=29,feature_scale=10.):
  with path.open() as f:self.rows=[json.loads(l) for l in f]
  with dev.open() as f:self.dev=[json.loads(l) for l in f]
  self.tk=tokenizer;self.device=device;self.feature_scale=feature_scale;self.examples_seen=0
  if not self.rows or not self.dev or any(r['label'] not in (0,1,2) for r in self.rows+self.dev):raise ValueError('Invalid human NLI rows')
  self.sampler=SourceSampler({'nli':list(range(len(self.rows)))},batch,seed+317);self.head=torch.nn.Linear(3*dim,3,bias=False).to(device)
  self.dev_tokens=None
 def backward(self,model,encoder_weight,amp=False):
  _,indexes=self.sampler.next();rows=[self.rows[i] for i in indexes];n=len(rows)
  enc=self.tk([r['sentence1'] for r in rows]+[r['sentence2'] for r in rows],padding=True,truncation=True,max_length=128,return_tensors='pt').to(self.device)
  with torch.set_grad_enabled(encoder_weight!=0):
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):v=model(enc['input_ids'],enc['attention_mask'].bool())[2].float()
  logits=self.head(features(v,n,encoder_weight,self.feature_scale));labels=torch.tensor([r['label'] for r in rows],device=self.device)
  loss=F.cross_entropy(logits,labels)
  if not torch.isfinite(loss):raise ValueError('Nonfinite human-NLI loss')
  loss.backward();self.examples_seen+=n
  return {'loss':float(loss.detach()),'accuracy':float((logits.argmax(-1)==labels).float().mean())*100,'examples_seen':self.examples_seen}
 def state_dict(self):return {'head':self.head.state_dict(),'sampler':self.sampler.state_dict(),'examples_seen':self.examples_seen}
 def load_state_dict(self,state):
  self.head.load_state_dict(state['head']);self.sampler.load_state_dict(state['sampler']);self.examples_seen=state['examples_seen']
 @torch.inference_mode()
 def evaluate(self,model,batch=128):
  was=model.training;model.eval();n=len(self.dev)
  if self.dev_tokens is None:
   texts=[r['sentence1'] for r in self.dev]+[r['sentence2'] for r in self.dev];self.dev_tokens=self.tk(texts,truncation=True,max_length=128)['input_ids']
  tokens=self.dev_tokens;order=sorted(range(len(tokens)),key=lambda i:len(tokens[i]));v=torch.empty((len(tokens),model.dims['out_dim']),device=self.device)
  for offset in range(0,len(order),batch):
   ids=order[offset:offset+batch];enc=self.tk.pad({'input_ids':[tokens[i] for i in ids]},padding=True,return_tensors='pt').to(self.device);v[ids]=model(enc['input_ids'],enc['attention_mask'].bool())[2].float()
  correct=self.head(features(v,n,scale=self.feature_scale)).argmax(-1)==torch.tensor([r['label'] for r in self.dev],device=self.device);label=torch.tensor([r['label'] for r in self.dev],device=self.device)
  by_label={str(k):float(correct[label==k].float().mean())*100 for k in (0,1,2) if (label==k).any()};model.train(was)
  return {'accuracy':float(correct.float().mean())*100,'macro_accuracy':sum(by_label.values())/len(by_label),'by_label':by_label,'examples':n,'scope':'Disjoint official human-NLI dev labels; disposable classifier, not an embedding test benchmark'}
