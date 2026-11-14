"""Exact representation-gradient replay for this deterministic Student encoder.

The model has no dropout or running-statistic layers. Replay uses the identical
microbatch inputs and precision from the first pass. No teacher targets.
Reference: Gao et al., https://arxiv.org/abs/2101.06983 .
"""
import torch

def cached_backward(model,enc,n_queries,positive_mask,temperature,micro_batch,amp=False):
 device=enc['input_ids'].device;count=len(enc['input_ids']);chunks=[];parts=[]
 if temperature<=0 or micro_batch<1 or not 0<n_queries<count:raise ValueError('Invalid cached contrastive batch')
 if positive_mask.shape!=(n_queries,count-n_queries):raise ValueError('Positive mask dimensions differ from vectors')
 # Fixed padding and partitions in both passes; no RNG-consuming modules exist.
 with torch.no_grad():
  for start in range(0,count,micro_batch):
   stop=min(count,start+micro_batch);chunks.append((start,stop))
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
    parts.append(model(enc['input_ids'][start:stop],enc['attention_mask'][start:stop].bool())[2].float())
 vectors=torch.cat(parts).detach().requires_grad_();logits=vectors[:n_queries]@vectors[n_queries:].T/temperature
 if not positive_mask.any(-1).all():raise ValueError('Every query needs a positive')
 loss=(logits.logsumexp(-1)-logits.masked_fill(~positive_mask,float('-inf')).logsumexp(-1)).mean()
 if not torch.isfinite(loss):raise ValueError('Nonfinite cached contrastive loss')
 gradients=torch.autograd.grad(loss,vectors)[0].detach()
 for start,stop in chunks:
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
   replay=model(enc['input_ids'][start:stop],enc['attention_mask'][start:stop].bool())[2]
  (replay.float()*gradients[start:stop]).sum().backward()
 return loss.detach()
