"""Direct positive-pair training; no teacher model, target vectors, or anchor loss.

Batches use one source at a time, sampled in proportion to its retained rows.
Known positive passages are marginalized, including positives outside the batch.
Preserves the checkpoint's float/ternary mode; does not convert QAT master weights.
"""
import argparse,collections,itertools,json,sys,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'tern-embed'))
from model import Student
from data_first_build import sha

def positive_pair_loss(queries,passages,mask,temperature=.05):
 if temperature<=0 or not mask.any(-1).all():raise ValueError('Need positive temperature and known positive for every query')
 logits=queries.float()@passages.float().T/temperature
 return (logits.logsumexp(-1)-logits.masked_fill(~mask,float('-inf')).logsumexp(-1)).mean()

def batch_documents(batch,use_negatives=False):
 docs={r['positive_visible_key']:r['positive'] for r in batch}
 if use_negatives:
  for r in batch:
   if r.get('negative'):docs.setdefault(r['negative_visible_key'],r['negative'])
 return docs

def batch_positive_mask(batch,doc_keys,known,device):
 # Intersect known positives with this document set, rather than testing every
 # query/document pair in Python. Includes positives introduced as negatives.
 lookup={key:i for i,key in enumerate(doc_keys)};queries=[];documents=[]
 for i,row in enumerate(batch):
  for key in known[row['query_visible_key']]:
   if key in lookup:queries.append(i);documents.append(lookup[key])
 mask=torch.zeros((len(batch),len(doc_keys)),dtype=torch.bool,device=device)
 if queries:mask[torch.tensor(queries,device=device),torch.tensor(documents,device=device)]=True
 return mask

def main():
 import math,signal
 from pair_training_state import SourceSampler,checkpoint,atomic_torch_save,atomic_json
 from pair_validation import PairValidation
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 budget=p.add_mutually_exclusive_group(required=True);budget.add_argument('--steps',type=int);budget.add_argument('--epochs',type=int)
 p.add_argument('--batch',type=int,default=128);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--temperature',type=float,default=.05);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=29);p.add_argument('--limit',type=int,help='Explicit tiny local pipeline smoke only')
 p.add_argument('--dev',type=Path);p.add_argument('--eval-every',type=int,default=1500);p.add_argument('--source-floor',type=float,default=2.,help='Maximum per-source dev nDCG drop from initialization')
 p.add_argument('--save-every',type=int,default=500);p.add_argument('--save-seconds',type=float,default=120);p.add_argument('--resume',type=Path);p.add_argument('--stop-after',type=int,help='Gracefully stop at this absolute step for recovery tests')
 p.add_argument('--amp',choices=['off','bf16'],default='off');p.add_argument('--growth-warmup-steps',type=int,default=0);p.add_argument('--expected-layers',type=int,choices=[2,3]);p.add_argument('--micro-batch',type=int,default=0,help='Exact representation-gradient replay; zero uses a full graph');p.add_argument('--use-negatives',action='store_true',help='Include audited explicit negatives, retaining all known-positive masks');p.add_argument('--validation-negatives',action='store_true');p.add_argument('--learn-row-scales',action='store_true');p.add_argument('--scale-lr-multiplier',type=float,default=100.)
 p.add_argument('--train-row-scales-only',action='store_true');p.add_argument('--aux-data',type=Path);p.add_argument('--aux-dev',type=Path);p.add_argument('--aux-encoder-weight',type=float,default=0.);p.add_argument('--aux-batch',type=int,default=64);p.add_argument('--aux-lr-multiplier',type=float,default=25.);p.add_argument('--aux-warmup-steps',type=int,default=100);p.add_argument('--aux-feature-scale',type=float,default=10.);a=p.parse_args()
 if (a.steps is not None and a.steps<1) or (a.epochs is not None and a.epochs<1) or a.batch<2:p.error('Need a positive training budget and batch >=2')
 if min(a.save_every,a.save_seconds,a.eval_every)<=0:p.error('Save/evaluation intervals must be positive')
 if a.micro_batch<0:p.error('Microbatch must be nonnegative')
 if a.train_row_scales_only and not a.learn_row_scales:p.error('Scale-only training requires learned ternary row scales')
 if bool(a.aux_data)!=bool(a.aux_dev) or (a.aux_encoder_weight and not a.aux_data):p.error('Auxiliary supervision requires both audited train/dev files')
 if not 0<=a.aux_encoder_weight<=1 or a.aux_batch<2 or a.aux_warmup_steps<0 or min(a.aux_lr_multiplier,a.aux_feature_scale)<=0:p.error('Invalid auxiliary settings')
 if (a.out/'latest.json').exists() and not a.resume:p.error('Existing training state: pass --resume or use a new output directory')
 torch.set_num_threads(2);torch.manual_seed(a.seed)
 with a.data.open() as f:rows=[json.loads(l) for l in itertools.islice(f,a.limit)]
 by_source=collections.defaultdict(list);known=collections.defaultdict(set)
 for i,r in enumerate(rows):by_source[r['source']].append(i);known[r['query_visible_key']].add(r['positive_visible_key'])
 if any(len(v)<2 for v in by_source.values()):p.error('Every source needs at least two pairs')
 sampler=SourceSampler(dict(by_source),a.batch,a.seed);total=a.steps or sampler.batches_per_epoch*a.epochs
 if a.learn_row_scales:
  if a.growth_warmup_steps:p.error('Test learned row scales only at a fixed depth')
  from learned_row_scales import load_training,inference_state,optimizer
  m=load_training(a.checkpoint,a.resume)
 else:m=Student.load(a.resume or a.checkpoint)
 m=m.to(a.device).train();assert m.dims['d']==384 and m.dims['layers'] in (2,3)
 if a.train_row_scales_only:
  from learned_row_scales import scales_only
  print(json.dumps({'trainable_ternary_row_scale_parameters':scales_only(m)}),flush=True)
 if a.expected_layers is not None and m.dims['layers']!=a.expected_layers:raise ValueError('Unexpected checkpoint depth')
 if a.growth_warmup_steps:
  if not 0<a.growth_warmup_steps<total:raise ValueError('Growth warmup must finish before the final update')
  from grow_depth import install_growth,set_growth
  install_growth(m)
 else:
  set_growth=lambda model,alpha:None
 growth_alpha=0. if a.growth_warmup_steps else 1.
 set_growth(m,growth_alpha)
 tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True);opt=optimizer(m,a.lr,a.scale_lr_multiplier) if a.learn_row_scales else torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=.01)
 auxiliary=None
 if a.aux_data:
  from nli_auxiliary import NliAuxiliary
  auxiliary=NliAuxiliary(a.aux_data,a.aux_dev,tk,m.dims['out_dim'],a.device,a.aux_batch,a.seed,a.aux_feature_scale)
  opt.add_param_group({'params':list(auxiliary.head.parameters()),'lr':a.lr*a.aux_lr_multiplier,'lr_multiplier':a.aux_lr_multiplier})
 a.out.mkdir(parents=True,exist_ok=True);config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()};config.update(teacher_targets=False,precision='ternary' if m.quantized else 'float',checkpoint_sha256=sha(a.checkpoint),data_sha256=sha(a.data),dev_sha256=sha(a.dev) if a.dev else None,loaded_rows=len(rows),source_sampling='complete source-batch epochs, shuffled and interleaved',total_steps=total,auxiliary_sha256=sha(a.aux_data) if a.aux_data else None,auxiliary_dev_sha256=sha(a.aux_dev) if a.aux_dev else None)
 validator=PairValidation([json.loads(l) for l in a.dev.open()],tk,a.validation_negatives) if a.dev else None
 progress={'examples_seen':0,'real_tokens':0,'padded_tokens':0,'best_score':float('-inf'),'last_validation':None,'baseline_validation':None};first_step=0
 if a.resume:
  saved=torch.load(a.resume,map_location='cpu',weights_only=False);old=saved['meta']['config']
  defaults={'growth_warmup_steps':0,'micro_batch':0,'use_negatives':False,'validation_negatives':False,'learn_row_scales':False,'scale_lr_multiplier':100.,'train_row_scales_only':False,'auxiliary_sha256':None,'auxiliary_dev_sha256':None,'aux_encoder_weight':0.,'aux_batch':64,'aux_lr_multiplier':25.,'aux_warmup_steps':100,'aux_feature_scale':10.}
  for key in ('checkpoint_sha256','data_sha256','dev_sha256','total_steps','batch','lr','temperature','seed','precision','amp','source_floor','limit','growth_warmup_steps','expected_layers','micro_batch','use_negatives','validation_negatives','learn_row_scales','scale_lr_multiplier','train_row_scales_only','auxiliary_sha256','auxiliary_dev_sha256','aux_encoder_weight','aux_batch','aux_lr_multiplier','aux_warmup_steps','aux_feature_scale'):
   previous=old.get(key,defaults.get(key))
   if previous!=config.get(key):raise ValueError('Resume configuration mismatch: '+key)
  state=saved['resume'];opt.load_state_dict(state['optimizer']);sampler.load_state_dict(state['sampler']);torch.set_rng_state(state['torch_rng'])
  if auxiliary:auxiliary.load_state_dict(state['nli_auxiliary'])
  if state['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
  progress=state['progress'];first_step=saved['meta']['step']
  growth_alpha=min(1.,first_step/a.growth_warmup_steps) if a.growth_warmup_steps else 1.;set_growth(m,growth_alpha)
 atomic_json(a.out/'config.json',config);started=time.monotonic();last_save=started;stop_requested=False
 def request_stop(signum,frame):
  nonlocal stop_requested
  stop_requested=True
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 def save_checkpoint(step):return checkpoint(m,opt,sampler,step,config,progress,a.out,extra_resume={'nli_auxiliary':auxiliary.state_dict()} if auxiliary else None)
 def export(path,step,validation=None):atomic_torch_save(path,{'dims':m.dims,'state':inference_state(m) if a.learn_row_scales else m.state_dict(),'quant':m.quantized,'meta':{'step':step,'config':config,'validation':validation,'growth_alpha':growth_alpha}})
 def validate(step):
  metrics=validator(m,a.device);progress['last_validation']=metrics
  if auxiliary:metrics['auxiliary_nli']=auxiliary.evaluate(m)
  if progress['baseline_validation'] is None:progress['baseline_validation']=metrics
  eligible=all(v['ndcg10']>=progress['baseline_validation']['by_source'][s]['ndcg10']-a.source_floor for s,v in metrics['by_source'].items())
  auxiliary_eligible=not auxiliary or metrics['auxiliary_nli']['macro_accuracy']>=progress['baseline_validation']['auxiliary_nli']['macro_accuracy']-1.
  score=(metrics['macro_ndcg10']+metrics['auxiliary_nli']['macro_accuracy'])/2 if auxiliary else metrics['macro_ndcg10']
  metrics.update(step=step,source_floor_passed=eligible,auxiliary_floor_passed=auxiliary_eligible,selection_score=score,growth_alpha=growth_alpha,export_eligible=growth_alpha==1.)
  if eligible and auxiliary_eligible and growth_alpha==1. and score>progress['best_score']:
   progress['best_score']=score;progress['best_step']=step;export(a.out/'best.pt',step,metrics)
  if growth_alpha==1.:export(a.out/f'milestone-{step:07d}.pt',step,metrics)
  with (a.out/'validation.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
  print(json.dumps({'validation':metrics}),flush=True)
 if first_step==0 and not a.resume:
  if auxiliary:
   for _ in range(a.aux_warmup_steps):
    opt.zero_grad(set_to_none=True);auxiliary.backward(m,0.,a.device.startswith('cuda') and a.amp=='bf16');head_grad=torch.nn.utils.clip_grad_norm_(auxiliary.head.parameters(),1.)
    if not torch.isfinite(head_grad):raise ValueError('Nonfinite NLI head gradient')
    opt.step()
   print(json.dumps({'auxiliary_head_warmup_steps':a.aux_warmup_steps,'encoder_unchanged_during_head_warmup':True}),flush=True)
  if validator:validate(0)
  save_checkpoint(0)
 with (a.out/'log.jsonl').open('a') as log:
  for step in range(first_step+1,total+1):
   growth_alpha=min(1.,step/a.growth_warmup_steps) if a.growth_warmup_steps else 1.;set_growth(m,growth_alpha)
   source,indexes=sampler.next();batch=[rows[i] for i in indexes];nq=len(batch)
   docs=batch_documents(batch,a.use_negatives);doc_keys=list(docs)
   enc=tk([r['query'] for r in batch]+list(docs.values()),padding=True,truncation=True,max_length=128,return_tensors='pt').to(a.device)
   mask=batch_positive_mask(batch,doc_keys,known,a.device)
   lr=a.lr*(.1+.9*(1+math.cos(math.pi*(step-1)/total))/2)
   for group in opt.param_groups:group['lr']=lr*group.get('lr_multiplier',1.)
   opt.zero_grad(set_to_none=True)
   if a.micro_batch:
    from contrastive_cache import cached_backward
    loss=cached_backward(m,enc,nq,mask,a.temperature,a.micro_batch,a.device.startswith('cuda') and a.amp=='bf16')
   else:
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda') and a.amp=='bf16'):
     v=m(enc['input_ids'],enc['attention_mask'].bool())[2];loss=positive_pair_loss(v[:nq],v[nq:],mask,a.temperature)
    if not torch.isfinite(loss):raise ValueError('Nonfinite pair loss')
    loss.backward()
   auxiliary_metrics=auxiliary.backward(m,a.aux_encoder_weight,a.device.startswith('cuda') and a.amp=='bf16') if auxiliary else None
   if auxiliary:
    head_grad=torch.nn.utils.clip_grad_norm_(auxiliary.head.parameters(),1.)
    if not torch.isfinite(head_grad):raise ValueError('Nonfinite NLI head gradient')
   grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
   opt.step();progress['examples_seen']+=nq;progress['real_tokens']+=int(enc['attention_mask'].sum());progress['padded_tokens']+=enc['input_ids'].numel()
   row={'step':step,'total_steps':total,'epoch':sampler.epoch+1,'source':str(source),'loss':float(loss.detach()),'grad_norm':float(grad),'lr':lr,'seconds':time.monotonic()-started,'growth_alpha':growth_alpha,'examples_seen':progress['examples_seen'],'real_tokens':progress['real_tokens'],'auxiliary_nli':auxiliary_metrics};log.write(json.dumps(row)+'\n');log.flush()
   if step%100==0 or step==total:print(json.dumps(row),flush=True)
   if validator and (step%a.eval_every==0 or step==total):validate(step)
   stopping=stop_requested or (a.stop_after is not None and step>=a.stop_after)
   if step%a.save_every==0 or time.monotonic()-last_save>=a.save_seconds or step==total or stopping:
    info=save_checkpoint(step);last_save=time.monotonic();print(json.dumps({'checkpoint':info}),flush=True)
   if stopping and step<total:print('Stopped cleanly after saving resumable checkpoint',flush=True);return
 if growth_alpha!=1.:raise RuntimeError('Only fully grown models may be exported')
 m.eval();export(a.out/'model.pt',total,progress['last_validation']);loaded=Student.load(a.out/'model.pt').to(a.device).eval()
 with torch.no_grad():
  before=m(enc['input_ids'],enc['attention_mask'].bool())[2];after=loaded(enc['input_ids'],enc['attention_mask'].bool())[2];exact=torch.equal(before,after)
  assert exact or (a.learn_row_scales and torch.allclose(before,after,atol=2e-6,rtol=2e-6))
 print(json.dumps({'exact_checkpoint_reload':exact,'canonical_checkpoint_max_abs_error':float((before-after).abs().max()),'quantized':loaded.quantized,'pipeline_smoke_only':bool(a.limit),'training_completed':True}),flush=True)
if __name__=='__main__':main()
