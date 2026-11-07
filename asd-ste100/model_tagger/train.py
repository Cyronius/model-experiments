"""Resumable local CPU baseline training on audited, memory-mapped tag trajectories."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from tagger import Tagger,Config,Tags,SteTokenizer,Lexicon,loss,rewrite,EncodingError
from packed import PackedDataset,Collator,Batches,KINDS
from rollouts import fingerprints,records,sha256
from ste_tokenizer import model_tokens
from safe_editor import rewrite as guarded_rewrite

STOP=False


def stop_signal(signum,frame):
    global STOP
    STOP=True


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def checkpoint(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary);temporary.replace(path)


def batch_sampler(index,batch_size,seed,epoch,shuffle=True,start=0,local_cap=None,repeats=None):
    """Batches, plus repeats[k]-1 extra shuffled copies of the rows of kind k (training-only oversampling)."""
    result=Batches(index,batch_size,seed,epoch,shuffle,start,local_cap)
    if repeats and max(repeats)>1:
        rng=np.random.default_rng(1000003+seed+epoch);lengths=index[:,1]+1
        for kind,count in enumerate(repeats):
            rows=np.flatnonzero(index[:,5]==kind)
            for _ in range(count-1):
                for low,high in ((0,16),(16,32),(32,64),(64,129),(129,257)):
                    positions=rows[(lengths[rows]>low)&(lengths[rows]<=high)]
                    if shuffle:rng.shuffle(positions)
                    result.batches.extend(positions[i:i+batch_size].tolist() for i in range(0,len(positions),batch_size))
        if shuffle:rng.shuffle(result.batches)
    return result


def loader(directory,split,tok,batch_size,workers,seed,epoch,shuffle=True,start=0,local_cap=None,repeats=None):
    index=np.load(directory/f'{split}.index.npy',mmap_mode='r')
    sampler=batch_sampler(index,batch_size,seed,epoch,shuffle,start,local_cap,repeats)
    generator=torch.Generator().manual_seed(seed+epoch)
    return torch.utils.data.DataLoader(PackedDataset(directory,split),batch_sampler=sampler,
        collate_fn=Collator(tok),num_workers=workers,persistent_workers=workers>0,generator=generator),len(sampler)


def metrics(logits,batch):
    predicted=logits.detach().argmax(-1);labels=batch['labels'];valid=batch['actions']
    edits=valid&labels.ne(Tags.KEEP);terminal=valid&batch['terminal'][:,None]
    return Counter(tokens=int(valid.sum()),correct=int(((predicted==labels)&valid).sum()),
                   edits=int(edits.sum()),correct_edits=int(((predicted==labels)&edits).sum()),
                   predicted_edits=int((predicted.ne(Tags.KEEP)&valid).sum()),
                   terminal_tokens=int(terminal.sum()),terminal_keep=int((predicted.eq(Tags.KEEP)&terminal).sum()))


def report_metrics(counts):
    return {'tag_accuracy':counts['correct']/max(1,counts['tokens']),
            'edit_tag_accuracy':counts['correct_edits']/max(1,counts['edits']),
            'exact_edit_tag_precision':counts['correct_edits']/max(1,counts['predicted_edits']),
            'terminal_keep_rate':counts['terminal_keep']/max(1,counts['terminal_tokens'])}


def objective(model,batch,tags,args):
    with torch.autocast('cpu',dtype=torch.bfloat16,enabled=args.precision=='bf16'):
        logits,cases=model(batch)
        weights=torch.where(batch['terminal'],args.terminal_weight,1.0)
        if model.training:
            kind_weights=torch.tensor(args.kind_weights,device=weights.device)
            weights=weights*kind_weights[batch['kind']]
        value=loss(logits,cases,batch['labels'],batch['label_cases'],batch['actions'],tags,args.keep_weight,weights)
    return value,logits


@torch.no_grad()
def evaluate(model,tok,args,max_batches=None):
    model.eval();batches,_=loader(args.data,'dev',tok,args.batch_size,args.workers,args.seed,0,True)
    total=steps=0;counts=Counter();slices={kind:Counter() for kind in KINDS}
    tags=Tags(tok)
    for batch in batches:
        value,logits=objective(model,batch,tags,args)
        if not torch.isfinite(value):raise RuntimeError('nonfinite evaluation loss')
        total+=value.item();steps+=1;counts.update(metrics(logits,batch))
        for i,kind in enumerate(KINDS):
            rows=batch['kind'].eq(i)
            if rows.any():slices[kind].update(metrics(logits[rows],{k:v[rows] for k,v in batch.items()}))
        if max_batches and steps>=max_batches:break
    del batches
    model.train()
    return {'loss':total/max(1,steps),'batches':steps,**report_metrics(counts),
            'kinds':{k:report_metrics(v) for k,v in slices.items() if v['tokens']}}


def decode_sample(model,tok,args,limit=4):
    lex=Lexicon();selected={k:[] for k in KINDS}
    for row in records(args.data/'pairs.jsonl.gz'):
        if row['split']=='dev' and len(selected[row['kind']])<limit:selected[row['kind']].append(row)
        if all(len(v)>=limit for v in selected.values()):break
    results=[]
    for kind,rows in selected.items():
        for row in rows:
            try:prediction=guarded_rewrite(model,tok,lex,row['src'],row['mode'])
            except EncodingError as error:prediction={'text':None,'resolved':False,'reason':str(error)}
            text=prediction['text']
            exact=text is not None and [t.text for t in model_tokens(text)]==[t.text for t in model_tokens(row['ste'])]
            results.append({'id':row['id'],'kind':kind,'src':row['src'],'target':row['ste'],
                            'prediction':prediction,'exact_target_tokens':exact})
    model.train()
    return results


def train(args):
    global STOP
    STOP=False
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    summary=json.loads((args.data/'summary.json').read_text())
    audit=json.loads((args.data/'validation.json').read_text())
    summary_hash=sha256(args.data/'summary.json')
    split_audit=json.loads((args.data/'split-validation.json').read_text())
    if not split_audit['passed'] or split_audit['summary_sha256']!=summary_hash or split_audit['code_sha256']!=sha256(Path(__file__).parent/'audit_splits.py'):
        raise ValueError('training requires a passed canonical-text/intermediate split audit')
    if not audit['passed'] or audit['summary_sha256']!=summary_hash or summary['code_sha256']!=fingerprints():
        raise ValueError('training requires a current, passed full rollout audit')
    if sha256(args.vocab/'vocab.json')!=summary['vocab_sha256']:raise ValueError('vocabulary differs')
    for name,expected in summary['files'].items():
        if sha256(args.data/name)!=expected:raise ValueError('training file hash mismatch: '+name)
    tok=SteTokenizer(args.vocab);tags=Tags(tok)
    run_codes={name:sha256(Path(__file__).parent/name) for name in ('train.py','tagger.py','packed.py','safe_editor.py','reviewed_rewrites.py')}
    settings={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','max_steps','max_hours')}
    settings['initial_checkpoint_sha256']=sha256(args.initialize_from) if args.initialize_from else None
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:raise ValueError('use an empty run directory or --resume')
    args.out.mkdir(parents=True,exist_ok=True)
    index=np.load(args.data/'train.index.npy',mmap_mode='r')
    batches_per_epoch=len(batch_sampler(index,args.batch_size,args.seed,0,local_cap=args.local_example_cap,repeats=args.kind_repeats))
    total_steps=sum(len(batch_sampler(index,args.batch_size,args.seed,e,local_cap=args.local_example_cap,repeats=args.kind_repeats)) for e in range(args.epochs))
    model=Tagger(tok)
    initial_hash=None
    if args.initialize_from:
        initial_hash=sha256(args.initialize_from)
        initial=torch.load(args.initialize_from,map_location='cpu',weights_only=True)
        if initial['vocab_sha256']!=summary['vocab_sha256'] or initial['model_config']!=asdict(model.config):
            raise ValueError('initial checkpoint vocabulary or architecture differs')
        model.load_state_dict(initial['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    epoch=next_batch=step=0;elapsed_before=0.;best=math.inf
    if args.resume:
        state=torch.load(args.out/'last.pt',map_location='cpu',weights_only=True)
        if state['settings']!=settings or state['rollout_summary_sha256']!=summary_hash or state['code_sha256']!=run_codes:
            raise ValueError('resume settings, dataset or implementation differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        epoch,next_batch,step=state['epoch'],state['next_batch'],state['step']
        best,elapsed_before=state['best_dev_loss'],state['elapsed_seconds']
        torch.set_rng_state(state['torch_rng_state'])
    config={'settings':settings,'model_config':asdict(model.config),'parameters':sum(p.numel() for p in model.parameters()),
            'rollout_summary_sha256':summary_hash,'rollout_audit_sha256':sha256(args.data/'validation.json'),'split_audit_sha256':sha256(args.data/'split-validation.json'),
            'vocab_sha256':summary['vocab_sha256'],'code_sha256':run_codes,'torch_version':torch.__version__,
            'batches_per_epoch':batches_per_epoch,'total_steps':total_steps,
            'initial_checkpoint_sha256':initial_hash,
            'limits':['Generated holdout diagnostics do not establish owner-verified meaning quality.',
                      'No paid APIs, GPU rentals or remote actions are used.']}
    atomic_json(args.out/'config.json',config)
    start=time.monotonic();last_time=start;last_step=step
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    def elapsed():return elapsed_before+time.monotonic()-start
    def save():
        checkpoint(args.out/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch,'next_batch':next_batch,'step':step,'best_dev_loss':best,
            'elapsed_seconds':elapsed(),'torch_rng_state':torch.get_rng_state(),
            'settings':settings,'rollout_summary_sha256':summary_hash,'code_sha256':run_codes,'model_config':asdict(model.config)})
    def status(state,extra=None):
        atomic_json(args.out/'status.json',{'state':state,'pid':os.getpid(),'epoch':min(epoch+1,args.epochs),'next_batch':next_batch,
            'step':step,'total_steps':total_steps,'elapsed_seconds':elapsed(),**(extra or {})})
    if epoch>=args.epochs or (args.max_steps and step>=args.max_steps):
        status('complete' if epoch>=args.epochs else 'stopped_step_limit');return
    status('running');print(json.dumps({'event':'started',**config}),flush=True)
    model.train();window_loss=0;window_counts=Counter();window_steps=0;final_state='complete'
    with (args.out/'metrics.jsonl').open('a',buffering=1) as log:
        if step==0:
            initial=evaluate(model,tok,args,max_batches=args.quick_eval_batches)
            entry={'event':'dev_initial','step':0,**initial}
            log.write(json.dumps(entry)+'\n');print(json.dumps(entry),flush=True)
        while epoch<args.epochs:
            batches,_=loader(args.data,'train',tok,args.batch_size,args.workers,args.seed,epoch,True,next_batch,args.local_example_cap,args.kind_repeats)
            for batch in batches:
                lr=args.lr*min(1.,(step+1)/args.warmup_steps)*(.1+.9*.5*(1+math.cos(math.pi*min(1.,step/max(1,total_steps)))))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                value,logits=objective(model,batch,tags,args)
                if not torch.isfinite(value):raise RuntimeError('nonfinite training loss')
                value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient norm')
                optimizer.step();step+=1;next_batch+=1
                window_loss+=value.item();window_steps+=1;window_counts.update(metrics(logits,batch))
                if step%args.log_every==0:
                    now=time.monotonic();rate=(step-last_step)*args.batch_size/max(.01,now-last_time)
                    entry={'event':'train','epoch':epoch+1,'step':step,'batch':next_batch,'loss':window_loss/max(1,window_steps),
                           'lr':lr,'examples_per_second':rate,'elapsed_seconds':elapsed(),**report_metrics(window_counts)}
                    log.write(json.dumps(entry)+'\n');print(json.dumps(entry),flush=True)
                    status('running',entry);window_loss=window_steps=0;window_counts.clear();last_time=now;last_step=step
                if step%args.checkpoint_every==0:save()
                if args.eval_every and step%args.eval_every==0:
                    result=evaluate(model,tok,args,max_batches=args.quick_eval_batches)
                    entry={'event':'dev_quick','step':step,'epoch':epoch+1,**result}
                    log.write(json.dumps(entry)+'\n');print(json.dumps(entry),flush=True);save()
                if STOP or (args.max_steps and step>=args.max_steps) or elapsed()>=args.max_hours*3600:
                    final_state='stopped_signal' if STOP else 'stopped_step_limit' if args.max_steps and step>=args.max_steps else 'stopped_time_limit'
                    save();status(final_state);print(json.dumps({'event':final_state,'step':step}),flush=True)
                    return
            del batches
            result=evaluate(model,tok,args)
            decoded=decode_sample(model,tok,args)
            atomic_json(args.out/f'dev-epoch-{epoch+1}.json',{'step':step,'metrics':result,'decode_sample':decoded})
            entry={'event':'dev_epoch','step':step,'epoch':epoch+1,**result}
            log.write(json.dumps(entry)+'\n');print(json.dumps(entry),flush=True)
            if args.retain_epochs:
                checkpoint(args.out/f'epoch-{epoch+1}.pt',{'model':model.state_dict(),'model_config':asdict(model.config),
                    'step':step,'dev_loss':result['loss'],'vocab_sha256':summary['vocab_sha256'],
                    'rollout_summary_sha256':summary_hash,'code_sha256':run_codes})
            if result['loss']<best:
                best=result['loss'];checkpoint(args.out/'best.pt',{'model':model.state_dict(),'model_config':asdict(model.config),
                    'step':step,'dev_loss':best,'vocab_sha256':summary['vocab_sha256'],'rollout_summary_sha256':summary_hash,'code_sha256':run_codes})
            epoch+=1;next_batch=0;save();status('running',{'last_dev':result})
        status(final_state,{'best_dev_loss':best});print(json.dumps({'event':'complete','step':step,'best_dev_loss':best}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--vocab',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--initialize-from',type=Path)
    parser.add_argument('--retain-epochs',action='store_true')
    parser.add_argument('--local-example-cap',type=int)
    parser.add_argument('--kind-weights',type=lambda value:[float(x) for x in value.split(',')],default=[1.,1.,1.,1.,1.,1.],help='original,messy,rule,local_rule,reviewed,doc_teacher training-only weights')
    parser.add_argument('--kind-repeats',type=lambda value:[int(x) for x in value.split(',')],default=[1,1,1,1,1,1],help='times each kind\'s training rows are batched per epoch, same order')
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=8);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--precision',choices=('fp32','bf16'),default='bf16');parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--keep-weight',type=float,default=.1);parser.add_argument('--terminal-weight',type=float,default=.25)
    parser.add_argument('--warmup-steps',type=int,default=500);parser.add_argument('--log-every',type=int,default=100)
    parser.add_argument('--checkpoint-every',type=int,default=1000);parser.add_argument('--eval-every',type=int,default=2000)
    parser.add_argument('--quick-eval-batches',type=int,default=64);parser.add_argument('--max-hours',type=float,default=24)
    parser.add_argument('--max-steps',type=int);parser.add_argument('--seed',type=int,default=0);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if len(args.kind_weights)!=len(KINDS) or min(args.kind_weights)<=0 or len(args.kind_repeats)!=len(KINDS) or min(args.kind_repeats)<1 or (args.local_example_cap is not None and args.local_example_cap<1):
        parser.error('invalid kind weights, kind repeats or local example cap')
    if min(args.epochs,args.batch_size,args.threads,args.warmup_steps,args.log_every,args.checkpoint_every,args.quick_eval_batches,args.max_hours,args.keep_weight,args.terminal_weight)<=0 or args.workers<0:
        parser.error('invalid training settings')
    try:
        train(args)
    except Exception as error:
        if args.out.exists():
            atomic_json(args.out/'status.json',{'state':'failed','pid':os.getpid(),'error_type':type(error).__name__,'error':str(error)})
        raise
