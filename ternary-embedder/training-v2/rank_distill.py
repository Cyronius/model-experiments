"""Dimension-independent ranking distillation with a gradual QAT transition."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'tern-embed'))
from model import Student,TernLinear,tern_codes,quant_act,ste

class GradualLinear(TernLinear):
    alpha=0.
    def forward(self,x):
        if not self.quant:return F.linear(x,self.weight,self.bias)
        if self.alpha==0.:return F.linear(x,self.weight,self.bias)
        codes,scale=tern_codes(self.weight)
        if self.alpha==1.:return F.linear(quant_act(x),ste(self.weight,codes*scale),self.bias)
        weight=self.weight+self.alpha*(ste(self.weight,codes*scale)-self.weight)
        value=x+self.alpha*(quant_act(x)-x)
        return F.linear(value,weight,self.bias)

def install(model):
    for layer in model.layers:
        for name in ('q','k','v','o','fc1','fc2'):
            old=getattr(layer,name)
            new=GradualLinear(old.in_features,old.out_features)
            new.load_state_dict(old.state_dict());new.quant=True
            setattr(layer,name,new)

def set_alpha(model,alpha):
    for m in model.modules():
        if isinstance(m,GradualLinear):m.alpha=alpha

def schedule(step,total,float_fraction,ramp_fraction):
    start=int(total*float_fraction);end=int(total*(float_fraction+ramp_fraction))
    return max(0.,min(1.,(step-start)/max(1,end-start)))

def ranking_loss(query,docs,tquery,tdocs,temperature=.05):
    target=(tquery.float()@tdocs.float().T/temperature).softmax(-1)
    return F.kl_div((query.float()@docs.float().T/temperature).log_softmax(-1),target,reduction='batchmean')

def positive_loss(query,docs,positive_mask,temperature=.05):
    logits=query.float()@docs.float().T/temperature
    if not positive_mask.any(-1).all():raise ValueError('Every query needs a known positive')
    # Marginalize known positives rather than penalizing another valid passage.
    return (logits.logsumexp(-1)-logits.masked_fill(~positive_mask,float('-inf')).logsumexp(-1)).mean()

def split_queries(rows,rng):
    groups=sorted(set(r['query'] for r in rows));rng.shuffle(groups)
    held=set(groups[:max(2,len(groups)//20)])
    valid=[i for i,r in enumerate(rows) if r['query'] in held]
    # Duplicate-question triples can otherwise leak a held query as a passage.
    train=[i for i,r in enumerate(rows) if not any(r[k] in held for k in ('query','positive','negative'))]
    return train,valid,held

def save_checkpoint(model,path,**metadata):
    """Never replace a usable checkpoint with an interrupted partial write."""
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    try:
        model.save(temporary,**metadata)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--anchor-checkpoint',type=Path,help='Optional independent checkpoint for vector preservation')
    p.add_argument('--checkpoint',type=Path,default=PROJECT/'retrieval-diagnostics/float-control/runs/float150/model.pt')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--float-fraction',type=float,default=.5);p.add_argument('--ramp-fraction',type=float,default=.25)
    p.add_argument('--positive-weight',type=float,default=0.,help='Optional verified-positive marginal contrastive loss')
    p.add_argument('--temperature',type=float,default=.05);p.add_argument('--anchor-weight',type=float,default=.1)
    p.add_argument('--qat-continuation',action='store_true',help='Keep an existing QAT checkpoint quantized for the entire run')
    p.add_argument('--restart-qat-lr',action='store_true',help='Restart the learning-rate schedule when QAT starts')
    p.add_argument('--hard-negatives',action='store_true');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if a.steps<4 or a.batch<2 or not 0<=a.float_fraction<1 or not 0<=a.ramp_fraction or a.float_fraction+a.ramp_fraction>=1:
        p.error('Need >=4 steps, batch>=2, and a final fully quantized training phase')
    if a.temperature<=0 or a.positive_weight<0:p.error('Temperature must be positive and positive weight nonnegative')
    if a.qat_continuation and (a.float_fraction!=0 or a.ramp_fraction!=0):p.error('QAT continuation requires zero float/ramp fractions')
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(a.threads);torch.manual_seed(a.seed)
    rng=np.random.default_rng(a.seed)
    rows=[json.loads(l) for l in a.data.read_text().splitlines()]
    teacher=np.load(a.teacher);meta=json.loads(a.teacher.with_suffix('.json').read_text())
    if meta['data_sha256']!=sha(a.data):raise ValueError('Teacher/data mismatch')
    if teacher.shape[:2]!=(len(rows),3) or not np.isfinite(teacher).all():raise ValueError('Invalid teacher vectors')
    # Query-group holdout prevents multiple positives of one query crossing splits.
    train,valid,held=split_queries(rows,rng)
    if len(train)<a.batch:p.error('Not enough distinct training rows for the batch')
    model=Student.load(a.checkpoint)
    if model.quantized!=a.qat_continuation:
        raise ValueError('Float/ramp needs the independent float control; QAT continuation needs an existing quantized checkpoint')
    baseline=Student.load(a.anchor_checkpoint or a.checkpoint).eval().to(a.device).requires_grad_(False)
    install(model);model=model.to(a.device).train()
    tk=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(dims=model.dims,checkpoint_sha256=sha(a.checkpoint),anchor_checkpoint_sha256=sha(a.anchor_checkpoint or a.checkpoint),data_sha256=sha(a.data),
                  teacher_sha256=sha(a.teacher),teacher_metadata=meta,train_rows=len(train),validation_rows=len(valid),
                  heldout_queries_excluded_from_all_training_text_roles=True,
                  heldout_query_sha256=hashlib.sha256(json.dumps(sorted(held)).encode()).hexdigest())
    (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    token_counts={'train_real':0,'train_padded':0}
    def forward(batch,collect=False):
        # Deduplicate passages, including positive/negative collisions, before softmax.
        docs=list(dict.fromkeys([rows[i]['positive'] for i in batch]+[rows[i]['negative'] for i in batch]))
        tdocs={}
        for i in batch:
            tdocs[rows[i]['positive']]=teacher[i,1]
            tdocs[rows[i]['negative']]=teacher[i,2]
        texts=[rows[i]['query'] for i in batch]+docs
        enc=tk(texts,padding=True,truncation=True,max_length=128,return_tensors='pt').to(a.device)
        if model.training:
            token_counts['train_real']+=int(enc['attention_mask'].sum())
            token_counts['train_padded']+=enc['input_ids'].numel()
        vectors=model(enc['input_ids'],enc['attention_mask'].bool())[2]
        with torch.no_grad():anchor=baseline(enc['input_ids'],enc['attention_mask'].bool())[2]
        nq=len(batch);tq=torch.from_numpy(teacher[batch,0].astype(np.float32)).to(a.device)
        td=torch.from_numpy(np.stack([tdocs[d] for d in docs]).astype(np.float32)).to(a.device)
        selected=list(range(len(docs))) if a.hard_negatives else [docs.index(d) for d in dict.fromkeys(rows[i]['positive'] for i in batch)]
        kd=ranking_loss(vectors[:nq],vectors[nq:][selected],tq,td[selected],a.temperature)
        preservation=1-(vectors.float()*anchor.float()).sum(-1).mean()
        if collect:
            scores=vectors[:nq].float()@vectors[nq:].float().T
            labels=torch.tensor([docs.index(rows[i]['positive']) for i in batch],device=a.device)
            positive=scores.gather(1,labels[:,None])
            ranks=1+(scores>positive).sum(1)
            return kd,ranks
        supervised=vectors.new_zeros(())
        if a.positive_weight:
            known={}
            for i in batch:known.setdefault(rows[i]['query'],set()).add(rows[i]['positive'])
            mask=torch.tensor([[docs[j] in known[rows[i]['query']] for j in selected] for i in batch],device=a.device,dtype=torch.bool)
            supervised=positive_loss(vectors[:nq],vectors[nq:][selected],mask,a.temperature)
        return kd+a.anchor_weight*preservation+a.positive_weight*supervised,kd,preservation
    @torch.no_grad()
    def validation():
        model.eval();values=[];ranks=[];source_ranks={}
        for i in range(0,len(valid),a.batch):
            b=valid[i:i+a.batch]
            if len(b)>1:
                kd,rr=forward(b,collect=True);values.append(float(kd));ranks.extend(rr.tolist())
                for index,rank in zip(b,rr.tolist()):source_ranks.setdefault(rows[index].get('source','msmarco'),[]).append(rank)
        model.train();return {'ranking_kl':float(np.mean(values)),'mrr':float(np.mean(1/np.array(ranks))),
                              'recall1':float(np.mean(np.array(ranks)==1)),'queries':len(ranks),
                              'by_source':{k:{'mrr':float(np.mean(1/np.array(v))),'queries':len(v)} for k,v in source_ranks.items()}}
    start=time.perf_counter();log=(a.out/'log.jsonl').open('w')
    def emit(row):
        print(json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
    initial_alpha=1. if a.qat_continuation else 0.
    set_alpha(model,initial_alpha);emit({'step':0,'alpha':initial_alpha,'validation':validation()})
    float_saved=a.qat_continuation;best_mrr=-1.;best_step=None
    for step in range(1,a.steps+1):
        alpha=1. if a.qat_continuation else schedule(step,a.steps,a.float_fraction,a.ramp_fraction)
        if alpha>0 and not float_saved:
            # Save an ordinary loadable float checkpoint, never a blended checkpoint.
            model.set_quant(False);save_checkpoint(model,a.out/'float.pt',step=step-1);model.set_quant(True);float_saved=True
        set_alpha(model,alpha)
        batch=rng.choice(train,size=a.batch,replace=False).tolist()
        progress=(step-1)/a.steps
        if a.restart_qat_lr and alpha>0:
            progress=(step-1-int(a.steps*a.float_fraction))/max(1,a.steps-int(a.steps*a.float_fraction))
        lr=a.lr*(.1+.9*(1+math.cos(math.pi*progress))/2)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):
            loss,kd,anchor=forward(batch)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
        opt.step()
        if step%100==0 or step==a.steps:
            metrics=validation()
            if alpha==1. and metrics['mrr']>best_mrr:
                best_mrr=metrics['mrr'];best_step=step
                save_checkpoint(model,a.out/'best.pt',step=step,validation=metrics,config=config)
            emit({'step':step,'alpha':alpha,'ranking_kl':float(kd.detach()),'anchor':float(anchor.detach()),
                  'lr':lr,'total_loss':float(loss.detach()),'validation':metrics,'seconds':time.perf_counter()-start,
                  'token_counts':dict(token_counts),'best_full_qat_step':best_step})
    assert alpha==1.
    model.eval();save_checkpoint(model,a.out/'model.pt',config=config,steps=a.steps,token_counts=token_counts)
    # A standard Student loader must exactly reproduce the final QAT arithmetic.
    loaded=Student.load(a.out/'model.pt').to(a.device).eval()
    enc=tk([rows[train[0]]['query']],return_tensors='pt',truncation=True,max_length=128).to(a.device)
    with torch.no_grad():
        assert torch.equal(model(enc['input_ids'],enc['attention_mask'].bool())[2],loaded(enc['input_ids'],enc['attention_mask'].bool())[2])
    emit({'checkpoint_reload_exact':True,'final_sha256':sha(a.out/'model.pt')})
if __name__=='__main__':main()
