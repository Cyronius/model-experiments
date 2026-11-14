"""Fit 384 output feature scales on audited human labels, with frozen encoders."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from pair_training_state import atomic_json,atomic_torch_save,digest
from data_first_build import visible_key
from pair_validation import PairValidation

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
DATA=ROOT/'work/human-similarity'
OUT=ROOT/'human-similarity-head/results'


def ranking_loss(similarity,labels,scale=20.):
    differences=scale*(similarity[:,None]-similarity[None,:])
    ordered=labels[:,None]<labels[None,:]
    return torch.logsumexp(torch.cat((similarity.new_zeros(1),differences[ordered])),0)


class FeatureScales(torch.nn.Module):
    def __init__(self,dims=384):
        super().__init__();self.log_scales=torch.nn.Parameter(torch.zeros(dims))
    def scales(self):return self.log_scales.clamp(-.5,.5).exp()
    def forward(self,vectors):return F.normalize(vectors*self.scales(),dim=-1)


class CachedHead(torch.nn.Module):
    def __init__(self,head,vectors,lookup):
        super().__init__();self.head=head;self.vectors=vectors;self.lookup=lookup;self.dims={'out_dim':384}
    def forward(self,ids,mask):
        indexes=[self.lookup[visible_key(row[:n])] for row,n in zip(ids.numpy(),mask.sum(-1).tolist())]
        return None,None,self.head(self.vectors[indexes])


def read(path):return json.loads(path.read_text())


def main():
    torch.set_num_threads(2)
    sys.path.insert(0,str(PROJECT/'retrieval-diagnostics'))
    from diagnose import student,cached_encode
    OUT.mkdir(parents=True,exist_ok=True)
    audit=read(DATA/'audit.json')
    for split,expected in audit['output_sha256'].items():assert digest(DATA/(split+'.jsonl'))==expected
    train=[json.loads(line) for line in (DATA/'train.jsonl').open()]
    dev=[json.loads(line) for line in (DATA/'validation.jsonl').open()]
    qa_path=ROOT/'work/broad-data/ready/dev.jsonl';assert digest(qa_path)==audit['qa_retention_dev_sha256']
    qa=[json.loads(line) for line in qa_path.open()]
    tokenizer=AutoTokenizer.from_pretrained(PROJECT/'tokenizer',local_files_only=True)
    validation=PairValidation(qa,tokenizer)
    texts=list(dict.fromkeys(validation.texts+[r[k] for rows in (train,dev) for r in rows for k in ('sentence1','sentence2')]))
    lookup={text:i for i,text in enumerate(texts)}
    token_ids=tokenizer(texts,truncation=True,max_length=128)['input_ids']
    token_lookup={visible_key(ids):i for i,ids in enumerate(token_ids)}
    train_a=torch.tensor([lookup[r['sentence1']] for r in train]);train_b=torch.tensor([lookup[r['sentence2']] for r in train]);labels=torch.tensor([r['score'] for r in train])
    dev_a=torch.tensor([lookup[r['sentence1']] for r in dev]);dev_b=torch.tensor([lookup[r['sentence2']] for r in dev]);dev_labels=np.array([r['score'] for r in dev])
    family=read(PROJECT/'models/campaign-20261025/frozen.json')['models']
    broad=read(ROOT/'broad-data/results/frozen.json')['models']
    sources={}
    for precision in ('float','ternary'):
        sources['parent-'+precision]=dict(family[precision],summary=str(ROOT/'delivery/family-results/summary.json'),arm=precision)
        sources['broad-'+precision]=dict(broad[precision+'-broad'],summary=str(ROOT/'broad-data/results/summary.json'),arm=precision+'-broad')
    settings=dict(trainable_feature_scales=384,maximum_steps=100,batch=128,lr=.01,seed=71,
        log_scale_bounds=[-.5,.5],identity_regularization=.1,qa_source_floor=1.,qa_macro_floor=.25,
        selection='Highest official STS validation Spearman subject to both QA retention floors; identity step0 eligible.',
        human_train_rows=len(train),human_dev_rows=len(dev),data_audit_sha256=digest(DATA/'audit.json'),
        source_models=sources,teacher_targets=False,architecture_parameters=17241984,
        scope='Exploratory local output metric fit after observing the broad-data failure. Benchmark tests remain observed development; no new reserved judgments may be opened.')
    plan=ROOT/'human-similarity-head/plan.json'
    if plan.exists():assert read(plan)==settings
    else:atomic_json(plan,settings)
    models={}
    for name,source in sources.items():
        record=OUT/(name+'-fit.json')
        if record.exists():models[name]=read(record)['selection'];continue
        path=Path(source['path']);assert digest(path)==source['sha256']
        precision=source['precision'];encoder=student(path,'quant' if precision=='ternary' else 'float')
        encoder.requires_grad_(False)
        identity=dict(sha256=source['sha256'],precision=precision,token_cap=128,tokenizer_sha256=digest(PROJECT/'tokenizer/tokenizer.json'))
        vectors=torch.from_numpy(cached_encode(encoder,tokenizer,texts,False,64,identity,DATA/'vectors'/(source['sha256']+'.npy')))
        head=FeatureScales();cached=CachedHead(head,vectors,token_lookup)
        optimizer=torch.optim.Adam(head.parameters(),lr=settings['lr'])
        torch.manual_seed(71);rng=np.random.default_rng(71);order=np.empty(0,dtype=np.int64)
        best_score=-float('inf');best_step=0;best_scale=head.scales().detach().clone();history=[];baseline=None
        for step in range(settings['maximum_steps']+1):
            if step%20==0 or step==settings['maximum_steps']:
                with torch.inference_mode():
                    similarity=(head(vectors[dev_a])*head(vectors[dev_b])).sum(-1).numpy()
                    semantic=float(spearmanr(dev_labels,similarity).statistic)*100
                    ranking=validation(cached,'cpu')
                if baseline is None:baseline=ranking
                eligible=ranking['macro_ndcg10']>=baseline['macro_ndcg10']-settings['qa_macro_floor'] and all(v['ndcg10']>=baseline['by_source'][s]['ndcg10']-settings['qa_source_floor'] for s,v in ranking['by_source'].items())
                history.append(dict(step=step,semantic_dev_spearman=semantic,qa_macro=ranking['macro_ndcg10'],eligible=eligible))
                if eligible and semantic>best_score:
                    best_score=semantic;best_step=step;best_scale=head.scales().detach().clone()
                print(name,json.dumps(history[-1]),flush=True)
            if step==settings['maximum_steps']:break
            if len(order)<2:order=rng.permutation(len(train))
            indexes=order[:settings['batch']];order=order[len(indexes):]
            optimizer.zero_grad(set_to_none=True)
            similarity=(head(vectors[train_a[indexes]])*head(vectors[train_b[indexes]])).sum(-1)
            loss=ranking_loss(similarity,labels[indexes])+.1*head.log_scales.square().mean()
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
        blob=torch.load(path,map_location='cpu',weights_only=False)
        if best_step==0:
            selection=dict(path=str(path),sha256=source['sha256'],precision=precision,layers=3,eligible_for_promotion=False,selected_identity=True,
                source_summary=source['summary'],source_summary_sha256=digest(Path(source['summary'])),source_arm=source['arm'])
            error=0.
        else:
            state=dict(blob['state']);state['out.weight']=best_scale[:,None]*state['out.weight']
            target=OUT/(name+'.pt')
            atomic_torch_save(target,dict(dims=blob['dims'],quant=blob['quant'],state=state,
                meta=dict(frozen_encoder=True,trained_feature_scales=384,head_step=best_step,source_sha256=source['sha256'],data_audit_sha256=digest(DATA/'audit.json'))))
            deployed=student(target,'quant' if precision=='ternary' else 'float')
            for key,value in deployed.state_dict().items():
                if key!='out.weight':assert torch.equal(value,encoder.state_dict()[key])
            sample=tokenizer(texts[:8],padding=True,truncation=True,max_length=128,return_tensors='pt')
            with torch.inference_mode():
                actual=deployed(sample['input_ids'],sample['attention_mask'].bool())[2]
                expected=F.normalize(vectors[:8]*best_scale,dim=-1)
            error=float((actual-expected).abs().max());assert error<1e-5
            selection=dict(path=str(target),sha256=digest(target),precision=precision,layers=3,eligible_for_promotion=False)
        atomic_json(record,dict(selection=selection,selected_step=best_step,selected_semantic_dev_spearman=best_score,
            history=history,full_encoder_unchanged=True,materialization_max_abs_error=error,scale_min=float(best_scale.min()),scale_max=float(best_scale.max())))
        models[name]=selection
    freeze=OUT/'frozen.json'
    if freeze.exists():assert read(freeze)['models']==models
    else:atomic_json(freeze,dict(models=models,local_only=True,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope=settings['scope'],selection=settings['selection'],plan_sha256=digest(plan)))
    if not (OUT/'summary.json').exists():
        subprocess.run([sys.executable,str(ROOT/'score_campaign.py'),'--profile','human-similarity-head','--arms',*models,
            '--local-frozen','--mteb','--dibs-inner'],check=True)
    print('Human-similarity head fit and fixed12 development evaluation complete.',flush=True)


if __name__=='__main__':main()
