"""One bounded CPU action-diffusion pilot plus an equal-update tagger control."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from tagger import ROOT,Tagger,Tags,Config,SteTokenizer,Lexicon,loss,source_units
from action_diffusion import ActionDiffusion,corrupt,rewrite as diffusion_rewrite
from safe_editor import rewrite as tagger_rewrite,validate
from copy_fidelity import preserved
from train import loader,atomic_json,checkpoint
from rollouts import fingerprints,records,sha256

OUT=ROOT/'data/out';DATA=OUT/'tagger-rollouts-v4';VOCAB=OUT/'fixes-v1/vocabulary'
INITIAL=OUT/'local-train-v2/best.pt';JOB=OUT/'diffusion-pilot-v1'

class BudgetExceeded(Exception):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--minutes',type=float,default=60)
    args=parser.parse_args()
    if args.updates<1 or args.minutes<=0:parser.error('positive budget required')
    if JOB.exists() and (JOB/'config.json').exists():raise ValueError('use a fresh experiment directory')
    JOB.mkdir(parents=True,exist_ok=True);start=time.monotonic();deadline=start+args.minutes*60
    torch.set_num_threads(4)
    def check_budget():
        if time.monotonic()>=deadline:raise BudgetExceeded('fixed wall-clock budget exhausted')
    def status(stage,**extra):atomic_json(JOB/'status.json',dict(stage=stage,elapsed_seconds=time.monotonic()-start,**extra))
    status('tests')
    with (JOB/'tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'model_tagger/tests')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.monotonic()))
    summary=json.loads((DATA/'summary.json').read_text());digest=sha256(DATA/'summary.json')
    for file in ('validation.json','split-validation.json'):
        audit=json.loads((DATA/file).read_text())
        if not audit['passed'] or audit['summary_sha256']!=digest:raise ValueError('missing current audit: '+file)
    if summary['code_sha256']!=fingerprints() or summary['vocab_sha256']!=sha256(VOCAB/'vocab.json'):raise ValueError('source or vocabulary changed')
    for name,expected in summary['files'].items():
        check_budget()
        if sha256(DATA/name)!=expected:raise ValueError('audited data changed: '+name)
    tok=SteTokenizer(VOCAB);tags=Tags(tok);lex=Lexicon()
    initial=torch.load(INITIAL,map_location='cpu',weights_only=True)
    if initial['vocab_sha256']!=summary['vocab_sha256']:raise ValueError('initial vocabulary differs')
    source=JOB/'source';source.mkdir(exist_ok=True);codes={}
    for directory in ('model_tagger','tokenizer','data','checker'):
        for path in (ROOT/directory).glob('*.py'):
            target=source/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);codes[str(path.relative_to(ROOT))]=sha256(path)
    config=dict(updates_per_arm=args.updates,total_update_cap=2*args.updates,wall_clock_cap_minutes=args.minutes,
                source_conditioning='fixed original source; noisy discrete edit-action canvas and continuous noise level',
                forward_process='independent absorbing masks at sampled noise level; force at least one masked valid action',
                reverse_process='confidence-ordered monotone unmasking; at most5steps; validated text only',
                seed=31,batch_size=32,threads=4,lr=3e-5,keep_weight=.3,terminal_weight=.5,kind_weights=[8,12,3,1],
                initial_checkpoint_sha256=sha256(INITIAL),vocab_sha256=summary['vocab_sha256'],rollout_summary_sha256=digest,
                code_sha256=codes,paid_resources=False,
                limitations=['Action-space masked denoising, not a full target-token canvas model.','Both arms use equal updates/batches, not exactly equal parameter count or FLOPs.','No diffusion ELBO or convergence guarantee claimed; finite masked-action reconstruction objective.','Existing content guard only permits articles and reviewed lexical equivalents.','Only the bounded first1000batches of a deterministic replay are used, not the whole training pool.'])
    atomic_json(JOB/'config.json',config)
    trained={};parameters={}
    try:
        for arm in ('diffusion','tagger-control'):
            torch.manual_seed(31);random.seed(31);np.random.seed(31)
            model=ActionDiffusion(tok,Config(**initial['model_config'])) if arm=='diffusion' else Tagger(tok,Config(**initial['model_config']))
            missing,unexpected=model.load_state_dict(initial['model'],strict=False)
            expected={'draft_action.weight','noise_level.weight','noise_level.bias'} if arm=='diffusion' else set()
            if set(missing)!=expected or unexpected:raise ValueError('unexpected initialization mismatch')
            parameters[arm]=sum(p.numel() for p in model.parameters())
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
            batches,_=loader(DATA,'train',tok,32,0,31,0,True,0,100000)
            corruption_rng=torch.Generator().manual_seed(731);model.train();step=0
            try:
                with (JOB/f'{arm}-metrics.jsonl').open('w',buffering=1) as log:
                    for batch in batches:
                        check_budget()
                        optimizer.zero_grad(set_to_none=True)
                        if arm=='diffusion':
                            noisy,noise,valid=corrupt(batch['labels'],batch['actions'],model.mask_action,corruption_rng)
                        else:valid=batch['actions']
                        with torch.autocast('cpu',dtype=torch.bfloat16):
                            scores,cases=model(batch,noisy,noise) if arm=='diffusion' else model(batch)
                            weights=torch.where(batch['terminal'],.5,1.)*torch.tensor([8.,12.,3.,1.])[batch['kind']]
                            objective=loss(scores,cases,batch['labels'],batch['label_cases'],valid,tags,.3,weights)
                        if not torch.isfinite(objective):raise RuntimeError('nonfinite loss')
                        objective.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        if not torch.isfinite(norm):raise RuntimeError('nonfinite gradients')
                        optimizer.step();step+=1
                        if step%100==0:
                            entry=dict(arm=arm,step=step,loss=objective.item(),masked_actions=int(valid.sum()))
                            log.write(json.dumps(entry)+'\n');status('training',**entry)
                        if step>=args.updates:break
            finally:
                checkpoint(JOB/f'{arm}.pt',dict(model=model.state_dict(),model_config=asdict(model.config),arm=arm,step=step,vocab_sha256=summary['vocab_sha256']))
                del optimizer,batches
            if step!=args.updates:raise RuntimeError('not enough batches for requested update budget')
            trained[arm]=model.eval()
        rows=[]
        for path in sorted((OUT/'id-joint-v3/shards').glob('dev-*.jsonl.gz')):rows.extend(records(path))
        rows.sort(key=lambda r:hashlib.sha256(('safe-editor-v1/'+str(r['id'])).encode()).digest())
        calibration=rows[:256];verification=rows[2048:2560]
        if {r['id'] for r in calibration}&{r['id'] for r in verification}:raise RuntimeError('calibration overlap')
        report=dict(config=config,parameters=parameters,calibration={},verification={})
        with gzip.open(JOB/'predictions.jsonl.gz','wt') as predictions:
            def score(arm,model,selected,margin,scope):
                counts=dict(n=len(selected),margin=margin,exact=0,clean_changed=0,resolved=0,copy_failures=0,guard_failures=0)
                rewrite=diffusion_rewrite if arm=='diffusion' else tagger_rewrite
                for row in selected:
                    for clean in (False,True):
                        check_budget();text=row['ste'] if clean else row['src']
                        result=rewrite(model,tok,lex,text,row['mode'],max_passes=5,edit_margin=margin)
                        counts['clean_changed' if clean else 'exact']+=(result['text']!=text if clean else result['text']==row['ste'])
                        counts['resolved']+=result['resolved']
                        counts['guard_failures']+=bool(validate(text,result['text']))
                        original=tok.encode_source(text,row['mode'])
                        counts['copy_failures']+=not preserved(result['text'],original['slots'])
                        predictions.write(json.dumps(dict(arm=arm,scope=scope,id=row['id'],clean=clean,source=text,target=row['ste'],margin=margin,result=result))+'\n')
                counts['clean_change_rate']=counts['clean_changed']/counts['n']
                return counts
            policies={}
            # Fix both policies before reading any reporting-only verification outputs.
            for arm,model in trained.items():
                candidates=[]
                for margin in (0.,.5,1.,2.,4.):
                    status('calibration',arm=arm,margin=margin)
                    candidates.append(score(arm,model,calibration,margin,'calibration'))
                eligible=[r for r in candidates if r['clean_change_rate']<=.05 and not r['copy_failures'] and not r['guard_failures']]
                policy=max(eligible,key=lambda r:(r['exact'],-r['clean_changed'])) if eligible else None
                report['calibration'][arm]=dict(candidates=candidates,chosen=policy)
                if policy:policies[arm]=policy['margin']
            atomic_json(JOB/'selection.json',dict(policies=policies,calibration_ids=[r['id'] for r in calibration],verification_ids=[r['id'] for r in verification]))
            for arm,margin in policies.items():
                status('verification',arm=arm)
                report['verification'][arm]=score(arm,trained[arm],verification,margin,'verification')
        # Frozen proposed software references are reporting-only, no policy tuning.
        report['software']={}
        for arm,margin in policies.items():
            counts=dict(clean_exact=0,repair_exact=0,literal_failures=0,guard_failures=0,unresolved=0);outputs=[]
            rewrite=diffusion_rewrite if arm=='diffusion' else tagger_rewrite
            for case in records(ROOT/'model_tagger/benchmarks/software-review-v1.jsonl'):
                check_budget();result=rewrite(trained[arm],tok,lex,case['source'],case['mode'],edit_margin=margin)
                counts[case['category']+'_exact']+=result['text'] in case['acceptable_outputs']
                counts['unresolved']+=not result['resolved']
                counts['guard_failures']+=bool(validate(case['source'],result['text']))
                counts['literal_failures']+=not preserved(result['text'],tok.encode_source(case['source'],case['mode'])['slots'])
                outputs.append(dict(id=case['id'],source=case['source'],result=result))
            report['software'][arm]=dict(counts=counts,predictions=outputs,review_status='assistant_reviewed_owner_pending')
        atomic_json(JOB/'comparison.json',report);status('complete',complete=True,comparison=str(JOB/'comparison.json'))
    except BudgetExceeded as exc:
        status('budget_exhausted',complete=False,reason=str(exc))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        JOB.mkdir(parents=True,exist_ok=True);atomic_json(JOB/'status.json',dict(stage='failed',error_type=type(exc).__name__,error=str(exc)));raise
