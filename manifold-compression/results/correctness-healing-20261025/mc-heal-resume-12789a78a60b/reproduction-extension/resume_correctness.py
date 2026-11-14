"""Continue the frozen pilot from its last adapter and optimizer, without new controls."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from correctness_heal import (load_source,project,install_lora,load_adapter,save_adapter,
    recipe_hashes,verify_source,load_tokenizer,encode_complete,selection_key,
    set_training,train_update,evaluate,legacy_generate,cleanup,save)


def save_training_state(model,optimizer,folder,metadata,steps,settings):
    save_adapter(model,folder/'adapter-last',metadata|{'step':steps})
    torch.save({'optimizer':optimizer.state_dict(),'steps_completed':steps,
        'data_sha256':metadata['data_sha256'],'settings':settings,
        'torch_rng_state':torch.get_rng_state(),
        'cuda_rng_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},folder/'optimizer-last.pt')


def restore_training_state(model,optimizer,folder,data_hash,settings):
    state=torch.load(folder/'optimizer-last.pt',map_location='cpu',weights_only=True)
    metadata=json.loads((folder/'adapter-last/adapter.json').read_text())
    if state['data_sha256']!=data_hash or metadata['data_sha256']!=data_hash or state['settings']!=settings:
        raise ValueError('Resume data/settings mismatch')
    if state['steps_completed']!=metadata['step']:raise ValueError('Adapter and optimizer steps differ')
    load_adapter(model,folder/'adapter-last');optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng_state'])
    if torch.cuda.is_available() and state['cuda_rng_state']:torch.cuda.set_rng_state_all(state['cuda_rng_state'])
    return state['steps_completed']


def run(args):
    data=json.loads(args.data.read_text());settings=data['settings']
    data_hash=hashlib.sha256(args.data.read_bytes()).hexdigest()
    prior=json.loads((args.prior/'result.json').read_text())
    if prior['status']!='complete' or prior.get('stop_reason')=='development regression gate':
        raise ValueError('Cannot continue an incomplete run or bypass its regression gate')
    if prior['data_sha256']!=data_hash or prior['settings']!=settings:raise ValueError('Frozen experiment changed')
    if len(prior['steps'])>=len(data['updates']):raise ValueError('All planned updates already completed')
    meta=json.loads((args.prior/'adapter-last/adapter.json').read_text())
    if meta['source']!=data['source'] or meta['recipe_sha256']!=recipe_hashes(args.recipe):raise ValueError('Source recipe changed')
    verify_source(args.source,data['source'])
    if args.out.exists():raise ValueError('Use a fresh continuation directory')
    args.out.mkdir(parents=True)
    for label in ['stock','full_width','unhealed_75']:
        for suffix in ['dev.json','test.json','legacy72.json']:
            shutil.copy2(args.prior/(label+'-'+suffix),args.out/(label+'-'+suffix))
    report=copy.deepcopy(prior);report['status']='running';report.pop('stop_reason',None)
    report['prior_segment']={'path':str(args.prior),'updates':len(prior['steps']),
        'selected_step':prior['selected_step'],'healed_selected':report['arms'].pop('healed_selected')}
    start=time.monotonic()
    def state(stage):
        report['stage']=stage;report['continuation_seconds']=time.monotonic()-start
        save(report,args.out/'result.json');print(stage,flush=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state('reconstructing saved student')
    tok=load_tokenizer(args.source);model,_=load_source(args.source,args.device)
    project(model,args.recipe,data['width']);info=install_lora(model,settings['rank'],settings['alpha'])
    if info['modules']!=meta['modules']:raise ValueError('Adapter module layout differs')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['lr'],
        betas=(.9,.95),weight_decay=0.,foreach=False)
    completed=restore_training_state(model,optimizer,args.prior,data_hash,settings)
    if completed!=len(prior['steps']):raise ValueError('Prior step count differs')
    encoded=[encode_complete(tok,r['messages'],data['max_length']) for r in data['train']]
    for raw,row in zip(data['train'],encoded,strict=True):
        if row is None or len(row['targets'])!=raw['assistant_tokens'] or len(row['input_ids'])!=raw['input_tokens']:
            raise ValueError('Training token counts changed')
    baseline=report['arms']['unhealed_75']['dev'];best_step=prior['selected_step'];best_path=args.prior/'adapter-best'
    best_metrics=baseline if best_step==0 else next(c['dev'] for c in prior['checkpoints'] if c['step']==best_step)
    best_key=selection_key(best_metrics,baseline,best_step)
    def assess(step,destination):
        nonlocal best_path,best_step,best_key
        model.gradient_checkpointing_disable();model.eval()
        save_adapter(model,destination,meta|{'step':step})
        state(f'development evaluation: step {step}')
        measured=evaluate(model,tok,data['dev'],destination/'dev.json')
        report['checkpoints'].append({'step':step,'dev':measured})
        key=selection_key(measured,baseline,step)
        if key is not None and key>best_key:best_key=key;best_path=destination;best_step=step
        report['selected_step']=best_step
        return measured['correct']<=baseline['correct']-4 or measured['loop_flags']>=baseline['loop_flags']+2
    for index in range(completed,len(data['updates'])):
        predicted=1.5*report['steps'][-1]['seconds']
        # Fifteen minutes cover a possible last development check and final
        # held-out generation, using observed durations from the first segment.
        if time.time()+predicted>args.deadline-900:
            report['stop_reason']='reserved final evaluation time';break
        step=index+1;set_training(model)
        metrics=train_update(model,optimizer,[encoded[i] for i in data['updates'][index]],args.device,
            settings['lr']*min(1.,step/settings['warmup_steps']))
        report['steps'].append({'step':step,**metrics});state('training')
        if step in settings['evaluation_steps'] or step==len(data['updates']):
            if assess(step,args.out/f'step-{step:04d}'):
                report['stop_reason']='development regression gate';break
    last=len(report['steps']);save_training_state(model,optimizer,args.out,meta,last,settings)
    del optimizer;model.gradient_checkpointing_disable();model.eval();cleanup()
    if last not in [c['step'] for c in report['checkpoints']]:assess(last,args.out/'adapter-last')
    load_adapter(model,best_path);save_adapter(model,args.out/'adapter-best',meta|{'step':best_step})
    state('selected checkpoint final evaluation')
    legacy=json.loads(args.legacy_data.read_text())['generation_eval']
    report['arms']['healed_selected']={'selected_step':best_step,
        'test':evaluate(model,tok,data['test'],args.out/'healed-selected-test.json'),
        'legacy72':legacy_generate(model,tok,legacy,args.out/'healed-selected-legacy72.json',4)}
    ids=torch.tensor([encoded[0]['input_ids'][:128]],device=args.device)
    with torch.no_grad():
        expected=model(ids,use_cache=False).logits.cpu()
        for p in model.parameters():
            if p.requires_grad:p.zero_()
    load_adapter(model,args.out/'adapter-best')
    with torch.no_grad():actual=model(ids,use_cache=False).logits.cpu()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    report['adapter_reload_exact']=True;report['selected_step']=best_step
    report['selected_improved_development']=best_step>0;report['status']='complete'
    report['planned_updates_completed']=last==len(data['updates'])
    state('complete')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['source','recipe','data','legacy-data','prior','out']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--deadline',type=float,required=True)
    run(ap.parse_args())
