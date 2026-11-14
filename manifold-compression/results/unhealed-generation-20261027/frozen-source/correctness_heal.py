"""Bounded correctness-focused LoRA repair of the saved Qwen27 projection.

No teacher distribution enters the training objective. Base parameters and the
projection matrix remain frozen. Saves portable adapter tensors and the exact
recipe; the export command reconstructs and writes a merged HF checkpoint.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file
from transformers import GenerationConfig

from audit_rotation import norm_gain, cache_check
from correctness_eval import grade
from healing_data import encode_complete, generation_metrics
from projection_models import project_residual_width, projected_output_matrix, install_output_matrix
from scale_data import load_tokenizer
from scale_probe import (load_source, save, cleanup, generate as legacy_generate,
                         validation_rows, evaluate_logits)
from summarize_scale_probe import repeated_line_run


class LowRankLinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.rank=rank;self.alpha=alpha
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))

    def forward(self,x):
        y=self.base(x)
        update=F.linear(F.linear(x.float(),self.a),self.b)*(self.alpha/self.rank)
        return y+update.to(y.dtype)


def install_lora(model,rank=16,alpha=32):
    model.requires_grad_(False)
    names=[]
    for name,module in list(model.named_modules()):
        if name.startswith('model.layers.') and isinstance(module,nn.Linear):
            parent,leaf=name.rsplit('.',1)
            setattr(model.get_submodule(parent),leaf,LowRankLinear(module,rank,alpha))
            names.append(name)
    if not names:raise ValueError('No body linear modules found')
    return {'rank':rank,'alpha':alpha,'modules':names,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}


def adapter_tensors(model):
    return {name:p.detach().cpu().contiguous() for name,p in model.named_parameters() if p.requires_grad}


def save_adapter(model,path,metadata):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    tensors=adapter_tensors(model)
    if not tensors or any(not (n.endswith('.a') or n.endswith('.b')) for n in tensors):
        raise ValueError('Only LoRA parameters may be saved as an adapter')
    save_file(tensors,str(path/'adapter.safetensors.tmp'))
    (path/'adapter.safetensors.tmp').replace(path/'adapter.safetensors')
    save(metadata,path/'adapter.json')


@torch.no_grad()
def load_adapter(model,path):
    tensors=load_file(str(Path(path)/'adapter.safetensors'))
    current={n:p for n,p in model.named_parameters() if p.requires_grad}
    if tensors.keys()!=current.keys():raise ValueError('Adapter parameter names differ')
    for name,p in current.items():
        if p.shape!=tensors[name].shape:raise ValueError('Adapter shape differs')
        p.copy_(tensors[name].to(device=p.device,dtype=p.dtype))


@torch.no_grad()
def merge_lora(model):
    for name,module in list(model.named_modules()):
        if not isinstance(module,LowRankLinear):continue
        for start in range(0,module.base.out_features,1024):
            merged=module.base.weight[start:start+1024].float()+(
                module.b[start:start+1024]@module.a)*(module.alpha/module.rank)
            module.base.weight[start:start+1024].copy_(merged.to(module.base.weight.dtype))
        parent,leaf=name.rsplit('.',1);setattr(model.get_submodule(parent),leaf,module.base)
    return model


def answer_loss(model,row,device,chunk=64):
    ids=torch.tensor([row['input_ids']],device=device)
    states=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,row['positions']]
    labels=torch.tensor(row['targets'],device=device)
    def part(h,y):return F.cross_entropy(model.lm_head(h).float(),y,reduction='sum')
    return sum(checkpoint(part,states[i:i+chunk],labels[i:i+chunk],use_reentrant=False)
               for i in range(0,len(labels),chunk))


def train_update(model,optimizer,rows,device,lr):
    optimizer.zero_grad(set_to_none=True)
    targets=sum(len(r['targets']) for r in rows)
    loss_sum=0.;start=time.monotonic()
    for row in rows:
        value=answer_loss(model,row,device)/targets
        if not torch.isfinite(value):raise RuntimeError('Nonfinite answer loss')
        value.backward();loss_sum+=value.detach().item()
    parameters=[p for p in model.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.step()
    return {'mean_answer_ce':loss_sum,'assistant_tokens':targets,
            'input_tokens':sum(len(r['input_ids']) for r in rows),'rows':len(rows),
            'grad_norm':norm.item(),'lr':lr,'seconds':time.monotonic()-start}


@torch.no_grad()
def evaluate(model,tok,cases,path,batch_size=4):
    model.eval();tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    result={'cases':[]};tick=time.monotonic();device=next(model.parameters()).device
    config=GenerationConfig(do_sample=False,use_cache=True,eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,bos_token_id=tok.bos_token_id,repetition_penalty=1.,no_repeat_ngram_size=0)
    for limit in sorted({c['max_new_tokens'] for c in cases}):
        config.max_new_tokens=limit
        group=[c for c in cases if c['max_new_tokens']==limit]
        for i in range(0,len(group),batch_size):
            subset=group[i:i+batch_size]
            texts=[tok.apply_chat_template(c['messages'],tokenize=False,add_generation_prompt=True,
                                           enable_thinking=False) for c in subset]
            inputs=tok(texts,add_special_tokens=False,padding=True,return_tensors='pt').to(device)
            output=model.generate(**inputs,generation_config=config)[:,inputs.input_ids.shape[1]:]
            for c,ids in zip(subset,output.tolist(),strict=True):
                if tok.eos_token_id in ids:ids=ids[:ids.index(tok.eos_token_id)+1]
                answer=tok.decode(ids,skip_special_tokens=True)
                metrics=generation_metrics(answer,ids,tok.eos_token_id)
                line_flag=repeated_line_run(answer)>=8
                result['cases'].append({'id':c['id'],'family':c['family'],'answer':answer,
                    'grade':grade(answer,c['check']) if c['check'] is not None else None,
                    **metrics,'repeated_line_flag':line_flag,'combined_loop_flag':metrics['loop_flag'] or line_flag})
            save(result,path)
    scored=[c for c in result['cases'] if c['grade'] is not None]
    result['summary']={'cases':len(cases),'scored':len(scored),
        'correct':sum(c['grade']['correct'] for c in scored),
        'format_pass':sum(c['grade']['format_pass'] for c in scored),
        'unparseable':sum(not c['grade']['parsed'] for c in scored),
        'loop_flags':sum(c['combined_loop_flag'] for c in result['cases']),
        'ended':sum(c['ended'] for c in result['cases']),'seconds':time.monotonic()-tick}
    save(result,path);return result['summary']


def selection_key(metrics,baseline,step):
    if metrics['loop_flags']>baseline['loop_flags'] or metrics['ended']<baseline['ended']-1:return None
    return (metrics['correct'],-metrics['loop_flags'],metrics['ended'],-step)


def recipe_hashes(recipe):
    return {n:hashlib.sha256((recipe/n).read_bytes()).hexdigest()
            for n in ['basis.pt','rms-scales.json','result.json']}


def verify_source(path, expected):
    if json.loads((path/'source.json').read_text())!=expected:
        raise ValueError('Pinned source identity differs')


@torch.no_grad()
def project(model,recipe,width):
    basis=torch.load(recipe/'basis.pt',weights_only=True,map_location='cpu')
    d=model.config.hidden_size
    if basis.shape!=(d,d):raise ValueError('Wrong basis dimensions')
    p=basis[:,:width].contiguous().to(next(model.parameters()).device)
    if width==d:scales=[1.]*(2*len(model.model.layers)+1)
    else:scales=json.loads((recipe/'rms-scales.json').read_text())[str(width)]
    gain=norm_gain(model.model.norm)
    project_residual_width(model,p,scales,chunk_size=1024)
    install_output_matrix(model,projected_output_matrix(gain,p,scales[-1]).to(torch.bfloat16))
    return model


def set_training(model):
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False


def smoke(device):
    """Exercise both Qwen attention types and BF16 adapter backward on-device."""
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    config=Qwen3_5TextConfig(vocab_size=257,hidden_size=256,intermediate_size=512,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=64,
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=128,linear_value_head_dim=128,
        layer_types=['linear_attention','full_attention'],tie_word_embeddings=False,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[8,8,16]})
    config._attn_implementation='eager'
    torch.manual_seed(73);torch.set_num_threads(4)
    model=Qwen3_5ForCausalLM(config).eval().to(device=device,dtype=torch.bfloat16)
    p=torch.linalg.qr(torch.randn(256,256)).Q[:,:192].contiguous().to(device)
    gain=norm_gain(model.model.norm)
    project_residual_width(model,p,[1.]*5,chunk_size=1024)
    install_output_matrix(model,projected_output_matrix(gain,p,1.).to(torch.bfloat16))
    ids=torch.randint(0,257,(1,96),device=device)
    with torch.no_grad():before=model(ids,use_cache=False).logits
    info=install_lora(model,4,8)
    with torch.no_grad():torch.testing.assert_close(model(ids,use_cache=False).logits,before,atol=0,rtol=0)
    row={'input_ids':ids[0].tolist(),'positions':list(range(63,95)),'targets':ids[0,64:].tolist()}
    set_training(model)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,foreach=False)
    first=answer_loss(model,row,device).item()/len(row['targets'])
    updates=[train_update(model,optimizer,[row],device,.001) for _ in range(3)]
    final=answer_loss(model,row,device).item()/len(row['targets'])
    if not final<first or not all(u['grad_norm']>0 for u in updates):raise RuntimeError('Runtime backward smoke gate failed')
    result={'device':str(device),'first_ce':first,'final_ce':final,'updates':updates,'adapter':info}
    del model,optimizer;cleanup();return result


def run(args):
    data=json.loads(args.data.read_text());tok=load_tokenizer(args.source)
    verify_source(args.source,data['source'])
    torch.manual_seed(data['settings']['seed']);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('Use a new output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','source':data['source'],'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
            'settings':data['settings'],'arms':{},'steps':[],'checkpoints':[]}
    start=time.monotonic()
    def state(stage):
        report['stage']=stage;report['elapsed_seconds']=time.monotonic()-start
        save(report,args.out/'result.json');print(stage,flush=True)
    legacy_data=json.loads(args.legacy_data.read_text());legacy=legacy_data['generation_eval']
    control_rows=validation_rows(legacy_data,tok)
    reference_dir=args.out/'teacher_logits'
    # Each arm runs the identical generation and presentation policy.
    for label in ['stock','full_width','unhealed_75']:
        state(label+': load')
        model,_=load_source(args.source,args.device)
        if label!='stock':project(model,args.recipe,model.config.hidden_size if label=='full_width' else data['width'])
        control=None
        if label in ['stock','full_width']:
            control=evaluate_logits(model,control_rows,reference_dir,label=='stock')
            if label=='full_width' and (control['mean_kl']>.02 or control['top1_agreement']<.95):
                save(control,args.out/'failed-full-width-gate.json')
                raise RuntimeError('Full-width BF16 equivalence gate failed')
        state(label+': generation')
        report['arms'][label]={'parameters':sum(p.numel() for p in model.parameters()),
            'numerical_control':control,
            'dev':evaluate(model,tok,data['dev'],args.out/(label+'-dev.json')),
            'test':evaluate(model,tok,data['test'],args.out/(label+'-test.json')),
            'legacy72':legacy_generate(model,tok,legacy,args.out/(label+'-legacy72.json'),4)}
        sample=encode_complete(tok,data['train'][0]['messages'],data['max_length'])
        ids=torch.tensor([sample['input_ids'][:128]],device=args.device)
        report['arms'][label]['cache']=cache_check(model,ids)
        if report['arms'][label]['cache']['max_token_kl']>.05:raise RuntimeError('BF16 cache gate failed')
        state(label+': complete')
        if label!='unhealed_75':del model;cleanup()
    if report['arms']['unhealed_75']['parameters']!=20103074288:raise ValueError('Wrong projected parameter count')
    settings=data['settings'];baseline=report['arms']['unhealed_75']['dev']
    with torch.no_grad():before=model(ids,use_cache=False).logits.cpu()
    info=install_lora(model,settings['rank'],settings['alpha']);report['adapter']=info
    with torch.no_grad():after=model(ids,use_cache=False).logits.cpu()
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    del before,after
    metadata={'source':data['source'],'width':data['width'],'data_sha256':report['data_sha256'],
              'recipe_sha256':recipe_hashes(args.recipe),**info}
    save_adapter(model,args.out/'step-0000',metadata|{'step':0})
    best_path=args.out/'step-0000';best_key=selection_key(baseline,baseline,0);best_step=0
    report['selected_step']=0
    encoded=[encode_complete(tok,r['messages'],data['max_length']) for r in data['train']]
    if any(r is None for r in encoded):raise ValueError('Frozen training example no longer fits')
    for raw,row in zip(data['train'],encoded,strict=True):
        if len(row['targets'])!=raw['assistant_tokens'] or len(row['input_ids'])!=raw['input_tokens']:
            raise ValueError('Frozen token counts differ')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=settings['lr'],
                                betas=(.9,.95),weight_decay=0.,foreach=False)
    state('training')
    for index,indices in enumerate(data['updates']):
        # Keep ten minutes for best-checkpoint evaluation and artifact sync.
        predicted=1.5*report['steps'][-1]['seconds'] if report['steps'] else 180
        if time.time()+predicted>args.deadline-600:
            report['stop_reason']='reserved final evaluation time';break
        step=index+1;set_training(model)
        metrics=train_update(model,optimizer,[encoded[i] for i in indices],args.device,
            settings['lr']*min(1.,step/settings['warmup_steps']))
        report['steps'].append({'step':step,**metrics});state('training')
        if step not in settings['evaluation_steps'] and step!=len(data['updates']):continue
        model.gradient_checkpointing_disable();model.eval()
        destination=args.out/f'step-{step:04d}'
        save_adapter(model,destination,metadata|{'step':step})
        state(f'development evaluation: step {step}')
        measured=evaluate(model,tok,data['dev'],destination/'dev.json')
        report['checkpoints'].append({'step':step,'dev':measured})
        key=selection_key(measured,baseline,step)
        if key is not None and key>best_key:best_key=key;best_path=destination;best_step=step
        report['selected_step']=best_step
        # A bad checkpoint is retained for diagnosis, never promoted.
        if measured['correct']<=baseline['correct']-4 or measured['loop_flags']>=baseline['loop_flags']+2:
            report['stop_reason']='development regression gate';break
    # Keep a resumable last optimizer, independently of selected-best adapter.
    torch.save({'optimizer':optimizer.state_dict(),'steps_completed':len(report['steps']),
                'data_sha256':report['data_sha256'],'settings':settings,
                'torch_rng_state':torch.get_rng_state(),
                'cuda_rng_state':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},args.out/'optimizer-last.pt')
    save_adapter(model,args.out/'adapter-last',metadata|{'step':len(report['steps'])})
    del optimizer;model.gradient_checkpointing_disable();model.eval();cleanup()
    # A time-limited run may finish between scheduled checkpoints. Give its
    # last completed update a development evaluation while final time remains.
    last=len(report['steps'])
    if last and last not in [c['step'] for c in report['checkpoints']] and time.time()<args.deadline-480:
        measured=evaluate(model,tok,data['dev'],args.out/'adapter-last/dev.json')
        report['checkpoints'].append({'step':last,'dev':measured})
        key=selection_key(measured,baseline,last)
        if key is not None and key>best_key:best_path=args.out/'adapter-last';best_step=last
    report['selected_step']=best_step
    load_adapter(model,best_path)
    save_adapter(model,args.out/'adapter-best',metadata|{'step':best_step})
    state('selected checkpoint final evaluation')
    report['arms']['healed_selected']={'selected_step':best_step,
        'test':evaluate(model,tok,data['test'],args.out/'healed-selected-test.json'),
        'legacy72':legacy_generate(model,tok,legacy,args.out/'healed-selected-legacy72.json',4)}
    # Verify persistence by zeroing then reloading every adapter tensor.
    with torch.no_grad():expected=model(ids,use_cache=False).logits.cpu()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.zero_()
    load_adapter(model,args.out/'adapter-best')
    with torch.no_grad():actual=model(ids,use_cache=False).logits.cpu()
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    report['adapter_reload_exact']=True
    report['status']='complete' if report['steps'] else 'no_training_time_remaining'
    report['selected_improved_development']=best_step>0
    state('complete')


def export(args):
    if args.out.exists():raise ValueError('Export destination already exists')
    meta=json.loads((args.adapter/'adapter.json').read_text())
    if meta['source']!={'repo':'Qwen/Qwen3.5-27B','revision':'fc05daec18b0a78c049392ed2e771dde82bdf654'}:
        raise ValueError('Unrecognized adapter source')
    verify_source(args.source,meta['source'])
    if meta['recipe_sha256']!=recipe_hashes(args.recipe):raise ValueError('Projection recipe differs')
    model,_=load_source(args.source,args.device)
    project(model,args.recipe,meta['width']);info=install_lora(model,meta['rank'],meta['alpha'])
    if info['modules']!=meta['modules']:raise ValueError('Different model module layout')
    load_adapter(model,args.adapter);merge_lora(model)
    model.save_pretrained(args.out);load_tokenizer(args.source).save_pretrained(args.out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['run','export'])
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--recipe',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path)
    ap.add_argument('--legacy-data',type=Path)
    ap.add_argument('--adapter',type=Path)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--deadline',type=float,default=float('inf'))
    args=ap.parse_args()
    run(args) if args.stage=='run' else export(args)
