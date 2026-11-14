"""Local Qwen ablations: basis geometry, calibration coverage and approximation.

No whole-model healing, no network access, no paid resources. Numerical
diagnostics and independently graded generations are reported separately.
"""
import argparse
import ast
import gc
import fcntl
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.utils.logging import disable_progress_bar

from audit_rotation import cache_check, compare
from correctness_eval import synthetic_cases, grade, json_value
from healing_data import chat_ids, encode_complete, generation_metrics
from horizon_data import run_reference, equivalent
from local_projection_models import build, fit_rms, norms, recipe_bases, rms_error, stored_scalars
from scale_probe import parameter_plan, text_config

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'models/manifold-audit/Qwen3.5-2B'
DEFAULT_OUT = ROOT / 'manifold-compression/results/local-projection-20261025'


def log(message):
    print(message, flush=True)


def save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def load_source(source=SOURCE):
    model, info = AutoModelForCausalLM.from_pretrained(source, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
        raise RuntimeError(f'Invalid source load: {info}')
    model.eval().requires_grad_(False)
    return model


def prefix_budget(rows, budget):
    chosen=[]; total=0
    for row in rows:
        chosen.append(row);total+=len(row['input_ids'])
        if total >= budget:return chosen
    raise ValueError(f'Only {total} tokens available for {budget}')


def code_cases():
    specs = [
        ('dedup','Write a Python function main_solution(values) that removes duplicate integers while preserving first-occurrence order. Return a list.',
         [({'values':[3,1,3,2,1]},[3,1,2]),({'values':[]},[]),({'values':[-1,-1,0]},[-1,0])]),
        ('intervals','Write a Python function main_solution(intervals) that merges overlapping or touching intervals. Return sorted lists [start,end] and do not mutate the input.',
         [({'intervals':[[5,8],[1,3],[3,6]]},[[1,8]]),({'intervals':[]},[]),({'intervals':[[1,10],[2,3],[12,14]]},[[1,10],[12,14]])]),
        ('search','Write a Python function main_solution(values, target) that uses binary search on a sorted list and returns the index of the target, or -1 when absent.',
         [({'values':[1,4,7,9],'target':7},2),({'values':[],'target':2},-1),({'values':[2,5,8],'target':6},-1)]),
        ('counts','Write a Python function main_solution(text) that counts words case-insensitively. A word is a maximal run of ASCII letters. Return a dictionary of lowercase word counts.',
         [({'text':'Red, RED blue!'}, {'red':2,'blue':1}),({'text':''},{}),({'text':'a1A--b'}, {'a':2,'b':1})]),
    ]
    return [{'id':'code-'+name,'family':'code_generation','messages':[{'role':'user','content':prompt+'\nReturn only Python code, with no explanation.'}],
             'check':{'kind':'program','tests':tests},'max_new_tokens':512} for name,prompt,tests in specs]


def prepare(out):
    if (out/'protocol.json').exists():raise ValueError('Refusing to overwrite frozen protocol')
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    old=json.loads((ROOT/'manifold-compression/results/scale-probe-20261024/data.json').read_text())
    def encode(row, limit):
        result=encode_complete(tok,row['messages'],limit)
        return None if result is None else {'id':row['id'],'messages':row['messages'],**result}
    standard=[encode(r,768) for r in old['calibration']]
    standard=[r for r in standard if r]
    validation=[encode(r,768) for r in old['validation'][:16]]
    validation=[r for r in validation if r]
    large=prefix_budget(standard,65536);small=prefix_budget(large,16384)
    # Coverage arm: independent long chat plus verified Horizon math and code I/O.
    source=next((ROOT/'target/manifold-audit/ultrachat/data').glob('train_sft-00000-*.parquet'))
    table=pq.read_table(source,columns=['prompt_id','messages'])
    order=list(range(len(table)));random.Random(25102581).shuffle(order)
    excluded={r['messages'][0]['content'].strip() for r in large+validation}
    long_rows=[];total=0
    for idx in order:
        raw=table.slice(idx,1).to_pylist()[0];m=raw['messages'][:2]
        if len(m)!=2 or [x['role'] for x in m]!=['user','assistant'] or m[0]['content'].strip() in excluded:continue
        row=encode({'id':'long-'+raw['prompt_id'],'messages':m},2048)
        if row and len(row['input_ids'])>768:
            long_rows.append(row);total+=len(row['input_ids']);excluded.add(m[0]['content'].strip())
        if total>=32768:break
    if total<32768:raise ValueError('Insufficient long complete conversations')
    horizon=json.loads((ROOT/'manifold-compression/results/correctness-healing-20261025/data.json').read_text())
    hgroups={}
    for kind in ('math','code'):
        rows=[];total=0
        for raw in horizon['train']:
            if not raw['kind'].startswith(kind):continue
            row=encode(raw,2048)
            if row and row['messages'][0]['content'].strip() not in excluded:
                rows.append(row);total+=len(row['input_ids']);excluded.add(row['messages'][0]['content'].strip())
            if total>=16384:break
        if total<16384:raise ValueError(f'Insufficient Horizon {kind}: {total}')
        hgroups[kind]=rows
    broad=long_rows+hgroups['math']+hgroups['code'];random.Random(25102582).shuffle(broad)
    cases=synthetic_cases(2510258,8)
    cases+=code_cases()
    for i,prompt in enumerate([
        'Explain photosynthesis and cellular respiration to a secondary-school student. Describe their inputs and outputs and how they are connected. Give a complete answer in 150 to 200 words.',
        'Explain why a metal spoon and a wooden spoon at the same room temperature can feel different. Distinguish temperature from heat transfer. Give a complete answer in 150 to 200 words.']):
        cases.append({'id':f'explanation-{i}','family':'explanation','messages':[{'role':'user','content':prompt}],
                      'check':None,'max_new_tokens':1024})
    train_prompts={r['messages'][0]['content'].strip() for r in large+broad}
    assert not train_prompts.intersection(r['messages'][0]['content'].strip() for r in validation+cases)
    data={'small':small,'large':large,'broad':broad,'validation':validation,'cases':cases}
    save(data,out/'data.json')
    plan=parameter_plan(text_config(SOURCE),(.75,))
    global_k=plan['arms'][0]['width'];slope=plan['parameters_per_residual_coordinate'];intercept=plan['fixed_parameters']
    # A norm gain per site already belongs to slope. Count 23 bridges plus the
    # final output matrix. Predictor/scale buffers get a small explicit allowance.
    local_k=max(k for k in range(16,2049,16)
                if slope*k+intercept+24*k*k+20000<=plan['arms'][0]['parameters'])
    protocol={'model':'Qwen/Qwen3.5-2B','source_manifest':json.loads((SOURCE/'download_manifest.json').read_text()),
        'device':'cpu','dtype':'float32','whole_model_healing':False,'cloud_spend_usd':0,
        'data_sha256':hashlib.sha256((out/'data.json').read_bytes()).hexdigest(),
        'calibration':{k:{'rows':len(v),'tokens':sum(len(r['input_ids']) for r in v),
                          'max_length':max(len(r['input_ids']) for r in v)} for k,v in data.items() if k!='cases'},
        'plan':plan,'global_width':global_k,'local_matched_width':local_k,
        'arms':['stock','shared-small','shared-large','shared-broad','layer-large-matched','layer-large-samewidth','shared-large-dynamic'],
        'controls':'Fresh source load per arm. Shared and local bases use identical large calibration corpus. Same-width local arm explicitly exceeds shared storage. No selection on held-out generations.',
        'rms_fit':'33 quadratic/linear features of retained coordinates; ridge 0.01; bounded by calibration quantiles. No model-loss training.',
        'generation':{'closed_cases':56,'code_cases':4,'explanations':2,'greedy':True,'thinking':False,'penalties':False,
                      'limits':'256 closed / 512 code / 1024 explanation; caps reported separately from repetition'},
        'scope':'Small mechanism screen; does not establish 27B transfer or broad intelligence parity.'}
    save(protocol,out/'protocol.json');log(json.dumps(protocol['calibration']))


@torch.no_grad()
def collect_packet(model, rows, path, cov=True):
    if path.exists():return
    d=model.config.hidden_size;ns=norms(model)
    covs=[torch.zeros(d,d) for _ in ns] if cov else None
    samples=[[] for _ in ns];groups=[];counts=0;handles=[]
    # Deterministic positions shared by all sites; never use validation to fit.
    for j,norm in enumerate(ns):
        def hook(mod,args,j=j):
            x=args[0].reshape(-1,d).float()
            if covs is not None:covs[j].addmm_(x.T,x)
            positions=torch.linspace(0,len(x)-1,min(32,len(x))).long().unique()
            samples[j].append(x[positions].clone())
        handles.append(norm.register_forward_pre_hook(hook))
    try:
        for i,row in enumerate(rows):
            ids=torch.tensor([row['input_ids']])
            # Zen 5 has native BF16 matrix instructions. Calibration alone uses
            # BF16 matmuls; covariance accumulation, eigensolves, projection and
            # all reported generation/numerical comparisons remain FP32/FP64.
            with torch.autocast('cpu',dtype=torch.bfloat16):
                model.model(input_ids=ids,use_cache=False)
            count=min(32,ids.numel());groups.extend([i]*count);counts+=ids.numel()
            if (i+1)%16==0:log(f'{path.stem}: {i+1}/{len(rows)}, {counts} tokens')
    finally:
        for h in handles:h.remove()
    packet={'covs':covs,'samples':torch.stack([torch.cat(s) for s in samples]),
            'groups':torch.tensor(groups),'tokens':counts,'rows':len(rows),
            'activation_forward':'CPU BF16 autocast; FP32 covariance accumulation'}
    torch.save(packet,path);log(f'Saved {path.name}: {counts} tokens')


@torch.no_grad()
def selected_logits(model, rows):
    outputs=[];targets=[]
    for row in rows:
        ids=torch.tensor([row['input_ids']])
        h=model.model(input_ids=ids,use_cache=False).last_hidden_state[0]
        positions=row['positions'][::max(1,len(row['positions'])//8)][:8]
        outputs.append(model.lm_head(h[positions]).float())
        targets.extend(row['input_ids'][p+1] for p in positions)
    return torch.cat(outputs),torch.tensor(targets)


def collect(out):
    data=json.loads((out/'data.json').read_text());model=load_source()
    for name in ('small','large','broad','validation'):
        collect_packet(model,data[name],out/f'{name}-states.pt',name!='validation')
    reference,targets=selected_logits(model,data['validation'])
    torch.save({'logits':reference,'targets':targets},out/'reference.pt')
    save(cache_check(model,torch.tensor([data['validation'][0]['input_ids'][:96]])),out/'stock-cache.json')


def eig(cov):
    vals,q=torch.linalg.eigh(cov.double())
    return q.flip(1).float().contiguous(),vals.flip(0).float()


def fit(out):
    protocol=json.loads((out/'protocol.json').read_text())
    validation=torch.load(out/'validation-states.pt',weights_only=True)['samples']
    k=protocol['global_width'];local_k=protocol['local_matched_width']
    summary={}
    data=json.loads((out/'data.json').read_text())
    for name in ('small','large','broad'):
        packet=torch.load(out/f'{name}-states.pt',weights_only=True)
        covs=packet['covs'];xs=packet['samples'];layers=(len(covs)-1)//2
        # Samples have up to 32 states per conversation. Restore token weighting
        # so many short math answers cannot overwhelm longer calibration rows.
        weights=torch.tensor([len(data[name][g]['input_ids'])/min(32,len(data[name][g]['input_ids']))
                              for g in packet['groups'].tolist()])
        normalized=[c.double()/c.double().trace() for c in covs]
        q,values=eig(sum(normalized))
        torch.save({'q':q,'values':values},out/f'{name}-pca.pt')
        qstack=q.unsqueeze(0).expand(layers,-1,-1)
        variants=[(f'shared-{name}',qstack,k,False)]
        if name=='large':
            local=[]
            for i in range(layers):
                cov=normalized[2*i]+normalized[2*i+1]
                if i==layers-1:cov=cov+normalized[-1]
                qi,_=eig(cov);local.append(qi)
                if (i+1)%4==0:log(f'Layer PCA {i+1}/{layers}')
            local=torch.stack(local)
            variants += [('layer-large-matched',local,local_k,True),('layer-large-samewidth',local,k,True)]
            # Full-width recipe is a mandatory numerical control for new bridges.
            torch.save({'basis':local,'width':q.shape[0],'local':True,'scales':[1.]*len(covs)},out/'fullwidth-recipe.pt')
        for arm,bases,width,islocal in variants:
            scales=[];predictors=[];errors=[]
            for site in range(len(covs)):
                p=bases[min(site//2,layers-1),:,:width].contiguous()
                scale,predictor=fit_rms(xs[site],p,weights=weights)
                scales.append(scale);predictors.append(predictor)
                errors.append(rms_error(validation[site],p,scale,predictor))
            recipe={'basis':bases,'width':width,'local':islocal,'scales':scales,'predictors':None,
                    'source':'Qwen/Qwen3.5-2B','calibration':name}
            torch.save(recipe,out/f'{arm}-recipe.pt')
            summary[arm]={'width':width,'local':islocal,'heldout_rms':errors,
                          'mean_rms_constant_mse':sum(r['constant_mse'] for r in errors)/len(errors),
                          'mean_rms_predicted_mse':sum(r['predicted_mse'] for r in errors)/len(errors)}
            if arm=='shared-large':
                recipe=dict(recipe,predictors=predictors)
                torch.save(recipe,out/'shared-large-dynamic-recipe.pt')
            log(f'Fit {arm}')
        del packet,covs,xs,normalized,variants;gc.collect()
    save(summary,out/'fit-summary.json')


def extract_program(answer):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',answer,re.S|re.I)
    code='\n\n'.join(blocks) if blocks else answer.strip()
    ast.parse(code)
    return code


def grade_program(answer,tests,preserve_inputs=False):
    try:code=extract_program(answer)
    except SyntaxError:return {'correct':False,'syntax_pass':False,'passed_tests':0,'tests':len(tests)}
    if preserve_inputs:
        class Rename(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                self.generic_visit(node)
                if node.name=='main_solution':node.name='__candidate_solution'
                return node
            def visit_Name(self,node):
                if node.id=='main_solution':node.id='__candidate_solution'
                return node
        code=ast.unparse(Rename().visit(ast.parse(code)))
        code+='\nimport copy as __copy\ndef main_solution(**inputs):\n    before=__copy.deepcopy(inputs)\n    value=__candidate_solution(**inputs)\n    return {"value":value,"unchanged":inputs==before}\n'
    passed=0
    for inputs,expected in tests:
        got=run_reference(code,inputs)
        if preserve_inputs:expected={'value':expected,'unchanged':True}
        passed+=bool(got and equivalent(got['result'],expected))
    return {'correct':passed==len(tests),'syntax_pass':True,'passed_tests':passed,'tests':len(tests)}


def grade_case(answer,case):
    check=case['check']
    if check is None:return None
    if check['kind']=='program':
        return grade_program(answer,check['tests'],case['id']=='code-intervals')
    result=grade(answer,check)
    if case['family']=='sorting':
        # Ordering is substantive; quoting an unambiguous integer is a separate
        # type/format defect. Keep the original strict result for transparency.
        result['typed_value_correct']=result['correct']
        try:
            values=json_value(answer)
            if not isinstance(values,list):raise ValueError('Not a list')
            converted=[]
            for v in values:
                if type(v) is int:converted.append(v)
                elif isinstance(v,str) and re.fullmatch(r'[+-]?\d+',v.strip()):converted.append(int(v))
                else:raise ValueError('Not an unambiguous integer')
            result['correct']=converted==check['expected']
            result['parsed']=True
            result['numeric_strings']=any(isinstance(v,str) for v in values)
        except (ValueError,TypeError):pass
    return result


@torch.no_grad()
def evaluate_generations(model,tok,cases,path):
    if path.exists():
        result=json.loads(path.read_text())
    else:result={'cases':[]}
    done={r['id'] for r in result['cases']}
    config=GenerationConfig(do_sample=False,use_cache=True,eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,bos_token_id=tok.bos_token_id,repetition_penalty=1.,no_repeat_ngram_size=0)
    tok.padding_side='left';tok.pad_token=tok.eos_token
    for cap in sorted({c['max_new_tokens'] for c in cases}):
        config.max_new_tokens=cap
        group=[c for c in cases if c['max_new_tokens']==cap]
        # Always keep the original batch composition on resume.
        for start in range(0,len(group),4):
            batch=group[start:start+4]
            if all(c['id'] in done for c in batch):continue
            texts=[tok.apply_chat_template(c['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for c in batch]
            inputs=tok(texts,add_special_tokens=False,padding=True,return_tensors='pt')
            sequences=model.generate(**inputs,generation_config=config)[:,inputs.input_ids.shape[1]:]
            for c,ids in zip(batch,sequences.tolist(),strict=True):
                if c['id'] in done:continue
                if tok.eos_token_id in ids:ids=ids[:ids.index(tok.eos_token_id)+1]
                answer=tok.decode(ids,skip_special_tokens=True)
                grading=grade_case(answer,c)
                result['cases'].append({'id':c['id'],'family':c['family'],'answer':answer,'grade':grading,
                    'max_new_tokens':cap,**generation_metrics(answer,ids,tok.eos_token_id)})
            save(result,path);log(f'{path.stem}: generation {len(result["cases"])}/{len(cases)}')
    closed=[c for c in result['cases'] if c['family'] not in ('code_generation','explanation')]
    programs=[c for c in result['cases'] if c['family']=='code_generation']
    result['summary']={'closed_correct':sum(c['grade']['correct'] for c in closed),'closed_total':len(closed),
        'programs_correct':sum(c['grade']['correct'] for c in programs),'programs_total':len(programs),
        'syntax_pass':sum(c['grade']['syntax_pass'] for c in programs),'ended':sum(c['ended'] for c in result['cases']),
        'capped':sum(not c['ended'] for c in result['cases']),'loop_flags':sum(c['loop_flag'] for c in result['cases'])}
    save(result,path);return result['summary']


@torch.no_grad()
def numerical(model,out,data):
    reference=torch.load(out/'reference.pt',weights_only=True)
    current,targets=selected_logits(model,data['validation'])
    assert torch.equal(targets,reference['targets'])
    comparison=compare([reference['logits']],[current])
    comparison['target_nll']=nn.functional.cross_entropy(current,targets).item()
    comparison['target_top1']=(current.argmax(-1)==targets).float().mean().item()
    return comparison


def control(out):
    data=json.loads((out/'data.json').read_text());recipe=torch.load(out/'fullwidth-recipe.pt',weights_only=True)
    model=build(load_source(),recipe_bases(recipe),recipe['scales'],local=True)
    result=numerical(model,out,data)
    result['cache']=cache_check(model,torch.tensor([data['validation'][0]['input_ids'][:96]]))
    save(result,out/'fullwidth-control.json')
    if not result['numerical_pass'] or not result['cache']['numerical_pass']:
        raise RuntimeError('Full-width bridge control failed')
    log('Full-width layer-basis control passed')


def roundtrip(out):
    from safetensors.torch import save_model, load_model
    data=json.loads((out/'data.json').read_text())
    recipe=torch.load(out/'layer-large-matched-recipe.pt',weights_only=True)
    def construct():
        return build(load_source(),recipe_bases(recipe),recipe['scales'],local=True)
    model=construct()
    rows=data['validation'][:2]
    with torch.no_grad():before,_=selected_logits(model,rows)
    checkpoint=out/'layer-large-matched.safetensors'
    save_model(model,str(checkpoint))
    del model;gc.collect()
    model=construct()
    missing,unexpected=load_model(model,str(checkpoint),strict=True)
    with torch.no_grad():after,_=selected_logits(model,rows)
    result=compare([before],[after])
    result.update(missing_keys=list(missing),unexpected_keys=list(unexpected),checkpoint=checkpoint.name,
                  checkpoint_bytes=checkpoint.stat().st_size)
    digest=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    result['sha256']=digest.hexdigest()
    save(result,out/'roundtrip.json')
    if missing or unexpected or result['max_abs_logit_error']!=0:
        raise RuntimeError('Full projected-state roundtrip differs')
    log('Full projected-state roundtrip: exact logits')


def _evaluate(out,arm):
    if not (out/'fullwidth-control.json').exists():raise RuntimeError('Run full-width control first')
    data=json.loads((out/'data.json').read_text())
    model=load_source();recipe=None
    if arm!='stock':
        recipe=torch.load(out/f'{arm}-recipe.pt',weights_only=True)
        build(model,recipe_bases(recipe),recipe['scales'],recipe['local'],recipe.get('predictors'),recipe.get('decoders'))
    result={'arm':arm,'parameters':sum(p.numel() for p in model.parameters()),'stored_scalars':stored_scalars(model),
            'validation':numerical(model,out,data)}
    result['cache']=cache_check(model,torch.tensor([data['validation'][0]['input_ids'][:96]]))
    if not result['cache']['numerical_pass']:raise RuntimeError(f'Cache mismatch: {result["cache"]}')
    save(result,out/f'{arm}-result.json')
    tok=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    result['generation']=evaluate_generations(model,tok,data['cases'],out/f'{arm}-generations.json')
    result['status']='complete';save(result,out/f'{arm}-result.json')


def evaluate(out,arm):
    # Separate CPU workers may traverse the queue in opposite orders. Lock each
    # arm, and recheck completion under the lock, to prevent duplicate writers.
    with (out/f'{arm}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=out/f'{arm}-result.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='complete':return
        return _evaluate(out,arm)


def reconstruction(out,data_name='large',steps=400):
    """Held-out nonlinear recoverability; this is not an end-to-end decoder arm."""
    train=torch.load(out/f'{data_name}-states.pt',weights_only=True)['samples']
    test=torch.load(out/'validation-states.pt',weights_only=True)['samples']
    recipe=torch.load(out/'shared-large-recipe.pt',weights_only=True)
    q=recipe['basis'][0];k=recipe['width'];p,r=q[:,:k],q[:,k:]
    result={};label='' if data_name=='large' else '-'+data_name
    for site in (0,24,48):
        x=train[site].float();xt=test[site].float()
        z=x@p;y=x@r;zt=xt@p;yt=xt@r
        zm=z.mean(0);zs=z.std(0).clamp_min(.01);ym=y.mean(0)
        z=(z-zm)/zs;zt=(zt-zm)/zs
        gram=z.T.double()@z.double()/len(z)
        w=torch.linalg.solve(gram+.1*torch.eye(k),z.T.double()@(y-ym).double()/len(z)).float()
        base=z@w+ym;base_test=zt@w+ym
        target=y-base;scale=target.std().clamp_min(.001)
        torch.manual_seed(251025+site)
        net=nn.Sequential(nn.Linear(k,64),nn.SiLU(),nn.Linear(64,y.shape[-1]))
        nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        curve=[]
        for step in range(steps):
            idx=torch.randint(len(z),(128,))
            loss=(net(z[idx])-target[idx]/scale).square().mean()
            opt.zero_grad();loss.backward();opt.step()
            if step+1 in {400,steps}:
                with torch.no_grad():
                    estimate=base_test+net(zt)*scale
                    curve.append({'step':step+1,'heldout_mse':(estimate-yt).square().mean().item(),
                                  'train_mse':(base+net(z)*scale-y).square().mean().item()})
        with torch.no_grad():
            nonlinear=base_test+net(zt)*scale
            denom=yt.square().mean().item()
            linear_mse=(base_test-yt).square().mean().item()
            nonlinear_mse=(nonlinear-yt).square().mean().item()
            result[str(site)]={'train_states':len(x),'test_states':len(xt),'discarded_zero_mse':denom,
                'linear_mse':linear_mse,'linear_plus_nonlinear_mse':nonlinear_mse,
                'linear_recovered_fraction':1-linear_mse/denom,'linear_plus_nonlinear_recovered_fraction':1-nonlinear_mse/denom,
                'extra_network_parameters':sum(v.numel() for v in net.parameters()),
                'data':data_name,'steps':steps,'fixed_projection':'shared-large','learning_curve':curve,
                'scope':'Independent calibration/test conversations; representation MSE only, not evidence of improved answers.'}
        torch.save({'site':site,'zm':zm,'zs':zs,'ym':ym,'w':w,'scale':scale,'net':net.state_dict()},out/f'decoder-site-{site}{label}.pt')
        save(result,out/f'reconstruction{label}.json');log(f'Reconstruction{label} site {site}: {result[str(site)]}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','collect','fit','control','roundtrip','evaluate','reconstruction','finish'])
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--arm',default='stock')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--reverse',action='store_true')
    ap.add_argument('--reconstruction-data',choices=['large','extended'],default='large')
    ap.add_argument('--decoder-steps',type=int,default=400)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    disable_progress_bar()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.manual_seed(251025)
    if args.stage=='finish':
        markers={'collect':'reference.pt','fit':'fit-summary.json','control':'fullwidth-control.json',
                 'roundtrip':'roundtrip.json','reconstruction':'reconstruction.json'}
        for stage,marker in markers.items():
            completed=(args.out/marker).exists()
            if stage=='reconstruction' and completed:
                completed=len(json.loads((args.out/marker).read_text()))==3
            if not completed:globals()[stage](args.out);gc.collect()
        if not json.loads((args.out/'fullwidth-control.json').read_text())['numerical_pass']:
            raise RuntimeError('Prior full-width control failed')
        if json.loads((args.out/'roundtrip.json').read_text())['max_abs_logit_error']!=0:
            raise RuntimeError('Prior reload control failed')
        arms=json.loads((args.out/'protocol.json').read_text())['arms']
        if args.reverse:arms=list(reversed(arms))
        for arm in arms:
            result=args.out/f'{arm}-result.json'
            if result.exists() and json.loads(result.read_text()).get('status')=='complete':continue
            evaluate(args.out,arm);gc.collect()
    elif args.stage=='evaluate':evaluate(args.out,args.arm)
    elif args.stage=='reconstruction':reconstruction(args.out,args.reconstruction_data,args.decoder_steps)
    else:globals()[args.stage](args.out)


if __name__=='__main__':main()
