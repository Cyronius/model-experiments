"""Matched 2x2 pilot: stock/projected Qwen3.5-2B, fresh rank16/rank64 LoRA."""
import argparse, gc, json, math, random, re
from pathlib import Path
import torch
from broad_healing import ROOT, SOURCE, RECIPE, sha, key, program
from curriculum_healing import OUT as PREVIOUS, PARENT, assess
from curriculum_tasks import FAMILIES, math_example
from curriculum_numeric_audit import audit_file
from correctness_eval import synthetic_cases
from correctness_heal import install_lora, save_adapter, load_adapter, train_update, LowRankLinear, answer_loss
from local_projection_models import from_recipe
from local_projection_probe import load_source, save
from scale_data import load_tokenizer
from audit_rotation import cache_check
from horizon_data import arithmetic

OUT=ROOT/'results/capacity-projection-20261026'
ARMS=['projected-r16','projected-r64','stock-r16','stock-r64']

def initialize(model,rank):
    info=install_lora(model,rank,2*rank)
    # The smaller adapter uses the exact first 16 directions of the larger
    # adapter in the same architecture. Both start with zero update and scale 2.
    for i,(name,m) in enumerate((n,m) for n,m in model.named_modules() if isinstance(m,LowRankLinear)):
        g=torch.Generator().manual_seed(26102680+i)
        a=torch.empty(64,m.base.in_features).uniform_(-1/math.sqrt(m.base.in_features),1/math.sqrt(m.base.in_features),generator=g)
        with torch.no_grad():m.a.copy_(a[:rank]);m.b.zero_()
    return info


def prepare():
    if (OUT/'data.json').exists():raise ValueError('Frozen study already exists')
    data=json.loads((PREVIOUS/'data.json').read_text());old=json.loads((PARENT/'data.json').read_text())
    prior=data['train']+old['train']+data['dev']+data['test']+data['legacy']+old['dev']+old['test']
    seen=set();multiply=set();prompts=set()
    for row in prior:
        prov=row.get('oracle',row.get('provenance',{}))
        if 'family' in prov and 'values' in prov:
            seen.add((prov['family'],tuple(prov['values'])))
            if prov['family']=='multiply':multiply.add(tuple(sorted(prov['values'])))
        if row.get('kind')=='math' and 'a' in prov:multiply.add(tuple(sorted((prov['a'],prov['b']))))
        messages=row['messages'];prompts.add(key(messages[:-1] if messages[-1]['role']=='assistant' else messages))
        for m in messages:
            match=re.search(r'(\d+) multiplied by (\d+)',m['content'])
            if match:multiply.add(tuple(sorted(map(int,match.groups()))))
    splits={}
    for split,n,seed in [('dev',1,26102681),('test',4,26102682)]:
        rng=random.Random(seed);cases=[]
        for fam in FAMILIES:
            count=0
            for attempt in range(10000):
                r=math_example(rng,fam);k=(fam,tuple(r['values']))
                if k in seen or (fam=='multiply' and tuple(sorted(r['values'])) in multiply):continue
                seen.add(k)
                if fam=='multiply':multiply.add(tuple(sorted(r['values'])))
                assert arithmetic(r['expression'])==arithmetic(r['expected'])
                cases.append({'id':f'cp-{split}-{fam}-{count}','family':'math_'+fam,'messages':[{'role':'user','content':r['transfer']+' Return only the final number or reduced fraction.'}],'check':{'kind':'numeric','expected':r['expected']},'max_new_tokens':256,'oracle':r})
                count+=1
                if count==n:break
            assert count==n
        splits[split]=cases
    # New numeric values for retention, excluding earlier prompts and products.
    counts=set();rng=random.Random(26102683)
    for attempt in range(10000):
        for c in synthetic_cases(261026830+attempt,1):
            if c['family'] in counts:continue
            if c['family']=='units':
                t=rng.randint(1001,9999);c['messages']=[{'role':'user','content':f'Convert {t/10:g} liters to milliliters. Return only the number.'}];c['check']['expected']=str(t*100)
            if key(c['messages']) in prompts:continue
            if c['family']=='arithmetic':
                m=re.search(r'(\d+) multiplied by (\d+)',c['messages'][0]['content']);pair=tuple(sorted(map(int,m.groups())))
                if pair in multiply:continue
                multiply.add(pair)
            counts.add(c['family']);splits['test'].append(c)
        if len(counts)==7:break
    assert len(counts)==7
    # Existing code tasks provide a fixed regression set, not novel algorithms.
    splits['dev'] += [c for c in data['dev'] if c['family']=='code_generation']
    splits['test'] += [c for c in data['test'] if c['family']=='code_generation']
    sci=[('charge','A neutral atom gains three electrons without changing its nucleus. What is its new net charge, and does it become a different element?','Charge -3 elementary-charge units; unchanged proton count means same element.'),
         ('current','Twelve coulombs of charge cross a wire section in four seconds. State the average current and the calculation.','3 amperes, I=Q/t=12/4.'),
         ('diffusion','Two regions of water have reached equal salt concentration. Does random molecular motion stop? Explain.','No; random thermal motion continues, with no net concentration-driven flow.'),
         ('friction','A sliding book stops on a rough table. Has its kinetic energy disappeared? Explain where the energy goes.','No; kinetic energy becomes internal/thermal energy in book and surroundings, conserving total energy.')]
    for name,p,r in sci:splits['test'].append({'id':'cp-science-'+name,'family':'explanation','messages':[{'role':'user','content':p}],'check':None,'manual_rubric':r,'max_new_tokens':512})
    # Existing development science is explicitly diagnostic, not fresh.
    splits['dev'] += [c for c in data['dev'] if c['family']=='explanation']
    train_prompts={key(r['messages'][:-1]) for r in data['train']}
    assert not train_prompts & {key(c['messages']) for s in splits.values() for c in s}
    updates=data['updates'][:8]
    protocol={'arms':ARMS,'initialization':'Fresh zero-output adapters on unhealed stock or identical frozen shared-broad projection. No parent adapter or optimizer state. Nested random A directions within each architecture.',
      'ranks':[16,64],'alpha_over_rank':2,'seed':26102680,'steps':8,'lr':3e-5,'optimizer':'fresh AdamW betas=(0.9,0.95), weight_decay=0, clip=1; no warmup',
      'assistant_tokens_per_arm':sum(len(data['train'][i]['targets']) for b in updates for i in b),'training_data_sha256':sha(PREVIOUS/'data.json'),
      'recipe_sha256':sha(RECIPE),'source_manifest':json.loads((PREVIOUS/'reproducibility.json').read_text())['source_manifest'],
      'primary_endpoint':'Fixed step 8, compare paired correctness gains from step 0 and stock-minus-projected gaps separately by rank. No best-checkpoint selection.',
      'capacity_interaction':'(projected64_final - projected16_final) - (stock64_final - stock16_final); positive means rank benefits projection more on that metric, conditional on this budget.',
      'evaluation_steps':[0,4,8],'test_evaluation_steps':[0,8],'train_precision':'CPU BF16 autocast, FP32 stored weights','eval_precision':'CPU FP32, four threads, greedy nonthinking, no repetition penalties',
      'train_fit_sample':'first scheduled row from each curriculum kind; teacher-forced CE only, not generation or generalization',
      'limitations':['Single seed, short fixed-token pilot, no claim of irrecoverable damage from a negative result.','Rank changes both capacity and optimization geometry; equal learning rate is not independently tuned.','Stock versus projected isolates the effect of the compression pipeline, not which individual projection operation damages capability.','New arithmetic operands and science wording; shared families/topics. Code and dev science are known regression sets.','This pilot restarts healing and does not directly compare against the previous 99k-token selected checkpoint.','Equal training tokens and optimizer steps, not equal FLOPs or trainable parameters.','A different matched-size projection method is a follow-up, not one of these arms.'],'cloud_spend':0}
    save({'train':data['train'],'updates':updates,**splits,'protocol':protocol},OUT/'data.json');save(protocol,OUT/'protocol.json')
    print(json.dumps(protocol),flush=True)


def evaluate(model,tok,cases,path):
    assess(model,tok,cases,path)
    audited=audit_file(path,cases);save(audited,path.with_name(path.stem+'-content-audit.json'));return audited['summary']


def fit_probe(model,rows):
    model.eval(); result={}
    with torch.no_grad():
        for row in rows:
            result[row['kind']]=float(answer_loss(model,row,'cpu')/len(row['targets']))
    return result


def run(arm):
    assert arm in ARMS
    out=OUT/arm
    if out.exists():raise ValueError('Arm exists; explicit checkpoint recovery required')
    out.mkdir(parents=True)
    data=json.loads((OUT/'data.json').read_text());p=data['protocol']
    assert sha(PREVIOUS/'data.json')==p['training_data_sha256'] and sha(RECIPE)==p['recipe_sha256']
    for name,v in p['source_manifest']['files'].items():assert sha(Path(SOURCE)/name)==v['sha256']
    torch.set_num_threads(4);torch.manual_seed(p['seed']);tok=load_tokenizer(SOURCE)
    model=from_recipe(SOURCE,torch.load(RECIPE,weights_only=True,map_location='cpu'),load_source) if arm.startswith('projected') else load_source(SOURCE)
    base_parameters=sum(v.numel() for v in model.parameters());rank=int(arm.split('-r')[1]);info=initialize(model,rank)
    info.update({'arm':arm,'base_parameters':base_parameters,'recipe_sha256':p['recipe_sha256'] if arm.startswith('projected') else None,'data_sha256':sha(OUT/'data.json')})
    save(info,out/'model.json')
    # Zero adapter updates must exactly preserve the frozen model path.
    row=data['train'][data['updates'][0][0]];ids=torch.tensor([row['input_ids'][:32]])
    model.eval()
    with torch.no_grad():baseline_logits=model(ids,use_cache=False).logits
    gate=cache_check(model,ids);assert gate['numerical_pass'];save({'cache':gate},out/'gates.json')
    save_adapter(model,out/'step-0',{**info,'step':0})
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['lr'],betas=(.9,.95),weight_decay=0,foreach=False)
    sample={}
    for batch in data['updates']:
        for i in batch:sample.setdefault(data['train'][i]['kind'],data['train'][i])
    probe={0:fit_probe(model,list(sample.values()))};dev={0:evaluate(model,tok,data['dev'],out/'dev-0.json')}
    test={0:evaluate(model,tok,data['test'],out/'test-0.json')};log=[]
    for step,batch in enumerate(data['updates'],1):
        model.train();model.config.use_cache=False
        with torch.autocast('cpu',dtype=torch.bfloat16):r=train_update(model,optimizer,[data['train'][i] for i in batch],'cpu',p['lr'])
        log.append({'step':step,**r});save(log,out/'training.json');print(arm,json.dumps(log[-1]),flush=True)
        if step in [4,8]:
            model.eval();save_adapter(model,out/f'step-{step}',{**info,'step':step})
            torch.save({'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),'step':step},out/f'step-{step}'/'optimizer.pt')
            probe[step]=fit_probe(model,list(sample.values()));dev[step]=evaluate(model,tok,data['dev'],out/f'dev-{step}.json')
    test[8]=evaluate(model,tok,data['test'],out/'test-8.json')
    with torch.no_grad():before=model(ids,use_cache=False).logits
    with torch.no_grad():
        for v in model.parameters():
            if v.requires_grad:v.zero_()
    load_adapter(model,out/'step-8')
    with torch.no_grad():torch.testing.assert_close(before,model(ids,use_cache=False).logits,atol=0,rtol=0)
    save({'cache':gate,'adapter_reload_exact':True},out/'gates.json')
    result={'arm':arm,'dev':dev,'test':test,'train_fit_ce':probe,'cloud_spend':0};save(result,out/'result.json');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare']+ARMS);args=parser.parse_args()
    prepare() if args.action=='prepare' else run(args.action)
