"""Local, bounded correctness repair of the broad-calibrated Qwen2B projection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import torch
from scale_data import load_tokenizer
from local_projection_probe import SOURCE, load_source, save, evaluate_generations
from local_projection_models import from_recipe, build
from correctness_heal import install_lora, save_adapter, load_adapter, train_update, answer_loss
from correctness_eval import synthetic_cases
from healing_data import encode_complete
from correctness_prepare import pure_reference
from horizon_data import run_reference, equivalent
from audit_rotation import cache_check
from broad_healing_science import SCIENCE

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'results/local-projection-20261025'
OUT=ROOT/'results/broad-healing-20261025'
RECIPE=OLD/'shared-broad-recipe.pt'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def key(messages):return json.dumps(messages,sort_keys=True)

def program(name,prompt,tests):
    return {'id':name,'family':'code_generation','messages':[{'role':'user','content':prompt+' Write a Python function named main_solution. Return only the complete code.'}],
            'check':{'kind':'program','tests':tests},'max_new_tokens':512}

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'data.json').exists():raise ValueError('Data is already frozen')
    tok=load_tokenizer(SOURCE)
    legacy=json.loads((OLD/'data.json').read_text())['cases']
    excluded=[key(c['messages']) for c in legacy]
    dev=synthetic_cases(25102531,3,excluded)
    test=synthetic_cases(25102532,6,excluded+[key(c['messages']) for c in dev])
    dev += [program('dev-chunks','Given values (a list) and n (a positive integer), return consecutive chunks of at most n elements, preserving order.',[[{'values':[1,2,3,4,5],'n':2},[[1,2],[3,4],[5]]],[{'values':[],'n':3},[]],[{'values':[7,8],'n':5},[[7,8]]]]),
            program('dev-factorial','Given a nonnegative integer n, return n factorial as an integer.',[[{'n':0},1],[{'n':5},120],[{'n':8},40320]])]
    test += [program('test-rotate','Given values (a list) and k (a nonnegative integer), return the list rotated right by k positions. Handle an empty list.',[[{'values':[1,2,3,4],'k':1},[4,1,2,3]],[{'values':[1,2,3],'k':5},[2,3,1]],[{'values':[],'k':7},[]]]),
             program('test-runs','Given text (a string), return run-length encoding as a list of [character, count] pairs for consecutive equal characters. Preserve run order.',[[{'text':'aaabbcaa'},[['a',3],['b',2],['c',1],['a',2]]],[{'text':''},[]],[{'text':'x'},[['x',1]]]]),
             program('test-transpose','Given matrix (a rectangular list of lists), return its transpose as lists. Return [] for an empty matrix.',[[{'matrix':[[1,2,3],[4,5,6]]},[[1,4],[2,5],[3,6]]],[{'matrix':[]},[]],[{'matrix':[[7],[8]]},[[7,8]]]]),
             program('test-brackets','Given text containing only ()[]{}, return a boolean indicating whether brackets are correctly matched and nested. An empty string is valid.',[[{'text':'([]){}'},True],[{'text':'([)]'},False],[{'text':''},True]])]
    for split,topics in [(dev,[('sound','Why can sound travel through air but not a vacuum? Explain in two short paragraphs.'),('shadow','Why does the length of a vertical pole\'s shadow change during a day? Explain in two short paragraphs.')]),(test,[('ice','Why does ice float in liquid water? Explain in two short paragraphs.'),('orbit','Why do astronauts in an orbiting spacecraft appear weightless even though Earth\'s gravity is still present? Explain in two short paragraphs.')])]:
        split.extend({'id':'science-'+n,'family':'explanation','messages':[{'role':'user','content':p}],'check':None,'max_new_tokens':1024} for n,p in topics)
    forbidden={key(c['messages']) for c in legacy+dev+test}
    assert len(forbidden)==len(legacy+dev+test)
    pools={k:[] for k in ['program','math','science','short']}
    def add(kind,messages,provenance):
        if key(messages[:-1]) in forbidden:return
        enc=encode_complete(tok,messages,1024)
        if enc:pools[kind].append({'kind':kind,'messages':messages,'provenance':provenance,**enc})
    def pair(p,a):return [{'role':'user','content':p},{'role':'assistant','content':a}]
    source=ROOT/'results/correctness-healing-20261025/data.json'
    candidates=json.loads(source.read_text())['train']
    random.Random(25102533).shuffle(candidates)
    groups=set()
    exclude=re.compile(r'duplicat|interval|binary.search|word.?count|count.?word|photosynth|respirat|spoon|chunk|factorial|rotat|run.length|consecutive|transpos|matrix|bracket|parenthe',re.I)
    for r in candidates:
        if r['kind']!='code' or r['problem_group'] in groups:continue
        code=r['reference_code'];prompt=r['messages'][0]['content'].split('\n\nGiven the following')[0]
        if len(code)>2200 or len(prompt)>2400 or exclude.search(code+'\n'+prompt) or not pure_reference(code):continue
        got=run_reference(code,r['check']['inputs'])
        if not got or not equivalent(got['result'],r['check']['expected_output']):continue
        before=len(pools['program'])
        add('program',pair(prompt+'\n\nImplement the specification as a complete Python function named main_solution. Return only code.', '```python\n'+code.strip()+'\n```'),
            {'source_id':r['id'],'problem_group':r['problem_group'],'source_file':r['source_file'],'source_row':r['source_row'],'verification':'reference re-executed in bwrap against supplied IO; not proof of natural-language specification','check':r['check']})
        if len(pools['program'])>before:groups.add(r['problem_group'])
        if len(groups)>=192:break
    if len(groups)<64:raise ValueError('Insufficient full programs')
    # Exact synthetic worked arithmetic; exclude all held-out operand pairs.
    held_pairs=set()
    for c in legacy+dev+test:
        for m in c['messages']:
            match=re.search(r'(\d+) multiplied by (\d+)',m['content'])
            if match:held_pairs.add(tuple(sorted(map(int,match.groups()))))
    rng=random.Random(25102534)
    seen=set()
    while len(pools['math'])<320:
        a,b=rng.randint(21,199),rng.randint(12,89)
        if tuple(sorted((a,b))) in held_pairs or (a,b) in seen:continue
        seen.add((a,b));t,u=divmod(b,10)
        answer=f'Break {b} into {t*10} + {u}. Then {a} × {t*10} = {a*t*10}, and {a} × {u} = {a*u}. Adding these partial products gives {a*t*10} + {a*u} = {a*b}.\n\nThe answer is {a*b}.'
        add('math',pair(f'Calculate {a} times {b}. Show the arithmetic and give the final answer.',answer),{'verification':'integer multiplication and distributive decomposition','a':a,'b':b,'expected':a*b})
    for name,p,a,url in SCIENCE:add('science',pair(p,a),{'topic':name,'source':url,'verification':'original explanation checked against primary source'})
    short=synthetic_cases(25102535,80,list(forbidden))
    for c in short:
        expected=c['check']['expected'];kind=c['check']['kind']
        if c['family']=='arithmetic':
            nums=re.search(r'(\d+) multiplied by (\d+)',c['messages'][0]['content'])
            if tuple(sorted(map(int,nums.groups()))) in held_pairs:continue
        answer=json.dumps(expected) if kind=='json' else repr(expected) if kind=='literal' else str(expected)
        add('short',c['messages']+[{'role':'assistant','content':answer}],{'verification':'deterministic synthetic oracle','check':c['check']})
    weights={'program':.40,'math':.25,'science':.20,'short':.15}
    rows=[];indices={};used=Counter();position=Counter()
    for k,pool in pools.items():
        random.Random('heal:'+k).shuffle(pool);indices[k]=list(range(len(rows),len(rows)+len(pool)));rows+=pool
    updates=[]
    for step in range(16):
        batch=[];count=0
        while count<4096:
            k=min(weights,key=lambda k:used[k]/weights[k]);idx=indices[k][position[k]%len(indices[k])];position[k]+=1
            n=len(rows[idx]['targets']);used[k]+=n;count+=n;batch.append(idx)
        updates.append(batch)
    protocol={'seed':25102536,'steps':16,'evaluation_steps':[8,16],'rank':16,'alpha':32,'lr':3e-5,'warmup_steps':2,
      'loss':'assistant correctness CE including EOS, no teacher KL','training':'FP32 parameters; CPU BF16 autocast; AdamW betas 0.9,0.95 wd0 clip1; max complete sequence1024',
      'selection':'Development closed+program correctness; no more loop flags and at most one fewer completion than step0. Tie: fewer loops, more completions, earlier step. Explanations reviewed separately, not selection.',
      'recipe_sha256':sha(RECIPE),'source':str(SOURCE),'recipe':str(RECIPE),'cloud_spend':0,
      'mix_target_tokens':weights,'pool_counts':{k:len(v) for k,v in pools.items()},'scheduled_target_tokens':dict(used),
      'scheduled_rows':dict(position),'unique_scheduled_rows':len(set(i for b in updates for i in b)),
      'limitations':['Only eight distinct science topics, resampled.','Worked math is synthetic multiplication, not a broad reasoning corpus.','Horizon programs checked against supplied IO, not exhaustive formal verification.','Fresh tasks share synthetic templates with training; this measures task-family transfer, not broad intelligence.','No claim that fresh evaluation was absent from stock pretraining or projection calibration.']}
    save({'train':rows,'updates':updates,'dev':dev,'test':test,'legacy':legacy,'protocol':protocol},OUT/'data.json')
    save(protocol,OUT/'protocol.json');print(json.dumps(protocol),flush=True)

def smoke():
    from test_projection_models import tiny_model
    model=tiny_model('qwen',True)
    p=torch.linalg.qr(torch.randn(32,32)).Q[:,:24]
    build(model,[p,p],[1.]*5)
    ids=torch.randint(0,101,(1,32))
    with torch.no_grad():before=model(ids,use_cache=False).logits
    install_lora(model,4,8)
    with torch.no_grad():torch.testing.assert_close(before,model(ids,use_cache=False).logits,rtol=0,atol=0)
    frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
    row={'input_ids':ids[0].tolist(),'positions':list(range(15,31)),'targets':ids[0,16:].tolist()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003,weight_decay=0)
    first=answer_loss(model,row,'cpu').item()
    model.train()
    for _ in range(4):
        with torch.autocast('cpu',dtype=torch.bfloat16):train_update(model,opt,[row],'cpu',.003)
    model.eval();final=answer_loss(model,row,'cpu').item()
    assert final<first
    for n,p in model.named_parameters():
        if n in frozen:assert torch.equal(p,frozen[n]),n
    return {'zero_adapter_exact':True,'base_unchanged':True,'first_ce_sum':first,'final_ce_sum':final}

def metric_key(s,base,step):
    if s['loop_flags']>base['loop_flags'] or s['ended']<base['ended']-1:return None
    return (s['closed_correct']+s['programs_correct'],-s['loop_flags'],s['ended'],-step)

def run():
    if (OUT/'training.json').exists():raise ValueError('Existing run: explicit resume required')
    data=json.loads((OUT/'data.json').read_text());p=data['protocol']
    assert sha(RECIPE)==p['recipe_sha256']
    save(smoke(),OUT/'smoke.json')
    torch.set_num_threads(4);torch.manual_seed(p['seed'])
    tok=load_tokenizer(SOURCE)
    recipe=torch.load(RECIPE,weights_only=True,map_location='cpu')
    model=from_recipe(SOURCE,recipe,load_source).eval()
    ids=torch.tensor([data['train'][0]['input_ids'][:64]])
    with torch.no_grad():before=model(ids,use_cache=False).logits
    info=install_lora(model,p['rank'],p['alpha'])
    with torch.no_grad():torch.testing.assert_close(before,model(ids,use_cache=False).logits,rtol=0,atol=0)
    del before
    save_adapter(model,OUT/'step-0',info)
    dev={0:evaluate_generations(model,tok,data['dev'],OUT/'dev-0.json')}
    evaluate_generations(model,tok,data['test'],OUT/'test-0.json')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],betas=(.9,.95),weight_decay=0,foreach=False)
    log=[]
    for step,batch in enumerate(data['updates'],1):
        model.train();model.config.use_cache=False
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=train_update(model,optimizer,[data['train'][i] for i in batch],'cpu',p['lr']*min(1,step/p['warmup_steps']))
        log.append({'step':step,**result});save(log,OUT/'training.json');print(json.dumps(log[-1]),flush=True)
        if step in p['evaluation_steps']:
            save_adapter(model,OUT/f'step-{step}',{**info,'step':step,'data_sha256':sha(OUT/'data.json'),'recipe_sha256':sha(RECIPE)})
            torch.save({'optimizer':optimizer.state_dict(),'step':step,'rng':torch.get_rng_state()},OUT/f'step-{step}'/'optimizer.pt')
            model.eval();dev[step]=evaluate_generations(model,tok,data['dev'],OUT/f'dev-{step}.json')
    eligible={s:metric_key(m,dev[0],s) for s,m in dev.items()};eligible={s:k for s,k in eligible.items() if k is not None}
    selected=max(eligible,key=eligible.get)
    save({'selected_step':selected,'dev':dev,'eligible_keys':eligible},OUT/'selection.json')
    load_adapter(model,OUT/f'step-{selected}');model.eval()
    with torch.no_grad():before=model(ids,use_cache=False).logits
    save_adapter(model,OUT/'selected',{**info,'selected_step':selected,'data_sha256':sha(OUT/'data.json'),'recipe_sha256':sha(RECIPE)})
    with torch.no_grad():
        for par in model.parameters():
            if par.requires_grad:par.zero_()
    load_adapter(model,OUT/'selected')
    with torch.no_grad():torch.testing.assert_close(before,model(ids,use_cache=False).logits,rtol=0,atol=0)
    gate=cache_check(model,ids)
    if not gate['numerical_pass']:raise RuntimeError('Cache gate failed')
    save({'adapter_reload_exact':True,'cache':gate,'zero_adapter_exact':True},OUT/'gates.json')
    test=evaluate_generations(model,tok,data['test'],OUT/'test-selected.json')
    legacy=evaluate_generations(model,tok,data['legacy'],OUT/'legacy-selected.json')
    result={'selected_step':selected,'dev':dev,'fresh_before':json.loads((OUT/'test-0.json').read_text())['summary'],'fresh_after':test,
            'legacy_before':json.loads((OLD/'shared-broad-generations.json').read_text())['summary'],'legacy_after':legacy,'adapter':info,'cloud_spend':0}
    save(result,OUT/'result.json');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','smoke']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run()
    else:print(smoke())
