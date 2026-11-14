"""Second bounded repair stage: broader checked curriculum, frozen projection."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import random
import re
import torch
from broad_healing import ROOT, OUT as PARENT, RECIPE, SOURCE, sha, key, smoke
from load_broad_healing import load as load_parent
from local_projection_probe import save, evaluate_generations
from correctness_heal import save_adapter, load_adapter, train_update
from correctness_eval import synthetic_cases
from healing_data import encode_complete
from scale_data import load_tokenizer
from correctness_prepare import pure_reference
from horizon_data import run_reference, equivalent, arithmetic
from curriculum_tasks import FAMILIES, math_example, training_algorithms, code_eval
from curriculum_science import SCIENCE
from broad_healing_science import SCIENCE as OLD_SCIENCE
from summarize_scale_probe import repeated_line_run
from audit_rotation import cache_check

OUT=ROOT/'results/curriculum-healing-20261026'

def pair(p,a):return [{'role':'user','content':p},{'role':'assistant','content':a}]
def math_key(r):return (r['family'],tuple(r['values']))

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'data.json').exists():raise ValueError('Frozen data exists')
    old=json.loads((PARENT/'data.json').read_text());tok=load_tokenizer(SOURCE)
    historical=old['dev']+old['test']+old['legacy']
    old_prompts={key(c['messages']) for c in historical}|{key(c['messages'][:-1]) for c in old['train']}
    held_pairs=set();trained_pairs=set()
    for c in historical:
        for m in c['messages']:
            match=re.search(r'(\d+) multiplied by (\d+)',m['content'])
            if match:held_pairs.add(tuple(sorted(map(int,match.groups()))))
    for r in old['train']:
        if r['kind']=='math':trained_pairs.add(tuple(sorted((r['provenance']['a'],r['provenance']['b']))))
        for m in r['messages'][:-1]:
            match=re.search(r'(\d+) multiplied by (\d+)',m['content'])
            if match:trained_pairs.add(tuple(sorted(map(int,match.groups()))))
    used=set();raw={}
    for split,n,seed in [('dev',2,2610261),('test',3,2610262),('train',100,2610263)]:
        rng=random.Random(seed);raw[split]=[]
        for fam in FAMILIES:
            count=0
            while count<n:
                r=math_example(rng,fam);k=math_key(r)
                if k in used:continue
                if fam=='multiply':
                    mp=tuple(sorted(r['values']))
                    if mp in held_pairs or (split!='train' and mp in trained_pairs):continue
                    # Keep commutative pairs disjoint, including new held-out cases.
                    if any(x[0]=='multiply' and tuple(sorted(x[1]))==mp for x in used):continue
                assert arithmetic(r['expression'])==arithmetic(r['expected'])
                used.add(k);raw[split].append(r);count+=1
    def closed(rows,split):
        return [{'id':f'{split}-{r["family"]}-{i}','family':'math_'+r['family'],
            'messages':[{'role':'user','content':r['transfer']+' Return only the final number or reduced fraction.'}],
            'check':{'kind':'numeric','expected':r['expected']},'max_new_tokens':256,'oracle':r} for i,r in enumerate(rows)]
    dev,test=closed(raw['dev'],'cdev'),closed(raw['test'],'ctest')
    # The old generator has only 99 unit-conversion prompts, of which the
    # previous experiment already used 97. Expand that domain rather than
    # asking a finite rejection sampler for nonexistent unseen examples.
    retained_pairs=held_pairs|trained_pairs|{tuple(sorted(r['values'])) for rows in raw.values() for r in rows if r['family']=='multiply'}
    retained_prompts=set(old_prompts)
    def retention(seed,n):
        result=[];counts=Counter();rng=random.Random(seed)
        for attempt in range(10000):
            if len(result)==7*n:return result
            for c in synthetic_cases(seed*10000+attempt,1):
                fam=c['family']
                if counts[fam]>=n:continue
                if fam=='units':
                    tenths=rng.randint(101,999)
                    c['messages']=[{'role':'user','content':f'Convert {tenths/10:g} liters to milliliters. Return only the number.'}]
                    c['check']['expected']=str(tenths*100)
                if key(c['messages']) in retained_prompts:continue
                if fam=='arithmetic':
                    match=re.search(r'(\d+) multiplied by (\d+)',c['messages'][0]['content'])
                    mp=tuple(sorted(map(int,match.groups())))
                    if mp in retained_pairs:continue
                    retained_pairs.add(mp)
                retained_prompts.add(key(c['messages']));counts[fam]+=1;result.append(c)
        raise ValueError('Retention sampler exhausted its bounded attempts')
    dev+=retention(2610264,1)
    test+=retention(2610265,2)
    dcode,tcode=code_eval();dev+=dcode;test+=tcode
    science_dev=[('compass','Why does a freely turning compass needle tend to point north? Explain briefly.','Earth has a magnetic field; the magnetized needle aligns approximately with its local horizontal component. Magnetic north differs from geographic north; no attraction to the North Star.'),
                 ('mirage','Why can a hot road appear to have a distant patch of water on it? Explain briefly.','Refraction through air layers with different temperatures bends light; the image can show sky. No actual puddle is required.')]
    science_test=[('thunder','Why do we usually see lightning before hearing its thunder? Explain briefly.','Light travels much faster than sound in air; the delay grows with distance, not because lightning must wait to make sound.'),
                  ('whale','Why is a whale classified as a mammal rather than a fish even though it lives in water? Explain briefly.','Classification uses inherited biological traits and ancestry; whales breathe air with lungs and nurse young with milk. Habitat and body shape alone do not make them fish.')]
    for cases,items in [(dev,science_dev),(test,science_test)]:
        for name,p,rubric in items:cases.append({'id':'cscience-'+name,'family':'explanation','messages':[{'role':'user','content':p}],'check':None,'max_new_tokens':1024,'manual_rubric':rubric})
    forbidden={key(c['messages']) for c in dev+test+historical}
    assert len({key(c['messages']) for c in dev+test})==len(dev+test)
    assert not ({key(c['messages']) for c in dev+test}&old_prompts)
    pools={k:[] for k in ['math','verified_code','horizon_code','science','retain']}
    def add(kind,messages,provenance):
        if key(messages[:-1]) in forbidden:raise ValueError('Train/eval prompt overlap')
        enc=encode_complete(tok,messages,1024)
        if enc:pools[kind].append({'kind':kind,'messages':messages,'provenance':provenance,**enc})
    for i,r in enumerate(raw['train']):
        p=r['prompt']+(' Show a brief calculation and the final answer.' if i%2==0 else ' Return only the final number or reduced fraction.')
        a=r['work']+'\n\nAnswer: '+r['expected'] if i%2==0 else r['expected']
        add('math',pair(p,a),{'verification':'exact rational arithmetic, separately parsed expression oracle',**r})
    verification={}
    for name,sig,task,code,tests in training_algorithms():
        for inputs,expected in tests:
            got=run_reference(code,inputs)
            if not got or not equivalent(got['result'],expected):raise ValueError('Curated code oracle disagreement: '+name)
        verification[name]={'tests':len(tests),'all_passed':True}
        # Different supplied examples, same algorithm. Do not count these as
        # additional independent algorithms or reference programs.
        for inputs,expected in tests[:8]:
            prompt=f'Use exactly this function signature: {sig}.\n{task}\nFor example, input {json.dumps(inputs)} must produce {json.dumps(expected)}.\nReturn only complete Python code.'
            add('verified_code',pair(prompt,'```python\n'+code+'\n```'),{'algorithm':name,'signature':sig,'tests':tests,'verification':'multiple independent input/output oracles; bwrap re-execution'})
    source=ROOT/'results/correctness-healing-20261025/data.json';candidates=json.loads(source.read_text())['train'];random.Random(2610266).shuffle(candidates)
    groups={r['provenance']['problem_group'] for r in old['train'] if r['kind']=='program'};new_groups=set()
    exclude=re.compile(r'duplicat|interval|binary.search|word.?count|count.?word|photosynth|respirat|spoon|chunk|factorial|rotat|run.length|consecutive|transpos|matrix|bracket|parenthe|palindrom|cumul.*prod|dot.product|scalar.product|non.?repeat|unique.*char|substring|prefix.sum|greatest.common|prime|digit.sum|reverse.*word',re.I)
    for r in candidates:
        if r['kind']!='code' or r['problem_group'] in groups|new_groups:continue
        code=r['reference_code'];prompt=r['messages'][0]['content'].split('\n\nGiven the following')[0]
        if len(code)>2400 or len(prompt)>2400 or exclude.search(code+'\n'+prompt) or not pure_reference(code):continue
        fn=next((n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='main_solution'),None)
        if fn is None:continue
        got=run_reference(code,r['check']['inputs'])
        if not got or not equivalent(got['result'],r['check']['expected_output']):continue
        sig='main_solution('+ast.unparse(fn.args)+')';before=len(pools['horizon_code'])
        add('horizon_code',pair(prompt+'\nUse exactly this function signature: '+sig+'. Return only complete Python code.','```python\n'+code.strip()+'\n```'),
            {'source_id':r['id'],'problem_group':r['problem_group'],'source_file':r['source_file'],'source_row':r['source_row'],'check':r['check'],'verification':'re-executed supplied reference IO; not independent proof of specification'})
        if len(pools['horizon_code'])>before:new_groups.add(r['problem_group'])
        if len(new_groups)>=256:break
    assert len(new_groups)==256
    for topic,url,p,a in SCIENCE:add('science',pair(p+' Explain in two short paragraphs.',a),{'topic':topic,'source':url,'verification':'original summary checked against OpenStax source'})
    for topic,p,a,url in OLD_SCIENCE:add('science',pair(p,a),{'topic':topic,'source':url,'verification':'source-checked prior science target; replay'})
    for r in old['train']:
        if r['kind']=='short':add('retain',r['messages'],{'verification':'previous independently generated task; replay','prior_provenance':r['provenance']})
    weights={'math':.35,'verified_code':.15,'horizon_code':.20,'science':.20,'retain':.10}
    train=[];indices={};positions=Counter();tokens=Counter()
    for kind,pool in pools.items():
        random.Random('curriculum:'+kind).shuffle(pool);indices[kind]=list(range(len(train),len(train)+len(pool)));train+=pool
    updates=[]
    for _ in range(16):
        batch=[];n=0
        while n<4096:
            kind=min(weights,key=lambda k:tokens[k]/weights[k]);idx=indices[kind][positions[kind]%len(indices[kind])];positions[kind]+=1
            count=len(train[idx]['targets']);n+=count;tokens[kind]+=count;batch.append(idx)
        updates.append(batch)
    parent_meta=json.loads((PARENT/'selected/adapter.json').read_text());assert parent_meta['selected_step']==16
    protocol={'parent':str(PARENT/'selected'),'parent_adapter_sha256':sha(PARENT/'selected/adapter.safetensors'),'parent_optimizer_sha256':sha(PARENT/'step-16/optimizer.pt'),
      'recipe_sha256':sha(RECIPE),'steps':16,'evaluation_steps':[8,16],'lr':3e-5,'optimizer':'resume parent AdamW state; no new warmup','rank':16,'alpha':32,'seed':2610267,
      'loss':'correct assistant targets including EOS; no teacher KL','training_precision':'CPU BF16 autocast with FP32 stored parameters','evaluation_precision':'CPU FP32, four threads, fixed original batches',
      'max_complete_sequence':1024,'mix':weights,'pool_counts':{k:len(v) for k,v in pools.items()},'scheduled_target_tokens':dict(tokens),'scheduled_presentations':dict(positions),
      'unique_scheduled_rows':len(set(i for b in updates for i in b)),
      'selection':'Highest development closed+program correctness, rejecting more combined word/line loop flags or more than one lost completion. Tie: fewer loops, more completions, earlier stage step. Parent eligible. Explanations reviewed separately.',
      'science_new_explanations':len(SCIENCE),'science_replayed_explanations':len(OLD_SCIENCE),'verified_algorithm_families':len(verification),'cloud_spend':0,
      'limitations':['Continuation plus new curriculum; no matched old-curriculum continuation control.','Arithmetic has new wording and operands but shared task families.','Science pool remains small and resampled.','Only curated algorithm subset has multiple independent oracle tests; Horizon subset has supplied-reference checks.','Old benchmarks have been inspected but are not used to select this checkpoint.','No claim of absence from stock pretraining or earlier calibration.']}
    save({'train':train,'updates':updates,'dev':dev,'test':test,'legacy':old['legacy'],'protocol':protocol,'algorithm_verification':verification},OUT/'data.json')
    save(protocol,OUT/'protocol.json');print(json.dumps(protocol),flush=True)


def assess(model,tok,cases,path):
    model.eval();evaluate_generations(model,tok,cases,path)
    data=json.loads(path.read_text())
    for c in data['cases']:c['combined_loop_flag']=c['loop_flag'] or repeated_line_run(c['answer'])>=8
    data['summary']['word_loop_flags']=sum(c['loop_flag'] for c in data['cases'])
    data['summary']['loop_flags']=sum(c['combined_loop_flag'] for c in data['cases'])
    save(data,path);return data['summary']


def run():
    if (OUT/'training.json').exists():raise ValueError('Explicit recovery required for existing run')
    data=json.loads((OUT/'data.json').read_text());p=data['protocol']
    assert sha(PARENT/'selected/adapter.safetensors')==p['parent_adapter_sha256']
    assert sha(PARENT/'step-16/optimizer.pt')==p['parent_optimizer_sha256']
    assert sha(RECIPE)==p['recipe_sha256']
    torch.set_num_threads(4);torch.manual_seed(p['seed']);tok=load_tokenizer(SOURCE);model=load_parent()
    metadata=json.loads((PARENT/'selected/adapter.json').read_text());info={k:metadata[k] for k in ['rank','alpha','modules','trainable_parameters']}
    optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],betas=(.9,.95),weight_decay=0,foreach=False)
    state=torch.load(PARENT/'step-16/optimizer.pt',weights_only=True,map_location='cpu');optimizer.load_state_dict(state['optimizer']);del state
    save_adapter(model,OUT/'step-0',{**info,'stage_step':0,'total_step':16,'recipe_sha256':sha(RECIPE)})
    dev={0:assess(model,tok,data['dev'],OUT/'dev-0.json')}
    assess(model,tok,data['test'],OUT/'test-0.json')
    log=[]
    for step,batch in enumerate(data['updates'],1):
        model.train();model.config.use_cache=False
        with torch.autocast('cpu',dtype=torch.bfloat16):r=train_update(model,optimizer,[data['train'][i] for i in batch],'cpu',p['lr'])
        log.append({'stage_step':step,'total_step':16+step,**r});save(log,OUT/'training.json');print(json.dumps(log[-1]),flush=True)
        if step in p['evaluation_steps']:
            save_adapter(model,OUT/f'step-{step}',{**info,'stage_step':step,'total_step':16+step,'data_sha256':sha(OUT/'data.json'),'recipe_sha256':sha(RECIPE)})
            torch.save({'optimizer':optimizer.state_dict(),'step':16+step,'rng':torch.get_rng_state()},OUT/f'step-{step}'/'optimizer.pt')
            dev[step]=assess(model,tok,data['dev'],OUT/f'dev-{step}.json')
    eligible={s:(m['closed_correct']+m['programs_correct'],-m['loop_flags'],m['ended'],-s) for s,m in dev.items()
              if m['loop_flags']<=dev[0]['loop_flags'] and m['ended']>=dev[0]['ended']-1}
    selected=max(eligible,key=eligible.get);save({'selected_stage_step':selected,'selected_total_step':16+selected,'dev':dev,'eligible_keys':eligible},OUT/'selection.json')
    load_adapter(model,OUT/f'step-{selected}');model.eval()
    save_adapter(model,OUT/'selected',{**info,'stage_step':selected,'total_step':16+selected,'parent_adapter_sha256':p['parent_adapter_sha256'],'data_sha256':sha(OUT/'data.json'),'recipe_sha256':sha(RECIPE)})
    ids=torch.tensor([data['train'][0]['input_ids'][:64]])
    with torch.no_grad():before=model(ids,use_cache=False).logits
    with torch.no_grad():
        for v in model.parameters():
            if v.requires_grad:v.zero_()
    load_adapter(model,OUT/'selected')
    with torch.no_grad():torch.testing.assert_close(before,model(ids,use_cache=False).logits,atol=0,rtol=0)
    gate=cache_check(model,ids);assert gate['numerical_pass'];save({'adapter_reload_exact':True,'cache':gate},OUT/'gates.json')
    fresh=assess(model,tok,data['test'],OUT/'test-selected.json')
    legacy=assess(model,tok,data['legacy'],OUT/'legacy-selected.json')
    result={'selected_stage_step':selected,'selected_total_step':16+selected,'dev':dev,'fresh_before':json.loads((OUT/'test-0.json').read_text())['summary'],
      'fresh_after':fresh,'legacy_before':json.loads((PARENT/'legacy-selected.json').read_text())['summary'],'legacy_after':legacy,'cloud_spend':0}
    save(result,OUT/'result.json');print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
