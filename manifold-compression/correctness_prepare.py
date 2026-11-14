"""Freeze a correctness-focused Qwen27 healing pilot, without paid actions."""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import random
import warnings

from correctness_eval import synthetic_cases,long_cases,LONG_PROMPTS
from horizon_data import digest

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/correctness-healing-20261025'
RECIPE=ROOT/'results/scale-probe-20261024/mc-scale-9b5ca68c9da9/qwen27'


def pure_reference(code):
    """Exclude environment-dependent reference programs from the pilot."""
    allowed={'math','collections','itertools','functools','heapq','bisect','re','string','fractions','decimal','typing','statistics'}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',SyntaxWarning)
        tree=ast.parse(code)
    for node in ast.walk(tree):
        # Set iteration has no stable printed order. Exclude these programs
        # rather than treating one arbitrary ordering as a unique gold answer.
        if isinstance(node,(ast.Set,ast.SetComp)):return False
        if isinstance(node,ast.Import) and any(a.name.split('.')[0] not in allowed for a in node.names):return False
        if isinstance(node,ast.ImportFrom) and (node.module or '').split('.')[0] not in allowed:return False
        if isinstance(node,ast.Name) and node.id in {'open','input','eval','exec','globals','locals','vars','__import__','getattr','setattr','delattr','compile','set','frozenset','hash','id'}:return False
        if isinstance(node,ast.Attribute) and node.attr.startswith('__'):return False
    return True


def prepare(source,out=OUT,max_steps=32):
    if (out/'data.json').exists():raise ValueError('Frozen data already exists; choose a new output directory')
    splits={s:[json.loads(l) for l in (source/(s+'.jsonl')).open()] for s in ['train','validation','test']}
    for s,rows in splits.items():
        splits[s]=[r for r in rows if r['kind']!='code' or pure_reference(r['reference_code'])]
    # This first correctness pilot excludes unverified general SFT labels.
    # Preserve the broader source audit so this decision remains visible.
    splits['train']=[r for r in splits['train'] if r['kind'] in ('math','code')]
    pools={k:[r for r in splits['train'] if r['kind']==k] for k in ['math','code']}
    for k,pool in pools.items(): random.Random('pilot:'+k).shuffle(pool)
    weights={'math':.10,'code':.90}
    used=collections.Counter();positions=collections.Counter();train=[];updates=[]
    for step in range(max_steps):
        update=[];tokens=inputs=0
        while tokens<4096 and inputs<65536:
            choices=[k for k in pools if positions[k]<len(pools[k])]
            if not choices:break
            k=min(choices,key=lambda k:used[k]/weights[k])
            row=pools[k][positions[k]];positions[k]+=1
            used[k]+=row['assistant_tokens'];tokens+=row['assistant_tokens'];inputs+=row['input_tokens']
            update.append(len(train));train.append(row)
        if not update:break
        updates.append(update)
    # Fresh deterministic tasks dominate the correctness evaluation. Horizon
    # code checks use previously withheld functions, output prediction only.
    def code_eval(split,n):
        chosen=[r for r in splits[split] if r['kind']=='code' and set(r['check']['target'])=={'output'}
                and r['assistant_tokens']<=192]
        random.Random('eval:'+split).shuffle(chosen)
        return [{'id':r['id'],'family':'horizon_code','problem_group':r['problem_group'],
                 'messages':r['messages'][:-1],'check':{'kind':'codeio','expected':r['check']['target']},
                 'max_new_tokens':256} for r in chosen[:n]]
    dev=synthetic_cases(2510251,8)+code_eval('validation',8)+long_cases('dev',LONG_PROMPTS[:2])
    test=synthetic_cases(2510252,16,[json.dumps(c['messages'],sort_keys=True) for c in dev])+code_eval('test',16)+long_cases('test',LONG_PROMPTS[2:])
    assert len(dev)==66 and len(test)==134
    groups={s:{r['problem_group'] for r in rows} for s,rows in splits.items()}
    assert not groups['train']&groups['test'] and not groups['train']&groups['validation']
    prompts=[digest(json.dumps(c['messages'],sort_keys=True)) for c in dev+test]
    assert len(prompts)==len(set(prompts))
    data={'schema':'correctness-healing-v1','source':{'repo':'Qwen/Qwen3.5-27B','revision':'fc05daec18b0a78c049392ed2e771dde82bdf654'},
          'width':3824,'max_length':2048,'train':train,'updates':updates,'dev':dev,'test':test,
          'settings':{'seed':251025,'rank':16,'alpha':32,'lr':3e-5,'warmup_steps':4,'max_steps':len(updates),
                      'loss':'assistant token cross-entropy, including EOS; no teacher KL',
                      'evaluation_steps':[4,8,16,32],'max_input_tokens_per_update':65536},
          'selection':'Highest development correctness among checkpoints with no more loop flags than before and at most one fewer ended response; tie-break fewer loops, more completions, earlier step.',
          'early_stop':'Stop at a measured checkpoint if development correctness loses at least 4/64 or loop flags increase by at least 2 relative to the unhealed student.',
          'limits':'Unverified general instruction SFT is excluded. Math is recomputed; CodeI/O agrees with its reference program. No claim that a reference program proves the full natural-language task. This narrow first pilot and its fresh synthetic tasks are diagnostic, not a general capability benchmark.'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    manifest={'data_sha256':hashlib.sha256((out/'data.json').read_bytes()).hexdigest(),
        'prepared_source_manifest':json.loads((source/'manifest.json').read_text()),
        'recipe_sha256':{n:hashlib.sha256((RECIPE/n).read_bytes()).hexdigest() for n in ['basis.pt','rms-scales.json','result.json']},
        'rows':len(train),'steps':len(updates),'assistant_tokens_by_kind':dict(used),
        'input_tokens':sum(r['input_tokens'] for r in train),'dev_cases':len(dev),'test_cases':len(test),
        'unpaid_preparation':True}
    (out/'preparation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['prepared_source_manifest','recipe_sha256']},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('target/manifold-healing/horizon-prepared'))
    ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args();prepare(args.source,args.out)
