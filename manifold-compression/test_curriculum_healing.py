import json
from collections import Counter
from fractions import Fraction
import random
import re
from curriculum_healing import OUT, PARENT, key
from curriculum_tasks import FAMILIES, math_example
from curriculum_science import SCIENCE
from horizon_data import arithmetic
from local_projection_probe import grade_case


def test_math_oracles_and_rational_targets():
    rng=random.Random(73)
    for family in FAMILIES:
        for _ in range(50):
            row=math_example(rng,family)
            assert arithmetic(row['expression'])==Fraction(row['expected'])


def test_frozen_split_and_target_invariants():
    data=json.loads((OUT/'data.json').read_text());old=json.loads((PARENT/'data.json').read_text())
    held={key(c['messages']) for split in ['dev','test','legacy'] for c in data[split]}
    train={key(r['messages'][:-1]) for r in data['train']}
    assert not train&held
    old_train={key(r['messages'][:-1]) for r in old['train']}
    assert not old_train&{key(c['messages']) for split in ['dev','test'] for c in data[split]}
    def multiplication_pairs(rows, training):
        result=set()
        for r in rows:
            prov=r.get('provenance',{}) if training else r.get('oracle',{})
            if prov.get('family')=='multiply':result.add(tuple(sorted(prov['values'])))
            if training and r.get('kind')=='math' and 'a' in prov:result.add(tuple(sorted((prov['a'],prov['b']))))
            for m in r['messages']:
                match=re.search(r'(\d+) multiplied by (\d+)',m['content'])
                if match:result.add(tuple(sorted(map(int,match.groups()))))
        return result
    trained=multiplication_pairs(data['train']+old['train'],True)
    evaluated=multiplication_pairs(data['dev']+data['test'],False)
    assert not trained&evaluated
    for r in data['train']:
        assert len(r['input_ids'])<=1024 and r['targets'][-1]==r['eos_token_id']
        assert r['targets']==r['input_ids'][r['answer_start']:]
        assert r['positions']==list(range(r['answer_start']-1,len(r['input_ids'])-1))
    assert len(data['updates'])==16
    assert len(data['dev'])==27 and len(data['test'])==44
    assert len(data['algorithm_verification'])==12
    assert all(r['all_passed'] and r['tests']>=6 for r in data['algorithm_verification'].values())
    assert all(sum(len(data['train'][i]['targets']) for i in b)>=4096 for b in data['updates'])


def test_science_provenance_budget():
    words=Counter()
    for _,url,prompt,answer in SCIENCE:words[url]+=len((prompt+' '+answer+' Explain in two short paragraphs.').split())
    assert len(SCIENCE)==24 and len(words)==12
    assert max(words.values())<=200


def test_final_numeric_content_overrides_intermediate_equations():
    from curriculum_numeric_audit import final_numeric
    assert final_numeric('First 1 = 19/19.\nThen 19/19 + 4/19 = 23/19.\n\n23/19')[0]==Fraction(23,19)
    assert final_numeric('1 3/7')[0]==Fraction(10,7)
    assert final_numeric('-1 3/7')[0]==Fraction(-10,7)
    assert final_numeric(r'\boxed{\frac{23}{19}}')[0]==Fraction(23,19)
    assert final_numeric('2 + 3 = 5\nFinal answer: 7')[0]==7
    import pytest
    with pytest.raises(ValueError):final_numeric('5 or 6')
    with pytest.raises(ValueError):final_numeric('5\nActually I cannot solve this.')


def test_content_selection_uses_dev_and_preserves_loop_gate(tmp_path):
    from curriculum_numeric_audit import select
    cases=[{'id':'a','family':'math_add','check':{'kind':'numeric','expected':'5'}},
           {'id':'b','family':'math_add','check':{'kind':'numeric','expected':'7'}}]
    (tmp_path/'data.json').write_text(json.dumps({'dev':cases}))
    (tmp_path/'selection.json').write_text(json.dumps({'selected_stage_step':8}))
    for step,answers in [(0,['0','0']),(8,['5','0']),(16,['2+3=5\n5','3+4=7\n7'])]:
        rows=[{'id':c['id'],'family':c['family'],'answer':a,'grade':{'correct':False,'parsed':False,'format_pass':False}} for c,a in zip(cases,answers)]
        (tmp_path/f'dev-{step}.json').write_text(json.dumps({'cases':rows,'summary':{'programs_correct':0,'loop_flags':0,'ended':2}}))
    assert select(tmp_path)['selected_stage_step']==16
    p=tmp_path/'dev-16.json';d=json.loads(p.read_text());d['summary']['loop_flags']=1;p.write_text(json.dumps(d))
    assert select(tmp_path)['selected_stage_step']==8
