import json
from pathlib import Path
from broad_healing import metric_key, OUT
from local_projection_probe import grade_case


def test_selection_rejects_correctness_gain_with_more_loops():
    base={'closed_correct':10,'programs_correct':0,'loop_flags':0,'ended':25}
    better={**base,'closed_correct':20,'loop_flags':1}
    assert metric_key(better,base,8) is None
    assert metric_key({**base,'ended':23},base,8) is None
    assert metric_key(base,base,0)>metric_key(base,base,8)


def test_frozen_training_masks_and_split_separation():
    data=json.loads((OUT/'data.json').read_text())
    held={json.dumps(c['messages'],sort_keys=True) for split in ['dev','test','legacy'] for c in data[split]}
    for r in data['train']:
        assert json.dumps(r['messages'][:-1],sort_keys=True) not in held
        assert len(r['input_ids'])<=1024
        assert r['targets'][-1]==r['eos_token_id']
        assert r['targets']==r['input_ids'][r['answer_start']:]
        assert r['positions']==list(range(r['answer_start']-1,len(r['input_ids'])-1))
        if r['kind']=='short':
            case={'check':r['provenance']['check'],'family':'short','id':'target-check'}
            assert grade_case(r['messages'][-1]['content'],case)['correct']
        if r['kind']=='math':
            p=r['provenance'];assert p['a']*p['b']==p['expected']
    assert len(data['updates'])==16
    assert all(sum(len(data['train'][i]['targets']) for i in b)>=4096 for b in data['updates'])
    assert len({r['provenance']['problem_group'] for r in data['train'] if r['kind']=='program'})==192


def test_positional_diagnostic_does_not_repair_wrong_algorithm(tmp_path):
    from summarize_broad_healing import positional_audit
    spec={'id':'code-example','family':'code_generation','check':{'kind':'program','tests':[[{'values':[2,3]},5],[{'values':[]},0]]}}
    path=tmp_path/'answers.json'
    path.write_text(json.dumps({'cases':[{'id':'code-example','answer':'def main_solution(a):\n    return sum(a)'}]}))
    assert positional_audit(path,[spec])['code-example']['correct']
    path.write_text(json.dumps({'cases':[{'id':'code-example','answer':'def main_solution(a):\n    return len(a)'}]}))
    assert not positional_audit(path,[spec])['code-example']['correct']
