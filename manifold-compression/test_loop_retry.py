import copy
import json

import pytest

from retry_healing_loops import compare
import run_loop_retry_pod as launch


def test_request_preserves_original_batches_and_only_changes_limit():
    request=launch.prepare_request()
    data=json.loads((launch.HEALING/'data.json').read_text())
    legacy=json.loads((launch.PRIOR/'reproduction/legacy-data.json').read_text())['generation_eval']
    splits={'dev':data['dev'],'test':data['test'],'legacy72':legacy}
    assert len(request['target_ids'])==8
    assert set(request['target_ids'])<=set(request['selected32_baseline'])
    seen=[]
    for b in request['batches']:
        original=[c for c in splits[b['split']] if c['max_new_tokens']==b['original_limit']]
        index=b['original_batch_index']*4
        expected=copy.deepcopy(original[index:index+4])
        for c in expected:c['max_new_tokens']=4096;c.setdefault('family','legacy_long_form')
        assert b['cases']==expected
        seen.extend(c['id'] for c in b['cases'])
    assert len(seen)==len(set(seen))==len(request['cases'])


def test_longer_completion_is_separate_from_repetition_and_prefix_match():
    old={'id':'example','answer':'repeat repeat','generated_tokens':1024,'ended':False,'loop_flag':True}
    new=old|{'answer':'repeat repeat then an ending','generated_tokens':1500,'ended':True,'combined_loop_flag':True}
    result=compare([new],{'example':old})[0]
    assert result['finished_after_old_length'] and result['previous_answer_is_exact_prefix']
    assert result['new_loop_flag']  # Finishing does not establish that repetition is fixed.
    other=new|{'answer':'different prefix'}
    assert not compare([other],{'example':old})[0]['previous_answer_is_exact_prefix']


def test_budget_rejected_before_api(monkeypatch):
    monkeypatch.setattr(launch.lifecycle,'api',lambda *a,**k:pytest.fail('Cloud API before budget guard'))
    for amount in [None,3,15]:
        with pytest.raises(ValueError,match='spending approval'):launch.execute(None,amount)
    p=launch.proposal(launch.prepare_request())
    assert p['maximum_planned_usage_usd']<p['budget_cap_usd']-.3
    assert p['maximum_minutes_from_create']==35 and not p['training_performed']
