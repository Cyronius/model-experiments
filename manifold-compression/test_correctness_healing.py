from fractions import Fraction
import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from correctness_eval import grade, synthetic_cases
from correctness_heal import (install_lora, answer_loss, train_update, set_training,
                              save_adapter, load_adapter, merge_lora, selection_key)
from correctness_prepare import OUT, pure_reference
from correctness_summary import paired
from healing_data import encode_complete
from horizon_data import arithmetic, solve_math, run_reference, parse_code, instruction_pair
from projection_models import project_residual_width, projected_output_matrix, install_output_matrix, load_causal_model
from scale_data import load_tokenizer
from test_projection_models import tiny_model


@pytest.mark.parametrize('text,expected',[
    ('493','493'),('The answer is **493**.','493'),('29 * 17 = 493.','493'),
    ('The result is 2,500 mL.','2500'),(r'$\boxed{\frac{2}{3}}$','2/3'),
    ('First compute the product.\n493','493')])
def test_numeric_correctness_ignores_presentation(text,expected):
    result=grade(text,{'kind':'numeric','expected':expected})
    assert result['correct'] and result['parsed']
    assert result['format_pass']==(text==expected)


@pytest.mark.parametrize('text',['493 or 494',r'\boxed{493} or \boxed{494}',
                                'It might be 493 but I am unsure.','Answer: 494'])
def test_no_keyword_or_ambiguous_number_credit(text):
    assert not grade(text,{'kind':'numeric','expected':'493'})['correct']


def test_structured_correctness_and_wrong_content():
    spec={'kind':'json','expected':{'a':[1,2],'ok':True}}
    assert grade('Here it is:\n```json\n{"ok":true,"a":[1,2]}\n```',spec)=={
        'correct':True,'format_pass':False,'parsed':True}
    for value in ['{"a":[1,3],"ok":true}','{"a":[1,2],"ok":1}',
                  '{"a":[1,2],"ok":true} {"a":[1,3],"ok":true}']:
        assert not grade(value,spec)['correct']
    assert grade('The program prints 42.',{'kind':'literal','expected':42})['correct']
    assert grade('Your parcel code is AB3D5FGH.',{'kind':'identifier','expected':'AB3D5FGH'})['correct']
    assert not grade('AB3D5FGH or ZX3D5FGH',{'kind':'identifier','expected':'AB3D5FGH'})['correct']


def test_independent_math_and_restricted_grammar():
    assert arithmetic('(2 + 3) * 4 / 6')==Fraction(10,3)
    assert solve_math('Two letters picked without replacement from banana. Give prob of picking 2 a.')==('1/5','numeric')
    assert solve_math('What is the hundreds digit of 7329?')==('3','numeric')
    assert solve_math('Calculate 2^3 + 5.')==('13','numeric')
    assert solve_math('Is 17 even?')==('No','boolean')
    for value in ['__import__("os").system("true")','i+1','2**100000','True']:
        with pytest.raises(ValueError):arithmetic(value)


def code_example(answer='{"output": 7}',feedback='Correct output!'):
    return ('User: Given the following input:\n\n{"x": 3}\n\nCan you predict the output? '
            'Please reason and put your final answer in the following json format: {"output": ...}\n\n'
            'Tip: Here is a reference code snippet to help you:\n\n'
            'def main_solution(x):\n    return x + 4\n\nAssistant: '+answer+'\n\nUser: '+feedback)


def test_code_extraction_requires_verification_and_preserves_task():
    row=parse_code(code_example())
    assert row['inputs']=={'x':3} and row['expected']==7
    assert 'Please reason' not in row['prompt']
    assert run_reference(row['code'],row['inputs'])=={'ok':True,'result':7}
    wrong=parse_code(code_example('{"output": 8}'))
    assert run_reference(wrong['code'],wrong['inputs'])['result']!=wrong['expected']
    assert parse_code(code_example(feedback='Wrong output!')) is None
    assert pure_reference(row['code'])
    assert not pure_reference('import os\ndef main_solution(): return os.getcwd()')
    assert not pure_reference('def main_solution(xs): return list(set(xs))')


def test_reference_execution_has_no_environment_or_home(monkeypatch):
    monkeypatch.setenv('PRIVATE_TEST_SENTINEL','not-for-corpus-code')
    code='import os\ndef main_solution():\n return [os.path.exists("/home"), os.getenv("PRIVATE_TEST_SENTINEL")]'
    assert run_reference(code,{})=={'ok':True,'result':[False,None]}
    assert run_reference('def main_solution():\n while True: pass',{}) is None


def test_instruction_final_response_only():
    prompt='Explain why ice floats.';answer='Ice is less dense than liquid water.'
    assert instruction_pair(f'### Instruction\n{prompt}\n### Reasoning\nPrivate analysis here.\n### Response\n{answer}')==(prompt,answer)
    assert instruction_pair('Prompt and answer without reliable boundaries') is None


def test_frozen_data_integrity_and_no_function_overlap():
    data=json.loads((OUT/'data.json').read_text());meta=json.loads((OUT/'preparation.json').read_text())
    assert hashlib.sha256((OUT/'data.json').read_bytes()).hexdigest()==meta['data_sha256']
    all_indices=[i for step in data['updates'] for i in step]
    assert all_indices==list(range(len(data['train'])))
    assert len(data['updates'])==32
    assert all(r['kind'] in ('math','code') for r in data['train'])
    groups={r['problem_group'] for r in data['train']}
    assert not groups & {r['problem_group'] for r in data['dev']+data['test'] if 'problem_group' in r}
    prompts=[json.dumps(c['messages'],sort_keys=True) for c in data['dev']+data['test']]
    assert len(prompts)==len(set(prompts))==200
    assert sum(c['check'] is not None for c in data['dev'])==64
    assert sum(c['check'] is not None for c in data['test'])==128
    tokenizer=load_tokenizer(OUT.parents[2]/'target/manifold-scale/qwen27')
    for raw in data['train'][::37]:
        row=encode_complete(tokenizer,raw['messages'],data['max_length'])
        assert row['targets'][-1]==tokenizer.eos_token_id
        assert len(row['targets'])==raw['assistant_tokens']
        assert row['targets']==[row['input_ids'][i+1] for i in row['positions']]
        assert row['positions'][0]==row['answer_start']-1
    first=synthetic_cases(2510251,8)
    second=synthetic_cases(2510252,16,[json.dumps(c['messages'],sort_keys=True) for c in first])
    assert len(first)==56 and len(second)==112


def projected_tiny():
    model=tiny_model('qwen',False)
    p=torch.linalg.qr(torch.randn(32,32)).Q[:,:24].contiguous()
    gain=model.model.norm.weight.detach().clone()+1
    project_residual_width(model,p,[1.]*5)
    install_output_matrix(model,projected_output_matrix(gain,p,1.))
    return model


def test_lora_zero_identity_training_freeze_save_reload_merge(tmp_path):
    model=projected_tiny();ids=torch.tensor([[1,7,9,3,2,5,8]])
    with torch.no_grad():initial=model(ids,use_cache=False).logits
    install_lora(model,rank=4,alpha=8)
    with torch.no_grad():torch.testing.assert_close(model(ids,use_cache=False).logits,initial,atol=0,rtol=0)
    frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
    row={'input_ids':ids[0].tolist(),'positions':[3,4,5],'targets':[2,5,8]}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003)
    set_training(model)
    first=answer_loss(model,row,'cpu').item()
    for _ in range(6):train_update(model,optimizer,[row],'cpu',.003)
    assert answer_loss(model,row,'cpu').item()<first
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    for n,p in model.named_parameters():
        if n in frozen:torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)
    model.gradient_checkpointing_disable();model.eval()
    with torch.no_grad():expected=model(ids,use_cache=False).logits
    save_adapter(model,tmp_path/'adapter',{'step':6})
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.zero_()
    load_adapter(model,tmp_path/'adapter')
    with torch.no_grad():torch.testing.assert_close(model(ids,use_cache=False).logits,expected,atol=0,rtol=0)
    merge_lora(model)
    with torch.no_grad():torch.testing.assert_close(model(ids,use_cache=False).logits,expected,atol=1e-6,rtol=1e-5)
    model.save_pretrained(tmp_path/'hf')
    restored=load_causal_model(tmp_path/'hf',dtype=torch.float32).eval()
    with torch.no_grad():torch.testing.assert_close(restored(ids,use_cache=False).logits,expected,atol=1e-6,rtol=1e-5)


def test_chunked_assistant_loss_gradient_matches_full_vocab_loss():
    model=projected_tiny();install_lora(model,4,8)
    row={'input_ids':[1,7,9,3,2,5,8],'positions':[3,4,5],'targets':[2,5,8]}
    total=answer_loss(model,row,'cpu',chunk=1);total.backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    logits=model(torch.tensor([row['input_ids']]),use_cache=False).logits[0,row['positions']]
    expected=F.cross_entropy(logits,torch.tensor(row['targets']),reduction='sum')
    expected.backward()
    torch.testing.assert_close(total,expected)
    for n,p in model.named_parameters():
        if p.requires_grad:torch.testing.assert_close(p.grad,grads[n],atol=1e-6,rtol=1e-5)


def test_correctness_selection_rejects_more_loops_or_early_stop():
    base={'correct':40,'loop_flags':1,'ended':65}
    assert selection_key(base,base,0)>selection_key(base,base,4)
    assert selection_key(base|{'correct':42},base,4)>selection_key(base,base,0)
    assert selection_key(base|{'correct':50,'loop_flags':2},base,4) is None
    assert selection_key(base|{'correct':50,'ended':63},base,4) is None


def test_paired_summary_exposes_traded_correct_answers():
    def run(first,second):
        return {'summary':{},'cases':[{'id':name,'grade':{'correct':ok,'parsed':True},
                'combined_loop_flag':False} for name,ok in [('a',first),('b',second)]]}
    result=paired(run(True,False),run(False,True))
    assert result['baseline_correct']==result['candidate_correct']==1
    assert result['gained']==['b'] and result['lost']==['a']
    assert result['loss_fraction_of_baseline_correct']==1
