import json
import torch
from torch import nn
from capacity_projection import OUT, initialize
from correctness_heal import LowRankLinear

def toy():
    m=nn.Module();m.model=nn.Module();m.model.layers=nn.ModuleList([nn.Linear(7,5),nn.Linear(5,3)])
    return m

def test_nested_capacity_preserves_start_and_trainability():
    torch.manual_seed(4);a=toy();torch.manual_seed(4);b=toy()
    x=torch.randn(2,7);expected=a.model.layers[0](x).detach()
    i=initialize(a,16);j=initialize(b,64)
    assert j['trainable_parameters']==4*i['trainable_parameters']
    for u,v in zip(a.model.layers,b.model.layers):
        assert isinstance(u,LowRankLinear)
        torch.testing.assert_close(u.a,v.a[:16],atol=0,rtol=0)
        assert u.alpha/u.rank==v.alpha/v.rank==2
        assert not u.base.weight.requires_grad
    torch.testing.assert_close(a.model.layers[0](x),expected,atol=0,rtol=0)
    torch.testing.assert_close(b.model.layers[0](x),expected,atol=0,rtol=0)
    a.model.layers[0](x).sum().backward()
    assert a.model.layers[0].b.grad.abs().sum()>0
    assert a.model.layers[0].base.weight.grad is None

def test_fixed_budget_and_disjoint_math():
    d=json.loads((OUT/'data.json').read_text())
    assert len(d['updates'])==8
    assert sum(len(d['train'][i]['targets']) for b in d['updates'] for i in b)==32975
    train={(r['provenance']['family'],tuple(r['provenance']['values'])) for r in d['train'] if r['kind']=='math'}
    held=[(r['oracle']['family'],tuple(r['oracle']['values'])) for split in ['dev','test'] for r in d[split] if 'oracle'in r]
    assert len(held)==40 and len(set(held))==40 and not train&set(held)
    assert len(d['test'])==47 and len(d['dev'])==12
    assert all(r['targets'][-1]==r['eos_token_id'] for r in d['train'])
