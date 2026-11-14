import json
from pathlib import Path

import pytest
import torch

from correctness_heal import install_lora,set_training,train_update
from resume_correctness import save_training_state,restore_training_state
from test_correctness_healing import projected_tiny
import extend_correctness_pod as extension
import restart_correctness_pod as restart


def optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003,
        betas=(.9,.95),weight_decay=0.,foreach=False)


def test_resume_preserves_adam_update_exactly(tmp_path):
    row={'input_ids':[1,7,9,3,2,5,8],'positions':[3,4,5],'targets':[2,5,8]}
    settings={'rank':4,'alpha':8,'lr':.003};meta={'data_sha256':'frozen-test'}
    model=projected_tiny();install_lora(model,4,8);set_training(model);opt=optimizer(model)
    for _ in range(2):train_update(model,opt,[row],'cpu',.003)
    save_training_state(model,opt,tmp_path,meta,2,settings)
    train_update(model,opt,[row],'cpu',.003)
    expected={n:p.detach().clone() for n,p in model.named_parameters()}
    restored=projected_tiny();install_lora(restored,4,8);set_training(restored);new_opt=optimizer(restored)
    assert restore_training_state(restored,new_opt,tmp_path,'frozen-test',settings)==2
    train_update(restored,new_opt,[row],'cpu',.003)
    for name,param in restored.named_parameters():torch.testing.assert_close(param,expected[name],atol=0,rtol=0)
    with pytest.raises(ValueError,match='mismatch'):
        restore_training_state(restored,new_opt,tmp_path,'wrong-data',settings)


def test_adoption_rejects_wrong_budget_identity_and_exited_job():
    with pytest.raises(ValueError,match='approval'):extension.validate_adoption({},None,0)
    with pytest.raises(ValueError,match='identity'):
        extension.validate_adoption({'name':'other','pod_id':'other'},15,0)
    with pytest.raises(ValueError,match='cannot be adopted'):
        extension.validate_adoption({'name':extension.JOB.name,'pod_id':'do2mevn0r9ac89','terminated':True},15,0)


def test_extension_dry_run_makes_no_cloud_change(monkeypatch):
    monkeypatch.setattr(extension.pod,'api',lambda *a,**k:pytest.fail('Cloud API called by preparation'))
    bundle=extension.prepare();assert bundle.exists()
    p=extension.proposal()
    assert p['total_budget_cap_usd']==15
    assert p['maximum_planned_total_usd']<15-1.5
    assert p['max_total_minutes_from_original_create']==150


def test_replacement_rejects_unapproved_budget_before_api(monkeypatch):
    monkeypatch.setattr(restart.lifecycle,'api',lambda *a,**k:pytest.fail('Cloud API before approval guard'))
    for cap in [None,10,20]:
        with pytest.raises(ValueError,match='explicit approval'):restart.execute(None,cap)


def test_replacement_inputs_and_combined_budget(monkeypatch):
    monkeypatch.setattr(restart.lifecycle,'api',lambda *a,**k:pytest.fail('Cloud API during offline preparation'))
    paths=restart.bundle_paths()
    assert paths['prior/optimizer-last.pt'].is_file()
    assert paths['prior/adapter-last/adapter.safetensors'].is_file()
    assert paths['prior/adapter-best/adapter.safetensors'].is_file()
    assert paths['data.json']==restart.PRIOR/'reproduction/data.json'
    assert paths['correctness_worker.py']==restart.PRIOR/'reproduction/scale_worker.py'
    assert paths['scale_worker.py']==restart.ROOT/'correctness_restart_worker.py'
    assert not any('credential' in n or 'ssh-key' in n for n in paths)
    p=restart.proposal()
    assert p['maximum_minutes_from_create']==75
    assert p['budget_cap_usd']+p['prior_run_cost_reserve_usd']==p['total_budget_cap_usd']==15
    assert p['maximum_planned_usage_usd']<p['budget_cap_usd']-.3
