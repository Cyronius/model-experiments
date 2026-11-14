import hashlib
import json
import tarfile

import pytest

import run_correctness_pod as pod


def test_no_paid_api_during_prepare(monkeypatch):
    monkeypatch.setattr(pod.lifecycle,'api',lambda *a,**kw:pytest.fail('Paid API called in dry run'))
    monkeypatch.setattr(pod.lifecycle,'fresh_quote',lambda:pytest.fail('Unexpected network call in dry run'))
    bundle=pod.prepare_bundle()
    with tarfile.open(bundle) as archive:
        names=archive.getnames()
        assert 'scale_worker.py' in names and 'recipe/basis.pt' in names
        assert not any('credential' in n or 'ssh-key' in n for n in names)
        manifest=json.load(archive.extractfile('bundle-manifest.json'))
        for name,digest in manifest['sha256'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
        worker=archive.extractfile('scale_worker.py').read()
        assert worker==(pod.ROOT/'correctness_worker.py').read_bytes()
    proposal=pod.proposal()
    assert proposal['maximum_planned_usage_usd']<proposal['budget_cap_usd']-1.5
    assert proposal['limits']['max_updates']==32
    assert proposal['authorization'].startswith('pending')


def test_wrong_budget_rejected_before_cloud_action(monkeypatch):
    # Restore module globals after the wrapper configures its shared controller.
    for name in ['JOB_ROOT','RESULTS','JOB_PREFIX','proposal']:
        monkeypatch.setattr(pod.lifecycle,name,getattr(pod.lifecycle,name))
    monkeypatch.setattr(pod.lifecycle,'fresh_quote',lambda:pytest.fail('API touched before cap check'))
    with pytest.raises(ValueError):pod.execute(None,None)
    with pytest.raises(ValueError):pod.execute(None,20)
