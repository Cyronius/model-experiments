import json
from pathlib import Path

import pytest

import run_retention_pod as pod


def ledger():
    return dict(status='authorized_not_launched', total_cap_usd=20, new_job_spend_usd=0,
                delegate_approval='claude', launch_owner='Codex sole owner')


def test_shared_budget_refuses_second_job_or_insufficient_remaining_cap():
    pod.validate_budget(ledger())
    for changed in ({'status': 'running'}, {'new_job_spend_usd': 13}, {'launch_owner': 'claude'}, {'total_cap_usd': 10}):
        with pytest.raises(ValueError):
            pod.validate_budget(ledger() | changed)


def fake_launch_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(pod, 'REPO', tmp_path)
    monkeypatch.setattr(pod, 'JOBS', tmp_path / 'jobs')
    monkeypatch.setattr(pod, 'INPUTS', tmp_path / 'results')
    monkeypatch.setattr(pod, 'LEDGER', tmp_path / 'budget.json')
    pod.INPUTS.mkdir()
    pod.LEDGER.write_text(json.dumps(ledger()))
    archive = tmp_path / 'bundle.tar'
    archive.write_bytes(b'test')
    monkeypatch.setattr(pod, 'quote', lambda: {'gpu': pod.GPUS[0], 'rate': 1.79})

    def process(argv, **kwargs):
        # The real guard writes readiness before the first paid request.
        p = Path(argv[-1])
        pod.write_json({'ready': True}, p.with_name('watchdog-ready.json'))
        return type('Process', (), {'poll': lambda self: None})()

    def ssh_key(argv, **kwargs):
        assert argv[0] == 'ssh-keygen'
        Path(argv[argv.index('-f') + 1] + '.pub').write_text('ssh-ed25519 test')

    monkeypatch.setattr(pod.subprocess, 'Popen', process)
    monkeypatch.setattr(pod.subprocess, 'run', ssh_key)
    return archive


def test_overpriced_allocation_is_deleted_and_never_retried(monkeypatch, tmp_path):
    archive = fake_launch_environment(monkeypatch, tmp_path)
    calls, deleted = [], []

    def api(method, path, payload):
        assert list(pod.JOBS.glob('*/watchdog-ready.json'))
        calls.append(payload)
        return {'id': 'only-new-pod', 'name': payload['name'], 'costPerHr': 3.}

    monkeypatch.setattr(pod, 'api', api)
    monkeypatch.setattr(pod, 'terminate', lambda m: deleted.append((m['pod_id'], m['name'])))
    monkeypatch.setattr(pod, 'matching_pod', lambda m: None)
    with pytest.raises(RuntimeError, match='rate'):
        pod.execute(archive)
    assert len(calls) == 1 and deleted == [('only-new-pod', calls[0]['name'])]
    assert calls[0]['gpuCount'] == 1 and calls[0]['volumeInGb'] == 0 and calls[0]['containerDiskInGb'] == 60
    assert json.loads(pod.LEDGER.read_text())['status'] == 'terminated'


def test_ambiguous_create_has_one_request_and_keeps_watchdog_reservation(monkeypatch, tmp_path):
    archive = fake_launch_environment(monkeypatch, tmp_path)
    calls, deleted = [], []

    def api(method, path, payload):
        calls.append(payload)
        raise ConnectionResetError('ambiguous create')

    monkeypatch.setattr(pod, 'api', api)
    monkeypatch.setattr(pod, 'terminate', lambda m: deleted.append(m['name']))
    with pytest.raises(ConnectionResetError):
        pod.execute(archive)
    assert len(calls) == 1 and deleted == [calls[0]['name']]
    state = json.loads(pod.LEDGER.read_text())
    assert state['status'] == 'termination_pending_watchdog' and state['reserved_job_max_usd'] == pod.JOB_CAP
    manifest = json.loads(next(pod.JOBS.glob('*/pod.json')).read_text())
    assert manifest['terminate_requested'] and manifest['deadline'] - manifest['created_intent'] == 4 * 3600


def test_watchdog_waits_out_ambiguous_create_and_confirms_absence(monkeypatch, tmp_path):
    path = tmp_path / 'pod.json'
    pod.write_json(dict(name='new-unique-job', created_intent=0, deadline=300, terminate_requested=True), path)
    clock = [100.]
    deleted = []
    monkeypatch.setattr(pod.time, 'time', lambda: clock[0])
    monkeypatch.setattr(pod.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(pod, 'terminate', lambda m: deleted.append(m['name']))
    monkeypatch.setattr(pod, 'matching_pod', lambda m: None)
    pod.watchdog(path)
    assert clock[0] > 300 and set(deleted) == {'new-unique-job'}
    assert (tmp_path / 'termination-confirmed.json').is_file()
