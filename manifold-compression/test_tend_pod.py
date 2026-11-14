import json
import subprocess
import sys
import time
import types

import pytest

import tend_pod
from run_scale_pod import write_json
from run_slim_pod import sha


@pytest.fixture
def job(tmp_path):
    path = tmp_path / 'target/job/pod.json'
    output = tmp_path / 'outputs/job'
    now = time.time()
    write_json({'name': 'job', 'pod_id': 'p1', 'created_intent': now - 3600, 'deadline': now + 3600,
                'allocated_rate': 0.5}, path)
    write_json({'status': 'running', 'spent_usd': 1.0, 'active_job': 'target/job/pod.json'}, tmp_path / 'ledger.json')
    write_json({'stage': 'train'}, output / 'status.json')
    calls = []
    alive = {'p1'}

    def terminate(manifest):
        calls.append(manifest['pod_id'])
        alive.discard(manifest['pod_id'])

    pod = types.SimpleNamespace(LEDGER=tmp_path / 'ledger.json', REPO=tmp_path, OUTPUTS=tmp_path / 'outputs',
        RATE=0.6, DISK_RATE=0.01, write_json=write_json, sha=sha, ssh_options=lambda job, key: [],
        api=lambda method, route: {}, terminate=terminate,
        matching_pod=lambda manifest: {'id': 'p1'} if 'p1' in alive else None)
    return pod, path, output, calls


def test_a_running_worker_keeps_its_pod(job):
    pod, path, _, calls = job
    assert not tend_pod.tend(pod) and calls == []
    assert not json.loads(path.read_text()).get('terminate_requested')


def test_a_finished_worker_loses_its_pod_and_the_ledger_is_charged_once(job):
    pod, path, output, calls = job
    (output / 'opd').mkdir()
    (output / 'opd/a.json').write_text('{}')
    write_json({'opd/a.json': sha(output / 'opd/a.json')}, output / 'artifact-manifest.json')
    write_json({'stage': 'complete'}, output / 'status.json')
    assert tend_pod.tend(pod) and calls == ['p1']
    manifest = json.loads(path.read_text())
    assert manifest['terminated'] and manifest['artifact_checksums_verified']
    assert json.loads((output / 'job-lifecycle.json').read_text()) == manifest
    ledger = json.loads(pod.LEDGER.read_text())
    assert ledger['status'] == 'terminated' and ledger['reserved_job_max_usd'] == 0
    assert ledger['spent_usd'] == pytest.approx(1.0 + 0.5, abs=1e-3)
    assert tend_pod.tend(pod) and calls == ['p1']
    assert json.loads(pod.LEDGER.read_text())['spent_usd'] == ledger['spent_usd']


def test_the_deadline_deletes_a_running_worker(job):
    pod, path, _, calls = job
    manifest = json.loads(path.read_text())
    write_json(manifest | {'deadline': time.time() + 100}, path)
    assert tend_pod.tend(pod) and calls == ['p1']


def test_an_unconfirmed_deletion_is_retried(job):
    pod, path, output, calls = job
    write_json({'stage': 'failed'}, output / 'status.json')
    pod.terminate = lambda manifest: calls.append(manifest['pod_id'])
    assert not tend_pod.tend(pod) and calls == ['p1']
    assert json.loads(path.read_text())['terminate_requested']
    assert json.loads(pod.LEDGER.read_text())['status'] == 'running'


def test_a_running_launcher_keeps_the_pod_even_when_the_worker_is_done(job, monkeypatch):
    pod, path, output, calls = job
    write_json({'stage': 'complete'}, output / 'status.json')
    monkeypatch.setattr(tend_pod, 'launcher_running', lambda launcher: launcher == 'run_x_pod')
    assert not tend_pod.tend(pod, 'run_x_pod') and calls == []
    assert tend_pod.tend(pod, 'run_y_pod') and calls == ['p1']


def test_a_ledger_with_no_job_yet_is_left_alone(job):
    pod, _, _, calls = job
    write_json({'status': 'authorized_not_launched', 'spent_usd': 0}, pod.LEDGER)
    assert not tend_pod.tend(pod) and calls == []


def test_the_launcher_is_found_by_its_command_line(tmp_path):
    script = tmp_path / 'run_fake_pod.py'
    script.write_text('import time\ntime.sleep(30)\n')
    process = subprocess.Popen([sys.executable, str(script), '--execute'])
    try:
        time.sleep(0.2)
        assert tend_pod.launcher_running('run_fake_pod') and not tend_pod.launcher_running('fake_pod')
    finally:
        process.kill()
        process.wait()
    assert not tend_pod.launcher_running('run_fake_pod')
