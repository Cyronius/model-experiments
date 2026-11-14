"""Prepare, launch, sync and terminate ONE pod for the separate slimming measurements.

Default only packages the bundle. --execute rents the pod: the user approved this batch with an $8
cap on 2026-10-29 (runpod_budget_20261029.json). One 48 GB GPU, at most 1.09 USD/h, deleted on
completion, failure or the 5-hour deadline by an independent watchdog.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid

from run_scale_pod import api, credential, IMAGE, matching_pod, ssh_options, terminate, write_json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
JOBS = REPO / 'target/manifold-slim'
OUTPUTS = ROOT / 'results/slim-pod-20261029'
LEDGER = ROOT / 'runpod_budget_20261029.json'
CAP = 8.0
# Cheapest first; every one has 48 GB, enough for the 9B in bf16 with 25 arms of activations.
GPUS = ('NVIDIA A40', 'NVIDIA RTX A6000', 'NVIDIA L40', 'NVIDIA RTX 6000 Ada Generation', 'NVIDIA L40S')
RATE = 1.09
DISK_GB = 80
DISK_RATE = DISK_GB * .10 / (30 * 24)
HOURS = 5
JOB_CAP = (RATE + DISK_RATE) * HOURS
CODE = ['moe_carve.py', 'slim_batch.py', 'h0_mlp_probe.py', 'slim_worker.py', 'slim_requirements.txt',
        'moe_carve_rules_v2.json', 'slim_rules.json', 'slim_prompts.json', 'test_moe_carve.py', 'test_slim_batch.py']
RUN = 'results/moe-carve-v2-20261029'
INPUTS = [f'{RUN}/corpus.json', f'{RUN}/slim/vocab64k.json', f'{RUN}/slim/trim.json']
# run_heal_pod.py reuses this launcher with its own values of the settings above and these four.
PREFIX, WORKER, REQUIREMENTS, ARCHIVE = 'mc-slim-', 'slim_worker.py', 'slim_requirements.txt', 'slim-bundle.tar'
MODEL = {'repo': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
         'path': 'models/manifold-audit/Qwen3.5-9B',
         'small_files': ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                         'vocab.json', 'merges.txt', 'model.safetensors.index.json']}


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            value.update(data)
    return value.hexdigest()


def quote():
    query = '{ gpuTypes { id securePrice lowestPrice(input: {gpuCount: 1, secureCloud: true}) { stockStatus uninterruptablePrice } } }'
    request = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps({'query': query}).encode(),
        headers={'Authorization': 'Bearer ' + credential(), 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        rows = json.load(response)['data']['gpuTypes']
    for gpu in GPUS:
        row = next((r for r in rows if r['id'] == gpu), None)
        if row is None:
            continue
        low = row.get('lowestPrice') or {}
        price = row.get('securePrice')
        if price is not None and float(price) <= RATE and low.get('stockStatus') in ('Low', 'Medium', 'High'):
            if low.get('uninterruptablePrice') is not None and float(low['uninterruptablePrice']) > RATE:
                continue
            return {'gpu': gpu, 'rate': float(price), 'stock_status': low['stockStatus'], 'checked_at': time.time()}
    raise RuntimeError(f'No 48 GB GPU in stock at <= {RATE} USD/h')


def model_manifest():
    """Pinned download list: the small files hashed here, the weights as the first carve run hashed them."""
    local = REPO / MODEL['path']
    files = {name: sha(local / name) for name in MODEL['small_files']}
    files.update(json.loads((ROOT / 'results/moe-carve-20261028/collect.json').read_text())['model_sha256'])
    return {k: v for k, v in MODEL.items() if k != 'small_files'} | {'files': files}


def bundle():
    files = {f'manifold-compression/{name}': sha(ROOT / name) for name in CODE + INPUTS}
    JOBS.mkdir(parents=True, exist_ok=True)
    manifest = JOBS / 'bundle-manifest.json'
    write_json({'files': files, 'model': model_manifest()}, manifest)
    path = JOBS / ARCHIVE
    with tarfile.open(path, 'w') as archive:
        for relative in files:
            archive.add(REPO / relative, arcname=relative, recursive=False)
        archive.add(manifest, arcname='bundle-manifest.json')
    write_json({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}, JOBS / 'bundle.json')
    return path


def watchdog(path):
    write_json({'pid': os.getpid(), 'ready_at': time.time()}, path.with_name('watchdog-ready.json'))
    while True:
        manifest = json.loads(path.read_text())
        if manifest.get('terminated'):
            return
        if time.time() >= manifest['deadline'] or manifest.get('terminate_requested'):
            try:
                terminate(manifest)
                # With an ambiguous create, keep deleting for the whole job window.
                confirm_after = manifest['created_intent'] + 120 if manifest.get('pod_id') else manifest['deadline']
                if matching_pod(manifest) is None and time.time() > confirm_after:
                    write_json({'confirmed_at': time.time()}, path.with_name('termination-confirmed.json'))
                    return
            except Exception as error:
                print('Deletion retry:', type(error).__name__, flush=True)
        time.sleep(10)


def validate_budget(ledger):
    if ledger['total_cap_usd'] != CAP or ledger['status'] != 'authorized_not_launched':
        raise ValueError('Ledger is not available for this launch')
    if ledger.get('spent_usd', 0) + JOB_CAP > ledger['total_cap_usd']:
        raise ValueError('Job could exceed the remaining budget')


def execute(archive):
    JOBS.mkdir(parents=True, exist_ok=True)
    with (JOBS / 'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(LEDGER.read_text())
        validate_budget(ledger)
        checked = quote()
        name = PREFIX + uuid.uuid4().hex[:12]
        job = JOBS / name
        job.mkdir()
        output = OUTPUTS / name
        output.mkdir(parents=True)
        key = job / 'ssh-key'
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(key), '-C', name], check=True)
        now = time.time()
        manifest = {'name': name, 'created_intent': now, 'deadline': now + HOURS * 3600, 'budget_cap_usd': CAP,
                    'job_max_estimated_usd': JOB_CAP, 'quote': checked, 'container_disk_gb': DISK_GB,
                    'volume_gb': 0, 'bundle_sha256': sha(archive)}
        path = job / 'pod.json'
        write_json(manifest, path)
        with (job / 'watchdog.log').open('a') as stream:
            guard = subprocess.Popen([sys.executable, __file__, '--watchdog', str(path)],
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(50):
            if guard.poll() is not None:
                raise RuntimeError('Watchdog failed before launch')
            if (job / 'watchdog-ready.json').exists():
                break
            time.sleep(.1)
        else:
            raise RuntimeError('No watchdog readiness before launch')
        ledger.update(status='launch_reserved', active_job=str(path.relative_to(REPO)),
                      reserved_job_max_usd=JOB_CAP, fresh_quote=checked)
        write_json(ledger, LEDGER)
        startup = '\n'.join(['set -eu', 'apt-get update -qq',
            'apt-get install -y -qq openssh-server rsync python3-venv',
            'mkdir -p /root/.ssh /run/sshd /workspace/manifold', 'chmod 700 /root/.ssh',
            'printf \'%s\\n\' "$PUBLIC_KEY" > /root/.ssh/authorized_keys',
            'chmod 600 /root/.ssh/authorized_keys', 'ssh-keygen -A',
            'exec /usr/sbin/sshd -D -e -o PasswordAuthentication=no'])
        payload = {'name': name, 'imageName': IMAGE, 'gpuTypeIds': [checked['gpu']], 'gpuCount': 1,
                   'cloudType': 'SECURE', 'computeType': 'GPU', 'interruptible': False,
                   'containerDiskInGb': DISK_GB, 'volumeInGb': 0, 'minRAMPerGPU': 48, 'minVCPUPerGPU': 8,
                   'ports': ['22/tcp'], 'supportPublicIp': True, 'dockerEntrypoint': ['/bin/bash'],
                   'dockerStartCmd': ['-c', startup], 'env': {'PUBLIC_KEY': key.with_suffix('.pub').read_text().strip()}}
        remote = None

        def sync():
            if not remote or time.time() >= manifest['deadline'] - 20:
                return False
            command, host = remote
            try:
                result = subprocess.run(['rsync', '-azL', '--partial', '--timeout=15', '--exclude=*.tmp',
                    '-e', shlex.join(command), host + ':/workspace/manifold/outputs/', str(output) + '/'],
                    timeout=min(300, max(1, manifest['deadline'] - time.time() - 20)),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False

        try:
            # Sole paid create. No retry on an ambiguous response.
            pod = api('POST', '/pods', payload)
            manifest.update(pod_id=pod['id'], allocated_rate=pod.get('costPerHr'))
            write_json(manifest, path)
            if pod.get('costPerHr') is None or not 0 < float(pod['costPerHr']) <= RATE + DISK_RATE + 1e-6:
                raise RuntimeError('Allocated rate missing or exceeds approved ceiling')
            ledger.update(status='running', pod_id=pod['id'])
            write_json(ledger, LEDGER)
            print('Created', pod['id'], checked['gpu'], 'deadline UTC epoch', manifest['deadline'], flush=True)
            while time.time() < min(now + 900, manifest['deadline'] - 180):
                pod = api('GET', '/pods/' + manifest['pod_id'])
                ip, port = pod.get('publicIp'), (pod.get('portMappings') or {}).get('22')
                if ip and port:
                    command = ['ssh', *ssh_options(job, key), '-p', str(port)]
                    host = 'root@' + ip
                    ready = subprocess.run([*command, host, 'test -d /workspace/manifold'], timeout=20,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if ready.returncode == 0:
                        remote = command, host
                        break
                time.sleep(10)
            if remote is None:
                raise TimeoutError('Pod not ready within 15 minutes')
            command, host = remote
            with archive.open('rb') as stream:
                subprocess.run([*command, host, 'tar xf - -C /workspace/manifold'], stdin=stream,
                               timeout=min(300, manifest['deadline'] - time.time() - 180), check=True)
            setup = '\n'.join(['set -eu', 'cd /workspace/manifold', 'mkdir -p outputs',
                'python -m venv --system-site-packages env',
                'env/bin/python -m pip install --no-cache-dir -r manifold-compression/' + REQUIREMENTS + ' > outputs/setup.log 2>&1',
                'MANIFOLD_JOB_DEADLINE=' + str(manifest['deadline'] - 600) +
                ' env/bin/python -u manifold-compression/' + WORKER + ' > outputs/worker.log 2>&1'])
            launch = 'cd /workspace/manifold || exit; nohup bash -c ' + shlex.quote(setup) + \
                     ' > /workspace/manifold/launch.log 2>&1 < /dev/null &'
            subprocess.run([*command, host, launch], timeout=20, check=True)
            last = None
            while time.time() < manifest['deadline'] - 300:
                sync()
                state_path = output / 'status.json'
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    if state['stage'] != last:
                        print('Worker:', state['stage'], json.dumps(state.get('outcome', {})), flush=True)
                        last = state['stage']
                    if state['stage'] in ('complete', 'failed'):
                        break
                elif time.time() > now + 1800:
                    raise TimeoutError('Worker did not start within 30 minutes')
                if guard.poll() is not None and not (job / 'termination-confirmed.json').exists():
                    raise RuntimeError('Independent watchdog exited unexpectedly')
                time.sleep(30)
            manifest['final_sync_succeeded'] = sync()
        finally:
            # Deletion precedes any extra copying, hashing, summaries or billing work.
            manifest['terminate_requested'] = True
            write_json(manifest, path)
            try:
                terminate(manifest)
                if manifest.get('pod_id') and matching_pod(manifest) is None:
                    manifest.update(terminated=True, terminated_at=time.time())
            finally:
                sums = output / 'artifact-manifest.json'
                if manifest.get('terminated') and sums.exists():
                    try:
                        manifest['artifact_checksums_verified'] = all(
                            (output / relative).is_file() and sha(output / relative) == expected
                            for relative, expected in json.loads(sums.read_text()).items())
                    except Exception as error:
                        manifest.update(artifact_checksums_verified=False, artifact_check_error=type(error).__name__)
                write_json(manifest, path)
                write_json(manifest, output / 'job-lifecycle.json')
                if manifest.get('terminated'):
                    rate = float(manifest.get('allocated_rate') or RATE + DISK_RATE)
                    estimate = (manifest['terminated_at'] - now) / 3600 * rate
                    ledger.update(status='terminated', spent_usd=ledger.get('spent_usd', 0) + estimate,
                                  billing_basis='elapsed time at the allocated rate, not the provider invoice',
                                  reserved_job_max_usd=0, terminated_at=manifest['terminated_at'])
                else:
                    ledger.update(status='termination_pending_watchdog', reserved_job_max_usd=JOB_CAP)
                write_json(ledger, LEDGER)
            print('Deletion confirmed:', bool(manifest.get('terminated')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--watchdog', type=Path)
    parser.add_argument('--quote', action='store_true')
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
    elif args.quote:
        print(json.dumps(quote()), flush=True)
    else:
        archive = bundle()
        execute(archive) if args.execute else print(str(archive))
