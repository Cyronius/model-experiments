"""One bounded, jointly approved A100 job with an independent deletion watchdog."""
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
INPUTS = ROOT / 'results/retention-objective-20261027'
JOBS = REPO / 'target/manifold-retention'
LEDGER = ROOT / 'runpod_budget_20261026.json'
GPUS = ('NVIDIA A100-SXM4-80GB', 'NVIDIA A100 80GB PCIe')
RATE = 1.79
DISK_GB = 60
DISK_RATE = DISK_GB * .10 / (30 * 24)
HOURS = 4
JOB_CAP = (RATE + DISK_RATE) * HOURS


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
        row = next(r for r in rows if r['id'] == gpu)
        low = row.get('lowestPrice') or {}
        price = row.get('securePrice')
        if price is not None and float(price) <= RATE and low.get('stockStatus') in ('Low', 'Medium', 'High'):
            if low.get('uninterruptablePrice') is not None and float(low['uninterruptablePrice']) > RATE:
                continue
            return {'gpu': gpu, 'rate': float(price), 'stock_status': low['stockStatus'], 'checked_at': time.time()}
    raise RuntimeError('No approved A10080 currently in stock at <=1.79 USD/h')


def bundle(reviewed=True):
    frozen_path = INPUTS / ('frozen.json' if reviewed else 'frozen-candidate.json')
    frozen = json.loads(frozen_path.read_text())
    if reviewed and (frozen['status'] != 'peer_reviewed' or frozen['reviewer'] != 'claude'):
        raise ValueError('Final peer review is required before packaging')
    files = {}
    for relative, expected in frozen['files'].items():
        path = REPO / relative
        if sha(path) != expected:
            raise ValueError('Frozen file changed: ' + relative)
        # Only the public pinned base weight is downloaded remotely; all other
        # reconstruction inputs and executable code are included verbatim.
        if relative.startswith('models/manifold-audit/Qwen3.5-2B/') and path.suffix == '.safetensors':
            continue
        files[relative] = expected
    files[str(frozen_path.relative_to(REPO))] = sha(frozen_path)
    JOBS.mkdir(parents=True, exist_ok=True)
    manifest = JOBS / 'bundle-manifest.json'
    write_json({'files': files, 'frozen_sha256': sha(frozen_path)}, manifest)
    path = JOBS / 'retention-bundle.tar'
    with tarfile.open(path, 'w') as archive:
        for relative in files:
            archive.add(REPO / relative, arcname=relative, recursive=False)
        archive.add(manifest, arcname='bundle-manifest.json')
    write_json({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                'frozen_sha256': sha(frozen_path)}, JOBS / 'bundle.json')
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
                # With an ambiguous POST, the name listing could lag. Keep the
                # deleter alive for the entire original job window if no id was
                # ever recorded, even when an early listing is empty.
                confirm_after = manifest['created_intent'] + 120 if manifest.get('pod_id') else manifest['deadline']
                if matching_pod(manifest) is None and time.time() > confirm_after:
                    write_json({'confirmed_at': time.time()}, path.with_name('termination-confirmed.json'))
                    return
            except Exception as error:
                print('Deletion retry:', type(error).__name__, flush=True)
        time.sleep(10)


def validate_budget(ledger):
    if ledger['total_cap_usd'] != 20 or ledger['status'] != 'authorized_not_launched':
        raise ValueError('Shared ledger is not available for the sole approved launch')
    if ledger.get('new_job_spend_usd', 0) + JOB_CAP > ledger['total_cap_usd']:
        raise ValueError('Job could exceed remaining shared budget')
    if not ledger.get('delegate_approval') or not ledger['launch_owner'].startswith('Codex'):
        raise ValueError('Missing delegated approval or sole ownership')


def execute(archive):
    JOBS.mkdir(parents=True, exist_ok=True)
    with (JOBS / 'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(LEDGER.read_text())
        validate_budget(ledger)
        checked = quote()
        name = 'mc-retention-' + uuid.uuid4().hex[:12]
        job = JOBS / name
        job.mkdir()
        output = INPUTS / name
        output.mkdir()
        key = job / 'ssh-key'
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(key), '-C', name], check=True)
        now = time.time()
        manifest = {'name': name, 'created_intent': now, 'deadline': now + HOURS * 3600,
                    'shared_budget_cap_usd': 20, 'job_max_estimated_usd': JOB_CAP, 'quote': checked,
                    'container_disk_gb': DISK_GB, 'volume_gb': 0, 'bundle_sha256': sha(archive)}
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
                   'containerDiskInGb': DISK_GB, 'volumeInGb': 0, 'minRAMPerGPU': 64, 'minVCPUPerGPU': 8,
                   'ports': ['22/tcp'], 'supportPublicIp': True, 'dockerEntrypoint': ['/bin/bash'],
                   'dockerStartCmd': ['-c', startup], 'env': {'PUBLIC_KEY': key.with_suffix('.pub').read_text().strip()}}
        remote = None

        def sync():
            if not remote or time.time() >= manifest['deadline'] - 20:
                return False
            command, host = remote
            try:
                result = subprocess.run(['rsync', '-az', '--partial', '--timeout=15', '--exclude=*.tmp',
                    '-e', shlex.join(command), host + ':/workspace/manifold/outputs/', str(output) + '/'],
                    timeout=min(45, max(1, manifest['deadline'] - time.time() - 20)),
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
            print('Created', pod['id'], 'deadline UTC epoch', manifest['deadline'], flush=True)
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
                'env/bin/python -m pip install --no-cache-dir -r manifold-compression/retention_requirements.txt > outputs/setup.log 2>&1',
                'MANIFOLD_JOB_DEADLINE=' + str(manifest['deadline'] - 120) +
                ' env/bin/python -u manifold-compression/retention_worker.py > outputs/worker.log 2>&1'])
            launch = 'cd /workspace/manifold || exit; nohup bash -c ' + shlex.quote(setup) + \
                     ' > /workspace/manifold/launch.log 2>&1 < /dev/null &'
            subprocess.run([*command, host, launch], timeout=20, check=True)
            last = None
            while time.time() < manifest['deadline'] - 60:
                sync()
                state_path = output / 'status.json'
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    if state['stage'] != last:
                        print('Worker:', state['stage'], flush=True)
                        last = state['stage']
                    if state['stage'] in ('complete', 'failed'):
                        break
                elif time.time() > now + 1200:
                    raise TimeoutError('Worker did not start within 20 minutes')
                if guard.poll() is not None and not (job / 'termination-confirmed.json').exists():
                    raise RuntimeError('Independent watchdog exited unexpectedly')
                time.sleep(20)
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
                        manifest['artifact_checksums_verified'] = all((output / relative).is_file() and sha(output / relative) == expected
                            for relative, expected in json.loads(sums.read_text()).items())
                    except Exception as error:
                        manifest.update(artifact_checksums_verified=False, artifact_check_error=type(error).__name__)
                write_json(manifest, path)
                write_json(manifest, output / 'job-lifecycle.json')
                if manifest.get('terminated'):
                    estimate = (manifest['terminated_at'] - now) / 3600 * (RATE + DISK_RATE)
                    ledger.update(status='terminated', estimated_new_job_spend_usd=estimate,
                                  new_job_spend_usd=estimate, billing_basis='conservative elapsed-time estimate, not provider invoice',
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
    parser.add_argument('--prepare-draft', action='store_true', help='Package unsigned candidate for local import validation only')
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
    elif args.quote:
        print(json.dumps(quote()), flush=True)
    elif args.prepare_draft:
        if args.execute:
            parser.error('Unsigned draft cannot execute')
        print(str(bundle(reviewed=False)))
    else:
        archive = bundle()
        execute(archive) if args.execute else print(str(archive))
