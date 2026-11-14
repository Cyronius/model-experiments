"""One pass over a pod whose launcher (run_slim_pod.execute) and watchdog died: sync its outputs, and once the
worker is done or the deadline is near, delete the pod and settle the ledger as the launcher would have.

Every step can be repeated, so a timer outside the launcher's process tree runs it until the pod is gone:
  systemd-run --user --unit mc-opd-tend --on-active=30s --on-unit-active=5min --working-directory=$PWD \\
      ../target/manifold-audit/env/bin/python tend_pod.py run_opd_pod --timer mc-opd-tend
While the launcher's own process runs, a pass does nothing: the launcher syncs and deletes the pod
itself, and the timer only stands by in case it stops. It never creates a pod, and deletes only the
ledger's active job, by its id and checked name.
"""
import argparse
import importlib
import json
import shlex
import subprocess
import time
import traceback
from pathlib import Path


def sync(pod, manifest, job, output):
    if time.time() >= manifest['deadline'] - 20:
        return False
    info = pod.api('GET', '/pods/' + manifest['pod_id']) or {}
    ip, port = info.get('publicIp'), (info.get('portMappings') or {}).get('22')
    if not ip or not port:
        return False
    command = ['ssh', *pod.ssh_options(job, job / 'ssh-key'), '-p', str(port)]
    try:
        result = subprocess.run(['rsync', '-azL', '--partial', '--timeout=15', '--exclude=*.tmp',
            '-e', shlex.join(command), f'root@{ip}:/workspace/manifold/outputs/', str(output) + '/'],
            timeout=min(300, max(1, manifest['deadline'] - time.time() - 20)),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def settle(pod, manifest, path, output, ledger):
    """The launcher's records after a confirmed deletion: checksums, lifecycle, and the ledger, charged once."""
    sums = output / 'artifact-manifest.json'
    if sums.exists():
        try:
            manifest['artifact_checksums_verified'] = all(
                (output / relative).is_file() and pod.sha(output / relative) == expected
                for relative, expected in json.loads(sums.read_text()).items())
        except Exception as error:
            manifest.update(artifact_checksums_verified=False, artifact_check_error=type(error).__name__)
    pod.write_json(manifest, path)
    pod.write_json(manifest, output / 'job-lifecycle.json')
    if ledger['status'] != 'terminated':
        rate = float(manifest.get('allocated_rate') or pod.RATE + pod.DISK_RATE)
        estimate = (manifest['terminated_at'] - manifest['created_intent']) / 3600 * rate
        ledger.update(status='terminated', spent_usd=ledger.get('spent_usd', 0) + estimate,
                      billing_basis='elapsed time at the allocated rate, not the provider invoice',
                      reserved_job_max_usd=0, terminated_at=manifest['terminated_at'],
                      tended_by='tend_pod.py after the launcher and its watchdog stopped')
        pod.write_json(ledger, pod.LEDGER)


def launcher_running(launcher):
    """Whether `<launcher>.py --execute` is still running."""
    for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = cmdline.read_bytes().split(b'\0')
        except OSError:
            continue
        if b'--execute' in args and any(a.rsplit(b'/', 1)[-1] == f'{launcher}.py'.encode() for a in args):
            return True
    return False


def tend(pod, launcher=None):
    """Returns True once the pod is deleted and the ledger settled."""
    ledger = json.loads(pod.LEDGER.read_text())
    if launcher and launcher_running(launcher):
        print(time.strftime('%H:%M:%S'), 'launcher running', flush=True)
        return False
    if 'active_job' not in ledger:
        print(time.strftime('%H:%M:%S'), 'no job launched', flush=True)
        return False
    path = pod.REPO / ledger['active_job']
    job, manifest = path.parent, json.loads(path.read_text())
    output = pod.OUTPUTS / manifest['name']
    if not manifest.get('terminated'):
        done = manifest.get('terminate_requested') or time.time() >= manifest['deadline'] - 300
        synced = sync(pod, manifest, job, output)
        state = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
        print(time.strftime('%H:%M:%S'), 'synced' if synced else 'sync failed', state.get('stage'), flush=True)
        if state.get('stage') in ('complete', 'failed'):
            # The pass that saw the end may have copied some files before their last write.
            synced = sync(pod, manifest, job, output)
            done = True
        if not done:
            return False
        manifest.setdefault('final_sync_succeeded', synced)
        manifest['terminate_requested'] = True
        pod.write_json(manifest, path)
        pod.terminate(manifest)
        if pod.matching_pod(manifest) is not None:
            print('Deletion not yet confirmed', flush=True)
            return False
        manifest.update(terminated=True, terminated_at=time.time())
        pod.write_json(manifest, path)
    settle(pod, manifest, path, output, ledger)
    print('Deletion confirmed', manifest['pod_id'], flush=True)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('launcher', help='the module that configured run_slim_pod, e.g. run_opd_pod')
    parser.add_argument('--timer', help='a systemd user timer to stop once the pod is gone')
    args = parser.parse_args()
    try:
        if tend(importlib.import_module(args.launcher).pod, args.launcher) and args.timer:
            subprocess.run(['systemctl', '--user', 'stop', args.timer + '.timer'])
    except Exception:
        # Logged, and retried at the next pass.
        traceback.print_exc()
