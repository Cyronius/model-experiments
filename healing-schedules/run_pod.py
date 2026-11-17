"""Rent ONE H100 pod that runs pod_worker.py: iterative pruning against one-shot pruning
(plan-iterative-prune.md). Results come home to target/healing-schedules/pod-outputs/<job>/.

Default only packages the bundle and prints its path; --quote asks the price. --execute rents the pod.
On 2026-11-16 the user said 'Use h100s. Go' to the $20 cap I proposed (runpod_budget_iterprune_20261116.json).
One H100 (secure cloud, at most RATE USD/h), deleted on completion, failure or the deadline by this launcher
and by an independent watchdog; the worker also deletes its own pod 15 minutes after the deadline if both
have died. The deadline is set so the whole job stays within the cap at the allocated rate.

The pod lifecycle is manifold-compression/run_slim_pod.py's, with this job's values. Nothing the provider
sees names this project: the pod is called hs-<random>, and the client has a plain name.

    setsid nohup dibs run runpod-healing network --label "healing-schedules: iterative pruning on a rented H100" --eta 8h -- \\
        target/manifold-rocm/env/bin/python healing-schedules/run_pod.py --ledger runpod_budget_iterprune_20261116.json --execute \\
        >> target/healing-schedules/pod/launch.log 2>&1 < /dev/null &
    stop early (deletes the pod; results copied so far stay):
        pkill -f '^[^ ]*python healing-schedules/run_pod.py .*--execute'
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'manifold-compression'))
import run_scale_pod as base
import run_slim_pod as pod

CAP = 20.0
LIGHT = 'healing-schedules/artifacts/damaged-light-h1792-n1473244736'
TEACHER = 'models/manifold-audit/Qwen3.5-2B'
# The teacher's weights are downloaded on the pod at the pinned revision and checked against download_manifest.json.
TEACHER_SMALL = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                 'model.safetensors.index.json', 'download_manifest.json']
EXTRA = ['protocol.json', 'pod_requirements.txt', 'metadata/Qwen3.5-2B.json',
         'baselines/mix-2048/canary-cpu.json', 'baselines/mix-2048/target-light.json']

pod.JOBS = REPO / 'target/healing-schedules/pod'
pod.OUTPUTS = REPO / 'target/healing-schedules/pod-outputs'
pod.CAP = CAP
# Cheapest first.
pod.GPUS = ('NVIDIA H100 PCIe', 'NVIDIA H100 NVL', 'NVIDIA H100 80GB HBM3')
pod.RATE = 3.39
pod.DISK_GB = 120
pod.DISK_RATE = pod.DISK_GB * .10 / (30 * 24)
pod.UPLOAD_SECONDS = 2400       # the light artifact is 2.8 GB
pod.PREFIX, pod.ARCHIVE = 'hs-', 'hs-bundle.tar'
# The launcher runs these two as manifold-compression/<name> on the pod; the bundle has that directory, empty.
pod.WORKER, pod.REQUIREMENTS = '../healing-schedules/pod_worker.py', '../healing-schedules/pod_requirements.txt'
# RunPod's own image: it starts sshd with PUBLIC_KEY (startSsh), which GraphQL needs because it has no entrypoint
# override for run_slim_pod's startup script. pod_requirements.txt replaces its torch.
pod.IMAGE = 'runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04'
# execute() starts its watchdog as `python <its own file> --watchdog <pod.json>`: make that this file, so the
# watchdog also talks to the provider through api() below.
pod.__file__ = str(Path(__file__).resolve())

GRAPHQL = 'https://api.runpod.io/graphql'
FIELDS = 'id name desiredStatus costPerHr runtime { ports { ip isIpPublic privatePort publicPort type } }'
# After sshd answers: the directory execute() waits for, rsync and venv, and no image torchvision/torchaudio.
PREPARE = ('mkdir -p /workspace/manifold && (command -v rsync >/dev/null || (apt-get update -qq && '
           'apt-get install -y -qq rsync python3-venv)) >/dev/null 2>&1 && '
           '(python -m pip uninstall -y -q torchvision torchaudio >/dev/null 2>&1 || true)')


def gql(query, variables=None):
    request = urllib.request.Request(GRAPHQL, data=json.dumps({'query': query, 'variables': variables or {}}).encode(),
        headers={'Authorization': 'Bearer ' + base.credential(), 'Content-Type': 'application/json',
                 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        out = json.load(response)
    if out.get('errors'):
        raise RuntimeError('graphql: ' + json.dumps(out['errors'])[:500])
    return out['data']


READY = {}             # pod id -> prepared ssh address
PREPARE_ON_GET = True  # the watchdog only deletes: it never prepares


def prepared(found):
    """Public ssh address once sshd answers and PREPARE has run there, else None."""
    if found['id'] in READY or not PREPARE_ON_GET:
        return READY.get(found['id'])
    port = next((p for p in ((found.get('runtime') or {}).get('ports') or [])
                 if p['privatePort'] == 22 and p['isIpPublic']), None)
    if port is None:
        return None
    job = pod.JOBS / found['name']
    command = ['ssh', *base.ssh_options(job, job / 'ssh-key'), '-p', str(port['publicPort']), 'root@' + port['ip'], PREPARE]
    try:
        ok = subprocess.run(command, timeout=600, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    if ok:
        READY[found['id']] = (port['ip'], port['publicPort'])
    return READY.get(found['id'])


def api(method, path, data=None):
    """The four REST v1 calls run_slim_pod / run_scale_pod make, over GraphQL: RunPod retired REST v1
    (rest.runpod.io) on 2026-11-15 and its name no longer resolves."""
    if method == 'POST' and path == '/pods':
        created = gql('mutation($in: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $in) '
                      '{ id name costPerHr } }', {'in': {
                          'cloudType': data['cloudType'], 'gpuCount': 1, 'gpuTypeId': data['gpuTypeIds'][0],
                          'name': data['name'], 'imageName': data['imageName'],
                          'containerDiskInGb': data['containerDiskInGb'], 'volumeInGb': 0,
                          'minVcpuCount': data['minVCPUPerGPU'], 'minMemoryInGb': data['minRAMPerGPU'],
                          'ports': '22/tcp', 'supportPublicIp': True, 'startSsh': True,
                          'env': [{'key': k, 'value': v} for k, v in data['env'].items()]}})['podFindAndDeployOnDemand']
        if created is None:
            raise RuntimeError('No pod created')
        return created
    if method == 'GET' and path == '/pods':
        return gql('query { myself { pods { id name } } }')['myself']['pods']
    if path.startswith('/pods/'):
        pod_id = path[len('/pods/'):]
        found = gql('query($id: String!) { pod(input: {podId: $id}) { %s } }' % FIELDS, {'id': pod_id})['pod']
        if found is None:
            raise urllib.error.HTTPError(GRAPHQL, 404, 'no such pod', None, None)
        if method == 'DELETE':
            gql('mutation($id: String!) { podTerminate(input: {podId: $id}) }', {'id': pod_id})
            return None
        address = prepared(found)
        return dict(found, publicIp=address[0] if address else None,
                    portMappings={'22': address[1]} if address else {})
    raise ValueError(f'Unsupported call {method} {path}')


base.api = pod.api = api      # base: matching_pod and terminate; pod: execute


def nobody(info):
    """Tar entries without this machine's user name."""
    info.uid = info.gid = 0
    info.uname = info.gname = 'root'
    return info


def bundle():
    names = [p.relative_to(REPO) for p in sorted(HERE.glob('*.py'))]
    names += [Path('healing-schedules') / x for x in EXTRA]
    names += [p.relative_to(REPO) for p in sorted((HERE / 'data/mix-2048').iterdir()) if p.is_file()]
    names += [p.relative_to(REPO) for p in sorted((REPO / LIGHT).iterdir()) if p.is_file()]
    names += [Path(TEACHER) / x for x in TEACHER_SMALL]
    files = {str(n): pod.sha(REPO / n) for n in names}
    pod.JOBS.mkdir(parents=True, exist_ok=True)
    manifest = pod.JOBS / 'bundle-manifest.json'
    pod.write_json({'files': files, 'budget': {'cap_usd': CAP, 'usd_per_hour': pod.RATE + pod.DISK_RATE,
                                               'hours': pod.HOURS}}, manifest)
    path = pod.JOBS / pod.ARCHIVE
    with tarfile.open(path, 'w') as archive:
        for relative in files:
            archive.add(REPO / relative, arcname=relative, recursive=False, filter=nobody)
        archive.add(manifest, arcname='bundle-manifest.json', filter=nobody)
        holder = tarfile.TarInfo('manifold-compression')
        holder.type, holder.mode, holder.uname, holder.gname = tarfile.DIRTYPE, 0o755, 'root', 'root'
        archive.addfile(holder)
    pod.write_json({'path': str(path), 'sha256': pod.sha(path), 'bytes': path.stat().st_size}, pod.JOBS / 'bundle.json')
    return path


def stop(signum, frame):
    raise KeyboardInterrupt      # so that execute()'s cleanup runs: it deletes the pod and closes the ledger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ledger', help='the budget file, in healing-schedules/')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--quote', action='store_true')
    parser.add_argument('--watchdog', type=Path)
    args = parser.parse_args()
    if args.watchdog:
        PREPARE_ON_GET = False
        pod.watchdog(args.watchdog)
    elif args.quote:
        print(json.dumps(pod.quote()), flush=True)
    else:
        pod.LEDGER = HERE / args.ledger
        hourly = pod.RATE + pod.DISK_RATE
        # As long as the money left allows, at most 6 hours.
        pod.HOURS = min(6, int((CAP - json.loads(pod.LEDGER.read_text()).get('spent_usd', 0)) / hourly * 10) / 10)
        pod.JOB_CAP = hourly * pod.HOURS
        archive = bundle()
        if args.execute:
            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            pod.execute(archive)
        else:
            print(str(archive), f'{pod.HOURS} h at most, {pod.JOB_CAP:.2f} USD at most')
