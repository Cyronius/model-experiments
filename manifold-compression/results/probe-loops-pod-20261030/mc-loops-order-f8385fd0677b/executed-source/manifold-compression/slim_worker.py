"""Disposable CUDA worker for the slimming batch. Has no RunPod credential or pod lifecycle authority.

Checks the bundle, downloads the pinned Qwen3.5-9B, then runs every stage in order. A stage whose
inputs failed is skipped; any other failure is recorded and the batch goes on. Results, logs,
status.json and an artifact manifest are kept in outputs/.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = REPO / 'outputs'
RUN = 'results/moe-carve-v2-20261029'
CARVE = ['moe_carve.py', '--rules', 'moe_carve_rules_v2.json']
SLIM = ['slim_batch.py']
# (stage, command, stages whose outputs it reads)
STAGES = [
    ('collect', CARVE[:1] + ['collect'] + CARVE[1:], []),
    ('build', CARVE[:1] + ['build'] + CARVE[1:], ['collect']),
    ('evaluate', CARVE[:1] + ['evaluate'] + CARVE[1:], ['build']),
    ('report', CARVE[:1] + ['report'] + CARVE[1:], ['evaluate']),
    ('route', CARVE[:1] + ['route'] + CARVE[1:], ['build']),
    ('vocab', SLIM + ['vocab'], []),
    ('cache', SLIM + ['cache'], []),
    ('state', SLIM + ['state'], []),
    ('deltanet', SLIM + ['deltanet'], []),
    ('combined', SLIM + ['combined'], ['route', 'cache', 'deltanet']),
    ('generate', SLIM + ['generate'], []),
]


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def status(stage, **info):
    save({'stage': stage, 'updated_at': time.time(), **info}, RESULTS / 'status.json')
    print(stage, info, flush=True)


def artifacts():
    save({str(p.relative_to(RESULTS)): sha(p) for p in RESULTS.rglob('*')
          if p.is_file() and p.suffix not in ('.log', '.tmp') and
          p.name not in ('status.json', 'artifact-manifest.json')}, RESULTS / 'artifact-manifest.json')


def run_stages(deadline, cwd, python, extra_carve=(), extra_slim=(), logs=None):
    """Run STAGES in order until deadline; returns {stage: 'done' | 'failed' | 'skipped' | 'no time'}."""
    logs = logs or RESULTS
    outcome = {}
    for stage, command, needs in STAGES:
        if any(outcome.get(n) != 'done' for n in needs):
            outcome[stage] = 'skipped'
            continue
        remaining = deadline - time.time()
        if remaining < 300:
            outcome[stage] = 'no time'
            continue
        status(stage, outcome=outcome)
        extra = extra_carve if command[0] == 'moe_carve.py' else extra_slim
        with (logs / f'{stage}.log').open('w') as stream:
            try:
                subprocess.run([python, '-u', *command, *extra], cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=remaining - 120, check=True)
                outcome[stage] = 'done'
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                outcome[stage] = 'failed'
                stream.write(f'\n{type(error).__name__}: {error}\n')
    return outcome


def main():
    deadline = float(os.environ['MANIFOLD_JOB_DEADLINE'])

    def expired(signum, frame):
        raise TimeoutError('Authorized worker deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(deadline - time.time())))
    RESULTS.mkdir(exist_ok=True)
    bundle = json.loads((REPO / 'bundle-manifest.json').read_text())
    for relative, expected in bundle['files'].items():
        if sha(REPO / relative) != expected:
            raise ValueError('Bundle checksum mismatch: ' + relative)
    for relative in bundle['files']:
        if relative.endswith(('.py', '.txt')):
            dest = RESULTS / 'executed-source' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / relative, dest)
    shutil.copy2(REPO / 'bundle-manifest.json', RESULTS / 'bundle-manifest.json')
    (RESULTS / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (RESULTS / 'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    import torch
    if not torch.cuda.is_available() or not torch.__version__.startswith('2.14.1'):
        raise RuntimeError('Wrong CUDA/PyTorch runtime')
    status('model download')
    from huggingface_hub import hf_hub_download
    source = bundle['model']
    target = REPO / source['path']
    for name, expected in source['files'].items():
        if not (target / name).exists():
            hf_hub_download(source['repo'], filename=name, revision=source['revision'], local_dir=target, token=False)
        if sha(target / name) != expected:
            raise ValueError('Pinned model mismatch: ' + name)
    # Stage outputs land in outputs/, so the launcher's sync carries them home.
    run = ROOT / RUN
    shutil.move(str(run), RESULTS / run.name)
    run.symlink_to(RESULTS / run.name, target_is_directory=True)
    outcome = run_stages(deadline - 60, ROOT, sys.executable, ['--device', 'cuda'], ['--device', 'cuda'])
    save(outcome, RESULTS / 'outcome.json')
    artifacts()
    status('complete', outcome=outcome)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        signal.alarm(0)
        artifacts()
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
