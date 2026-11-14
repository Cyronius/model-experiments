"""Disposable CUDA worker for the healing pilot. Has no RunPod credential or pod lifecycle authority.

Checks the bundle, downloads the pinned Qwen3.5-9B, then runs heal, final and generate in order
(slim_worker's stage runner). The heal/ folder lives in outputs/ so the launcher's sync carries it
home; the low-rank factors and the full-precision adapter stay beside it on the pod. run() serves
other single-folder jobs too (cmoe_worker.py).
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import slim_worker as w

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RUN = 'results/moe-carve-v2-20261029'
HEAL = ['slim_heal.py']
# (stage, command, stages whose outputs it reads)
STAGES = [
    ('heal', HEAL + ['heal'], []),
    ('final', HEAL + ['final'], ['heal']),
    ('generate', HEAL + ['generate'], ['heal']),
]


def main(stages, folder):
    deadline = float(os.environ['MANIFOLD_JOB_DEADLINE'])

    def expired(signum, frame):
        raise TimeoutError('Authorized worker deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(deadline - time.time())))
    w.RESULTS.mkdir(exist_ok=True)
    bundle = json.loads((REPO / 'bundle-manifest.json').read_text())
    for relative, expected in bundle['files'].items():
        if w.sha(REPO / relative) != expected:
            raise ValueError('Bundle checksum mismatch: ' + relative)
    for relative in bundle['files']:
        if relative.endswith(('.py', '.txt', '.json')) and '/results/' not in relative:
            dest = w.RESULTS / 'executed-source' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / relative, dest)
    shutil.copy2(REPO / 'bundle-manifest.json', w.RESULTS / 'bundle-manifest.json')
    (w.RESULTS / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (w.RESULTS / 'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    import torch
    if not torch.cuda.is_available() or not torch.__version__.startswith('2.14.1'):
        raise RuntimeError('Wrong CUDA/PyTorch runtime')
    w.status('model download')
    from huggingface_hub import hf_hub_download
    source = bundle['model']
    target = REPO / source['path']
    for name, expected in source['files'].items():
        if not (target / name).exists():
            hf_hub_download(source['repo'], filename=name, revision=source['revision'], local_dir=target, token=False)
        if w.sha(target / name) != expected:
            raise ValueError('Pinned model mismatch: ' + name)
    # The job's folder moves to outputs/, so the launcher's sync carries its results home.
    path = ROOT / RUN / folder
    shutil.move(str(path), w.RESULTS / folder)
    path.symlink_to(w.RESULTS / folder, target_is_directory=True)
    w.STAGES = stages
    outcome = w.run_stages(deadline - 60, ROOT, sys.executable, (), ['--device', 'cuda'])
    w.save(outcome, w.RESULTS / 'outcome.json')
    w.artifacts()
    w.status('complete', outcome=outcome)


def run(stages, folder):
    try:
        main(stages, folder)
    except BaseException as error:
        signal.alarm(0)
        w.artifacts()
        w.status('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    run(STAGES, 'heal')
