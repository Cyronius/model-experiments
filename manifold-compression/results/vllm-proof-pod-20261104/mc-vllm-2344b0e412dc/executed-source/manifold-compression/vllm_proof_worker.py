"""Disposable CUDA worker for the vLLM proof (vllm_proof_rules.json). No training. No RunPod credential.

Checks the bundle, downloads the pinned Qwen3.5-9B (plus the processor files vLLM reads, hashed and
recorded), installs vLLM into its own environment, then runs vllm_proof.py generate there and
vllm_proof.py compare in the bundle's environment. Everything lands in outputs/vllm_proof/.
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
VLLM = 'vllm==0.33.0'
VENV = Path('/workspace/vllm-env')


def stage(name, command, deadline, env=None):
    w.status(name)
    with (w.RESULTS / f'{name}.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True, env=env,
                       timeout=max(60, deadline - time.time() - 120))


def main():
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
        if relative.endswith(('.py', '.txt', '.json', '.jsonl')) and '/results/' not in relative:
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
    from huggingface_hub import hf_hub_download, snapshot_download
    source = bundle['model']
    target = REPO / source['path']
    for name, expected in source['files'].items():
        if not (target / name).exists():
            hf_hub_download(source['repo'], filename=name, revision=source['revision'], local_dir=target, token=False)
        if w.sha(target / name) != expected:
            raise ValueError('Pinned model mismatch: ' + name)
    # vLLM also reads the processor and generation configs; same revision, recorded by hash.
    snapshot_download(source['repo'], revision=source['revision'], local_dir=target, token=False,
                      allow_patterns=['*.json', '*.jinja', '*.txt'])
    w.save({p.name: w.sha(p) for p in sorted(target.iterdir()) if p.is_file() and p.suffix != '.safetensors'},
           w.RESULTS / 'model-small-files.json')
    out = w.RESULTS / 'vllm_proof'
    out.mkdir(exist_ok=True)
    stage('vllm_install', ['bash', '-c', f'python -m venv {VENV} && {VENV}/bin/python -m pip install --no-cache-dir {VLLM} '
                                         f'&& {VENV}/bin/python -m pip freeze > {w.RESULTS}/vllm-environment.txt'], deadline)
    stage('generate', [str(VENV / 'bin/python'), '-u', 'vllm_proof.py', 'generate', '--model', str(target), '--out', str(out)],
          deadline)
    stage('compare', [sys.executable, '-u', 'vllm_proof.py', 'compare', '--model', str(target), '--out', str(out)], deadline)
    report = json.loads((out / 'compare.json').read_text())
    w.artifacts()
    w.status('complete', outcome=report['verdict'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        signal.alarm(0)
        w.artifacts()
        w.status('failed', error=f'{type(error).__name__}: {error}')
        raise
