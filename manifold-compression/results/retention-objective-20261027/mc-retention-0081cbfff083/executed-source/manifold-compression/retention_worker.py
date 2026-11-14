"""Disposable CUDA worker. Has no RunPod credential or pod lifecycle authority."""
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
        if relative.endswith(('.py', '.txt')) or relative.endswith('/frozen.json'):
            dest = RESULTS / 'executed-source' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / relative, dest)
    shutil.copy2(REPO / 'bundle-manifest.json', RESULTS / 'bundle-manifest.json')
    (RESULTS / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    (RESULTS / 'gpu.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True))
    import torch
    from retention_run import configure, runtime_smoke
    configure(8)
    if not torch.cuda.is_available() or not torch.__version__.startswith('2.14.1'):
        raise RuntimeError('Wrong CUDA/PyTorch runtime')
    status('runtime smoke')
    save(runtime_smoke('cuda'), RESULTS / 'runtime-smoke.json')
    from test_retention_run import hybrid_batch_equivalence
    save(hybrid_batch_equivalence('cuda'), RESULTS / 'runtime-batch-equivalence.json')
    torch.cuda.empty_cache()
    from retention_prepare import SOURCE, SOURCE_ID, OUT
    from huggingface_hub import hf_hub_download
    status('pinned model download')
    model_manifest = json.loads((SOURCE / 'download_manifest.json').read_text())
    for name, record in model_manifest['files'].items():
        dest = SOURCE / name
        if not dest.exists():
            hf_hub_download(SOURCE_ID['repo'], filename=name, revision=SOURCE_ID['revision'],
                            local_dir=SOURCE, token=False)
        if sha(dest) != record['sha256']:
            raise ValueError('Pinned source mismatch: ' + name)
    env = dict(os.environ, RETENTION_RUN_OUT=str(RESULTS / 'cuda'), TOKENIZERS_PARALLELISM='false')

    def command(argv, label):
        remaining = int(deadline - time.time())
        if remaining < 180:
            raise TimeoutError('Insufficient time for ' + label)
        status(label)
        with (RESULTS / (label + '.log')).open('w') as stream:
            subprocess.run([sys.executable, '-u', *argv], cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, timeout=remaining - 30, check=True)

    command(['retention_run.py', '--out', str(RESULTS / 'cuda'), '--deadline', str(deadline - 60)], 'retention')
    metrics = RESULTS / 'tf-metrics'
    metrics.mkdir()
    source_metrics = ROOT / 'results/tf-metrics-20261027'
    for name in ('manifest.jsonl', 'prepare-report.json'):
        shutil.copy2(source_metrics / name, metrics / name)
    score_seconds = []
    skipped = []
    for arm in ('stock', 'A0', 'A1', 'A2', 'S1', 'S2'):
        estimate = max(score_seconds or [900]) * 1.5
        if arm in ('S1', 'S2') and time.time() + estimate + 600 > deadline:
            skipped.append(arm)
            continue
        start = time.time()
        tail = ['--arm', 'stock'] if arm == 'stock' else ['--loader', f'retention_run:load_{arm}', '--name', arm]
        command(['tf_metrics.py', 'score', '--device', 'cuda', '--threads', '8', '--out', str(metrics), *tail], 'score-' + arm)
        score_seconds.append(time.time() - start)
    stock = json.loads((metrics / 'scores-stock.json').read_text())
    stock.update(arm='S0', alias_of='stock', alias_reason='S0 is the identical unmodified stock model; no second scoring pass')
    save(stock, metrics / 'scores-S0.json')
    save({'order': ['stock', 'A0', 'A1', 'A2', 'S1', 'S2'], 'skipped_for_deadline': skipped,
          'score_seconds': score_seconds, 'complete_all_arms': not skipped}, metrics / 'completion.json')
    command(['retention_run.py', '--out', str(RESULTS / 'cuda'), '--deadline', str(deadline - 60),
             '--regression-only'], 'regressions')
    artifacts()
    result = json.loads((RESULTS / 'cuda/result.json').read_text())
    status('complete', regression_status=result.get('regression_status'), skipped_0b=skipped,
           local_review='pending; test files sealed; no automatic promotion')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        signal.alarm(0)
        artifacts()
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
