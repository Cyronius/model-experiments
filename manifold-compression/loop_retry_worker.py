"""Disposable inference-only worker for an explicitly approved loop diagnostic."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from correctness_worker import ROOT,status,artifacts


def main():
    deadline=float(os.environ['SCALE_JOB_DEADLINE'])
    def expired(signum,frame):raise TimeoutError('Approved diagnostic deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(deadline-time.time())))
    manifest=json.loads((ROOT/'bundle-manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        h=hashlib.sha256()
        with (ROOT/name).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=digest:raise ValueError('Bundle checksum mismatch: '+name)
    status('diagnostic setup')
    for name in [*manifest['sha256'],'bundle-manifest.json']:
        if name.startswith('adapter/'):continue
        dest=ROOT/'results/reproduction'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    (ROOT/'results/environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (ROOT/'results/gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Expected CUDA GPU')
    from huggingface_hub import snapshot_download
    source=json.loads((ROOT/'data.json').read_text())['source'];folder=ROOT/'models/qwen27'
    status('downloading',model=source['repo'])
    snapshot_download(source['repo'],revision=source['revision'],local_dir=folder,token=False,
        allow_patterns=['config.json','generation_config.json','tokenizer*','special_tokens_map.json',
                        'chat_template*','model*.safetensors','model.safetensors.index.json'])
    (folder/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    if deadline-time.time()<600:raise TimeoutError('Less than ten minutes remain after setup')
    status('evaluating previously flagged prompts at 4096 tokens')
    with (ROOT/'results/inference.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'retry_healing_loops.py'),
            '--source',str(folder),'--recipe',str(ROOT/'recipe'),'--data',str(ROOT/'data.json'),
            '--request',str(ROOT/'loop-retry.json'),'--adapter',str(ROOT/'adapter'),
            '--out',str(ROOT/'results/loop-retry'),'--deadline',str(deadline-60)],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,int(deadline-time.time())-30))
    artifacts();status('complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        signal.alarm(0);artifacts();status('failed',error=f'{type(error).__name__}: {error}');raise
