"""Restore a completed pilot on one newly approved disposable GPU pod."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from correctness_worker import ROOT, status, artifacts


def main():
    deadline=float(os.environ['SCALE_JOB_DEADLINE'])
    def expired(signum,frame):raise TimeoutError('Authorized continuation deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(deadline-time.time())))
    manifest=json.loads((ROOT/'bundle-manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        h=hashlib.sha256()
        with (ROOT/name).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=digest:raise ValueError('Bundle checksum mismatch: '+name)
    status('continuation setup')
    # Prior optimizer/adapters already have verified local copies; do not copy
    # them back again on every result sync. Preserve hashes and executable inputs.
    for name in [*manifest['sha256'],'bundle-manifest.json']:
        if name.startswith('prior/'):continue
        dest=ROOT/'results/reproduction'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    (ROOT/'results/environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (ROOT/'results/gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    import torch
    from correctness_heal import smoke
    if not torch.cuda.is_available():raise RuntimeError('Expected CUDA GPU')
    status('runtime backward smoke')
    (ROOT/'results/runtime-smoke.json').write_text(json.dumps(smoke('cuda:0'),indent=2)+'\n')
    from huggingface_hub import snapshot_download
    source=json.loads((ROOT/'data.json').read_text())['source'];folder=ROOT/'models/qwen27'
    status('downloading',model=source['repo'])
    snapshot_download(source['repo'],revision=source['revision'],local_dir=folder,token=False,
        allow_patterns=['config.json','generation_config.json','tokenizer*','special_tokens_map.json',
                        'chat_template*','model*.safetensors','model.safetensors.index.json'])
    (folder/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    if deadline-time.time()<1800:raise TimeoutError('Less than thirty minutes remain after setup')
    os.environ['CORRECTNESS_PRIOR']=str(ROOT/'prior')
    from correctness_resume_worker import main as resume
    resume()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        signal.alarm(0);artifacts();status('failed',error=f'{type(error).__name__}: {error}');raise
