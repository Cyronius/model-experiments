"""Disposable GPU worker for the approved correctness pilot; no cloud secret."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def status(stage,**extra):
    path=ROOT/'results/status.json';path.parent.mkdir(exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps({'stage':stage,'updated_at':time.time(),**extra},indent=2)+'\n')
    temp.replace(path);print(stage,extra,flush=True)


def artifacts():
    results=ROOT/'results'
    hashes={str(p.relative_to(results)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in results.rglob('*') if p.is_file() and 'teacher_logits' not in p.parts
        and p.suffix not in ('.log','.tmp') and p.name not in ('status.json','artifact-manifest.json')}
    (results/'artifact-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')


def main():
    deadline=float(os.environ['SCALE_JOB_DEADLINE'])
    def expired(signum,frame):raise TimeoutError('Authorized job deadline reached')
    signal.signal(signal.SIGALRM,expired)
    signal.alarm(max(1,int(deadline-time.time())))
    manifest=json.loads((ROOT/'bundle-manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Bundle checksum mismatch: '+name)
    status('starting')
    # Save executable source, frozen data and the complete reconstruction recipe.
    for name in [*manifest['sha256'],'bundle-manifest.json']:
        target=ROOT/'results/reproduction'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    (ROOT/'results/environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    (ROOT/'results/gpu.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True))
    import torch
    from correctness_heal import smoke
    if not torch.cuda.is_available():raise RuntimeError('Expected CUDA GPU')
    status('runtime backward smoke')
    (ROOT/'results/runtime-smoke.json').write_text(json.dumps(smoke('cuda:0'),indent=2)+'\n')
    from huggingface_hub import snapshot_download
    source=json.loads((ROOT/'data.json').read_text())['source']
    folder=ROOT/'models/qwen27'
    status('downloading',model=source['repo'])
    snapshot_download(source['repo'],revision=source['revision'],local_dir=folder,token=False,
        allow_patterns=['config.json','generation_config.json','tokenizer*','special_tokens_map.json',
                        'chat_template*','model*.safetensors','model.safetensors.index.json'])
    (folder/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    remaining=int(deadline-time.time())
    if remaining<1200:raise TimeoutError('Less than twenty minutes remain after setup')
    status('controls and healing',model='qwen27')
    with (ROOT/'results/healing.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'correctness_heal.py'),'run',
            '--source',str(folder),'--recipe',str(ROOT/'recipe'),'--data',str(ROOT/'data.json'),
            '--legacy-data',str(ROOT/'legacy-data.json'),'--out',str(ROOT/'results/healing'),
            '--deadline',str(deadline-60)],stdout=log,stderr=subprocess.STDOUT,
            timeout=remaining-30,check=True)
    result=json.loads((ROOT/'results/healing/result.json').read_text())
    if result['status']!='complete':raise RuntimeError('Pilot did not complete any training updates')
    from correctness_summary import summarize
    summarize(ROOT/'results/healing')
    artifacts();status('complete',selected_step=result['selected_step'],updates=len(result['steps']))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        signal.alarm(0)
        artifacts();status('failed',error=f'{type(error).__name__}: {error}')
        raise
