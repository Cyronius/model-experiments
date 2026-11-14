"""CPU-only training, reload, split, and input-budget checks; not a quality benchmark."""
import collections
import json
import math
import os
from pathlib import Path
import subprocess
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from broad_data_fetch import atomic,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent


def main():
    import torch
    torch.set_num_threads(2)
    ready=ROOT/'work/broad-data/ready';out=ROOT/'work/broad-data/preflight';out.mkdir(exist_ok=True)
    audit=json.loads((ready/'audit.json').read_text());plan=json.loads((ROOT/'broad-data/runpod-job.json').read_text())
    dev_keys=set();dev_groups=set();tiny=[];source_counts=collections.Counter()
    for line in (ready/'dev.jsonl').open():
        r=json.loads(line);dev_keys.update((r['query_visible_key'],r['positive_visible_key']))
        if r.get('source_group'):dev_groups.add(r['source_group'])
        if source_counts[r['source']]<2:tiny.append(r);source_counts[r['source']]+=1
    n=0;overlap=0;group_overlap=0
    for line in (ready/'train.jsonl').open():
        r=json.loads(line);n+=1;overlap+=r['query_visible_key'] in dev_keys or r['positive_visible_key'] in dev_keys
        group_overlap+=bool(r.get('source_group') and r['source_group'] in dev_groups)
    assert n==audit['train_rows'] and overlap==0 and group_overlap==0
    tiny_path=out/'tiny-dev.jsonl';tiny_path.write_text(''.join(json.dumps(r)+'\n' for r in tiny))
    results={}
    for precision in ('ternary','float'):
        target=out/precision
        if not (target/'model.pt').exists():
            command=[sys.executable,str(ROOT/'pair_train.py'),'--data',str(ready/'train.jsonl'),'--dev',str(tiny_path),
                     '--checkpoint',str(ROOT/'work/broad-data/starts'/(precision+'.pt')),'--out',str(target),'--steps','4',
                     '--batch','8','--micro-batch','4','--limit','128','--lr','1e-5','--eval-every','4','--save-every','2',
                     '--source-floor','100','--device','cpu','--amp','off','--expected-layers','3']
            with (out/(precision+'.log')).open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        sys.path.insert(0,str(PROJECT/'tern-embed'))
        from model import Student
        model=Student.load(target/'model.pt').eval()
        assert model.quantized==(precision=='ternary') and model.dims['layers']==3
        assert sum(p.numel() for p in model.parameters())==17241984
        logs=[json.loads(line) for line in (target/'log.jsonl').open()]
        assert len(logs)==4 and all(math.isfinite(r['loss']) for r in logs)
        checkpoint=torch.load(target/'checkpoint-0000004.pt',map_location='cpu',weights_only=False)
        assert checkpoint['resume']['optimizer'] and checkpoint['resume']['sampler'] and checkpoint['resume']['torch_rng'] is not None
        results[precision]=dict(steps=4,losses=[r['loss'] for r in logs],reload_passed=True,precision_preserved=True,complete_resume_state=True,
                                checkpoint_sha256=digest(target/'model.pt'))
    report=dict(status='CPU preflight passed; not evidence of improved quality',paid_services_used=False,train_rows_verified=n,train_dev_visible_overlap=overlap,train_dev_source_group_overlap=group_overlap,
                token_cap=128,architecture_parameters=17241984,models=results,plan_sha256=digest(ROOT/'broad-data/runpod-job.json'))
    atomic(ROOT/'broad-data/preflight.json',report)
    print('Split and CPU train/reload preflight passed for both precisions.',flush=True)


if __name__=='__main__':main()
