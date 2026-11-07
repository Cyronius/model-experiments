"""Audited local v2 rebuild, deterministic resume pilot, training and dev comparison.

This pipeline uses only local CPU processing. Every required audit gates training.
Stage logs and status persist across the calling chat session.
"""
from pathlib import Path
import json
import subprocess
import sys
import time
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/out'
RUN=OUT/'local-v2-pipeline'
PY=OUT/'.venv/bin/python'
VOCAB=OUT/'fixes-v1/vocabulary'
INITIAL=VOCAB/'model.pt'
PAIRS=OUT/'id-joint-v3'
RAW=OUT/'tagger-rollouts-v4'
DATA=RAW
TRAIN=OUT/'local-train-v2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def status(stage,state,**extra):
    path=RUN/'status.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(stage=stage,state=state,updated=time.time(),**extra),indent=2)+'\n')
    temporary.replace(path)

def execute(stage,script,*arguments):
    print(f'STAGE {stage}',flush=True);status(stage,'running')
    with (RUN/(stage+'.log')).open('w') as log:
        subprocess.run([str(PY),'-u',str(ROOT/script),*map(str,arguments)],cwd=ROOT.parent,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    status(stage,'passed')

def require(path):
    report=json.loads(path.read_text())
    if not report['passed']:raise RuntimeError('audit failed: '+str(path))
    return report


def train_args(data,out):
    return ['--data',data,'--vocab',VOCAB,'--out',out,'--initialize-from',INITIAL,
            '--epochs',3,'--lr',6e-5,'--warmup-steps',200,'--keep-weight',1,
            '--terminal-weight',1,'--local-example-cap',200000,'--kind-weights','8,12,3,1',
            '--threads',8,'--workers',2,'--batch-size',32,'--precision','bf16',
            '--checkpoint-every',500,'--eval-every',1000,'--log-every',100,'--max-hours',8]


def main():
    global DATA
    RUN.mkdir(parents=True,exist_ok=True)
    code_files=[p for directory in ('model_tagger','tokenizer','data','checker') for p in (ROOT/directory).glob('*.py')]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in code_files}
    (RUN/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    execute('quality-validate','data/validate_quality.py','--parent',OUT/'joint-surface-v1','--out',OUT/'surface-clean-v3')
    require(OUT/'surface-clean-v3/quality-validation.json')
    execute('id-export','tokenizer/prepare.py','export','--pairs',OUT/'surface-clean-v3','--vocab',VOCAB,'--out',PAIRS,'--workers',12)
    execute('id-validate','tokenizer/prepare.py','validate','--vocab',VOCAB,'--out',PAIRS)
    require(PAIRS/'validation.json')
    execute('rollout-export','model_tagger/rollouts.py','export','--pairs',PAIRS,'--vocab',VOCAB,'--out',RAW,'--workers',12)
    execute('rollout-validate','model_tagger/rollouts.py','validate','--pairs',PAIRS,'--vocab',VOCAB,'--out',RAW,'--workers',12)
    require(RAW/'validation.json')
    execute('split-audit','model_tagger/audit_splits.py','--data',RAW,'--vocab',VOCAB)
    split=json.loads((RAW/'split-validation.json').read_text())
    if not split['passed']:
        DATA=OUT/'tagger-rollouts-v5'
        execute('quarantine','model_tagger/quarantine_rollouts.py','--parent',RAW,'--pairs',PAIRS,'--out',DATA)
        execute('quarantine-validate','model_tagger/rollouts.py','validate','--pairs',PAIRS,'--vocab',VOCAB,'--out',DATA,'--workers',12)
        execute('quarantine-split-audit','model_tagger/audit_splits.py','--data',DATA,'--vocab',VOCAB)
    require(DATA/'validation.json');require(DATA/'split-validation.json')
    execute('benchmark-isolation','model_tagger/evaluate_review.py','isolation','--pairs',PAIRS,'--output',RUN/'benchmark-isolation.json')
    require(RUN/'benchmark-isolation.json')
    # Exercise actual initialization, optimizer/checkpoint and exact RNG/batch resume.
    common=train_args(DATA,OUT/'local-v2-pilot-resume')+['--quick-eval-batches',4]
    execute('pilot-first','model_tagger/train.py',*common,'--max-steps',4)
    execute('pilot-resume','model_tagger/train.py',*common,'--max-steps',8,'--resume')
    execute('pilot-continuous','model_tagger/train.py',*train_args(DATA,OUT/'local-v2-pilot-continuous'),'--quick-eval-batches',4,'--max-steps',8)
    import torch
    a=torch.load(OUT/'local-v2-pilot-resume/last.pt',map_location='cpu',weights_only=True)
    b=torch.load(OUT/'local-v2-pilot-continuous/last.pt',map_location='cpu',weights_only=True)
    def equal(x,y):
        if isinstance(x,torch.Tensor):return torch.equal(x,y)
        if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(i,j) for i,j in zip(x,y))
        return x==y
    if not all(equal(a[k],b[k]) for k in ('model','optimizer','torch_rng_state','step','next_batch')):
        raise RuntimeError('resumed pilot differs from continuous training')
    (RUN/'pilot-validation.json').write_text(json.dumps(dict(passed=True,updates=8,model_optimizer_rng_exact=True),indent=2)+'\n')
    if manifest!={str(p.relative_to(ROOT)):sha(p) for p in code_files}:raise RuntimeError('source changed during pipeline')
    execute('train','model_tagger/train.py',*train_args(DATA,TRAIN))
    if json.loads((TRAIN/'status.json').read_text())['state']!='complete':
        raise RuntimeError('training stopped before completion; review and resume explicitly')
    execute('calibrate','model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',TRAIN/'best.pt','--pairs',PAIRS,'--output',TRAIN/'calibration','--sample',512)
    if not (TRAIN/'calibration/policy.json').exists():raise RuntimeError('no threshold meets clean-change target')
    execute('verify','model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',TRAIN/'best.pt','--pairs',PAIRS,'--output',TRAIN/'verification','--sample',512,'--offset',512,'--locked-policy',TRAIN/'calibration/policy.json')
    execute('baseline-verify','model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',INITIAL,'--pairs',PAIRS,'--output',TRAIN/'baseline-verification','--sample',512,'--offset',512,'--locked-policy',OUT/'fixes-v1/calibration/policy.json')
    execute('benchmark-new','model_tagger/evaluate_review.py','score','--vocab',VOCAB,'--checkpoint',TRAIN/'best.pt','--policy',TRAIN/'calibration/policy.json','--output',TRAIN/'software-review.json')
    execute('benchmark-baseline','model_tagger/evaluate_review.py','score','--vocab',VOCAB,'--checkpoint',INITIAL,'--policy',OUT/'fixes-v1/calibration/policy.json','--output',TRAIN/'software-review-baseline.json')
    report={name:json.loads((TRAIN/path).read_text()) for name,path in [('new_dev','verification/calibration.json'),('baseline_dev','baseline-verification/calibration.json'),('software','software-review.json'),('software_baseline','software-review-baseline.json')]}
    (TRAIN/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    status('complete','complete',training=str(TRAIN),data=str(DATA),comparison=str(TRAIN/'comparison.json'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        current=json.loads((RUN/'status.json').read_text()) if (RUN/'status.json').exists() else {}
        status(current.get('stage','startup'),'failed',error_type=type(exc).__name__,error=str(exc))
        raise
