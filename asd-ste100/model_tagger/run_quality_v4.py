"""Audited reviewed-data rebuild, local tagger run and matched dev comparison."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/out';JOB=OUT/'quality-v4-pipeline'
PY=OUT/'.venv/bin/python';VOCAB=OUT/'fixes-v1/vocabulary';PAIRS=OUT/'id-quality-v4b';RAW=OUT/'tagger-rollouts-v6';TRAIN=OUT/'local-train-v4'
BENCH=ROOT/'model_tagger/benchmarks/software-review-v4.jsonl';INITIAL=OUT/'local-train-v2/best.pt'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def status(stage,state,**extra):
    JOB.mkdir(parents=True,exist_ok=True);tmp=JOB/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,state=state,updated=time.time(),**extra),indent=2)+'\n');tmp.replace(JOB/'status.json')

def run(stage,script,*args):
    status(stage,'running');print(stage,flush=True)
    with (JOB/(stage+'.log')).open('w') as log:
        subprocess.run([str(PY),'-u',str(ROOT/script),*map(str,args)],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    status(stage,'passed')

def require(path):
    result=json.loads(Path(path).read_text())
    if not result.get('passed'):raise RuntimeError('required audit failed: '+str(path))
    return result

def calibrate(stage,checkpoint,margins='0,0.25,0.5,1,2,3,4,6'):
    out=JOB/stage
    run(stage,'model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',checkpoint,'--pairs',PAIRS,
        '--output',out,'--sample',512,'--margins',margins)
    result=json.loads((out/'calibration.json').read_text())
    if result['chosen'] is None:raise RuntimeError('no confidence setting meets clean-control constraint: '+stage)
    return dict(name=stage,checkpoint=str(checkpoint),policy=str(out/'policy.json'),chosen=result['chosen'])

def main():
    JOB.mkdir(parents=True,exist_ok=True);src=JOB/'source';src.mkdir(exist_ok=True);hashes={}
    for directory in ('model_tagger','tokenizer','data','checker'):
        for path in (ROOT/directory).glob('*.py'):
            target=src/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            hashes[str(path.relative_to(ROOT))]=sha(path)
    for path in (BENCH,ROOT/'model_tagger/benchmarks/reviewed-software-seeds-v1.json',ROOT/'data/out/data-quality-v4/manual-review.json'):
        hashes[str(path.relative_to(ROOT))]=sha(path)
    (JOB/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    require(PAIRS/'revision-validation.json')
    with (JOB/'tests.log').open('w') as log:
        for folder in ('model_tagger','tokenizer'):
            subprocess.run([str(PY),'-m','unittest','discover','-s',str(ROOT/folder/'tests')],stdout=log,stderr=subprocess.STDOUT,check=True)
    run('rollout-export','model_tagger/rollouts.py','export','--pairs',PAIRS,'--vocab',VOCAB,'--out',RAW,'--workers',8)
    run('rollout-validate','model_tagger/rollouts.py','validate','--pairs',PAIRS,'--vocab',VOCAB,'--out',RAW,'--workers',8)
    require(RAW/'validation.json')
    run('split-audit','model_tagger/audit_splits.py','--data',RAW,'--vocab',VOCAB)
    if not require(RAW/'split-validation.json'):
        raise RuntimeError('canonical split audit failed')
    run('benchmark-isolation','model_tagger/evaluate_review.py','isolation','--pairs',PAIRS,'--benchmark',BENCH,
        '--output',JOB/'benchmark-isolation.json')
    require(JOB/'benchmark-isolation.json')
    if hashes!={name:sha(ROOT/name) for name in hashes}:raise RuntimeError('source/reference changed during preparation')
    run('train','model_tagger/train.py','--data',RAW,'--vocab',VOCAB,'--out',TRAIN,
        '--initialize-from',INITIAL,'--epochs',2,'--retain-epochs','--lr',3e-5,'--warmup-steps',200,
        '--keep-weight',.4,'--terminal-weight',.6,'--local-example-cap',50000,'--kind-weights','8,12,3,1,12',
        '--threads',8,'--workers',2,'--batch-size',32,'--precision','bf16','--checkpoint-every',500,
        '--eval-every',1000,'--quick-eval-batches',64,'--log-every',100,'--max-hours',3)
    if json.loads((TRAIN/'status.json').read_text())['state']!='complete':raise RuntimeError('training stopped before completion')
    candidates=[calibrate('baseline-calibration',INITIAL)]
    for epoch in (1,2):candidates.append(calibrate(f'v4-epoch-{epoch}',TRAIN/f'epoch-{epoch}.pt'))
    winner=max(candidates[1:],key=lambda c:(c['chosen']['exact'],-c['chosen']['clean_changed']))
    (JOB/'selection.json').write_text(json.dumps(dict(winner=winner,candidates=candidates,
        rule='max exact on calibration rows0:512 subject to <=5% clean changes; freeze before verification/benchmark'),indent=2)+'\n')
    report=dict(candidates=candidates,selected=winner,verification={},software={})
    for candidate in (candidates[0],winner):
        name=candidate['name'];out=JOB/(name+'-verification')
        run(name+'-verification','model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',candidate['checkpoint'],
            '--pairs',PAIRS,'--output',out,'--sample',1024,'--offset',1024,'--locked-policy',candidate['policy'])
        report['verification'][name]=json.loads((out/'calibration.json').read_text())['candidates'][0]
        dest=JOB/(name+'-software.json')
        run(name+'-software','model_tagger/evaluate_review.py','score','--vocab',VOCAB,'--checkpoint',candidate['checkpoint'],
            '--policy',candidate['policy'],'--benchmark',BENCH,'--output',dest)
        report['software'][name]=json.loads(dest.read_text())['counts']
    report['limits']=['Assistant-reviewed seed frames and benchmark references remain pending owner review.','No test-set tuning or production promotion.','Manually approved sample does not certify all corpus meanings.']
    (JOB/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    status('complete','complete',comparison=str(JOB/'comparison.json'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        previous=json.loads((JOB/'status.json').read_text()) if (JOB/'status.json').exists() else {}
        status(previous.get('stage','startup'),'failed',error_type=type(exc).__name__,error=str(exc));raise
