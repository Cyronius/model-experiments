"""Local loss-balance experiment; run inside one Dibs-coordinated CPU job.

Select all policies/checkpoints on the original calibration slice only. Fresh dev
rows1024:2048 and the frozen software review set are reporting-only. No test data.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/out';JOB=OUT/'balance-v3';VOCAB=OUT/'fixes-v1/vocabulary'
PAIRS=OUT/'id-joint-v3';DATA=OUT/'tagger-rollouts-v4';TRAIN=OUT/'local-train-v3'
PY=OUT/'.venv/bin/python'
GRID='0,0.25,0.5,0.75,1,1.5,2'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def status(stage,state,**extra):
    JOB.mkdir(parents=True,exist_ok=True)
    tmp=JOB/'status.tmp';tmp.write_text(json.dumps(dict(stage=stage,state=state,updated=time.time(),**extra),indent=2)+'\n');tmp.replace(JOB/'status.json')

def run(stage,script,*args):
    status(stage,'running');print(stage,flush=True)
    with (JOB/f'{stage}.log').open('w') as log:
        subprocess.run([str(PY),'-u',str(ROOT/script),*map(str,args)],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    status(stage,'passed')


def calibrate(name,checkpoint,margins=GRID):
    destination=JOB/name
    existing=destination/'calibration.json'
    report=json.loads(existing.read_text()) if existing.exists() else None
    expected_margins=[float(x) for x in margins.split(',')]
    if report is not None:
        if report['checkpoint_sha256']!=sha(Path(checkpoint)) or report['offset']!=0 or len(report['sample_ids'])!=512 or [c['margin'] for c in report['candidates']]!=expected_margins:
            raise RuntimeError('existing calibration differs: '+name)
    else:
        run(name,'model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',checkpoint,
            '--pairs',PAIRS,'--output',destination,'--sample',512,'--margins',margins)
        report=json.loads(existing.read_text())
    return dict(name=name,checkpoint=str(checkpoint),directory=str(destination),chosen=report['chosen'])


def verify(candidate):
    name=candidate['name'];dest=JOB/(name+'-verification')
    run(name+'-verification','model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',candidate['checkpoint'],
        '--pairs',PAIRS,'--output',dest,'--sample',1024,'--offset',1024,'--locked-policy',Path(candidate['directory'])/'policy.json')
    report=json.loads((dest/'calibration.json').read_text())
    return report['candidates'][0]


def main():
    JOB.mkdir(parents=True,exist_ok=True)
    # Retain exact implementations used for this experiment.
    import shutil
    source=JOB/'source';source.mkdir(exist_ok=True)
    manifest={}
    for directory in ('model_tagger','tokenizer','data','checker'):
        for path in (ROOT/directory).glob('*.py'):
            target=source/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            manifest[str(path.relative_to(ROOT))]=sha(path)
    (JOB/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (JOB/'tests.log').open('w') as log:
        for folder in ('model_tagger','tokenizer'):
            subprocess.run([str(PY),'-m','unittest','discover','-s',str(ROOT/folder/'tests')],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Keep all source/data/split checks in train.py; reuse the passed immutable dataset.
    candidates=[calibrate('v2-calibration',OUT/'local-train-v2/best.pt'),
                calibrate('baseline-calibration',VOCAB/'model.pt','3,3.5,4,4.5,5,5.5,6')]
    run('train','model_tagger/train.py','--data',DATA,'--vocab',VOCAB,'--out',TRAIN,
        '--initialize-from',VOCAB/'model.pt','--epochs',3,'--retain-epochs','--lr',6e-5,'--warmup-steps',200,
        '--keep-weight',.3,'--terminal-weight',.5,'--local-example-cap',200000,'--kind-weights','8,12,3,1',
        '--threads',8,'--workers',2,'--batch-size',32,'--precision','bf16','--checkpoint-every',500,
        '--eval-every',1000,'--quick-eval-batches',64,'--log-every',100,'--max-hours',4)
    if json.loads((TRAIN/'status.json').read_text())['state']!='complete':raise RuntimeError('training stopped before completion')
    for epoch in (1,2,3):
        candidates.append(calibrate(f'v3-epoch-{epoch}',TRAIN/f'epoch-{epoch}.pt','0,0.5,1,1.5,2,2.5,3,4,5,6'))
    eligible=[candidate for candidate in candidates if candidate['chosen'] is not None]
    if not eligible:raise RuntimeError('no policy meets calibration clean-change constraint')
    # Winner frozen before any new verification or software-review result is observed.
    winner=max(eligible,key=lambda c:(c['chosen']['exact'],-c['chosen']['clean_changed']))
    (JOB/'selection.json').write_text(json.dumps(dict(winner=winner,candidates=candidates,selection_scope='calibration rows0:512 only; <=5% clean changes; max noisy exact',verification_scope='fresh dev rows1024:2048; reporting only'),indent=2)+'\n')
    report=dict(selected=winner,candidates=candidates,verification={})
    for candidate in eligible:
        report['verification'][candidate['name']]=verify(candidate)
    # Software set is inspected once for the calibration-selected winner, never for selection.
    run('selected-software','model_tagger/evaluate_review.py','score','--vocab',VOCAB,'--checkpoint',winner['checkpoint'],
        '--policy',Path(winner['directory'])/'policy.json','--output',JOB/'software-review.json')
    report['software']=json.loads((JOB/'software-review.json').read_text())
    report['limitations']=['Dev calibration reused for tuning; verification rows are distinct.','Software references are assistant-reviewed, owner review pending.','No production promotion or changes to frozen test results.']
    (JOB/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    status('complete','complete',comparison=str(JOB/'comparison.json'),selection=str(JOB/'selection.json'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        current=json.loads((JOB/'status.json').read_text()) if (JOB/'status.json').exists() else {}
        status(current.get('stage','startup'),'failed',error_type=type(exc).__name__,error=str(exc));raise
