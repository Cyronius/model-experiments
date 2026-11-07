"""Audited v5 rebuild with the October 30 vocabulary and the documentation pairs, local run and comparison.

Stages as run_quality_v4.py: fresh ID export of surface-clean-v3 with fixes-v2, the v5
revision (v4b rules plus the doc_teacher pairs), rollout export and replay, canonical
split audit (whole implicated groups quarantined if it fails), benchmark isolation,
then two CPU epochs from the migrated v2 checkpoint. Confidence policies are selected
on dev rows 0:512 only, then frozen for dev rows 1024:2048, the software review set and
the documentation dev pairs (id-doc-v5). The baseline is v2 migrated to the new
vocabulary (fixes-v2/vocabulary/model.pt). No test-set use, no paid resources.
"""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/out';JOB=OUT/'doc-v5-pipeline'
PY=OUT/'.venv/bin/python';VOCAB=OUT/'fixes-v2/vocabulary';SURFACE=OUT/'surface-clean-v3';IDS=OUT/'id-joint-v5'
PAIRS=OUT/'id-quality-v5';DOCS=OUT/'id-doc-v5';RAW=OUT/'tagger-rollouts-v7';QUARANTINED=OUT/'tagger-rollouts-v8'
TRAIN=OUT/'local-train-v5';INITIAL=VOCAB/'model.pt'
BENCH=ROOT/'model_tagger/benchmarks/software-review-v4.jsonl'
MARGINS='0,0.25,0.5,1,1.25,1.5,2,3,4,6'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def status(stage,state,**extra):
    JOB.mkdir(parents=True,exist_ok=True);tmp=JOB/'status.tmp'
    tmp.write_text(json.dumps(dict(stage=stage,state=state,updated=time.time(),**extra),indent=2)+'\n');tmp.replace(JOB/'status.json')

def run(stage,script,*args):
    status(stage,'running');print(stage,flush=True)
    with (JOB/(stage+'.log')).open('w') as log:
        subprocess.run([str(PY),'-u',str(ROOT/script),*map(str,args)],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    status(stage,'passed')

def require(path):
    result=json.loads(Path(path).read_text())
    if not result.get('passed'):raise RuntimeError('required audit failed: '+str(path))
    return result

def calibrate(stage,checkpoint):
    out=JOB/stage
    run(stage,'model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',checkpoint,'--pairs',PAIRS,
        '--output',out,'--sample',512,'--margins',MARGINS)
    result=json.loads((out/'calibration.json').read_text())
    if result['chosen'] is None:raise RuntimeError('no confidence setting meets clean-control constraint: '+stage)
    return dict(name=stage,checkpoint=str(checkpoint),policy=str(out/'policy.json'),chosen=result['chosen'])

def main():
    JOB.mkdir(parents=True,exist_ok=True);src=JOB/'source';src.mkdir(exist_ok=True);hashes={}
    for directory in ('model_tagger','tokenizer','data','checker','data/doc_pilot'):
        for path in (ROOT/directory).glob('*.py'):
            target=src/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            hashes[str(path.relative_to(ROOT))]=sha(path)
    for path in (BENCH,ROOT/'model_tagger/benchmarks/reviewed-software-seeds-v1.json',OUT/'data-quality-v4/manual-review.json',
                 ROOT/'lexicon/tech_words.txt',OUT/'doc-teacher-v5/final/accepted.jsonl',VOCAB/'vocab.json',INITIAL):
        hashes[str(path.relative_to(ROOT))]=sha(path)
    (JOB/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with (JOB/'tests.log').open('w') as log:
        for folder in ('model_tagger','tokenizer'):
            subprocess.run([str(PY),'-m','unittest','discover','-s',str(ROOT/folder/'tests')],stdout=log,stderr=subprocess.STDOUT,check=True)
    run('id-export','tokenizer/prepare.py','export','--pairs',SURFACE,'--vocab',VOCAB,'--out',IDS,'--workers',12)
    run('id-validate','tokenizer/prepare.py','validate','--vocab',VOCAB,'--out',IDS)
    require(IDS/'validation.json')
    run('revision-build','model_tagger/data_revision_v5.py','build')
    run('revision-validate','model_tagger/data_revision_v5.py','validate')
    require(PAIRS/'revision-validation.json')
    run('rollout-export','model_tagger/rollouts.py','export','--pairs',PAIRS,'--vocab',VOCAB,'--out',RAW,'--workers',12)
    run('rollout-validate','model_tagger/rollouts.py','validate','--pairs',PAIRS,'--vocab',VOCAB,'--out',RAW,'--workers',12)
    require(RAW/'validation.json')
    run('split-audit','model_tagger/audit_splits.py','--data',RAW,'--vocab',VOCAB)
    data=RAW
    if not json.loads((RAW/'split-validation.json').read_text())['passed']:
        data=QUARANTINED
        run('quarantine','model_tagger/quarantine_rollouts.py','--parent',RAW,'--pairs',PAIRS,'--out',data)
        run('quarantine-validate','model_tagger/rollouts.py','validate','--pairs',PAIRS,'--vocab',VOCAB,'--out',data,'--workers',12)
        run('quarantine-split-audit','model_tagger/audit_splits.py','--data',data,'--vocab',VOCAB)
    require(data/'validation.json');require(data/'split-validation.json')
    run('benchmark-isolation','model_tagger/evaluate_review.py','isolation','--pairs',PAIRS,'--benchmark',BENCH,
        '--output',JOB/'benchmark-isolation.json')
    require(JOB/'benchmark-isolation.json')
    if hashes!={name:sha(ROOT/name) for name in hashes}:raise RuntimeError('source/reference changed during preparation')
    run('train','model_tagger/train.py','--data',data,'--vocab',VOCAB,'--out',TRAIN,
        '--initialize-from',INITIAL,'--epochs',2,'--retain-epochs','--lr',3e-5,'--warmup-steps',200,
        '--keep-weight',.4,'--terminal-weight',.6,'--local-example-cap',50000,'--kind-weights','8,12,3,1,12,8',
        '--threads',8,'--workers',2,'--batch-size',32,'--precision','bf16','--checkpoint-every',500,
        '--eval-every',1000,'--quick-eval-batches',64,'--log-every',100,'--max-hours',3)
    if json.loads((TRAIN/'status.json').read_text())['state']!='complete':raise RuntimeError('training stopped before completion')
    candidates=[calibrate('baseline-calibration',INITIAL)]
    for epoch in (1,2):candidates.append(calibrate(f'v5-epoch-{epoch}',TRAIN/f'epoch-{epoch}.pt'))
    winner=max(candidates[1:],key=lambda c:(c['chosen']['exact'],-c['chosen']['clean_changed']))
    (JOB/'selection.json').write_text(json.dumps(dict(winner=winner,candidates=candidates,
        rule='max exact on calibration rows0:512 subject to <=5% clean changes; frozen before verification, software and doc dev'),indent=2)+'\n')
    doc_dev=json.loads((DOCS/'summary.json').read_text())['splits']['dev']
    report=dict(candidates=candidates,selected=winner,verification={},software={},doc_dev={})
    for candidate in (candidates[0],winner):
        name=candidate['name']
        for kind,pairs,sample,offset in (('verification',PAIRS,1024,1024),('doc_dev',DOCS,doc_dev,0)):
            out=JOB/f'{name}-{kind}'
            run(f'{name}-{kind}','model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',candidate['checkpoint'],
                '--pairs',pairs,'--output',out,'--sample',sample,'--offset',offset,'--locked-policy',candidate['policy'])
            report[kind][name]=json.loads((out/'calibration.json').read_text())['candidates'][0]
        dest=JOB/(name+'-software.json')
        run(name+'-software','model_tagger/evaluate_review.py','score','--vocab',VOCAB,'--checkpoint',candidate['checkpoint'],
            '--policy',candidate['policy'],'--benchmark',BENCH,'--output',dest)
        report['software'][name]=json.loads(dest.read_text())['counts']
    report['limits']=['Documentation pairs are teacher candidates passed by gates and two model judges, not owner gold.',
                      'Assistant-reviewed seed frames and benchmark references remain pending owner review.',
                      'The baseline is v2 migrated to the new vocabulary: new output words start suppressed.',
                      'No test-set use or production promotion.']
    (JOB/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    status('complete','complete',comparison=str(JOB/'comparison.json'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        previous=json.loads((JOB/'status.json').read_text()) if (JOB/'status.json').exists() else {}
        status(previous.get('stage','startup'),'failed',error_type=type(exc).__name__,error=str(exc));raise
