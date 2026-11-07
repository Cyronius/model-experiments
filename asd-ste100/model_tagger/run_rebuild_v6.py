"""Tagger rebuild v6: random initialization, documentation rows mixed in from the start.

v5b showed that the v2 lineage cannot absorb the documentation pairs (its encoder no
longer learns new edits) while a fresh tagger trained on them alone does. v6 trains a
fresh tagger on the audited tagger-rollouts-v7 data: local rule rows capped at 200,000
per epoch (a new draw each epoch), documentation rows batched four times per epoch,
the other kinds once, LR 3e-4 with 500 warmup steps, KEEP 0.4 / terminal 0.6, kind
weights as v5, four epochs, every epoch retained. Each epoch is calibrated on dev rows
0:512 (max exact with at most 5% clean changes) and verified with its frozen policy on
dev rows 1024:2048, the documentation dev pairs and the software review set, plus the
teacher-forced edit categories (edit_confusion.py) and an unguarded decode of the
documentation dev pairs (decode_pairs.py). The v5 epoch 2 decode is the baseline; the
other baselines are in doc-v5-pipeline and doc-v5b-pipeline. No test-set use, no
paid resources.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/out';JOB=OUT/'rebuild-v6-pipeline'
PY=OUT/'.venv/bin/python';VOCAB=OUT/'fixes-v2/vocabulary';PAIRS=OUT/'id-quality-v5';DOCS=OUT/'id-doc-v5'
DATA=OUT/'tagger-rollouts-v7';TRAIN=OUT/'local-train-v6';BASELINE=OUT/'local-train-v5/epoch-2.pt'
BENCH=ROOT/'model_tagger/benchmarks/software-review-v4.jsonl'
MARGINS='0,0.25,0.5,1,1.25,1.5,2,3,4,6';EPOCHS=4

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

def diagnostics(name,checkpoint,report):
    dest=JOB/(name+'-edit-confusion.json')
    run(name+'-edit-confusion','model_tagger/edit_confusion.py','--vocabulary',VOCAB,'--checkpoint',checkpoint,'--data',DATA,'--output',dest)
    report['edit_confusion'][name]=json.loads(dest.read_text())['splits']
    dest=JOB/(name+'-doc-decode.json')
    run(name+'-doc-decode','model_tagger/decode_pairs.py','--vocabulary',VOCAB,'--checkpoint',checkpoint,'--pairs',DOCS,'--output',dest)
    report['doc_decode'][name]=json.loads(dest.read_text())['counts']

def main():
    JOB.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (*(ROOT/'model_tagger').glob('*.py'),BENCH,BASELINE,VOCAB/'vocab.json',DATA/'summary.json')}
    (JOB/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with (JOB/'tests.log').open('w') as log:
        for folder in ('model_tagger','tokenizer'):
            subprocess.run([str(PY),'-m','unittest','discover','-s',str(ROOT/folder/'tests')],stdout=log,stderr=subprocess.STDOUT,check=True)
    require(DATA/'validation.json');require(DATA/'split-validation.json');require(OUT/'doc-v5-pipeline/benchmark-isolation.json')
    report=dict(candidates=[],verification={},doc_dev={},software={},edit_confusion={},doc_decode={})
    diagnostics('v5-epoch-2',BASELINE,report)
    run('train','model_tagger/train.py','--data',DATA,'--vocab',VOCAB,'--out',TRAIN,
        '--epochs',EPOCHS,'--retain-epochs','--lr',3e-4,'--warmup-steps',500,
        '--keep-weight',.4,'--terminal-weight',.6,'--local-example-cap',200000,'--kind-weights','8,12,3,1,12,8',
        '--kind-repeats','1,1,1,1,1,4','--threads',8,'--workers',2,'--batch-size',32,'--precision','bf16',
        '--checkpoint-every',1000,'--eval-every',2000,'--quick-eval-batches',64,'--log-every',100,'--max-hours',6)
    if json.loads((TRAIN/'status.json').read_text())['state']!='complete':raise RuntimeError('training stopped before completion')
    if hashes!={name:sha(ROOT/name) for name in hashes}:raise RuntimeError('source/reference changed during the run')
    doc_dev=json.loads((DOCS/'summary.json').read_text())['splits']['dev']
    for epoch in range(1,EPOCHS+1):
        name=f'v6-epoch-{epoch}';checkpoint=TRAIN/f'epoch-{epoch}.pt'
        diagnostics(name,checkpoint,report)
        out=JOB/name
        run(name,'model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',checkpoint,'--pairs',PAIRS,
            '--output',out,'--sample',512,'--margins',MARGINS)
        chosen=json.loads((out/'calibration.json').read_text())['chosen']
        report['candidates'].append(dict(name=name,checkpoint=str(checkpoint),policy=str(out/'policy.json'),chosen=chosen))
        if chosen is None:continue  # No margin keeps clean changes at or under 5%: recorded, not verified.
        for kind,pairs,sample,offset in (('verification',PAIRS,1024,1024),('doc_dev',DOCS,doc_dev,0)):
            run(f'{name}-{kind}','model_tagger/calibrate_editor.py','--vocabulary',VOCAB,'--checkpoint',checkpoint,
                '--pairs',pairs,'--output',JOB/f'{name}-{kind}','--sample',sample,'--offset',offset,'--locked-policy',out/'policy.json')
            report[kind][name]=json.loads((JOB/f'{name}-{kind}'/'calibration.json').read_text())['candidates'][0]
        dest=JOB/(name+'-software.json')
        run(name+'-software','model_tagger/evaluate_review.py','score','--vocab',VOCAB,'--checkpoint',checkpoint,
            '--policy',out/'policy.json','--benchmark',BENCH,'--output',dest)
        report['software'][name]=json.loads(dest.read_text())['counts']
    eligible=[c for c in report['candidates'] if c['chosen']]
    report['selected']=max(eligible,key=lambda c:(c['chosen']['exact'],-c['chosen']['clean_changed']))['name'] if eligible else None
    report['selection_rule']='max exact on calibration rows 0:512 with at most 5% clean changes; every epoch is verified, so later numbers are diagnostic'
    report['limits']=['Documentation pairs are teacher candidates passed by gates and two model judges, not owner gold.',
                      'Assistant-reviewed seed frames and benchmark references remain pending owner review.',
                      'No test-set use or production promotion.']
    (JOB/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    status('complete','complete',comparison=str(JOB/'comparison.json'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        previous=json.loads((JOB/'status.json').read_text()) if (JOB/'status.json').exists() else {}
        status(previous.get('stage','startup'),'failed',error_type=type(exc).__name__,error=str(exc));raise
