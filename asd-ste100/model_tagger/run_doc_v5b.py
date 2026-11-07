"""Documentation-focused fine-tune of v5: can the tagger learn the teacher's replacements at all?

Same audited data as run_doc_v5.py (tagger-rollouts-v7, both audits required), started
from v5 epoch 2. The documentation rows are batched six times per epoch, the local rule
rows are capped at 25,000, and the peak learning rate is 1e-4 instead of 3e-5, for three
epochs. (Kind weights alone did little in v5: the loss is normalized per batch and most
batches hold one kind.) Every epoch is calibrated on dev rows 0:512 as before and then
verified with its frozen policy on dev rows 1024:2048, the documentation dev pairs and
the software review set; edit_confusion.py records what each checkpoint emits where an
edit is due, on documentation dev and train states. No test-set use, no paid resources.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/out';JOB=OUT/'doc-v5b-pipeline'
PY=OUT/'.venv/bin/python';VOCAB=OUT/'fixes-v2/vocabulary';PAIRS=OUT/'id-quality-v5';DOCS=OUT/'id-doc-v5'
DATA=OUT/'tagger-rollouts-v7';TRAIN=OUT/'local-train-v5b';INITIAL=OUT/'local-train-v5/epoch-2.pt'
BENCH=ROOT/'model_tagger/benchmarks/software-review-v4.jsonl';PREVIOUS=OUT/'doc-v5-pipeline/comparison.json'
MARGINS='0,0.25,0.5,1,1.25,1.5,2,3,4,6';EPOCHS=3

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

def confusion(name,checkpoint):
    dest=JOB/(name+'-edit-confusion.json')
    run(name+'-edit-confusion','model_tagger/edit_confusion.py','--vocabulary',VOCAB,'--checkpoint',checkpoint,
        '--data',DATA,'--output',dest)
    return json.loads(dest.read_text())['splits']

def main():
    JOB.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (*(ROOT/'model_tagger').glob('*.py'),BENCH,INITIAL,VOCAB/'vocab.json',DATA/'summary.json')}
    (JOB/'source-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with (JOB/'tests.log').open('w') as log:
        for folder in ('model_tagger','tokenizer'):
            subprocess.run([str(PY),'-m','unittest','discover','-s',str(ROOT/folder/'tests')],stdout=log,stderr=subprocess.STDOUT,check=True)
    require(DATA/'validation.json');require(DATA/'split-validation.json');require(JOB.parent/'doc-v5-pipeline/benchmark-isolation.json')
    report=dict(previous=json.loads(PREVIOUS.read_text()),candidates=[],verification={},doc_dev={},software={},edit_confusion={})
    report['edit_confusion']['v5-epoch-2']=confusion('v5-epoch-2',INITIAL)
    run('train','model_tagger/train.py','--data',DATA,'--vocab',VOCAB,'--out',TRAIN,
        '--initialize-from',INITIAL,'--epochs',EPOCHS,'--retain-epochs','--lr',1e-4,'--warmup-steps',200,
        '--keep-weight',.4,'--terminal-weight',.6,'--local-example-cap',25000,'--kind-weights','8,12,3,1,12,8',
        '--kind-repeats','1,1,1,1,1,6','--threads',8,'--workers',2,'--batch-size',32,'--precision','bf16',
        '--checkpoint-every',500,'--eval-every',1000,'--quick-eval-batches',64,'--log-every',100,'--max-hours',2)
    if json.loads((TRAIN/'status.json').read_text())['state']!='complete':raise RuntimeError('training stopped before completion')
    if hashes!={name:sha(ROOT/name) for name in hashes}:raise RuntimeError('source/reference changed during the run')
    doc_dev=json.loads((DOCS/'summary.json').read_text())['splits']['dev']
    for epoch in range(1,EPOCHS+1):
        name=f'v5b-epoch-{epoch}';checkpoint=TRAIN/f'epoch-{epoch}.pt'
        report['edit_confusion'][name]=confusion(name,checkpoint)
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
    report['limits']=['Documentation pairs are teacher candidates passed by gates and two model judges, not owner gold.',
                      'Every v5b epoch is verified, so this is a diagnostic comparison, not a frozen selection.',
                      'No test-set use or production promotion.']
    (JOB/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    status('complete','complete',comparison=str(JOB/'comparison.json'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        previous=json.loads((JOB/'status.json').read_text()) if (JOB/'status.json').exists() else {}
        status(previous.get('stage','startup'),'failed',error_type=type(exc).__name__,error=str(exc));raise
