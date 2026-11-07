"""Canvas editor smoke test: train on the documentation pairs only and compare with the doc-only tagger.

Trains a small canvas editor (about 8M parameters; the tagger is 27.9M) on the 6,897
documentation train pairs of id-quality-v5, then decodes the 679 documentation dev
pairs of id-doc-v5 with the canvas editor and, for the same counts, with the fresh
doc-only tagger from the v5b probes (doc-v5b-pipeline/probes/fresh-doc.pt). No
test-set use, no paid resources.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/out';JOB=None
PY=OUT/'.venv/bin/python';VOCAB=OUT/'fixes-v2/vocabulary';TRAIN_PAIRS=OUT/'id-quality-v5';DOCS=OUT/'id-doc-v5'
TAGGER=OUT/'doc-v5b-pipeline/probes/fresh-doc.pt'

def status(stage,state,**extra):
    JOB.mkdir(parents=True,exist_ok=True);tmp=JOB/'status.tmp'
    tmp.write_text(json.dumps(dict(stage=stage,state=state,updated=time.time(),**extra),indent=2)+'\n');tmp.replace(JOB/'status.json')

def run(stage,script,*args):
    status(stage,'running');print(stage,flush=True)
    with (JOB/(stage+'.log')).open('w') as log:
        subprocess.run([str(PY),'-u',str(ROOT/script),*map(str,args)],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    status(stage,'passed')

def main():
    global JOB
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',default='canvas-smoke-v1')
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--checkpoint',default='last.pt',help='which saved model to decode')
    parser.add_argument('--tagger-from',help='reuse the tagger decode of an earlier job instead of rerunning it')
    args=parser.parse_args();JOB=OUT/args.job
    JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'tests.log').open('w') as log:
        for folder in ('model_canvas','model_tagger','tokenizer'):
            subprocess.run([str(PY),'-m','unittest','discover','-s',str(ROOT/folder/'tests')],stdout=log,stderr=subprocess.STDOUT,check=True)
    run('train','model_canvas/train.py','--vocabulary',VOCAB,'--train-pairs',TRAIN_PAIRS,'--dev-pairs',DOCS,
        '--out',JOB/'model','--epochs',args.epochs,'--eval-every',5,'--quick-decode',128,'--max-hours',2)
    run('canvas-decode','model_canvas/decode.py','--vocabulary',VOCAB,'--checkpoint',JOB/'model'/args.checkpoint,
        '--pairs',DOCS,'--output',JOB/'canvas-decode.json')
    if args.tagger_from:(JOB/'tagger-decode.json').write_bytes((OUT/args.tagger_from/'tagger-decode.json').read_bytes())
    else:run('tagger-decode','model_tagger/decode_pairs.py','--vocabulary',VOCAB,'--checkpoint',TAGGER,
            '--pairs',DOCS,'--output',JOB/'tagger-decode.json')
    report={name:json.loads((JOB/f'{name}-decode.json').read_text())['counts'] for name in ('canvas','tagger')}
    report['training']=json.loads((JOB/'model/status.json').read_text())
    report['limits']=['Documentation pairs are teacher candidates passed by gates and two model judges, not owner gold.',
                      'The canvas checkpoint is selected on held-out train groups (best.pt) or is the last epoch (last.pt); dev is report-only.',
                      'The canvas editor is about 8M parameters, the doc-only tagger 27.9M.']
    (JOB/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    status('complete','complete',comparison=str(JOB/'comparison.json'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        if JOB is None:raise
        previous=json.loads((JOB/'status.json').read_text()) if (JOB/'status.json').exists() else {}
        status(previous.get('stage','startup'),'failed',error_type=type(exc).__name__,error=str(exc));raise
