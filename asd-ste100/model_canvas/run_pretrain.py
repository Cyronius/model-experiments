"""Canvas editor pretraining test: local-rule pairs first, then the documentation pairs.

Stage 1 trains the canvas editor from scratch on the non-documentation train rows of
id-quality-v5 (local_rule, rule, reviewed, original, messy), with drop_article noise
capped and the pairs that delete an article before a code literal skipped. Stage 2
fine-tunes on the 6,897 documentation train pairs and keeps the epoch with the most
exact rewrites of the held-out documentation train groups (best.pt). A control runs
stage 2 from scratch with the same selection, so the effect of pretraining is
separate from the effect of decode selection. The 679 documentation dev pairs of
id-doc-v5 are decoded with each model and compared with the smoke v2 canvas and the
doc-only tagger. No test-set use, no paid resources.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/out'
PY = OUT / '.venv/bin/python'
VOCAB = OUT / 'fixes-v2/vocabulary'
TRAIN_PAIRS = OUT / 'id-quality-v5'
DOCS = OUT / 'id-doc-v5'
PRETRAIN_KINDS = 'local_rule,rule,reviewed,original,messy'
JOB = None


def status(stage, state, **extra):
    JOB.mkdir(parents=True, exist_ok=True)
    tmp = JOB / 'status.tmp'
    tmp.write_text(json.dumps(dict(stage=stage, state=state, updated=time.time(), **extra), indent=2) + '\n')
    tmp.replace(JOB / 'status.json')


def run(stage, script, *args):
    status(stage, 'running')
    print(stage, flush=True)
    with (JOB / (stage + '.log')).open('w') as log:
        subprocess.run([str(PY), '-u', str(ROOT / script), *map(str, args)], cwd=ROOT.parent,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    status(stage, 'passed')


def train(stage, *args):
    run(stage, 'model_canvas/train.py', '--vocabulary', VOCAB, '--train-pairs', TRAIN_PAIRS, '--dev-pairs', DOCS,
        '--out', JOB / stage, '--quick-decode', 128, *args)


def decode(stage, checkpoint):
    run(stage + '-decode', 'model_canvas/decode.py', '--vocabulary', VOCAB, '--checkpoint', checkpoint,
        '--pairs', DOCS, '--output', JOB / f'{stage}-decode.json')


def main():
    global JOB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', default='canvas-pretrain-v1')
    parser.add_argument('--pretrain-epochs', type=int, default=2)
    parser.add_argument('--drop-article-cap', type=int, default=120000)
    parser.add_argument('--finetune-epochs', type=int, default=25)
    parser.add_argument('--skip-control', action='store_true')
    args = parser.parse_args()
    JOB = OUT / args.job
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB / 'tests.log').open('w') as log:
        for folder in ('model_canvas', 'model_tagger', 'tokenizer'):
            subprocess.run([str(PY), '-m', 'unittest', 'discover', '-s', str(ROOT / folder / 'tests')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    selection = ('--kind', 'doc_teacher', '--epochs', args.finetune_epochs, '--eval-every', 1,
                 '--select', 'decode', '--select-rows', 512, '--max-hours', 1)
    train('pretrain', '--kind', PRETRAIN_KINDS, '--cap', f'local_rule:drop_article={args.drop_article_cap}',
          '--epochs', args.pretrain_epochs, '--eval-every', 1, '--warmup-steps', 1000, '--max-hours', 2)
    decode('pretrain', JOB / 'pretrain/last.pt')
    train('finetune', *selection, '--init-from', JOB / 'pretrain/last.pt', '--lr', 3e-4, '--warmup-steps', 100)
    decode('finetune', JOB / 'finetune/best.pt')
    names = ['pretrain', 'finetune']
    if not args.skip_control:
        train('control', *selection)
        decode('control', JOB / 'control/best.pt')
        names.append('control')
    report = {name: json.loads((JOB / f'{name}-decode.json').read_text())['counts'] for name in names}
    report['smoke_v2'] = json.loads((OUT / 'canvas-smoke-v2/canvas-decode.json').read_text())['counts']
    report['doc_only_tagger'] = json.loads((OUT / 'canvas-smoke-v2/tagger-decode.json').read_text())['counts']
    report['training'] = {name: json.loads((JOB / name / 'status.json').read_text()) for name in names}
    report['limits'] = ['Documentation pairs are teacher candidates passed by gates and two model judges, not owner gold.',
                        'best.pt is selected by decoding held-out documentation train groups; dev is report-only.',
                        'The canvas editor is about 8M parameters, the doc-only tagger 27.9M.']
    (JOB / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    status('complete', 'complete', comparison=str(JOB / 'comparison.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if JOB is None:
            raise
        previous = json.loads((JOB / 'status.json').read_text()) if (JOB / 'status.json').exists() else {}
        status(previous.get('stage', 'startup'), 'failed', error_type=type(exc).__name__, error=str(exc))
        raise
