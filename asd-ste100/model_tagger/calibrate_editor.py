"""Dev-only confidence calibration and reproducible diagnostics for the safe editor."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from tagger import Config, Tagger, SteTokenizer, Lexicon, rewrite as raw_rewrite
from safe_editor import rewrite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sample', type=int, default=256)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--locked-policy', type=Path)
    p.add_argument('--margins', type=lambda value: tuple(float(x) for x in value.split(',')), default=(0.,2.,4.,6.))
    args = p.parse_args()
    torch.set_num_threads(4)
    tok = SteTokenizer(args.vocabulary)
    checkpoint_path = args.checkpoint or args.vocabulary / 'model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['vocab_sha256'] != hashlib.sha256((args.vocabulary/'vocab.json').read_bytes()).hexdigest():
        raise ValueError('checkpoint/vocabulary fingerprint mismatch')
    model = Tagger(tok, Config(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'])
    lex = Lexicon()
    rows = []
    for path in sorted((args.pairs / 'shards').glob('dev-*.jsonl.gz')):
        with gzip.open(path, 'rt') as f:
            for line in f:
                row = json.loads(line)
                if row['split'] != 'dev':
                    raise ValueError('non-dev calibration row')
                rows.append(row)
    rows.sort(key=lambda row: hashlib.sha256(('safe-editor-v1/' + str(row['id'])).encode()).digest())
    selected = rows[args.offset:args.offset + args.sample]
    if not selected or any(not (0 <= x < float('inf')) for x in args.margins):
        raise ValueError('empty dev sample or invalid margins')
    results, predictions = [], []
    margins = (json.loads(args.locked_policy.read_text())['edit_margin'],) if args.locked_policy else args.margins
    for margin in margins:
        counts = dict(margin=margin, n=len(selected), exact=0, clean_changed=0,
                      noisy_resolved=0, clean_resolved=0, retries=0, identity_exact=sum(r['src'] == r['ste'] for r in selected))
        for row in selected:
            for clean in (False, True):
                source = row['ste'] if clean else row['src']
                result = rewrite(model, tok, lex, source, row['mode'], edit_margin=margin)
                counts['clean_changed' if clean else 'exact'] += (result['text'] != source if clean else result['text'] == row['ste'])
                counts['clean_resolved' if clean else 'noisy_resolved'] += result['resolved']
                counts['retries'] += result['passes'] > 1
                predictions.append(dict(id=row['id'], clean=clean, margin=margin, source=source,
                                        target=row['ste'], **result))
        counts['clean_change_rate'] = counts['clean_changed'] / len(selected)
        results.append(counts)
        print(json.dumps(counts), flush=True)
    eligible = [r for r in results if r['clean_change_rate'] <= .05]
    chosen = max(eligible, key=lambda r: (r['exact'], -r['clean_changed'], r['margin'])) if eligible else None
    diagnostics = []
    prompts = ['Open file.', 'Use the `logging` module.', 'Utilize the existing cache.',
               'Do not delete `config.json`.', 'Set `timeout_ms` to 500.',
               'If the test fails, stop. Otherwise, continue.',
               'Return an empty list if the file is missing.',
               'Write a C++ program.', 'Use foo(x).', 'Use ``a ` b``.',
               'Copy /tmp/input.txt to /tmp/output.txt.']
    if chosen:
        for text in prompts:
            diagnostics.append(dict(source=text, **rewrite(model, tok, lex, text, edit_margin=chosen['margin'])))
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(scope='dev-only SHA sample; synthetic-heavy, not owner gold',
                  locked_policy=str(args.locked_policy) if args.locked_policy else None, offset=args.offset,
                  sample_ids=[r['id'] for r in selected], candidates=results, chosen=chosen,
                  diagnostics=diagnostics, checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
    (args.output/'calibration.json').write_text(json.dumps(report, indent=2) + '\n')
    with gzip.open(args.output/'predictions.jsonl.gz', 'wt') as f:
        for row in predictions:
            f.write(json.dumps(row) + '\n')
    if chosen:
        (args.output/'policy.json').write_text(json.dumps(dict(edit_margin=chosen['margin'], max_passes=5,
                                                            calibration='calibration.json', checkpoint_sha256=report['checkpoint_sha256'],
                                                            vocab_sha256=checkpoint['vocab_sha256']), indent=2)+'\n')


if __name__ == '__main__':
    main()
