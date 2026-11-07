"""Decode the dev pairs of an ID export with the canvas editor; same counts as model_tagger/decode_pairs.py.

For every dev row it rewrites the source and the reference. Counts: exact reference,
source unchanged, closer to / farther from the reference than the source (word-level
difflib distance), reference changed (a clean input edited), no output (encoding or
slot-copy failure); plus guard_flagged (safe_editor.validate finds an issue: a changed
content word, literal or duplicated article) and checker_clean outputs.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import torch
from canvas import CanvasEditor, Config, SteTokenizer, Lexicon, rewrite
from decode_pairs import distance
from safe_editor import validate


def dev_rows(pairs):
    rows = []
    for path in sorted((pairs / 'shards').glob('dev-*.jsonl.gz')):
        with gzip.open(path, 'rt') as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def decode_rows(model, tok, lex, rows, samples=0):
    counts, kept = Counter(), []
    for row in rows:
        result = rewrite(model, tok, lex, row['src'], row['mode'])
        text = result['text']
        if text is None:
            counts['no_output'] += 1
            continue
        counts['exact'] += text == row['ste']
        counts['unchanged'] += text == row['src']
        before, after = distance(row['src'], row['ste']), distance(text, row['ste'])
        counts['closer_to_reference'] += after < before
        counts['farther'] += after > before
        counts['guard_flagged'] += bool(validate(row['src'], text))
        counts['checker_clean'] += result['resolved']
        counts['clean_changed'] += rewrite(model, tok, lex, row['ste'], row['mode'])['text'] != row['ste']
        if len(kept) < samples and text not in (row['src'], row['ste']):
            kept.append(dict(id=row['id'], source=row['src'], output=text, reference=row['ste']))
    return dict(counts), kept


def load(checkpoint_path, vocabulary):
    tok = SteTokenizer(vocabulary)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['vocab_sha256'] != hashlib.sha256((vocabulary / 'vocab.json').read_bytes()).hexdigest():
        raise ValueError('checkpoint/vocabulary fingerprint mismatch')
    model = CanvasEditor(tok, Config(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'])
    return model.eval(), tok


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=24)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model, tok = load(args.checkpoint, args.vocabulary)
    rows, start = dev_rows(args.pairs), time.time()
    counts, samples = decode_rows(model, tok, Lexicon(), rows, args.samples)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  pairs=str(args.pairs), n=len(rows), seconds=round(time.time() - start), counts=counts, samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
