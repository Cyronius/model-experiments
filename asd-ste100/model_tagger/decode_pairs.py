"""Raw (unguarded) decode of the dev pairs of an ID export: how close the model's rewrite comes to the reference.

For every dev row it runs tagger.rewrite (the checker-stopped edit loop, no confidence
margin or content guard) on the source and on the reference. Counts: exact reference,
source unchanged, closer to / farther from the reference than the source (word-level
difflib distance), reference changed (a clean input edited), no output (the source does
not encode), guard_flagged (safe_editor.validate finds an issue) and checker_clean
outputs. Diagnostic only; selection stays with calibrate_editor.py.
"""
import argparse
from collections import Counter
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import torch
from tagger import Config, Tagger, SteTokenizer, Lexicon, rewrite, EncodingError
from ste_tokenizer import model_tokens
from safe_editor import validate


def words(text):
    return [t.text for t in model_tokens(text)]


def distance(a, b):
    return 1 - difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


def decode(model, tok, lex, row):
    try:
        out = rewrite(model, tok, lex, row['src'], row['mode'])
    except EncodingError:
        return None, False
    return (out.get('text'), out.get('resolved', False)) if isinstance(out, dict) else (out, False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(4)
    tok = SteTokenizer(args.vocabulary)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['vocab_sha256'] != hashlib.sha256((args.vocabulary / 'vocab.json').read_bytes()).hexdigest():
        raise ValueError('checkpoint/vocabulary fingerprint mismatch')
    model = Tagger(tok, Config(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    lex = Lexicon()
    rows = []
    for path in sorted((args.pairs / 'shards').glob('dev-*.jsonl.gz')):
        with gzip.open(path, 'rt') as f:
            rows.extend(json.loads(line) for line in f)
    counts, samples = Counter(), []
    with torch.no_grad():
        for row in rows:
            text, resolved = decode(model, tok, lex, row)
            if text is None:
                counts['no_output'] += 1
                continue
            counts['exact'] += text == row['ste']
            counts['unchanged'] += text == row['src']
            before, after = distance(row['src'], row['ste']), distance(text, row['ste'])
            counts['closer_to_reference'] += after < before
            counts['farther'] += after > before
            counts['guard_flagged'] += bool(validate(row['src'], text))
            counts['checker_clean'] += bool(resolved)
            clean, _ = decode(model, tok, lex, {**row, 'src': row['ste']})
            counts['clean_changed'] += clean != row['ste']
            if len(samples) < args.samples and text not in (row['src'], row['ste']):
                samples.append(dict(id=row['id'], source=row['src'], output=text, reference=row['ste']))
    report = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  pairs=str(args.pairs), n=len(rows), counts=dict(counts), samples=samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
