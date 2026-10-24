"""Audit all emitted pairs, including reconstruction, target filters and split leakage."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from build_pairs import BUCKETS, ROOT, digest, read_jsonl
from checker import Lexicon, check, errors
from text_utils import normalized, protected_values, tokenize


def validate(directory: Path, run_dir: Path) -> dict:
    judged = {str(r['id']): r for r in read_jsonl(run_dir / 'judged.jsonl')}
    gold = {normalized(r[field]) for r in read_jsonl(ROOT / 'lexicon/pairs.jsonl') for field in ('ste', 'non_ste')}
    lex = Lexicon()
    seen_ids, text_splits, origin_splits, group_splits = set(), {}, {}, {}
    kinds, splits, buckets = Counter(), Counter(), Counter()
    target_checks = set()
    for split in ('train', 'dev', 'test'):
        for row in read_jsonl(directory / f'{split}.jsonl'):
            pair_id = row['id']

            def require(condition, message):
                if not condition:
                    raise ValueError(f'{pair_id}: {message}')

            require(pair_id not in seen_ids, 'duplicate ID')
            seen_ids.add(pair_id)
            require(row['split'] == split, 'wrong split label')
            require(pair_id == digest('\0'.join((row['mode'], normalized(row['src']), normalized(row['ste']))))[:24], 'ID does not match content')
            original = judged[row['origin_id']]
            require(original.get('score') == 5 and original.get('changed') is False, 'target lacks faithful judge score')
            require(row['ste'] == original['ste'] and row['mode'] == original['mode'], 'target differs from judged row')
            require(normalized(row['src']) != normalized(row['ste']), 'unchanged pair')
            require(protected_values(row['src']) == protected_values(row['ste']), 'protected content differs')
            for field in ('src', 'ste'):
                text = normalized(row[field])
                require(text not in gold, 'dictionary evaluation text in training corpus')
                require(text_splits.setdefault(text, split) == split, 'text crosses splits')
            require(origin_splits.setdefault(row['origin_id'], split) == split, 'origin crosses splits')
            require(group_splits.setdefault(row['group'], split) == split, 'group crosses splits')
            if row['origin_id'] not in target_checks:
                names = {w.lower() for w in re.findall(r'(?<=[a-z,] )[A-Z][a-z]+', original['src'])}
                require(not any(errors(check(para, lex, row['mode'], extra_allowed=names))
                                for para in re.split(r'\n\s*\n', row['ste'])), 'target has checker errors')
                target_checks.add(row['origin_id'])
            for text_field, prefix in (('src', 'src'), ('ste', 'tgt')):
                toks = tokenize(row[text_field])
                require(row[prefix + '_tokens'] == [t.text for t in toks], 'tokens differ from surface text')
                require(row[prefix + '_offsets'] == [[t.start, t.end] for t in toks], 'offsets differ from surface text')
                require(row['protected_' + prefix] == [i for i, t in enumerate(toks) if t.protected], 'protected indices differ')
            bucket = row['bucket']
            require(bucket in BUCKETS, 'invalid bucket')
            src, tgt = row['src_tokens'], row['tgt_tokens']
            require(len(row['source_canvas']) == len(row['target_canvas']) == bucket, 'canvas length differs')
            require([v for v in row['source_canvas'] if v is not None] == src, 'source canvas does not reconstruct')
            require(row['source_canvas'][::2][:len(src)] == src and all(v is None for v in row['source_canvas'][1::2]), 'source not on even positions')
            require([v for v in row['target_canvas'] if v is not None] == tgt, 'target canvas does not reconstruct')
            positions = row['target_positions']
            require(len(positions) == len(tgt) and positions == sorted(set(positions)), 'bad target positions')
            require(all(row['target_canvas'][p] == t for p, t in zip(positions, tgt)), 'target placement differs')
            i = j = 0
            for operation, si, tj in row['edits']:
                require((si, tj) == (i, j), 'discontinuous alignment')
                if operation in {'keep', 'replace'}:
                    require(i < len(src) and j < len(tgt), 'alignment out of range')
                    require((src[i] == tgt[j]) == (operation == 'keep'), 'wrong keep/replace label')
                    i += 1; j += 1
                elif operation == 'delete':
                    i += 1
                elif operation == 'insert':
                    j += 1
                else:
                    require(False, 'unknown edit')
            require((i, j) == (len(src), len(tgt)), 'alignment does not cover text')
            kinds[row['kind']] += 1; splits[split] += 1; buckets[str(bucket)] += 1
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if summary['pairs'] != len(seen_ids) or summary['splits'] != dict(splits) or summary['kinds'] != dict(kinds) or summary['buckets'] != dict(buckets):
        raise ValueError('summary counts differ from dataset')
    return {'pairs_checked': len(seen_ids), 'targets_checked': len(target_checks),
            'splits': dict(splits), 'status': 'passed'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(validate(args.pairs, args.run_dir), indent=2))
