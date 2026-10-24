"""Build reproducible, checker-filtered training splits from one judged pod run.

  python asd-ste100/data/build_pairs.py --run-dir asd-ste100/data/out/full1/ste-out \
      --out asd-ste100/data/out/pairs-v1

Dictionary examples stay evaluation-only. Output contains surface tokens, Levenshtein
edits and fixed canvases in 64/128/256 buckets; model vocabulary IDs are a later step.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

from noise_rules import Noiser, ROOT
from text_utils import normalized, protected_values, tokenize

sys.path.insert(0, str(ROOT / 'checker'))
from checker import Lexicon, check, errors  # noqa: E402

BUCKETS = (64, 128, 256)
PREAMBLE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:"
    r"(?:style|rewrite|rewritten text|output|verbose(?: and formal)?|casual(?: and chatty)?|"
    r"terse|bureaucratic|a quick (?:note|message)(?: from[^:\n.]*)?)\s*[:\n]"
    r"|here['’]s? (?:the text|a (?:quick|rewritten)|the rewrite)"
    r"|here is (?:the|a) (?:text|rewrite|rewritten|quick)"
    r"|(?:verbose|casual|terse|bureaucratic) style\b)", re.I)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def align(source: list[str], target: list[str]) -> tuple[list[list], list[int], int]:
    """Unit-cost Levenshtein alignment; ties prefer substitution, then delete, then insert.

    Each edit is [operation, source_cursor, target_cursor]. Target indices are strictly
    increasing and anchored at 2*i for matching/replaced source tokens where possible.
    Insertions can push later targets right; this is accounted for in bucket selection.
    """
    n, m = len(source), len(target)
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = source[i - 1] != target[j - 1]
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j = n, m
    edits = []
    while i or j:
        if i and j and dp[i][j] == dp[i - 1][j - 1] + (source[i - 1] != target[j - 1]):
            edits.append(['keep' if source[i - 1] == target[j - 1] else 'replace', i - 1, j - 1])
            i, j = i - 1, j - 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            edits.append(['delete', i - 1, j])
            i -= 1
        else:
            edits.append(['insert', i, j - 1])
            j -= 1
    edits.reverse()
    positions = []
    previous = -1
    for operation, i, j in edits:
        if operation == 'delete':
            continue
        preferred = 2 * i if operation != 'insert' else max(0, 2 * i - 1)
        previous = max(previous + 1, preferred)
        positions.append(previous)
    required = max(2 * n, previous + 1)
    return edits, positions, required


class Groups:
    """Connected components by origin ID and by any identical input/output text."""
    def __init__(self):
        self.parents = {}

    def find(self, key):
        self.parents.setdefault(key, key)
        parent = key
        while self.parents[parent] != parent:
            parent = self.parents[parent]
        while key != parent:
            next_key = self.parents[key]
            self.parents[key] = parent
            key = next_key
        return parent

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def assign_splits(rows: list[dict], seed: int, dev_fraction: float, test_fraction: float):
    groups = Groups()
    for row in rows:
        origin = 'id:' + row['origin_id']
        for text in (row['src'], row['ste']):
            groups.join(origin, 'text:' + normalized(text))
    for row in rows:
        group = groups.find('id:' + row['origin_id'])
        group_hash = digest(f'{seed}\0{group}')
        fraction = int(group_hash[:16], 16) / 2**64
        row['split'] = 'test' if fraction < test_fraction else 'dev' if fraction < test_fraction + dev_fraction else 'train'
        row['group'] = group_hash[:20]


def build(run_dir: Path, out_dir: Path, rule_variants: int = 2, seed: int = 0,
          dev_fraction: float = .05, test_fraction: float = .05,
          lex_dir: Path = ROOT / 'lexicon') -> dict:
    if rule_variants < 0 or min(dev_fraction, test_fraction) < 0 or dev_fraction + test_fraction >= 1:
        raise ValueError('invalid variants or split fractions')
    # Fail before writing any outputs if a required source is absent or malformed.
    judged_path, messy_path = run_dir / 'judged.jsonl', run_dir / 'messy.jsonl'
    judged, messy = read_jsonl(judged_path), read_jsonl(messy_path)
    if out_dir.resolve() == run_dir.resolve():
        raise ValueError('output directory must differ from the pod run directory')
    gold_path = lex_dir / 'pairs.jsonl'
    gold = {normalized(row[field]) for row in read_jsonl(gold_path) for field in ('ste', 'non_ste')}
    noiser, lex = Noiser(lex_dir), Lexicon(lex_dir)
    counts, rejects, candidates = Counter(), [], []
    eligible, seen_ids = {}, set()

    def reject(kind, origin, reason, **extra):
        counts[f'rejected.{kind}.{reason}'] += 1
        rejects.append({'kind': kind, 'origin_id': origin, 'reason': reason, **extra})

    for row in sorted(judged, key=lambda r: (str(r['id']), json.dumps(r, sort_keys=True))):
        origin = str(row['id'])
        if origin in seen_ids:
            # Ambiguous resume output must be fixed at the source, not selected by file order.
            raise ValueError(f'duplicate judged id: {origin}')
        seen_ids.add(origin)
        mode, target, source = row.get('mode'), row.get('ste', ''), row.get('src', '')
        if row.get('score') != 5 or row.get('changed') is not False:
            reject('target', origin, 'judge_not_faithful')
            continue
        if mode not in {'PROC', 'DESC'} or not target.strip() or not source.strip():
            reject('target', origin, 'invalid')
            continue
        if normalized(target) in gold:
            reject('target', origin, 'gold_overlap')
            continue
        if PREAMBLE.search(target):
            reject('target', origin, 'preamble')
            continue
        names = {w.lower() for w in re.findall(r'(?<=[a-z,] )[A-Z][a-z]+', source)}
        violations = [v for para in re.split(r'\n\s*\n', target) for v in errors(check(para, lex, mode, extra_allowed=names))]
        if violations:
            reject('target', origin, 'checker', rules=sorted({v.rule for v in violations}))
            continue
        eligible[origin] = row

    def add(row, source, kind, noise=None):
        origin, target, mode = str(row['id']), row['ste'], row['mode']
        if not isinstance(source, str) or not source.strip():
            reject(kind, origin, 'empty')
            return
        if PREAMBLE.search(source):
            reject(kind, origin, 'preamble')
            return
        if normalized(source) in gold:
            reject(kind, origin, 'gold_overlap')
            return
        if normalized(source) == normalized(target):
            reject(kind, origin, 'unchanged')
            return
        if protected_values(source) != protected_values(target):
            reject(kind, origin, 'protected_mismatch')
            return
        ratio = len(source.split()) / max(1, len(target.split()))
        if not .3 <= ratio <= 3:
            reject(kind, origin, 'length_ratio')
            return
        src_tokens, tgt_tokens = tokenize(source), tokenize(target)
        if 2 * len(src_tokens) > BUCKETS[-1] or len(tgt_tokens) > BUCKETS[-1]:
            reject(kind, origin, 'oversize')
            return
        src_values, tgt_values = [t.text for t in src_tokens], [t.text for t in tgt_tokens]
        edits, positions, required = align(src_values, tgt_values)
        bucket = next((size for size in BUCKETS if size >= required), None)
        if bucket is None:
            reject(kind, origin, 'canvas_overflow')
            return
        source_canvas, target_canvas = [None] * bucket, [None] * bucket
        source_canvas[:2 * len(src_values):2] = src_values
        for position, token in zip(positions, tgt_values):
            target_canvas[position] = token
        pair_key = '\0'.join((mode, normalized(source), normalized(target)))
        candidates.append({
            'id': digest(pair_key)[:24], 'origin_id': origin, 'kind': kind, 'noise': noise,
            'mode': mode, 'src': source, 'ste': target,
            'meaning_review': 'teacher_target_score5' if kind == 'original' else 'unjudged_' + kind,
            'src_tokens': src_values, 'tgt_tokens': tgt_values,
            'src_offsets': [[t.start, t.end] for t in src_tokens],
            'tgt_offsets': [[t.start, t.end] for t in tgt_tokens],
            'protected_src': [i for i, t in enumerate(src_tokens) if t.protected],
            'protected_tgt': [i for i, t in enumerate(tgt_tokens) if t.protected],
            'edits': edits, 'target_positions': positions, 'bucket': bucket,
            'source_canvas': source_canvas, 'target_canvas': target_canvas,
        })

    for origin, row in sorted(eligible.items()):
        add(row, row['src'], 'original')
        for variant in noiser.variants(row['ste'], row['mode'], rule_variants, seed):
            add(row, variant['src'], 'rule', variant['noise'])
    for row in sorted(messy, key=lambda r: (str(r['id']), json.dumps(r, sort_keys=True))):
        origin = str(row['id'])
        target = eligible.get(origin)
        if target is None:
            reject('messy', origin, 'ineligible_target')
        elif row.get('ste') != target['ste'] or row.get('mode') != target['mode']:
            reject('messy', origin, 'target_mismatch')
        else:
            add(target, row.get('messy', ''), 'messy', row.get('style'))

    # Exclude conflicting supervision; prefer original provenance for identical pairs.
    labels = defaultdict(set)
    for row in candidates:
        labels[(row['mode'], normalized(row['src']))].add(normalized(row['ste']))
    rows, seen = [], set()
    rank = {'original': 0, 'messy': 1, 'rule': 2}
    # Link duplicates too, so their other variants cannot leak into another split.
    assign_splits(candidates, seed, dev_fraction, test_fraction)
    for row in sorted(candidates, key=lambda r: (rank[r['kind']], r['id'], r['origin_id'])):
        if len(labels[(row['mode'], normalized(row['src']))]) > 1:
            reject(row['kind'], row['origin_id'], 'conflicting_target')
        elif row['id'] in seen:
            reject(row['kind'], row['origin_id'], 'duplicate')
        else:
            seen.add(row['id'])
            rows.append(row)
    rows.sort(key=lambda r: r['id'])
    inputs = [judged_path, messy_path, gold_path, lex_dir / 'lexicon.jsonl',
              lex_dir / 'technical.jsonl', lex_dir / 'tech_words.txt']
    summary = {
        'schema': 'ste-surface-pairs-v1', 'seed': seed, 'rule_variants': rule_variants,
        'dev_fraction': dev_fraction, 'test_fraction': test_fraction,
        'input_judged': len(judged), 'input_messy': len(messy), 'eligible_targets': len(eligible),
        'pairs': len(rows), 'splits': dict(Counter(r['split'] for r in rows)),
        'kinds': dict(Counter(r['kind'] for r in rows)),
        'buckets': dict(Counter(str(r['bucket']) for r in rows)),
        'modes': dict(Counter(r['mode'] for r in rows)),
        'noise': dict(Counter(r['noise'] for r in rows if r['kind'] == 'rule')),
        'rejections': dict(sorted(counts.items())),
        'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.exists()},
        'code_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                        (Path(__file__), ROOT / 'data/noise_rules.py', ROOT / 'data/text_utils.py', ROOT / 'checker/checker.py')},
        'limits': ['checker covers a subset of STE', 'surface tokens are not final model vocabulary IDs',
                   'messy back-translations and rule noise have no meaning judge score',
                   'protected-span heuristics do not detect every proper name or unit'],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'dev', 'test'):
        write_jsonl(out_dir / f'{split}.jsonl', (row for row in rows if row['split'] == split))
    write_jsonl(out_dir / 'rejected.jsonl', rejects)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rule-variants', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dev-fraction', type=float, default=.05)
    ap.add_argument('--test-fraction', type=float, default=.05)
    args = ap.parse_args()
    try:
        result = build(args.run_dir, args.out, args.rule_variants, args.seed, args.dev_fraction, args.test_fraction)
    except ValueError as exc:
        ap.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
