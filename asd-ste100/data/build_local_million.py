"""Build a large, explicitly checker-only corpus with free local rule noising.

Uses a disk-backed staging database and sharded output. No teacher or judge is called.
Raw public text is a target only when it passes the configured checker/coverage gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3

from build_pairs import BUCKETS, Groups, PREAMBLE, ROOT, align, digest, read_jsonl
from checker import Lexicon, check
from harvest_sources import IMPERATIVE, TECHNICAL, chunks, records, sha256, texts
from noise_rules import Noiser
from text_utils import normalized, protected_values, tokenize

SAFE_OPERATIONS = {'drop_article', 'merge_sentences', 'verbose', 'perfect_tense', 'passive', 'nominalize'}


def raw_sources(source_dir):
    """Interleave pinned raw files directly, avoiding a second multi-GB text cache."""
    manifest = json.loads((source_dir / 'download_manifest.json').read_text(encoding='utf-8'))
    entries = {e['key']: e for e in manifest['datasets']}

    def dataset_rows(key):
        entry = entries[key]
        for item in manifest['files']:
            if item['dataset'] != key:
                continue
            path = source_dir / item['path']
            if sha256(path) != item.get('sha256'):
                raise ValueError(f'raw file checksum differs: {path}')
            for index, row in enumerate(records(path)):
                doc = entry['repo'] + ':' + digest(json.dumps(row, sort_keys=True, ensure_ascii=False))[:24]
                for role, text in texts(row, entry['format']):
                    for chunk in chunks(text):
                        if not 5 <= len(chunk.split()) <= 60 or len(chunk) > 4000 or len(tokenize(chunk)) * 2 > 256:
                            continue
                        letters = [c for c in chunk if c.isalpha()]
                        if not letters or sum(c.isascii() for c in letters) / len(letters) < .95:
                            continue
                        mode = 'PROC' if role in {'user', 'system'} or IMPERATIVE.search(chunk) else 'DESC'
                        provenance = {'dataset': entry['repo'], 'revision': entry['revision'], 'license': entry['license'],
                                      'url': entry['card'], 'file': item['name'], 'record': index, 'role': role,
                                      'domain': entry['domain'], 'document_id': doc}
                        yield digest(normalized(chunk))[:24], chunk, mode, key, json.dumps(provenance), [doc]

    iterators = {key: iter(dataset_rows(key)) for key in sorted(entries)}
    schedule = [key for key in sorted(entries) for _ in range(entries[key]['export_weight'])]
    while schedule:
        for key in schedule[:]:
            try:
                yield next(iterators[key])
            except StopIteration:
                schedule = [value for value in schedule if value != key]


def align_fast(source, target):
    """Trim equal edges before dynamic programming; retain an optimal global edit path."""
    prefix = 0
    while prefix < min(len(source), len(target)) and source[prefix] == target[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(source), len(target)) - prefix and source[-suffix - 1] == target[-suffix - 1]:
        suffix += 1
    ns, nt = len(source) - suffix, len(target) - suffix
    middle, _, _ = align(source[prefix:ns], target[prefix:nt])
    edits = [['keep', i, i] for i in range(prefix)]
    edits += [[op, i + prefix, j + prefix] for op, i, j in middle]
    edits += [['keep', ns + i, nt + i] for i in range(suffix)]
    positions, previous = [], -1
    for op, i, j in edits:
        if op != 'delete':
            previous = max(previous + 1, 2 * i if op != 'insert' else max(0, 2 * i - 1))
            positions.append(previous)
    return edits, positions, max(2 * len(source), previous + 1)


def target_quality(text, mode, lex, max_unknown=.2):
    if PREAMBLE.search(text):
        return 'preamble'
    tokens = tokenize(text)
    # Do not turn dangling headings or unfenced program fragments into clean prose targets.
    prose = ''.join(' ' if t.protected else t.text + ' ' for t in tokens)
    if text.rstrip().endswith(':') or re.match(r'^\s*(?:Step\s+\d+|Input|Output|Example|Solution|Explanation)\s*:', text, re.I):
        return 'fragment'
    if re.search(r'[{};=]|^\s*[!#]|\b(?:CALL|def|import|include)\b', prose):
        return 'code_fragment'
    if not 5 <= len(text.split()) <= 60 or 2 * len(tokens) > 256:
        return 'length'
    violations = check(text, lex, mode)
    if any(v.severity in {'error', 'warn'} for v in violations):
        return 'checker'
    words = [t for t in tokens if not t.protected and t.text.isalpha()]
    if not words:
        return 'no_prose'
    unknown = sum(t.text.lower() not in lex.allowed and t.text.lower() not in lex.technical for t in words)
    if unknown / len(words) > max_unknown:
        return 'unknown_coverage'
    return None


def initialize_stage(source_dir, output, seed, variants, max_unknown, exclude):
    output.mkdir(parents=True, exist_ok=True)
    config = {'sources_manifest': sha256(source_dir / 'download_manifest.json'), 'seed': seed,
              'variants': variants, 'max_unknown': max_unknown, 'safe_operations': sorted(SAFE_OPERATIONS),
              'source_directory': str(source_dir.resolve()), 'source_mode': 'stream_pinned_raw',
              'lexicon_sha256': sha256(ROOT / 'lexicon/lexicon.jsonl'),
              'code_sha256': {p.name: sha256(p) for p in [Path(__file__), ROOT / 'data/noise_rules.py', ROOT / 'data/text_utils.py', ROOT / 'checker/checker.py']},
              'exclude': str(exclude.resolve()) if exclude else None}
    config_path = output / 'build_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('build configuration changed; choose a new output directory')
    config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    db = sqlite3.connect(output / 'pairs.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS pairs (id TEXT PRIMARY KEY, origin_id TEXT, mode TEXT, src TEXT,
          ste TEXT, noise TEXT, dataset TEXT, documents TEXT, provenance TEXT);
        CREATE TABLE IF NOT EXISTS processed (origin_id TEXT PRIMARY KEY, reason TEXT);
        CREATE TABLE IF NOT EXISTS reserved (text_hash TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS source_labels (mode TEXT, source_hash TEXT, target_hash TEXT,
          PRIMARY KEY(mode,source_hash));
        CREATE TABLE IF NOT EXISTS target_documents (origin_id TEXT, document_id TEXT,
          PRIMARY KEY(origin_id,document_id));
    ''')
    for row in read_jsonl(ROOT / 'lexicon/pairs.jsonl'):
        for field in ('ste', 'non_ste'):
            db.execute('INSERT OR IGNORE INTO reserved VALUES (?)', (digest(normalized(row[field])),))
    if exclude:
        for path in sorted(exclude.glob('*.jsonl')):
            if path.name not in {'train.jsonl', 'dev.jsonl', 'test.jsonl'}:
                continue
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    for field in ('src', 'ste'):
                        db.execute('INSERT OR IGNORE INTO reserved VALUES (?)', (digest(normalized(row[field])),))
    db.commit()
    return db


def stage(source_dir: Path, output: Path, n=1_000_000, variants=6, seed=0, max_unknown=.2, exclude=None):
    db = initialize_stage(source_dir, output, seed, variants, max_unknown, exclude)
    lex, noiser = Lexicon(), Noiser()
    reserved = {r[0] for r in db.execute('SELECT text_hash FROM reserved')}
    done = {r[0] for r in db.execute('SELECT origin_id FROM processed')}
    count = db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
    mode_counts = Counter(dict(db.execute('SELECT mode,COUNT(*) FROM pairs GROUP BY mode')))
    dataset_counts = Counter(dict(db.execute('SELECT dataset,COUNT(*) FROM pairs GROUP BY dataset')))
    accepted_origins = {r[0] for r in db.execute('SELECT DISTINCT origin_id FROM pairs')}
    max_mode, max_dataset = math.ceil(n * .7), math.ceil(n * .6)
    scanned = 0
    for origin, text, mode, dataset, provenance, docs in raw_sources(source_dir):
        if count >= n:
            break
        if origin in done:
            if origin in accepted_origins:
                db.executemany('INSERT OR IGNORE INTO target_documents VALUES (?,?)', [(origin, doc) for doc in docs])
            continue
        scanned += 1
        if mode_counts[mode] >= max_mode or dataset_counts[dataset] >= max_dataset:
            # Quota skips are not permanently marked: a later larger build may use them.
            continue
        done.add(origin)
        reason = 'reserved' if digest(normalized(text)) in reserved else target_quality(text, mode, lex, max_unknown)
        if reason:
            db.execute('INSERT INTO processed VALUES (?,?)', (origin, reason))
        else:
            generated = noiser.variants(text, mode, variants, seed, SAFE_OPERATIONS)
            accepted = 0
            for variant in generated:
                noisy = variant['src']
                if PREAMBLE.search(noisy) or digest(normalized(noisy)) in reserved or len(tokenize(noisy)) * 2 > 256:
                    continue
                if protected_values(text) != protected_values(noisy):
                    continue
                source_hash, target_hash = digest(normalized(noisy)), digest(normalized(text))
                previous = db.execute('SELECT target_hash FROM source_labels WHERE mode=? AND source_hash=?', (mode, source_hash)).fetchone()
                if previous and previous[0] != target_hash:
                    # Multiple clean phrasings can yield the same corruption. Keep one stable label.
                    continue
                pair_id = digest('\0'.join((mode, normalized(noisy), normalized(text))))[:24]
                cur = db.execute('INSERT OR IGNORE INTO pairs VALUES (?,?,?,?,?,?,?,?,?)',
                                 (pair_id, origin, mode, noisy, text, variant['noise'], dataset, json.dumps(docs), provenance))
                db.execute('INSERT OR IGNORE INTO source_labels VALUES (?,?,?)', (mode, source_hash, target_hash))
                accepted += cur.rowcount; count += cur.rowcount
                mode_counts[mode] += cur.rowcount; dataset_counts[dataset] += cur.rowcount
                if count >= n or mode_counts[mode] >= max_mode or dataset_counts[dataset] >= max_dataset:
                    break
            db.execute('INSERT INTO processed VALUES (?,?)', (origin, 'accepted' if accepted else 'no_safe_noise'))
            if accepted:
                accepted_origins.add(origin)
                db.executemany('INSERT OR IGNORE INTO target_documents VALUES (?,?)', [(origin, doc) for doc in docs])
        if scanned % 1000 == 0:
            db.commit()
        if scanned % 20000 == 0:
            print(f'scanned {scanned:,} new sources; staged {count:,}/{n:,} pairs', flush=True)
    db.commit()
    summary = {'requested_pairs': n, 'staged_pairs': count, 'modes': dict(mode_counts),
               'datasets': dict(dataset_counts), 'target_outcomes': dict(db.execute('SELECT reason,COUNT(*) FROM processed GROUP BY reason')),
               'targets': db.execute('SELECT COUNT(DISTINCT origin_id) FROM pairs').fetchone()[0],
               'noise': dict(db.execute('SELECT noise,COUNT(*) FROM pairs GROUP BY noise')),
               'status': 'ready' if count >= n else 'shortfall', 'teacher_calls': 0, 'judge_calls': 0}
    (output / 'stage_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    db.close()
    return summary


def encoded_pair(item):
    pair_id, origin, mode, src, tgt, noise, dataset, docs, provenance, split, group = item
    st, tt = tokenize(src), tokenize(tgt)
    sv, tv = [t.text for t in st], [t.text for t in tt]
    edits, positions, required = align_fast(sv, tv)
    bucket = next((b for b in BUCKETS if b >= required), None)
    if bucket is None:
        raise ValueError(f'{pair_id}: canvas overflow after staging')
    source_canvas, target_canvas = [None] * bucket, [None] * bucket
    source_canvas[:2 * len(sv):2] = sv
    for p, token in zip(positions, tv):
        target_canvas[p] = token
    row = {'id': pair_id, 'origin_id': origin, 'mode': mode, 'src': src, 'ste': tgt,
           'kind': 'local_rule', 'noise': noise, 'target_review': 'checker_only',
           'meaning_review': 'unjudged_rule', 'provenance': json.loads(provenance),
           'document_ids': json.loads(docs), 'split': split, 'group': group,
           'src_tokens': sv, 'tgt_tokens': tv, 'src_offsets': [[t.start,t.end] for t in st],
           'tgt_offsets': [[t.start,t.end] for t in tt], 'protected_src': [i for i,t in enumerate(st) if t.protected],
           'protected_tgt': [i for i,t in enumerate(tt) if t.protected], 'edits': edits,
           'target_positions': positions, 'source_canvas': source_canvas, 'target_canvas': target_canvas, 'bucket': bucket}
    return split, bucket, json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'


def encode_batch(batch):
    return [encoded_pair(row) for row in batch]


def export(output: Path, seed=0, workers=4, shard_size=25000):
    db = sqlite3.connect(output / 'pairs.sqlite3')
    groups = Groups()
    target_docs = {}
    for origin, doc in db.execute('SELECT origin_id,document_id FROM target_documents ORDER BY origin_id,document_id'):
        target_docs.setdefault(origin, []).append(doc)
    print('Grouping documents and shared texts before split assignment', flush=True)
    for origin, src, tgt, docs in db.execute('SELECT origin_id,src,ste,documents FROM pairs ORDER BY id'):
        anchor = 'id:' + origin
        groups.join(anchor, 'text:' + digest(normalized(src)))
        groups.join(anchor, 'text:' + digest(normalized(tgt)))
        for doc in target_docs.get(origin, json.loads(docs)):
            groups.join(anchor, 'doc:' + doc)
    # A source that has different targets would give conflicting supervision. Fail explicitly.
    labels = {}
    for mode, src, tgt in db.execute('SELECT mode,src,ste FROM pairs'):
        key = (mode, digest(normalized(src)))
        target = digest(normalized(tgt))
        if labels.setdefault(key, target) != target:
            raise ValueError('conflicting target labels in staged pairs')
    del labels
    group_info = {}
    for (origin,) in db.execute('SELECT DISTINCT origin_id FROM pairs'):
        group = digest(f'{seed}\0{groups.find("id:" + origin)}')
        fraction = int(group[:16], 16) / 2**64
        split = 'test' if fraction < .05 else 'dev' if fraction < .1 else 'train'
        group_info[origin] = (split, group[:20])
    del groups

    def batches():
        batch = []
        for row in db.execute('SELECT * FROM pairs ORDER BY id'):
            split, group = group_info[row[1]]
            row = (*row[:7], json.dumps(target_docs.get(row[1], json.loads(row[7]))), row[8])
            batch.append((*row, split, group))
            if len(batch) == 256:
                yield batch
                batch = []
        if batch:
            yield batch

    streams, paths, counts, bucket_counts = {}, [], Counter(), Counter()
    shard_dir = output / 'shards'
    shard_dir.mkdir(exist_ok=True)

    def write(results):
        for split, bucket, line in results:
            if counts[split] % shard_size == 0:
                if split in streams:
                    streams[split].close()
                path = shard_dir / f'{split}-{counts[split] // shard_size:05d}.jsonl'
                paths.append(path)
                streams[split] = path.open('w', encoding='utf-8', newline='\n')
            streams[split].write(line)
            counts[split] += 1; bucket_counts[str(bucket)] += 1
        if sum(counts.values()) % 25600 == 0:
            print(f'exported {sum(counts.values()):,} aligned pairs', flush=True)

    try:
        if workers == 1:
            for batch in batches():
                write(encode_batch(batch))
        else:
            # Bounded batches avoid a future per pair or copying the entire corpus into memory.
            iterator = iter(batches())
            with ProcessPoolExecutor(max_workers=workers) as pool:
                while True:
                    wave = []
                    for _ in range(workers * 2):
                        batch = next(iterator, None)
                        if batch is None:
                            break
                        wave.append(pool.submit(encode_batch, batch))
                    if not wave:
                        break
                    for future in wave:
                        write(future.result())
    finally:
        for stream in streams.values():
            stream.close()
    summary = {'schema': 'ste-local-surface-pairs-v1', 'pairs': sum(counts.values()), 'splits': dict(counts),
               'buckets': dict(bucket_counts), 'seed': seed, 'shard_size': shard_size,
               'target_review': 'checker_only', 'meaning_review': 'unjudged_rule',
               'shards': [{'path': str(p.relative_to(output)), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(paths)],
               'stage': json.loads((output / 'stage_summary.json').read_text(encoding='utf-8')),
               'limitations': ['not human-verified or meaning-judged STE', 'checker implements only a subset of the rules',
                               'up to 20% unknown prose words are allowed and need future protection',
                               'synthetic source datasets do not establish factual correctness', 'surface tokens still need model vocabulary IDs']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    db.close()
    return {k:v for k,v in summary.items() if k != 'shards'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['stage', 'export'])
    ap.add_argument('--sources', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=1_000_000)
    ap.add_argument('--variants', type=int, default=6)
    ap.add_argument('--max-unknown', type=float, default=.2)
    ap.add_argument('--exclude', type=Path, default=ROOT / 'data/out/pairs-v1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--shard-size', type=int, default=25000)
    args = ap.parse_args()
    if min(args.n, args.variants, args.workers, args.shard_size) <= 0 or not 0 <= args.max_unknown <= 1:
        ap.error('invalid build limits')
    if args.command == 'stage' and args.sources is None:
        ap.error('--sources is required for staging')
    result = stage(args.sources, args.out, args.n, args.variants, args.seed, args.max_unknown, args.exclude) if args.command == 'stage' else export(args.out, args.seed, args.workers, args.shard_size)
    print(json.dumps(result, indent=2))
