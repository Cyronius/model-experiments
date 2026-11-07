"""Combine surface corpora before tokenizer filtering; no model dependencies required."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile

from build_pairs import Groups
from text_utils import normalized

SPLITS = ('train', 'dev', 'test')
DERIVED = {'src_tokens', 'tgt_tokens', 'src_offsets', 'tgt_offsets', 'protected_src',
           'protected_tgt', 'edits', 'target_positions', 'source_canvas', 'target_canvas'}


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def records(path):
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def manifest(root):
    summary_path = root / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    items = summary.get('shards')
    if items is None:
        items = [{'path': f'{split}.jsonl'} for split in SPLITS]
    result, seen = [], set()
    for item in items:
        path = root / item['path'].replace('\\', '/')
        if not path.resolve().is_relative_to(root.resolve()) or path.resolve() in seen:
            raise ValueError('unsafe or repeated manifest path')
        seen.add(path.resolve())
        actual = sha256(path)
        if item.get('sha256', actual) != actual:
            raise ValueError(f'shard hash mismatch: {path}')
        result.append({'path': path.relative_to(root).as_posix(), 'sha256': actual})
    return {'summary_sha256': sha256(summary_path), 'shards': result}


def rows(root, info):
    for item in info['shards']:
        yield from records(root / item['path'])


def pair_key(row):
    return digest(json.dumps([row['mode'], normalized(row['src']), normalized(row['ste'])], ensure_ascii=False))


def source_key(row):
    return digest(json.dumps([row['mode'], normalized(row['src'])], ensure_ascii=False))


def links(corpus, row):
    if row.get('mode') not in ('PROC', 'DESC') or row.get('split') not in SPLITS:
        raise ValueError('invalid mode or split')
    if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('id', 'origin_id', 'src', 'ste')):
        raise ValueError('missing pair identity or text')
    if not row.get('meaning_review'):
        raise ValueError('missing meaning review label')
    if 'source_ids' in row:
        raise ValueError('joint grouping requires unfiltered surface records')
    yield 'origin:' + json.dumps([corpus, row['origin_id']])
    yield 'text:' + digest(normalized(row['src']))
    yield 'text:' + digest(normalized(row['ste']))
    docs = row.get('document_ids', [])
    if not isinstance(docs, list) or any(not isinstance(d, str) or not d for d in docs):
        raise ValueError('invalid document provenance')
    primary = row.get('provenance', {}).get('document_id')
    if primary is not None and primary not in docs:
        raise ValueError('primary document absent from document_ids')
    # Recorded local IDs already include source identity. Share them across corpora.
    for doc in docs:
        yield 'doc:' + doc


def member(corpus, row):
    return {'corpus': corpus, 'record': {k: v for k, v in row.items() if k not in DERIVED}}


def member_hash(corpus, row):
    return digest(json.dumps(member(corpus, row), sort_keys=True, ensure_ascii=False))


def build(corpora, output, seed=0, dev=.05, test=.05, shard_size=25000):
    if not corpora or len({name for name, _ in corpora}) != len(corpora):
        raise ValueError('supply unique corpus names')
    if not (0 <= dev < 1 and 0 <= test < 1 and dev + test < 1) or shard_size < 1:
        raise ValueError('invalid split fractions or shard size')
    inputs = {name: manifest(root) for name, root in sorted(corpora)}
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a new or empty output directory')
    if any(output.resolve() == root.resolve() for _, root in corpora):
        raise ValueError('output equals input')
    output.mkdir(parents=True, exist_ok=True)
    groups, scanned = Groups(), Counter()
    with tempfile.TemporaryDirectory(prefix='joint-', dir=output) as work:
        with closing(sqlite3.connect(Path(work) / 'stage.sqlite3')) as db:
            db.execute('CREATE TABLE rows (corpus TEXT, id TEXT, pair TEXT, source TEXT, target TEXT, anchor TEXT, rank INT, row TEXT, UNIQUE(corpus,id))')
            for corpus, root in sorted(corpora):
                for row in rows(root, inputs[corpus]):
                    keys = list(links(corpus, row))
                    for key in keys:
                        groups.join(keys[0], key)
                    rank = {'original': 0, 'messy': 1, 'rule': 2, 'local_rule': 3}.get(row['kind'], 4)
                    db.execute('INSERT INTO rows VALUES (?,?,?,?,?,?,?,?)',
                               (corpus, row['id'], pair_key(row), source_key(row),
                                normalized(row['ste']), keys[0], rank, json.dumps(row, ensure_ascii=False)))
                    scanned[corpus] += 1
                    if scanned[corpus] % 50000 == 0:
                        print(f'grouped {corpus}: {scanned[corpus]:,}', flush=True)
            db.execute('CREATE INDEX pair_index ON rows(pair)')
            db.execute('CREATE INDEX source_index ON rows(source)')
            db.execute('CREATE TABLE conflicts AS SELECT source FROM rows GROUP BY source HAVING COUNT(DISTINCT target)>1')
            db.execute('CREATE UNIQUE INDEX conflict_index ON conflicts(source)')
            # Component sizes include rejected/conflicting and duplicate surface records.
            counts = Counter()
            for anchor, count in db.execute('SELECT anchor,COUNT(*) FROM rows GROUP BY anchor'):
                counts[groups.find(anchor)] += count
            total = sum(scanned.values())
            large = {g for g, n in counts.items() if n > max(1, total * .02)}
            assignments = {}
            for g in counts:
                group = digest(f'{seed}\0{g}')
                fraction = int(group[:16], 16) / 2**64
                split = 'train' if g in large else 'test' if fraction < test else 'dev' if fraction < test + dev else 'train'
                assignments[g] = (group, split)
            streams, paths, splits, reviews = {}, [], Counter(), Counter()
            def close(split):
                text, raw = streams.pop(split)
                text.close()
                raw.close()
            try:
                query = 'SELECT pair FROM rows WHERE source NOT IN (SELECT source FROM conflicts) GROUP BY pair ORDER BY pair'
                for (pair,) in db.execute(query):
                    entries = list(db.execute('SELECT corpus,row,anchor FROM rows WHERE pair=? ORDER BY rank,corpus,id', (pair,)))
                    row = json.loads(entries[0][1])
                    group, split = assignments[groups.find(entries[0][2])]
                    row.update(id=pair, group=group, split=split, schema='ste-joint-surface-pairs-v1',
                               joint_members=[member(c, json.loads(r)) for c, r, _ in entries])
                    if splits[split] % shard_size == 0:
                        if split in streams:
                            close(split)
                        path = output / 'shards' / f'{split}-{splits[split] // shard_size:05d}.jsonl.gz'
                        path.parent.mkdir(exist_ok=True)
                        paths.append(path)
                        raw = path.open('wb')
                        streams[split] = (io.TextIOWrapper(gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=1), encoding='utf-8', newline='\n'), raw)
                    streams[split][0].write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                    splits[split] += 1
                    if sum(splits.values()) % 25000 == 0:
                        print(f'exported {sum(splits.values()):,} joint pairs', flush=True)
                    reviews[row['meaning_review']] += 1
            finally:
                for split in list(streams):
                    close(split)
            conflicting = db.execute('SELECT COUNT(*) FROM rows WHERE source IN (SELECT source FROM conflicts)').fetchone()[0]
            unique = db.execute('SELECT COUNT(DISTINCT pair) FROM rows WHERE source NOT IN (SELECT source FROM conflicts)').fetchone()[0]
    summary = {'schema': 'ste-joint-surface-pairs-v1', 'inputs': inputs, 'seed': seed,
               'code_sha256': sha256(Path(__file__)),
               'dev_fraction': dev, 'test_fraction': test, 'scanned': dict(scanned),
               'pairs': sum(splits.values()), 'splits': dict(splits), 'meaning_review': dict(reviews),
               'conflicting_records': conflicting, 'duplicates_merged': total - conflicting - unique,
               'components': len(counts), 'largest_component_records': max(counts.values(), default=0),
               'large_components_assigned_train': len(large),
               'shards': [{'path': p.relative_to(output).as_posix(), 'sha256': sha256(p)} for p in paths],
               'limitations': ['Surface grouping precedes tokenizer filtering; re-export IDs for these splits.',
                               'Existing tokenizer was fitted on an older train split; refit on joint train only.',
                               'Generated holdouts are not owner-verified gold; review labels are unchanged.']}
    dump(output / 'summary.json', summary)
    return summary


def validate(corpora, output):
    """Check all original links, including discarded records, against emitted assignments."""
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    actual_inputs = {c: manifest(p) for c, p in corpora}
    if actual_inputs != summary['inputs']:
        raise ValueError('input fingerprints differ')
    info = manifest(output)
    with tempfile.TemporaryDirectory(prefix='joint-audit-') as work:
        with closing(sqlite3.connect(Path(work) / 'audit.sqlite3')) as db:
            db.executescript('CREATE TABLE pairs (pair TEXT PRIMARY KEY, group_id TEXT, split TEXT);'
                             'CREATE TABLE members (corpus TEXT, id TEXT, pair TEXT, hash TEXT, payload_hash TEXT, PRIMARY KEY(corpus,id));'
                             'CREATE TABLE originals (corpus TEXT,id TEXT,pair TEXT,source TEXT,target TEXT,PRIMARY KEY(corpus,id));'
                             'CREATE TABLE links (key TEXT,pair TEXT); CREATE INDEX links_key ON links(key);')
            splits = Counter()
            for row in rows(output, info):
                key = pair_key(row)
                if row['id'] != key or row['split'] not in SPLITS or not row.get('joint_members'):
                    raise ValueError('invalid output identity/split/members')
                db.execute('INSERT INTO pairs VALUES (?,?,?)', (key, row['group'], row['split']))
                for index, item in enumerate(row['joint_members']):
                    original = item['record']
                    if pair_key(original) != key:
                        raise ValueError('member differs from canonical pair')
                    canonical = {k: v for k, v in row.items() if k != 'joint_members'}
                    for field in ('id', 'schema', 'group', 'split'):
                        if field in original:
                            canonical[field] = original[field]
                        else:
                            canonical.pop(field, None)
                    payload_hash = digest(json.dumps(canonical, sort_keys=True, ensure_ascii=False)) if index == 0 else None
                    db.execute('INSERT INTO members VALUES (?,?,?,?,?)',
                               (item['corpus'], original['id'], key, member_hash(item['corpus'], original), payload_hash))
                # Canonical data/labels must be the preferred original, unchanged.
                first = row['joint_members'][0]['record']
                metadata = {k: v for k, v in row.items() if k not in DERIVED | {'joint_members', 'schema', 'id', 'group', 'split'}}
                expected = {k: v for k, v in first.items() if k not in {'schema', 'id', 'group', 'split'}}
                if metadata != expected:
                    raise ValueError('canonical provenance or review labels changed')
                splits[row['split']] += 1
            counts = Counter()
            for corpus, root in corpora:
                for row in rows(root, actual_inputs[corpus]):
                    pair = pair_key(row)
                    db.execute('INSERT INTO originals VALUES (?,?,?,?,?)', (corpus, row['id'], pair, source_key(row), normalized(row['ste'])))
                    db.executemany('INSERT INTO links VALUES (?,?)', ((k, pair) for k in links(corpus, row)))
                    found = db.execute('SELECT hash,payload_hash FROM members WHERE corpus=? AND id=?', (corpus, row['id'])).fetchone()
                    if found and found[0] != member_hash(corpus, row):
                        raise ValueError('member attribution or review changed')
                    if found and found[1] and found[1] != digest(json.dumps(row, sort_keys=True, ensure_ascii=False)):
                        raise ValueError('canonical surface payload changed')
                    counts[corpus] += 1
                    if counts[corpus] % 50000 == 0:
                        print(f'audited input {corpus}: {counts[corpus]:,}', flush=True)
            db.executescript('CREATE TABLE conflicts AS SELECT source FROM originals GROUP BY source HAVING COUNT(DISTINCT target)>1;'
                             'CREATE INDEX originals_pair ON originals(pair); CREATE INDEX link_pairs ON links(pair);'
                             'CREATE UNIQUE INDEX conflicts_source ON conflicts(source);')
            # Propagate output assignments across every surface link, even removed examples.
            db.execute('CREATE TABLE assignments AS SELECT l.key,p.group_id,p.split FROM links l JOIN pairs p ON l.pair=p.pair GROUP BY l.key,p.group_id,p.split')
            db.execute('CREATE INDEX assignments_key ON assignments(key)')
            while True:
                if db.execute('SELECT key FROM assignments GROUP BY key HAVING COUNT(DISTINCT group_id)>1 OR COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
                    raise ValueError('surface text/origin/document leakage')
                db.execute('CREATE TEMP TABLE propagated AS SELECT DISTINCT b.key,a.group_id,a.split FROM assignments a JOIN links x ON a.key=x.key JOIN links b ON b.pair=x.pair WHERE NOT EXISTS (SELECT 1 FROM assignments z WHERE z.key=b.key)')
                added = db.execute('SELECT COUNT(*) FROM propagated').fetchone()[0]
                db.execute('INSERT INTO assignments SELECT * FROM propagated')
                db.execute('DROP TABLE propagated')
                if not added:
                    break
            missing = db.execute('SELECT 1 FROM originals o LEFT JOIN members m ON o.corpus=m.corpus AND o.id=m.id WHERE o.source NOT IN (SELECT source FROM conflicts) AND m.id IS NULL LIMIT 1').fetchone()
            extra = db.execute('SELECT 1 FROM members m LEFT JOIN originals o ON m.corpus=o.corpus AND m.id=o.id WHERE o.id IS NULL OR o.source IN (SELECT source FROM conflicts) LIMIT 1').fetchone()
            conflicting = db.execute('SELECT COUNT(*) FROM originals WHERE source IN (SELECT source FROM conflicts)').fetchone()[0]
            duplicates = sum(counts.values()) - conflicting - sum(splits.values())
            if conflicting != summary['conflicting_records'] or duplicates != summary['duplicates_merged']:
                raise ValueError('conflict/duplicate accounting differs')
            if missing or extra or dict(counts) != summary['scanned'] or dict(splits) != summary['splits'] or sum(splits.values()) != summary['pairs']:
                raise ValueError('record accounting differs')
    result = {'passed': True, 'pairs': sum(splits.values()), 'surface_records': sum(counts.values()), 'splits': dict(splits),
              'scope': 'joint grouping, deduplication and metadata; upstream checker/alignment audits remain required'}
    dump(output / 'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'validate'))
    parser.add_argument('--corpus', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dev', type=float, default=.05)
    parser.add_argument('--test', type=float, default=.05)
    parser.add_argument('--shard-size', type=int, default=25000)
    args = parser.parse_args()
    corpora = [(name, Path(path)) for name, path in (v.split('=', 1) for v in args.corpus)]
    result = build(corpora, args.out, args.seed, args.dev, args.test, args.shard_size) if args.command == 'build' else validate(corpora, args.out)
    print(json.dumps(result, indent=2))
