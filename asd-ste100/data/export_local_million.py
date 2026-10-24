"""Export staged local pairs as deterministic gzip shards with document-grouped splits."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import io
import json
from pathlib import Path
import sqlite3

from build_local_million import encode_batch
from build_pairs import Groups, digest
from harvest_sources import sha256
from text_utils import normalized


def export(output: Path, seed=0, workers=4, shard_size=25000):
    config = json.loads((output / 'build_config.json').read_text(encoding='utf-8'))
    if config['seed'] != seed:
        raise ValueError('export seed must match staging seed')
    db = sqlite3.connect(output / 'pairs.sqlite3')
    db.execute('CREATE INDEX IF NOT EXISTS pairs_origin ON pairs(origin_id)')
    db.commit()
    groups, target_docs, origins_seen = Groups(), {}, set()
    for origin, doc in db.execute('SELECT origin_id,document_id FROM target_documents ORDER BY origin_id,document_id'):
        target_docs.setdefault(origin, []).append(doc)
    print('Grouping selected documents and shared texts', flush=True)
    for count, (origin, src, tgt, docs) in enumerate(db.execute('SELECT origin_id,src,ste,documents FROM pairs ORDER BY id'), 1):
        anchor = 'id:' + origin
        groups.join(anchor, 'text:' + digest(normalized(src)))
        if origin not in origins_seen:
            groups.join(anchor, 'text:' + digest(normalized(tgt)))
            for doc in target_docs.get(origin, json.loads(docs)):
                groups.join(anchor, 'doc:' + doc)
            origins_seen.add(origin)
        if count % 200000 == 0:
            print(f'grouped {count:,} pairs', flush=True)
    group_info, group_counts = {}, Counter()
    for origin, count in db.execute('SELECT origin_id,COUNT(*) FROM pairs GROUP BY origin_id'):
        group = digest(f'{seed}\0{groups.find("id:" + origin)}')[:20]
        group_info[origin] = group
        group_counts[group] += count
    total = sum(group_counts.values())
    # Boilerplate can link many documents. Keep large components in train rather than
    # letting a hash move a large fraction of the corpus into a holdout.
    large = {group for group, count in group_counts.items() if count > max(1, total * .02)}
    for origin, group in group_info.items():
        fraction = int(group[:16], 16) / 2**64
        split = 'train' if group in large else 'test' if fraction < .05 else 'dev' if fraction < .1 else 'train'
        group_info[origin] = split, group
    del groups, origins_seen

    def batches():
        batch = []
        for row in db.execute('SELECT * FROM pairs ORDER BY id'):
            split, group = group_info[row[1]]
            docs = json.dumps(target_docs.get(row[1], json.loads(row[7])))
            batch.append((*row[:7], docs, row[8], split, group))
            if len(batch) == 256:
                yield batch
                batch = []
        if batch:
            yield batch

    streams, paths, counts, buckets = {}, [], Counter(), Counter()
    shard_dir = output / 'shards'
    shard_dir.mkdir(exist_ok=True)

    def write(results):
        for split, bucket, line in results:
            if counts[split] % shard_size == 0:
                if split in streams:
                    close_stream(streams[split])
                path = shard_dir / f'{split}-{counts[split] // shard_size:05d}.jsonl.gz'
                paths.append(path)
                raw = path.open('wb')
                compressed = gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=1)
                # Closing TextIOWrapper closes gzip, whose fileobj must also be closed.
                streams[split] = io.TextIOWrapper(compressed, encoding='utf-8', newline='\n')
                streams[split]._ste_raw_file = raw
            streams[split].write(line)
            counts[split] += 1; buckets[str(bucket)] += 1
        if sum(counts.values()) % 51200 == 0:
            print(f'exported {sum(counts.values()):,}/{total:,} aligned pairs', flush=True)

    def close_stream(stream):
        stream.close()
        stream._ste_raw_file.close()

    try:
        if workers == 1:
            for batch in batches():
                write(encode_batch(batch))
        else:
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
            close_stream(stream)
    summary = {'schema': 'ste-local-surface-pairs-v1', 'pairs': sum(counts.values()), 'splits': dict(counts),
               'buckets': dict(buckets), 'seed': seed, 'shard_size': shard_size, 'compression': 'gzip-level-1-mtime-0',
               'target_review': 'checker_only', 'meaning_review': 'unjudged_rule',
               'largest_group_pairs': max(group_counts.values(), default=0), 'large_groups_assigned_train': len(large),
               'shards': [{'path': str(p.relative_to(output)), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(paths)],
               'stage': json.loads((output / 'stage_summary.json').read_text(encoding='utf-8')),
               'code_sha256': {'export_local_million.py': sha256(Path(__file__))},
               'limitations': ['checker-only targets; no teacher rewrite or meaning judge', 'checker implements only a subset of STE',
                               'at most 20% unknown prose words; future tokenizer must protect unknowns',
                               'synthetic source datasets can contain incorrect facts', 'surface tokens still need model vocabulary IDs']}
    db.close()
    summary['staging_db_sha256'] = sha256(output / 'pairs.sqlite3')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return {key:value for key,value in summary.items() if key != 'shards'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shard-size', type=int, default=25000)
    args = ap.parse_args()
    if min(args.workers, args.shard_size) <= 0:
        ap.error('workers and shard size must be positive')
    print(json.dumps(export(args.out, args.seed, args.workers, args.shard_size), indent=2))
