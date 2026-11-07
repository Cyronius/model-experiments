"""Reproducible review samples from manifest-listed retained model pairs.

The uniform sample is separate from deliberately oversampled diagnostic slices.
Sampling itself makes no semantic-quality judgments and never changes the corpus.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tokenizer'))
from prepare import input_paths, records
from ste_tokenizer import sha256


def sample(pairs, output, seed='spot-20261024-v1', uniform=60, per_kind=20, per_operation=5):
    if min(uniform, per_kind, per_operation) < 1:
        raise ValueError('sample sizes must be positive')
    if output.exists() and any(output.iterdir()):
        raise ValueError('use an empty output directory')
    paths = list(input_paths(pairs))
    heaps, counts, splits, kinds = {}, Counter(), Counter(), Counter()
    def add(group, capacity, row):
        priority = int(hashlib.sha256(f'{seed}\0{group}\0{row["id"]}'.encode()).hexdigest(), 16)
        heap = heaps.setdefault(group, [])
        # Full SHA-256 priority plus pair ID makes selection independent of shard ordering.
        entry = (-priority, row['id'], row)
        if len(heap) < capacity:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    for path, _ in paths:
        for row in records(path):
            compact = {k: row[k] for k in ('id','src','ste','mode','kind','noise','split','meaning_review')}
            for k in ('target_review', 'provenance', 'document_ids'):
                if k in row:
                    compact[k] = row[k]
            add('uniform', uniform, compact)
            if row['kind'] != 'local_rule':
                add('kind:' + row['kind'], per_kind, compact)
            else:
                add('local-operation:' + str(row['noise']), per_operation, compact)
            counts[(row['kind'], str(row['noise']))] += 1
            splits[row['split']] += 1
            kinds[row['kind']] += 1
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for group, heap in sorted(heaps.items()):
        for index, (_, _, row) in enumerate(sorted(heap, key=lambda x: (-x[0], x[1])), 1):
            samples.append(dict(row, sample_group=group, review_id=f'{group}/{index:02d}'))
    with (output / 'sample.jsonl').open('w', encoding='utf-8') as stream:
        for row in samples:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'seed': seed, 'population': sum(kinds.values()), 'population_kinds': dict(kinds),
                'population_splits': dict(splits),
                'population_operations': {f'{k}/{op}': n for (k, op), n in sorted(counts.items())},
                'sample_groups': {k: len(v) for k, v in sorted(heaps.items())},
                'sample_memberships': len(samples), 'unique_sample_pairs': len({r['id'] for r in samples}),
                'input_summary_sha256': sha256(pairs / 'summary.json'),
                'sample_sha256': sha256(output / 'sample.jsonl'), 'code_sha256': sha256(Path(__file__)),
                'review_status': 'unreviewed',
                'limits': ['Only uniform group is a uniform pair-level sample; other groups oversample rare slices.',
                           'Rows can occur in more than one sample group.',
                           'Review is an assistant spot check, not owner-verified gold or a full corpus audit.']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', default='spot-20261024-v1')
    args = parser.parse_args()
    print(json.dumps(sample(args.pairs, args.out, args.seed), indent=2))
