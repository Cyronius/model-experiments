"""Combine audited clean relevance pairs with fresh or repeated science pairs.

This is local preparation only. It neither rents a GPU nor reads test labels.
The repeat control has identical pair counts and source weights to the fresh arm.
"""
import collections
import json
import random
from pathlib import Path

from data_first_build import sha

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def keys(row):
    result = {row['query_visible_key'], row['positive_visible_key']}
    if row.get('negative'):
        result.add(row['negative_visible_key'])
    return result


def main():
    out = ROOT / 'work/science-mix-clean'
    out.mkdir(exist_ok=True)
    if (out / 'audit.json').exists():
        raise RuntimeError('Audited output already exists; do not silently rebuild')
    base = ROOT / 'work/benchmark-clean'
    science = ROOT / 'work/science-global'
    base_audit = json.loads((ROOT / 'benchmark-clean/data-audit.json').read_text())
    science_audit = json.loads((ROOT / 'science-global/data-audit.json').read_text())
    inputs = {}
    for directory, audit in ((base, base_audit), (science, science_audit)):
        for split in ('train', 'dev'):
            path = directory / (split + '.jsonl')
            expected = audit['output_sha256'][split]
            if sha(path) != expected:
                raise RuntimeError('Audited input changed: ' + str(path))
            inputs[str(path.relative_to(PROJECT))] = expected
    counts, identities, old_science = {}, {}, []
    fresh_count = 0
    for split in ('train', 'dev'):
        counter = collections.Counter()
        seen = set()
        with (out / (split + '.jsonl')).open('w') as target:
            for directory in (base, science):
                for line in (directory / (split + '.jsonl')).open():
                    row = json.loads(line)
                    target.write(line)
                    counter[row['source']] += 1
                    seen.update(keys(row))
                    if split == 'train':
                        if directory == science:
                            fresh_count += 1
                        elif row['source'] == 'sentence-transformers/s2orc':
                            old_science.append(line)
        counts[split], identities[split] = dict(counter), seen
    if identities['train'] & identities['dev']:
        raise RuntimeError('Training and held-out visible inputs overlap')
    auxiliary_overlap = {}
    for aux_split, relevance_split in (('train', 'dev'), ('dev', 'train')):
        overlap = set()
        path = ROOT / 'work/nli-aux-global' / (aux_split + '.jsonl')
        inputs[str(path.relative_to(PROJECT))] = sha(path)
        for line in path.open():
            row = json.loads(line)
            overlap.update({row['visible1'], row['visible2']} & identities[relevance_split])
        auxiliary_overlap[aux_split + '_vs_relevance_' + relevance_split] = len(overlap)
    if any(auxiliary_overlap.values()):
        raise RuntimeError('Auxiliary and relevance holdouts overlap')
    repeat_counts = collections.Counter()
    rng = random.Random(67)
    with (out / 'train-repeat.jsonl').open('w') as target:
        for line in (base / 'train.jsonl').open():
            target.write(line)
            repeat_counts[json.loads(line)['source']] += 1
        for _ in range(fresh_count):
            target.write(rng.choice(old_science))
            repeat_counts['sentence-transformers/s2orc'] += 1
    if dict(repeat_counts) != counts['train']:
        raise RuntimeError('Fresh and repeat controls have different source counts')
    audit = {
        'scope': 'Prepared only; clean base plus fresh science, and matched repeat control.',
        'teacher_targets': False,
        'inputs_sha256': inputs,
        'base_audit_sha256': sha(ROOT / 'benchmark-clean/data-audit.json'),
        'science_audit_sha256': sha(ROOT / 'science-global/data-audit.json'),
        'output_sha256': {name: sha(out / (name + '.jsonl')) for name in ('train', 'dev', 'train-repeat')},
        'rows': {split: sum(counter.values()) for split, counter in counts.items()},
        'by_source': counts,
        'repeat_rows': sum(repeat_counts.values()),
        'repeated_science_pairs': fresh_count,
        'available_old_science_pairs': len(old_science),
        'repeat_seed': 67,
        'train_dev_shared_visible_inputs': 0,
        'auxiliary_cross_holdout_overlap': auxiliary_overlap,
        'known_legacy_test_overlap': 0,
        'limits': 'Inherited exact model-visible exclusions, not an exhaustive semantic near-duplicate audit or claim about historical pretraining.',
    }
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    (ROOT / 'science-mix/clean-data-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
