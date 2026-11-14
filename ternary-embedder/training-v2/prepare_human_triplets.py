"""Prepare audited human entailment/contradiction pairs without a paid job.

Uses only the already audited official NLI train split. Inspired by supervised
SimCSE: https://aclanthology.org/2021.emnlp-main.552/ . No teacher targets.
"""
import collections
import hashlib
import json
from pathlib import Path

from pair_training_state import atomic_json, digest

ROOT = Path(__file__).resolve().parent


def triplets(rows):
    """Attach a deterministic same-premise contradiction to every entailment.

    Preserve all known positives and exclude contradictory visible identities.
    Neutral labels are never used as negative relevance judgments.
    """
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        if row['label'] not in (0, 1, 2):
            raise ValueError('Unexpected NLI label')
        grouped[row['visible1']][row['label']].append(row)
    result = []
    rejected = collections.Counter()
    for premise in sorted(grouped):
        labels = grouped[premise]
        if not labels[0] or not labels[2]:
            rejected['premises_without_both_labels'] += 1
            continue
        positive_keys = {r['visible2'] for r in labels[0]}
        negative_rows = [r for r in labels[2]
                         if r['visible2'] not in positive_keys and r['visible2'] != premise]
        if not negative_rows:
            rejected['premises_without_nonconflicting_contradiction'] += 1
            continue
        negative = min(negative_rows, key=lambda r: (r['visible2'], r['sentence2']))
        for positive in sorted(labels[0], key=lambda r: (r['visible2'], r['sentence2'])):
            if positive['visible2'] == premise:
                rejected['identical_positive'] += 1
                continue
            result.append(dict(
                query=positive['sentence1'], positive=positive['sentence2'],
                source='human-nli-triplets', relation='human entailment with same-premise contradiction',
                query_visible_key=premise, positive_visible_key=positive['visible2'],
                query_key=hashlib.sha256(positive['sentence1'].encode()).hexdigest(),
                positive_key=hashlib.sha256(positive['sentence2'].encode()).hexdigest(),
                source_row=positive['source_row'], negative=negative['sentence2'],
                negative_visible_key=negative['visible2'], negative_source='human same-premise contradiction',
                negative_source_row=negative['source_row'], positive_label=0, negative_label=2))
    return result, dict(rejected)


def main():
    source = ROOT / 'work/nli-aux-global'
    output = ROOT / 'work/human-triplets'
    if (output / 'audit.json').exists():
        raise RuntimeError('Already audited; preserve the prepared input')
    audit = json.loads((source / 'audit.json').read_text())
    for split in ('train', 'dev'):
        assert digest(source / (split + '.jsonl')) == audit['output_sha256'][split]
    rows = [json.loads(line) for line in (source / 'train.jsonl').open()]
    prepared, rejected = triplets(rows)
    training_keys = {r[k] for r in prepared for k in
                     ('query_visible_key', 'positive_visible_key', 'negative_visible_key')}
    held_keys = set()
    for line in (source / 'dev.jsonl').open():
        row = json.loads(line)
        held_keys.update((row['visible1'], row['visible2']))
    relevance_dev = ROOT / 'work/science-mix-clean/dev.jsonl'
    for line in relevance_dev.open():
        row = json.loads(line)
        held_keys.update((row['query_visible_key'], row['positive_visible_key']))
        if row.get('negative'):
            held_keys.add(row['negative_visible_key'])
    assert not training_keys & held_keys
    assert prepared and all(r['negative_visible_key'] != r['positive_visible_key'] for r in prepared)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'train.jsonl'
    with target.open('w') as f:
        for row in prepared:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = dict(prepared_only=True, paid_resource_allocated=False, teacher_targets=False,
                  source_audit_sha256=digest(source / 'audit.json'),
                  inputs_sha256={str(p.relative_to(ROOT.parent)): digest(p) for p in
                                 (source / 'train.jsonl', source / 'dev.jsonl', relevance_dev)},
                  rows=len(prepared), premises=len({r['query_visible_key'] for r in prepared}),
                  output_sha256={'train': digest(target)}, rejected=rejected,
                  same_premise_contradictions=True, neutral_labels_used_as_negatives=False,
                  train_relevance_or_auxiliary_dev_visible_overlap=0,
                  benchmark_exclusions='Inherited from pinned audited human-NLI inputs, including all observed benchmark inputs.',
                  reserved_fiqa_scidocs_judgments_read=False,
                  limits='Human contradiction is a logical relation, not a graded similarity score. Quality is untested; no new relevance labels have been generated.')
    atomic_json(output / 'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
