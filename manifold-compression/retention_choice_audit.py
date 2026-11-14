"""Post-hoc choice-position diagnostics; never replaces the frozen scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, device='cuda', arms=('stock', 'A0', 'A1', 'A2', 'S1', 'S2')):
    manifest_path = directory / 'manifest.jsonl'
    manifest_hash = sha(manifest_path)
    items = {row['id']: row for row in
             map(json.loads, manifest_path.read_text().splitlines()) if row['kind'] == 'mc'}
    sources = sorted({row['source'] for row in items.values()})
    report = {
        'scope': 'Post-hoc descriptive choice-position audit; no rescoring, selection, or changes to frozen results.',
        'manifest_sha256': manifest_hash,
        'code_sha256': sha(Path(__file__)),
        'device': device, 'requested_arms': list(arms),
        'sources': {}, 'arms': {}, 'missing': [],
    }
    for source in sources:
        rows = [row for row in items.values() if row['source'] == source]
        report['sources'][source] = {
            'items': len(rows),
            'choice_counts': dict(sorted(Counter(len(row['choice_ids']) for row in rows).items())),
            'gold_position_counts': dict(sorted(Counter(row['answer'] for row in rows).items())),
            'uniform_random_accuracy': sum(1 / len(row['choice_ids']) for row in rows) / len(rows),
        }
    for arm in arms:
        path = directory / f'scores-{arm}.json'
        if not path.exists():
            report['missing'].append(arm)
            continue
        scores = json.loads(path.read_text())
        if (scores['arm'] != arm or scores['manifest_sha256'] != manifest_hash
                or scores['device'] != device or scores['status'] != 'complete'):
            raise ValueError('Wrong score identity or incomplete score file: ' + arm)
        rows = [row for row in scores['rows'] if row['kind'] == 'mc']
        if len(rows) != len(items) or {row['id'] for row in rows} != items.keys():
            raise ValueError('Missing or duplicate multiple-choice rows: ' + arm)
        result = {'score_sha256': sha(path), 'sources': {}}
        for source in sources:
            selected = [row for row in rows if row['source'] == source]
            if {row['id'] for row in selected} != {id_ for id_, row in items.items() if row['source'] == source}:
                raise ValueError('Source identity differs: ' + arm)
            result['sources'][source] = {}
            for normalization in ('sum', 'per_token', 'per_char'):
                counts = Counter()
                correct = 0
                for row in selected:
                    item = items[row['id']]
                    pick = row['picks'][normalization]
                    if not 0 <= pick < len(item['choice_ids']):
                        raise ValueError('Invalid choice index')
                    truth = int(pick == item['answer'])
                    if truth != row['correct'][normalization]:
                        raise ValueError('Recorded correctness does not match chosen answer')
                    counts[pick] += 1
                    correct += truth
                result['sources'][source][normalization] = {
                    'pick_counts': dict(sorted(counts.items())),
                    'accuracy': correct / len(selected),
                    'most_common_pick_fraction': max(counts.values()) / len(selected),
                }
        report['arms'][arm] = result
    destination = directory / 'choice-position-audit.json'
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'arms': list(report['arms']), 'missing': report['missing']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument('--run', type=Path, help='Run directory containing tf-metrics/')
    location.add_argument('--metrics-dir', type=Path, help='Directory containing manifest.jsonl and score files')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--arms', nargs='+', default=['stock', 'A0', 'A1', 'A2', 'S1', 'S2'])
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error('Arm names must be unique')
    audit(args.metrics_dir if args.metrics_dir is not None else args.run / 'tf-metrics', args.device, args.arms)
