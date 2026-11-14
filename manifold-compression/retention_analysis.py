"""Descriptive paired endpoint analysis; never selects or promotes a checkpoint."""
import argparse
import json
from pathlib import Path

from retention_prepare import OUT, save, sha
from tf_metrics import compare_scores, paired_bootstrap

PAIRS = [('A0', 'A1'), ('A0', 'A2'), ('A1', 'A2'), ('S0', 'S1'), ('S0', 'S2'), ('S1', 'S2')]


def generation(run, split, draws=10000):
    path = run / 'review' / (split + '-review.json')
    scored = json.loads(path.read_text())
    if scored['status'] != ('development_review_complete' if split == 'dev' else 'test_review_complete'):
        raise ValueError('Generation grading incomplete')
    if scored['data_sha256'] != sha(OUT / 'data.json'):
        raise ValueError('Frozen data differs')
    outputs = {}
    for before, after in PAIRS:
        b = {r['id']: r for r in scored['normalized'][before]}
        a = {r['id']: r for r in scored['normalized'][after]}
        if b.keys() != a.keys():
            raise ValueError('Unpaired generation rows')
        pair = {}
        for category in ('primary', 'retention', 'program', 'science'):
            ids = sorted(i for i in b if b[i]['category'] == category)
            if category == 'science':
                base = sum(b[i]['correct'] for i in ids) / len(ids)
                arm = sum(a[i]['correct'] for i in ids) / len(ids)
                metrics = {'base': base, 'arm': arm, 'difference': arm - base,
                           'ci95': None, 'items': len(ids),
                           'scope': 'Only4 development or8 test items: report individual factual transitions, not an interval'}
            else:
                metrics = paired_bootstrap([b[i]['group'] for i in ids], [b[i]['correct'] for i in ids],
                                           [a[i]['correct'] for i in ids], draws=draws, seed=26102741)
            metrics['gained_ids'] = [i for i in ids if not b[i]['correct'] and a[i]['correct']]
            metrics['lost_ids'] = [i for i in ids if b[i]['correct'] and not a[i]['correct']]
            # Sensitivity to shared template families, where multiple clusters exist.
            families = [b[i]['family'] for i in ids]
            if category != 'science' and len(set(families)) > 1:
                metrics['family_cluster_sensitivity'] = paired_bootstrap(families,
                    [b[i]['correct'] for i in ids], [a[i]['correct'] for i in ids], draws=draws, seed=26102741)
            pair[category] = metrics
        pair['new_loop_ids'] = [i for i in b if not b[i]['loop'] and a[i]['loop']]
        pair['lost_completion_ids'] = [i for i in b if b[i]['ended'] and not a[i]['ended']]
        outputs[before + '-vs-' + after] = pair
    result = {'split': split, 'review_sha256': sha(path), 'pairs': outputs,
              'scope': 'Descriptive fixed-endpoint comparisons, one training seed. Bootstrap intervals condition on sampled groups/templates; not a broad generalization guarantee or multiple-testing-corrected claims.',
              'program_gate_scope': 'Fresh Horizon cases share one family: frozen gate limits net category loss. Historical program vetoes limit individual losses.',
              'no_checkpoint_selection': True, 'parent_remains_default': True}
    save(result, run / 'review' / (split + '-paired-analysis.json'))
    for name, pair in outputs.items():
        print(name, {key: {'delta': round(value['difference'], 4), 'ci95': value['ci95']}
                     for key, value in pair.items() if isinstance(value, dict)}, flush=True)


def broad(run, draws=10000):
    folder = run / 'tf-metrics'
    pairs = [('stock', 'A0'), ('stock', 'A1'), ('stock', 'A2'), ('A0', 'A1'), ('A0', 'A2'),
             ('A1', 'A2'), ('stock', 'S1'), ('stock', 'S2'), ('S1', 'S2')]
    missing, paths = [], {}
    for before, after in pairs:
        name = before + '-vs-' + after
        sources = [folder / f'scores-{arm}.json' for arm in (before, after)]
        if any(not p.exists() for p in sources):
            missing.append(name)
            continue
        a, b = [json.loads(p.read_text()) for p in sources]
        if a['status'] != 'complete' or b['status'] != 'complete':
            missing.append(name)
            continue
        result = compare_scores(a, b, draws, 26102742)
        result.update(scope='Descriptive paired source-specific metrics; no pooled score or checkpoint selection',
                      score_sha256={p.name: sha(p) for p in sources})
        destination = folder / ('compare-' + name + '.json')
        save(result, destination)
        paths[name] = destination.name
    save({'comparisons': paths, 'missing': missing, 'complete': not missing,
          'no_auto_promotion': True, 'multiplicity': 'Exploratory per-source intervals; no multiple-comparison correction'},
         folder / 'paired-analysis-index.json')
    print(json.dumps({'comparisons': list(paths), 'missing': missing}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['generation', 'broad'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    generation(args.run, args.split, args.draws) if args.stage == 'generation' else broad(args.run, args.draws)
