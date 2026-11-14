"""Add paired recipe and precision-matched baseline contrasts to frozen reports.

Useful for workers that were already running when reporting was extended.
Does not train, select checkpoints, or open any new evaluation labels.
"""
import argparse
import json
from pathlib import Path

from summarize_pair_results import interval

ROOT = Path(__file__).resolve().parent


def enrich(profile):
    out = ROOT / profile / 'results'
    path = out / 'summary.json'
    report = json.loads(path.read_text())
    reference = json.loads((ROOT / 'round2/query-rows.json').read_text())
    rows = {}
    for name, model in report['models'].items():
        quality = json.loads((out / (name + '-quality.json')).read_text())
        assert quality['selection']['sha256'] == model['selection']['sha256']
        rows[name] = json.loads((out / (name + '-rows.json')).read_text())
        precision = model['selection']['precision']
        prior = reference['original'] if precision == 'ternary' else {
            t: json.loads((ROOT / 'analysis' / ('precision-float-' + t + '-rows.json')).read_text())
            for t in ('SciFact', 'NFCorpus')}
        model['original_reference_precision'] = 'ternary'
        model['matched_original_delta'] = interval(prior, rows[name])
        model['matched_original_scope'] = 'Original two-layer checkpoint of the same precision; not equal historical training compute.'
    pairs = [(p + '-pairs', p + '-hard') for p in ('float', 'ternary')]
    pairs += [(p + '-0.0', p + '-' + w) for p in ('float', 'ternary') for w in ('0.1', '0.25')]
    pairs += [(p + '-repeat', p + '-fresh') for p in ('float', 'ternary')]
    pairs += [('fixed', 'learned'), ('fixed', 'scales-only')]
    contrasts = {}
    for left, right in pairs:
        if left not in rows or right not in rows:
            continue
        a, b = report['models'][left], report['models'][right]
        contrasts[right + ' minus ' + left] = {
            'core': interval(rows[left], rows[right]),
            'stsbenchmark_delta': b['stsbenchmark'] - a['stsbenchmark'],
            'mean12_delta': b['mteb_12_mean'] - a['mteb_12_mean']}
    report['matched_recipe_contrasts'] = contrasts
    temporary = path.with_suffix('.json.enriched-tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)
    print(profile, json.dumps(contrasts, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    enrich(parser.parse_args().profile)
