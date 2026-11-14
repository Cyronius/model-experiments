"""Combine measured pilot results; no missing run is treated as a pass."""
import json
import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root.parents[1]))
from healing_data import check_answer, scoring_spec

case_definitions = {c['id']: c for c in json.loads((root / 'data.json').read_text())['generation_eval']}
rows = []
comparisons = {}
for arch in ('qwen35', 'smollm2'):
    generations = {}
    labels = ['stock', 'legacy-before', 'legacy-after', 'corrected-before', 'corrected-after']
    if arch == 'qwen35':
        labels.append('recovered-500m')
    for label in labels:
        file = root / f'{arch}-{label}-generation.json'
        generation = json.loads(file.read_text())
        if 'summary' not in generation:
            raise ValueError(f'Incomplete generation: {arch} {label}')
        for c in generation['cases']:
            spec = scoring_spec(case_definitions[c['id']])
            if spec != c['check']:
                c['original_check_pass'] = c['check_pass']
                c['check'] = spec
            c['check_pass'] = check_answer(c['answer'], spec)
        generation['rubric_revision'] = 2
        generation['summary']['check_passes'] = sum(c['check_pass'] for c in generation['cases'])
        file.write_text(json.dumps(generation, indent=2) + '\n')
        generations[label] = generation
        row = {'arch': arch, 'label': label, **generation['summary']}
        if label == 'recovered-500m':
            row.update(generation['validation'])
        elif label != 'stock':
            arm, when = label.split('-')
            training = json.loads((root / f'{arch}-{arm}-training.json').read_text())
            assert len(training['steps']) == 32 and training['roundtrip']['numerical_pass']
            row.update(training[when])
        rows.append(row)
    by_label = {label: {c['id']: c for c in run['cases']} for label, run in generations.items()}
    comparisons[arch] = {}
    for arm in ('legacy', 'corrected'):
        before, after = by_label[f'{arm}-before'], by_label[f'{arm}-after']
        comparisons[arch][arm] = {
            'check_improvements': [k for k in before if not before[k]['check_pass'] and after[k]['check_pass']],
            'check_regressions': [k for k in before if before[k]['check_pass'] and not after[k]['check_pass']],
            'failed_after_while_stock_passes': [k for k in after if not after[k]['check_pass'] and by_label['stock'][k]['check_pass']],
            'loop_flags_after': [k for k, c in after.items() if c['loop_flag']],
            'capped_after': [k for k, c in after.items() if not c['ended']],
        }
result = {'rows': rows, 'paired_cases': comparisons, 'spend_usd': 0}
(root / 'combined.json').write_text(json.dumps(result, indent=2) + '\n')
print('| Model | State | Held-out KL | Checks | Completed | Repetition flags |')
print('|---|---|---:|---:|---:|---:|')
for r in rows:
    kl = f"{r['kl']:.4f}" if 'kl' in r else '0 (reference)'
    print(f"| {r['arch']} | {r['label']} | {kl} | {r['check_passes']}/{r['cases']} | {r['completed']}/{r['cases']} | {r['loop_flags']} |")
print(json.dumps(comparisons, indent=2))
