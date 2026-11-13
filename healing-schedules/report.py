"""Summarize completed/censored arms without selecting a winner from missing data.

The primary question is how the seven schedules compare, so every arm is scored on the
fraction of the damaged-to-target validation gap it closes, at the end of its budget and on a
ladder of thresholds (25/50/75/100% of the gap; 100% is stock 0.8B quality). Tokens, FLOPs and
wall time to each rung give an efficiency comparison even when no arm reaches stock 0.8B.

Branch runs (follow-up experiment, run.py --init-from) are scored against the trunk's damaged
start, so their rungs are the same thresholds as the independent arms'. Only branches from one
checkpoint are compared with each other.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def first_confirmed(history, threshold):
    """run.py's rule for any threshold: the second of two consecutive evaluations at or below it."""
    hits = [i for i, row in enumerate(history) if row['validation']['ce'] <= threshold]
    paired = [i for i in hits if i > 0 and i - 1 in hits]
    return history[paired[0]] if paired else None


LADDER = (0.25, 0.5, 0.75, 1.0)


def gap_closed(start, ce, target):
    """Fraction of the validation gap from the damaged start to the target that is closed."""
    return (start - ce) / (start - target)


def ladder(history, target, start=None):
    """First confirmed evaluation (two consecutive at or below) for each gap fraction."""
    start = history[0]['validation']['ce'] if start is None else start
    out = {}
    for fraction in LADDER:
        hit = first_confirmed(history, start - fraction * (start - target))
        out[f'gap{round(fraction * 100)}'] = (dict(tokens=hit['tokens'], training_flops=hit['training_flops'],
                                                    elapsed_seconds=hit['elapsed_seconds'], step=hit['step'])
                                               if hit else None)
    return out


def collect(root, secondary_offset=0.10):
    rows, signatures = [], set()
    for path in sorted(root.glob('*/summary.json')):
        s = json.loads(path.read_text())
        p = json.loads((path.parent / 'provenance.json').read_text())
        signatures.add((p['damaged_manifest_sha256'], p['data_manifest_sha256'],
                        p['arguments']['budget_flops'], p['target_ce'], json.dumps(p['source_sha256'], sort_keys=True),
                        (p.get('branch') or {}).get('checkpoint_sha256')))
        history = json.loads((path.parent / 'curve.json').read_text())
        hit = s['first_confirmed_target']
        second = first_confirmed(history, p['target_ce'] + secondary_offset)
        branch = p.get('branch')
        start = branch['start_validation_ce'] if branch else history[0]['validation']['ce']
        rungs = ladder(history, p['target_ce'], start)
        flat = {f'{k}_{field}': (v or {}).get(field) for k, v in rungs.items()
                for field in ('tokens', 'training_flops', 'elapsed_seconds')}
        rows.append(dict(run=path.parent.name, strategy=p['arguments']['strategy'], seed=p['arguments']['seed'],
                         status=s['status'], training_flops=s['training_flops'],
                         total_accounted_flops=s['total_accounted_flops'], tokens=s['tokens'], steps=s['steps'],
                         validation_ce=history[-1]['validation']['ce'], test_ce=s['final_test']['ce'],
                         start_validation_ce=start,
                         branch_gap=(branch or {}).get('gap_fraction'), trunk_strategy=(branch or {}).get('trunk_strategy'),
                         gap_closed_final=gap_closed(start, history[-1]['validation']['ce'], p['target_ce']),
                         best_validation_ce=min(h['validation']['ce'] for h in history),
                         decision_flops=s.get('decision_flops', 0), **flat,
                         test_perplexity=s['final_test']['perplexity'], test_teacher_kl=s['final_test']['teacher_kl'],
                         reached_target=s['reached_target'],
                         target_training_flops=hit['training_flops'] if hit else None,
                         target_total_flops=hit['total_accounted_flops'] if hit else None,
                         target_tokens=hit['tokens'] if hit else None,
                         target_elapsed_seconds=hit['elapsed_seconds'] if hit else None,
                         secondary_target_reached=second is not None,
                         secondary_target_tokens=second['tokens'] if second else None,
                         secondary_target_training_flops=second['training_flops'] if second else None,
                         elapsed_seconds=s['elapsed_seconds'], peak_host_rss_bytes=s['peak_host_rss_bytes'],
                         peak_device_allocated_bytes=s['peak_device_allocated_bytes']))
    if len(signatures) > 1:
        raise ValueError('Runs differ in damage, data, budget, target, code or branch point; analyze separately')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = collect(args.runs)
    if not rows:
        raise ValueError('No completed summary files; no empirical comparison available')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with (args.output / 'runs.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for strategy in sorted({r['strategy'] for r in rows}):
        selected = [r for r in rows if r['strategy'] == strategy]
        successful = [r for r in selected if r['status'] == 'budget']
        values = [r['test_ce'] for r in successful]
        gaps = [r['gap_closed_final'] for r in successful]
        summary[strategy] = dict(runs=len(selected), completed_budgets=len(successful),
                                target_reached=sum(r['reached_target'] for r in selected),
                                gap_closed_final_mean=statistics.mean(gaps) if gaps else None,
                                rungs_reached={f'gap{round(f * 100)}': sum(r[f'gap{round(f * 100)}_tokens'] is not None
                                                                           for r in selected) for f in LADDER},
                                test_ce_mean=statistics.mean(values) if values else None,
                                test_ce_sample_sd=statistics.stdev(values) if len(values) > 1 else None)
    (args.output / 'summary.json').write_text(json.dumps(dict(strategies=summary,
        note='No significance claim. Actual FLOPs differ by a residual smaller than the next step. Inspect curves and all seeds; censored arms are excluded from budget-complete means.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
