"""Paired grouped bootstrap and the frozen G1 progression/floor decisions.

Run only after all generation and independent science reviews are complete:
    python unhealed_analysis.py
Writes analysis.json once. Results are exploratory; a pass is not promotion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from unhealed_review import ARMS, OUT, boolean, digest, finish, index, read, write_new

METRICS = ('correct', 'loop', 'nontermination')


def interval(draws):
    return [float(v) for v in np.quantile(draws, [.025, .975])]


def lower_gate(ci, threshold, strict=False):
    if ci[0] > threshold if strict else ci[0] >= threshold:
        return 'pass'
    if ci[1] <= threshold if strict else ci[1] < threshold:
        return 'fail'
    return 'inconclusive'


def upper_gate(ci, threshold):
    if ci[1] <= threshold:
        return 'pass'
    return 'fail' if ci[0] > threshold else 'inconclusive'


def bootstrap_means(rows, values, rng, draws, sensitivity=False):
    """Resample whole groups, jointly across arms/metrics, within fixed strata.

    Family proportions stay fixed for arithmetic/replay. Within a stratum,
    sampled cluster sums/counts estimate the case-weighted mean. This also
    supports unequal cluster sizes without accidentally weighting clusters equally.
    """
    if not rows or len(rows) != len(values):
        raise ValueError('Cannot bootstrap empty/misaligned cases')
    category = rows[0]['category']
    if any(row['category'] != category for row in rows):
        raise ValueError('Bootstrap one reporting category at a time')
    fixed_families = category in ('arithmetic', 'replay') and not sensitivity
    strata = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        stratum = row['family'] if fixed_families else 'all'
        group = (row['family'] if category in ('arithmetic', 'replay') else row['id']) if sensitivity else row['group']
        strata[stratum][group].append(i)
    result = np.zeros((draws,) + values.shape[1:], dtype=np.float64)
    for name in sorted(strata):
        groups = strata[name]
        positions = [groups[k] for k in sorted(groups)]
        counts = np.asarray([len(g) for g in positions], dtype=np.float64)
        sums = np.asarray([values[g].sum(axis=0) for g in positions])
        multiplicity = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=draws)
        total = np.tensordot(multiplicity, sums, axes=(1, 0))
        denominator = multiplicity @ counts
        shape = (draws,) + (1,) * (values.ndim - 1)
        result += total / denominator.reshape(shape) * counts.sum() / len(rows)
    return result


def aligned(value):
    if value.get('status') != 'complete' or set(value['arms']) != set(ARMS):
        raise ValueError('All three complete arms are required')
    review = value['science_review']
    if sorted(review['reviewers']) != ['claude', 'codex'] or review['adjudicated'] is not True:
        raise ValueError('Independent science reviews/adjudication are required')
    maps = {arm: index(value['arms'][arm], arm) for arm in ARMS}
    ids = sorted(maps['stock'])
    if any(set(maps[arm]) != set(ids) for arm in ARMS):
        raise ValueError('Arms do not cover the same cases')
    identity = ('part', 'family', 'group', 'category', 'rehearsal_exposed', 'primary_mask', 'training_overlap', 'tf_scored')
    expected_excluded = set(value['rules']['quarantine']['excluded_ids'])
    for id_ in ids:
        case = maps['stock'][id_]
        for arm in ARMS:
            row = maps[arm][id_]
            if any(row[k] != case[k] for k in identity):
                raise ValueError(f'{id_}: unpaired identity/mask')
            for k in ('correct', 'loop', 'ended', 'rehearsal_exposed'):
                boolean(row[k], k)
        if case['rehearsal_exposed'] != (id_ in expected_excluded):
            raise ValueError('Quarantine mask differs from frozen rules')
    rows = [maps['stock'][k] for k in ids]
    counts = dict(Counter(row['category'] for row in rows if not row['rehearsal_exposed']))
    if counts != value['rules']['reporting']['expected_eligible_counts']:
        raise ValueError('Eligible reporting categories/counts differ from rules')
    if dict(Counter(row['part'] for row in rows)) != value['rules']['reporting']['expected_generated_counts']:
        raise ValueError('Generated part counts differ from rules')
    values = np.asarray([[[maps[arm][k]['correct'], maps[arm][k]['loop'], not maps[arm][k]['ended']]
                          for arm in ARMS] for k in ids], dtype=np.float64)
    return rows, values, maps


def changes(rows, values):
    result = {}
    for i, name in enumerate(METRICS):
        base, candidate = values[:, 1, i], values[:, 2, i]
        result[name] = {'zero_to_one': [r['id'] for r, x in zip(rows, candidate - base) if x > 0],
                        'one_to_zero': [r['id'] for r, x in zip(rows, candidate - base) if x < 0]}
    return result


def analyze(value):
    rows, values, maps = aligned(value)
    rules = value['rules']
    draws, seed = rules['bootstrap']['draws'], rules['bootstrap']['seed']
    rng = np.random.default_rng(seed)
    eligible = [i for i, row in enumerate(rows) if not row['rehearsal_exposed']]
    n = len(eligible)
    categories = sorted({rows[i]['category'] for i in eligible})
    summaries, samples, means, lengths, sensitivities = {}, {}, {}, {}, {}
    for cat in categories:
        positions = [i for i in eligible if rows[i]['category'] == cat]
        group_rows, group_values = [rows[i] for i in positions], values[positions]
        samples[cat] = bootstrap_means(group_rows, group_values, rng, draws)
        means[cat], lengths[cat] = group_values.mean(axis=0), len(positions)
        delta = samples[cat][:, 2, 0] - samples[cat][:, 1, 0]
        summaries[cat] = {'n': len(positions), 'groups': len({r['group'] for r in group_rows}),
                          'arms': {arm: {name: {'count': int(group_values[:, a, m].sum()),
                                                'rate': float(means[cat][a, m])}
                                         for m, name in enumerate(METRICS)} for a, arm in enumerate(ARMS)},
                          'accuracy_delta': float(means[cat][2, 0] - means[cat][1, 0]),
                          'accuracy_delta_ci95': interval(delta), 'transitions': changes(group_rows, group_values)}
    # Sensitivities use a separate fixed stream so adding a descriptive estimate
    # cannot perturb any primary/gated bootstrap draws.
    sensitivity_rng = np.random.default_rng(seed + 1)
    for cat in ('arithmetic', 'replay', 'science_new'):
        pos = [i for i in eligible if rows[i]['category'] == cat]
        sampled = bootstrap_means([rows[i] for i in pos], values[pos], sensitivity_rng, draws, sensitivity=True)
        sensitivities[cat] = {'unit': 'family' if cat != 'science_new' else 'item',
                              'accuracy_delta_ci95': interval(sampled[:, 2, 0] - sampled[:, 1, 0])}
    primary = rules['progression']['primary_categories']
    macro_draws = np.mean([samples[cat][:, 2, 0] - samples[cat][:, 1, 0] for cat in primary], axis=0)
    macro_ci = interval(macro_draws)
    macro = {'delta': float(np.mean([means[cat][2, 0] - means[cat][1, 0] for cat in primary])),
             'ci95': macro_ci,
             'status': lower_gate(macro_ci, rules['progression']['macro_improvement_lower_ci_strictly_greater_than'], strict=True)}
    gates = {'complete_and_reviewed': 'pass', 'macro': macro['status']}
    for cat in primary:
        gates['noninferiority:' + cat] = lower_gate(summaries[cat]['accuracy_delta_ci95'],
                                                    rules['progression']['per_category_improvement_lower_ci_at_least'])
    # These are net event rates; retain case-count weights across fixed categories.
    overall_draws = sum(samples[cat] * lengths[cat] / n for cat in categories)
    overall_means = sum(means[cat] * lengths[cat] / n for cat in categories)
    bad_events = {}
    for m, event in enumerate(METRICS[1:], 1):
        ci = interval(overall_draws[:, 2, m] - overall_draws[:, 1, m])
        status = upper_gate(ci, rules['progression']['bad_event_rate_upper_ci_at_most'])
        gates[event] = status
        bad_events[event] = {'n': n, 'delta': float(overall_means[2, m] - overall_means[1, m]),
                             'ci95': ci, 'status': status,
                             'arms': {arm: float(overall_means[a, m]) for a, arm in enumerate(ARMS)}}
    for cat, minimum in [('science_new', rules['science']['progression_new_minimum_net_correct_change']),
                         ('horizon', rules['horizon']['minimum_net_correct_change'])]:
        count_delta = summaries[cat]['arms']['P1']['correct']['count'] - summaries[cat]['arms']['P0']['correct']['count']
        gates[cat + ':count'] = 'pass' if count_delta >= minimum else 'fail'
    floors = {}
    fraction = rules['stock_floors']['retained_fraction']
    for cat in rules['stock_floors']['categories']:
        floors[cat] = {}
        for arm in rules['stock_floors']['arms']:
            a = ARMS.index(arm)
            ci = interval(samples[cat][:, a, 0] - fraction * samples[cat][:, 0, 0])
            stock_count = summaries[cat]['arms']['stock']['correct']['count']
            floors[cat][arm] = {'margin': float(means[cat][a, 0] - fraction * means[cat][0, 0]),
                                'ci95': ci, 'stock_successes': stock_count,
                                'status': 'unassessable' if stock_count == 0 else lower_gate(ci, rules['stock_floors']['pass_requires_lower_ci_at_least'])}
    status = 'fail' if 'fail' in gates.values() else 'inconclusive' if 'inconclusive' in gates.values() else 'pass'
    quarantined = {r['id']: {arm: maps[arm][r['id']] for arm in ARMS} for r in rows if r['rehearsal_exposed']}
    return {'status': 'complete', 'bootstrap': rules['bootstrap'], 'eligible_cases': n,
            'progression': {'status': status, 'gates': gates, 'interpretation': rules['progression'][status]},
            'macro': macro, 'categories': summaries, 'bad_events': bad_events,
            'stock_floors': floors, 'sensitivity': sensitivities,
            'transitions': changes([rows[i] for i in eligible], values[eligible]),
            'quarantined_diagnostics': quarantined,
            'labels': {'intervals': 'Exploratory paired percentile bootstrap; no multiplicity correction.',
                       'quarantine': rules['quarantine']['policy'], 'promotion': 'None; parent remains default.',
                       'transitions': 'zero_to_one = gains for correctness, new bad events for loop/nontermination; one_to_zero = reverse.'}}


def main():
    value = read(OUT / 'normalized.json')
    # Reconstruct from frozen generation/grades and independent reviews, rather
    # than trusting editable normalized scores or merely their claimed hashes.
    if value != finish(write=False):
        raise ValueError('Normalized results differ from their source artifacts')
    result = analyze(value)
    result['normalized_sha256'] = digest(OUT / 'normalized.json')
    result['sources'] = value['sources']
    write_new(OUT / 'analysis.json', result)
    print('G1 progression:', result['progression']['status'], '(not model promotion)')


if __name__ == '__main__':
    main()
