import copy
import json
from pathlib import Path

import numpy as np
import pytest
import unhealed_analysis as analysis


def fixture():
    rules = json.loads((Path(__file__).parent / 'unhealed_rules.json').read_text())
    rules['bootstrap']['draws'] = 2000
    # Small complete synthetic suite, with fixed count checks adjusted explicitly.
    cats = ['arithmetic', 'replay', 'gsm8k', 'humaneval_primary', 'humaneval_overlap',
            'horizon', 'science_new', 'science_legacy']
    parts = {'arithmetic': 'track-a', 'replay': 'track-a', 'horizon': 'track-a',
             'science_legacy': 'track-a', 'gsm8k': 'gsm8k', 'humaneval_primary': 'humaneval',
             'humaneval_overlap': 'humaneval', 'science_new': 'science-new'}
    rules['quarantine']['excluded_ids'] = ['exposed']
    rules['reporting']['expected_eligible_counts'] = {c: 20 for c in cats}
    rules['reporting']['expected_generated_counts'] = {'track-a': 81, 'gsm8k': 20, 'humaneval': 40, 'science-new': 20}
    arms = {}
    for arm in analysis.ARMS:
        rows = []
        for cat in cats:
            for i in range(20):
                rows.append({'id': f'{cat}-{i}', 'part': parts[cat], 'category': cat,
                             'family': f'{cat}-{i % 2}', 'group': f'{cat}-{i // 2}' if cat == 'science_new' else f'{cat}-{i}',
                             'rehearsal_exposed': False, 'primary_mask': cat == 'humaneval_primary',
                             'training_overlap': cat == 'humaneval_overlap', 'tf_scored': None,
                             'correct': arm != 'P0', 'loop': False, 'ended': True})
        rows.append(rows[0] | {'id': 'exposed', 'rehearsal_exposed': True,
                              'correct': arm == 'P0', 'loop': arm == 'P1', 'ended': arm != 'P1'})
        arms[arm] = rows
    return {'status': 'complete', 'arms': arms, 'rules': rules,
            'science_review': {'reviewers': ['claude', 'codex'], 'adjudicated': True}}


def test_known_complete_improvement_and_quarantine_cannot_affect_gates():
    value = fixture()
    result = analysis.analyze(value)
    assert result['progression']['status'] == 'pass'
    assert result['macro']['delta'] == 1
    assert result['macro']['ci95'] == [1, 1]
    assert result['eligible_cases'] == 160
    assert result['bad_events']['loop']['ci95'] == [0, 0]
    assert len(result['quarantined_diagnostics']) == 1
    for cat in result['stock_floors'].values():
        assert cat['P0']['status'] == 'fail'
        assert cat['P1']['status'] == 'pass'
    for arm in value['arms']:
        value['arms'][arm][-1].update(correct=False, loop=True, ended=False)
    changed = analysis.analyze(value)
    for key in ('progression', 'macro', 'bad_events', 'stock_floors', 'categories'):
        assert result[key] == changed[key]


def test_horizon_count_gate_separate_from_primary_humaneval():
    value = fixture()
    for arm in ('P0', 'P1'):
        for row in value['arms'][arm]:
            if row['category'] == 'horizon':
                row['correct'] = arm == 'P0'
    result = analysis.analyze(value)
    assert result['macro']['status'] == 'pass'
    assert result['progression']['gates']['horizon:count'] == 'fail'
    assert result['progression']['status'] == 'fail'


def test_zero_stock_is_unassessable_without_changing_progression():
    value = fixture()
    for row in value['arms']['stock']:
        row['correct'] = False
    result = analysis.analyze(value)
    assert result['progression']['status'] == 'pass'
    assert all(v['status'] == 'unassessable' for c in result['stock_floors'].values() for v in c.values())


def test_inconclusive_macro_and_separate_noninferiority_event_failures():
    value = fixture()
    for rows in value['arms'].values():
        for row in rows:
            row['correct'] = False
    value['arms']['P1'][0]['correct'] = True
    result = analysis.analyze(value)
    assert result['macro']['ci95'][0] == 0 < result['macro']['ci95'][1]
    assert result['progression']['status'] == 'inconclusive'
    value = fixture()
    for arm in ('P0', 'P1'):
        for row in value['arms'][arm]:
            if row['category'] == 'replay' and not row['rehearsal_exposed']:
                row['correct'] = arm == 'P0'
    result = analysis.analyze(value)
    assert result['macro']['status'] == 'pass'
    assert result['progression']['gates']['noninferiority:replay'] == 'fail'
    value = fixture()
    for row in value['arms']['P1']:
        if row['category'] == 'humaneval_overlap':
            row.update(loop=True, ended=False)
    result = analysis.analyze(value)
    assert result['macro']['status'] == 'pass'
    assert result['bad_events']['loop']['status'] == 'fail'
    assert result['bad_events']['nontermination']['status'] == 'fail'


def test_shared_pairs_groups_and_fixed_family_weights():
    # Two identical outcomes per concept: group sampling must not split a pair.
    rows = [{'id': str(i), 'family': 'f', 'group': str(i // 2), 'category': 'science_new'} for i in range(4)]
    values = np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=float)
    sampled = analysis.bootstrap_means(rows, values, np.random.default_rng(7), 1000)
    assert set(sampled[:, 0]) == {0, .5, 1}
    assert np.array_equal(sampled[:, 0], sampled[:, 1])  # Arms paired in every draw.
    # Family proportions must not drift when a small family differs systematically.
    rows = [{'id': str(i), 'family': 'small' if i == 0 else 'large',
             'group': str(i), 'category': 'arithmetic'} for i in range(4)]
    values = np.asarray([[1], [0], [0], [0]], dtype=float)
    sampled = analysis.bootstrap_means(rows, values, np.random.default_rng(8), 1000)
    assert np.all(sampled == .25)
    sensitivity = analysis.bootstrap_means(rows, values, np.random.default_rng(8), 1000, sensitivity=True)
    assert set(sensitivity[:, 0]) == {0, .25, 1}


def test_interval_boundaries_and_missing_or_unpaired_data_fail_closed():
    assert analysis.lower_gate([0, .1], 0, strict=True) == 'inconclusive'
    assert analysis.lower_gate([0, 0], 0, strict=True) == 'fail'
    assert analysis.lower_gate([-.05, .1], -.05) == 'pass'
    assert analysis.upper_gate([0, .02], .02) == 'pass'
    value = fixture()
    for mutate in [lambda v: v['arms']['P1'].pop(),
                   lambda v: v['arms']['P1'][0].update(group='different'),
                   lambda v: v['arms']['stock'][0].update(rehearsal_exposed=True),
                   lambda v: v['science_review'].update(adjudicated=False),
                   lambda v: v['arms']['P1'].append(copy.deepcopy(v['arms']['P1'][0]))]:
        broken = copy.deepcopy(value)
        mutate(broken)
        with pytest.raises(ValueError):
            analysis.analyze(broken)
