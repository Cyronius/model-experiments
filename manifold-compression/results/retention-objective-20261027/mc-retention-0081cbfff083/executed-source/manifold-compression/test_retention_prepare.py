import random

from curriculum_tasks import FAMILIES, math_example
from retention_prepare import MIX, conservative_mutations, math_group, numeric_splits, schedule


def test_canonical_math_groups_exclude_commuted_instances():
    assert math_group('multiply', [17, 29]) == math_group('multiply', [29, 17])
    assert math_group('add_subtract', [12, 13, 4]) == math_group('add_subtract', [13, 12, 4])
    assert math_group('fraction', [1, 2, 1, 3]) == math_group('fraction', [2, 6, 2, 4])


def test_numeric_sets_are_disjoint_and_keep_exact_oracles():
    from horizon_data import arithmetic
    rng = random.Random(26102711)
    first = math_example(rng, FAMILIES[0])
    blocked = math_group(first['family'], first['values'])
    sets = numeric_splits({blocked}, set(), dev_n=2, test_n=3)
    dev, test = sets['dev'], sets['test']
    assert len(dev) == 15 * 2 and len(test) == 15 * 3
    groups = [r['group'] for r in dev + test]
    assert len(set(groups)) == len(groups) and blocked not in groups
    for row in dev + test:
        if 'oracle' in row:
            assert arithmetic(row['oracle']['expression']) == arithmetic(row['check']['expected'])


def test_token_schedule_matches_mix_and_records_repetition():
    rows = [{'kind': kind, 'targets': [1] * n} for kind in MIX for n in [11, 31, 70]]
    updates, report = schedule(rows, steps=4, tokens_per_update=256)
    assert len(updates) == 4
    assert all(sum(len(rows[i]['targets']) for i in update) >= 256 for update in updates)
    for kind, weight in MIX.items():
        assert abs(report['tokens_by_kind'][kind] - report['answer_tokens'] * weight) < 100
    assert report['maximum_row_presentations'] > 1
    assert schedule(rows, 4, 256) == (updates, report)


def test_conservative_mutations_do_not_assume_empty_or_negative_is_valid():
    original = {'xs': [2, 4, 4, 8], 'n': 1, 'flag': True, 'name': 'x'}
    mutations = list(conservative_mutations(original))
    assert mutations
    for row in mutations:
        assert len(row['xs']) == 4 and row['xs'][1] == row['xs'][2]
        assert row['xs'] == sorted(row['xs']) and min(row['xs']) > 0
        assert row['n'] > 0 and row['flag'] is True and row['name'] == 'x'
