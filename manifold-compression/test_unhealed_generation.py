import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import unhealed_generation as ug
from eval_data import table
from local_projection_models import build
from test_allocation_pilot import basis
from test_projection_models import tiny_model


def test_humaneval_grader_passes_every_reference_and_rejects_non_solutions():
    raw = {r['task_id'].replace('/', '-'): r for r in table(*ug.HUMANEVAL)}
    cases = ug.humaneval_cases()
    assert len(cases) == 164

    def grades(case):
        r, g = raw[case['id']], lambda answer: ug.grade_humaneval(answer, case['check'])['correct']
        return (g('Sure.\n```python\n' + r['prompt'] + r['canonical_solution'] + '```\nDone.'),
                g(r['canonical_solution']), g('```\n' + r['canonical_solution'] + '```'),
                g('```python\n' + r['prompt'] + '    pass\n```'), g(''),
                g('```python\nimport sys\nsys.exit(0)\n```'),
                g('```python\n' + r['prompt'] + r['canonical_solution']),
                g('First:\n```python\nx = 1\n```\nThen:\n```python\n' + r['prompt'] + r['canonical_solution']))
    with ThreadPoolExecutor(16) as pool:
        results = list(pool.map(grades, cases))
    # Columns: fenced, bare body, fenced body, stub, empty, early exit, unclosed final fence, closed then unclosed.
    for column, expected in enumerate((True, True, True, False, False, False, True, True)):
        assert all(row[column] is expected for row in results), column


def test_unclosed_fence_is_recorded():
    check = ug.humaneval_cases()[23]['check']
    grade = ug.grade_humaneval('```python\ndef strlen(string: str) -> int:\n    return len(string)\n', check)
    assert grade['correct'] and not grade['fences_closed']
    assert ug.grade_humaneval('```python\ndef strlen(string):\n    return len(string)\n```', check)['fences_closed']


def test_horizon_gate_uses_the_track_a_grader_on_fence_tolerant_code():
    case = next(c for c in ug.track_a_cases() if c['id'] == 'rh-test-program-0')
    code = ('from itertools import groupby\n\ndef main_solution(n, m):\n    s = n\n    for _ in range(m - 1):\n'
            "        s = ''.join(c + str(len(list(g))) for c, g in groupby(s))\n    return s\n")
    closed = ug.grade_case('```python\n' + code + '```', case)
    assert closed['correct'] and closed['track_a_grade']['grade']['correct'] and closed['fences_closed']
    unclosed = ug.grade_case('```python\n' + code, case)
    assert unclosed['correct'] and not unclosed['track_a_grade']['grade']['correct'] and not unclosed['fences_closed']
    wrong = ug.grade_case('```python\ndef main_solution(n, m):\n    return n\n', case)
    assert not wrong['correct'] and not wrong['track_a_grade']['grade']['correct']


def test_sandbox_rejects_forged_output_and_timeouts():
    assert ug.run_program('x = 1\n')
    assert not ug.run_program('import os\nos.write(1, b"\\npass\\n")\nraise SystemExit(0)\n')
    assert not ug.run_program('raise AssertionError\n')
    assert not ug.run_program('while True:\n    pass\n')
    assert not ug.run_program('import socket\nsocket.create_connection(("1.1.1.1", 80), 1)\n')


def test_gsm8k_final_answer_content_is_primary_with_format_and_flexible_recorded():
    check = {'expected': '1234'}
    flexible_only = ug.grade_gsm8k('So she earns $1,234 per week.', check)
    assert not flexible_only['correct'] and flexible_only['flexible_correct']
    bare = ug.grade_gsm8k('5 + 6 = 11\n1234', check)
    assert bare['correct'] and bare['bare_final_line'] and bare['final_answer_method'] == 'standalone_final_number'
    explicit = ug.grade_gsm8k('Adding up, the answer is 1,234.', check)
    assert explicit['correct'] and not explicit['bare_final_line']
    assert explicit['final_answer_method'] == 'legacy_explicit_answer_parser'
    boxed = ug.grade_gsm8k('Total\n**1234**', check)
    assert boxed['correct'] and not boxed['bare_final_line']
    # An incidental trailing number misleads last-number extraction, not the final-answer parser.
    assert not ug.grade_gsm8k('Total: 1234\n1234\n\nThis took 2 steps.', check)['flexible_correct']
    assert not ug.grade_gsm8k('1234 eggs, minus 4, gives 1230', check)['correct']
    assert not ug.grade_gsm8k('no number', check)['correct']
    # Codex's regressions: whole numeric tokens are never split into a matching fragment.
    for answer, expected in (('3/2', '2'), ('.5', '5'), ('1e3', '3'), ('so x = 1e3', '3')):
        grade = ug.grade_gsm8k(answer, {'expected': expected})
        assert not grade['correct'] and not grade['flexible_correct'], answer
    assert ug.grade_gsm8k('so x = 1e3', {'expected': '1000'})['flexible_correct']
    assert ug.gsm8k_answer('work\n#### 1,050') == '1050'


def test_gsm8k_sample_is_fixed_and_drawn_only_from_the_screened_pool():
    (first, pool), (second, _) = ug.gsm8k_cases(), ug.gsm8k_cases()
    assert first == second and len(first) == ug.GSM8K_COUNT
    assert pool['pool'] == pool['problems'] - pool['tf_scored'] - pool['rehearsal_exposed'] == 1319 - 400 - 2
    assert not {c['id'] for c in first} & set(ug.REHEARSAL_EXPOSED)
    assert not ug.tf_scored('gsm8k') & {c['id'] for c in first}
    assert len({c['id'] for c in first}) == len(first)
    blocked = {c['messages'][0]['content'].removesuffix(ug.GSM8K_SUFFIX) for c in first[:50]}
    screened, pool = ug.gsm8k_cases(blocked=lambda text: text.split('\n')[0] in blocked)
    assert pool['pool'] == 1319 - 400 - 2 - 50
    assert not {c['id'] for c in screened} & {c['id'] for c in first[:50]}


def test_track_a_part_is_the_full_test_split_plus_dev_programs_and_science():
    cases = ug.track_a_cases()
    counts = {}
    for c in cases:
        counts[c['category']] = counts.get(c['category'], 0) + 1
    assert counts == {'primary': 128, 'retention': 112, 'program': 18, 'science': 12}
    assert {c['id'] for c in cases if c['rehearsal_exposed']} == {'rh-test-retain-arithmetic-0', 'rh-test-program-3'}


def test_science_items_need_essential_clauses(tmp_path):
    item = {'id': 's', 'family': 'f', 'messages': [], 'reference_answer': 'r', 'source_url': 'u',
            'manual_rubric': {'essential': ['a'], 'optional': []}}
    path = tmp_path / 'science.json'
    path.write_text(json.dumps({'items': [item]}))
    assert ug.science_cases(path)[0]['category'] == 'science'
    path.write_text(json.dumps({'items': [{**item, 'manual_rubric': {'essential': [], 'optional': ['a']}}]}))
    with pytest.raises(ValueError, match='essential'):
        ug.science_cases(path)


def frozen_setup(tmp_path, monkeypatch):
    root, repo = tmp_path / 'mc', tmp_path / 'repo'
    root.mkdir()
    (repo / 'weights').mkdir(parents=True)
    (root / 'code.py').write_text('x = 1\n')
    (repo / 'weights/source.bin').write_text('source')
    (repo / 'weights/recipe.pt').write_text('recipe')
    monkeypatch.setattr(ug, 'OUT', tmp_path)
    monkeypatch.setattr(ug, 'ROOT', root)
    monkeypatch.setattr(ug, 'REPO', repo)
    monkeypatch.setattr(ug, 'dependencies', lambda: ['code.py'])
    monkeypatch.setattr(ug, 'PEER_FILES', [])
    monkeypatch.setattr(ug, 'DRIVER', 'code.py')
    monkeypatch.setattr(ug, 'check_suite', lambda suite, rules: None)
    monkeypatch.setattr(ug, 'arm_inputs', lambda arm: ['weights/source.bin'] + (['weights/recipe.pt'] if arm != 'stock' else []))
    (tmp_path / 'suite.json').write_text(json.dumps({'parts': {}}))
    (tmp_path / 'rules.json').write_text(json.dumps(rules_for_test()))
    return root, repo


def rules_for_test(**runtime):
    return {k: {} for k in ug.RULE_KEYS} | {'arms': ['stock', 'P0', 'P1'], 'runtime': {
        'device': 'cpu', 'precision': 'float32', 'threads': 16, 'batch': 16, 'arm_order': ['P0', 'P1', 'stock'],
        'part_order': ug.PART_ORDER, 'maximum_total_hours': 8,
        'maximum_hours_per_part': {'stock': 1.25, 'P0': 1.0, 'P1': 1.0}, **runtime}}


def freeze_args(tmp_path, **changes):
    import argparse
    return argparse.Namespace(**({'rules': tmp_path / 'rules.json', 'reviewers': ['codex', 'claude']} | changes))


def test_freeze_refuses_incomplete_rules_one_reviewer_or_bad_runtime(tmp_path, monkeypatch):
    frozen_setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='reviewers'):
        ug.stage_freeze(freeze_args(tmp_path, reviewers=['claude']))
    for runtime in ({'maximum_hours_per_part': {'stock': 1.25, 'P0': 1}}, {'maximum_hours_per_part': {'stock': 0, 'P0': 1, 'P1': 1}},
                    {'maximum_hours_per_part': {'stock': 4, 'P0': 1, 'P1': 1}}, {'maximum_total_hours': 0},
                    {'maximum_total_hours': 9}, {'arm_order': ['stock', 'P0', 'P1']}, {'part_order': ug.PART_ORDER[::-1]},
                    {'device': 'cuda'}, {'threads': 0}):
        (tmp_path / 'rules.json').write_text(json.dumps(rules_for_test(**runtime)))
        with pytest.raises(ValueError):
            ug.stage_freeze(freeze_args(tmp_path))
    (tmp_path / 'rules.json').write_text(json.dumps(rules_for_test() | {'arms': ['stock', 'P1']}))
    with pytest.raises(ValueError, match='exactly'):
        ug.stage_freeze(freeze_args(tmp_path))
    (tmp_path / 'rules.json').write_text(json.dumps({'arms': ['stock']}))
    with pytest.raises(ValueError, match='lacks'):
        ug.stage_freeze(freeze_args(tmp_path))
    assert not (tmp_path / 'protocol.json').exists()


def test_generation_refuses_draft_protocol_or_changed_suite_code_and_arm_inputs(tmp_path, monkeypatch):
    root, repo = frozen_setup(tmp_path, monkeypatch)
    ug.stage_freeze(freeze_args(tmp_path))
    with pytest.raises(ValueError, match='not rewritten'):
        ug.stage_freeze(freeze_args(tmp_path))
    protocol = json.loads((tmp_path / 'protocol.json').read_text())
    assert ug.require_frozen('P1') == protocol
    with pytest.raises(RuntimeError, match='not a frozen arm'):
        ug.require_frozen('P1-norefit')
    for changed in (tmp_path / 'suite.json', root / 'code.py', repo / 'weights/recipe.pt'):
        original = changed.read_text()
        changed.write_text(original + ' ')
        with pytest.raises(RuntimeError, match='changed'):
            ug.require_frozen('P1')
        changed.write_text(original)
    (repo / 'weights/recipe.pt').write_text('other')
    ug.require_frozen('stock')  # stock does not depend on the recipe
    (tmp_path / 'protocol.json').write_text(json.dumps({**protocol, 'status': 'draft'}))
    with pytest.raises(RuntimeError, match='not frozen'):
        ug.require_frozen()


def test_shared_clock_starts_once_and_cannot_be_reset(tmp_path, monkeypatch):
    frozen_setup(tmp_path, monkeypatch)
    ug.stage_freeze(freeze_args(tmp_path))
    protocol = json.loads((tmp_path / 'protocol.json').read_text())
    first = ug.shared_deadline(protocol)
    monkeypatch.setattr(ug.time, 'time', lambda: first - 3600)
    assert ug.shared_deadline(protocol) == first
    monkeypatch.setattr(ug.time, 'time', lambda: first + 1)
    with pytest.raises(TimeoutError):
        ug.shared_deadline(protocol)
    clock = json.loads((tmp_path / 'clock.json').read_text())
    (tmp_path / 'clock.json').write_text(json.dumps(clock | {'protocol_sha256': 'other'}))
    monkeypatch.setattr(ug.time, 'time', lambda: first - 3600)
    with pytest.raises(RuntimeError, match='another protocol'):
        ug.shared_deadline(protocol)


def test_humaneval_primary_mask_is_the_complement_of_training_overlap():
    blocked = {'HumanEval-3', 'HumanEval-7'}
    cases = ug.humaneval_cases(lambda text: any(f'def {n}(' in text for n in ('below_zero', 'filter_by_substring')))
    masked = {c['id'] for c in cases if not c['primary_mask']}
    assert masked == blocked | {'HumanEval-0', 'HumanEval-23'}
    assert all(c['primary_mask'] == (not c['training_overlap'] and not c['rehearsal_exposed']) for c in cases)


def test_grading_validates_metadata_and_rejects_duplicate_rows(tmp_path, monkeypatch):
    frozen_setup(tmp_path, monkeypatch)
    case = {'id': 'gsm8k-1', 'family': 'gsm8k', 'category': 'gsm8k', 'group': 'gsm8k-1',
            'check': {'kind': 'gsm8k', 'expected': '4'}, 'rehearsal_exposed': False}
    (tmp_path / 'suite.json').write_text(json.dumps({'parts': {'gsm8k': [case]}}))
    ug.stage_freeze(freeze_args(tmp_path))
    protocol = json.loads((tmp_path / 'protocol.json').read_text())
    meta = {'arm': 'P1', 'part': 'gsm8k', 'protocol_sha256': ug.sha(tmp_path / 'protocol.json'),
            'input_sha256': {n: protocol['input_sha256'][n] for n in ug.arm_inputs('P1')},
            'torch': protocol['torch'], 'threads': 16, 'batch': 16, 'deadline': 2e9, 'completed_at': 1.9e9}
    row = {k: case[k] for k in ('id', 'family', 'category', 'group')} | {
        'answer': '2 + 2\n4', 'ended': True, 'combined_loop_flag': False}
    out = tmp_path / 'generations'
    out.mkdir()

    def grade(meta, rows):
        (out / 'P1-gsm8k.meta.json').write_text(json.dumps(meta))
        (out / 'P1-gsm8k.json').write_text(json.dumps({'status': 'complete', 'cases': rows}))
        import argparse
        ug.stage_grade(argparse.Namespace(arm='P1'))

    for wrong in ({'threads': 8}, {'arm': 'P0'}, {'input_sha256': {}}, {'protocol_sha256': 'x'}):
        with pytest.raises(ValueError, match='metadata'):
            grade(meta | wrong, [row])
    for late in ({'completed_at': 2.1e9}, {'completed_at': None}):
        with pytest.raises(ValueError, match='deadline'):
            grade(meta | late, [row])
    with pytest.raises(ValueError, match='one-to-one'):
        grade(meta, [row, row | {'answer': '5'}])
    grade(meta, [row])
    graded = json.loads((tmp_path / 'grades/P1.json').read_text())
    assert graded['cases']['gsm8k-1']['grade']['correct']


def test_norefit_control_equals_the_projected_mlp_restricted_to_kept_neurons():
    torch.manual_seed(5)
    model = build(tiny_model('qwen', True), [basis()] * 2, [1.03] * 5)
    keep = [torch.randperm(48)[:30].sort().values for _ in range(2)]
    mlp = model.model.layers[1].mlp
    x = torch.randn(3, 7, mlp.gate_proj.weight.shape[1])
    mask = torch.zeros(48)
    mask[keep[1]] = 1
    expected = mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x) * mask)
    ug.prune_without_refit(model, {'keep': keep, 'mlp_width': 30})
    torch.testing.assert_close(model.model.layers[1].mlp(x), expected, atol=1e-6, rtol=1e-5)
    assert model.model.layers[0].mlp.down_proj.weight.shape[1] == 30
    assert model.config.intermediate_size == 30


def test_the_sealed_suite_matches_its_inputs_and_the_frozen_rules():
    suite = json.loads((ug.OUT / 'suite.json').read_text())
    rules = json.loads((ug.ROOT / 'unhealed_rules.json').read_text())
    ug.check_suite(suite, rules)
    with pytest.raises(ValueError, match='inputs differ'):
        ug.check_suite(suite | {'inputs': suite['inputs'] | {'science_new': 'other'}}, rules)
    shrunk = suite | {'parts': suite['parts'] | {'gsm8k': suite['parts']['gsm8k'][:-1]}}
    with pytest.raises(ValueError):
        ug.check_suite(shrunk, rules)


def generation_fixture(tmp_path, monkeypatch, hours, load):
    """A frozen one-case suite with a synthetic model loader and generator; no real model runs."""
    import retention_run
    frozen_setup(tmp_path, monkeypatch)
    case = {'id': 'gsm8k-1', 'family': 'gsm8k', 'category': 'gsm8k', 'group': 'gsm8k-1', 'max_new_tokens': 8,
            'messages': [{'role': 'user', 'content': 'x'}], 'check': {'kind': 'gsm8k', 'expected': '4'},
            'rehearsal_exposed': False}
    (tmp_path / 'suite.json').write_text(json.dumps({'parts': {'gsm8k': [case]}}))
    (tmp_path / 'rules.json').write_text(json.dumps(rules_for_test(
        maximum_hours_per_part={'stock': hours, 'P0': hours, 'P1': hours})))
    ug.stage_freeze(freeze_args(tmp_path))
    monkeypatch.setitem(ug.ARMS, 'P1', load)
    monkeypatch.setattr(retention_run, 'configure', lambda threads: None)

    def fake_generate(model, tok, cases, path, deadline, batch_size):
        ug.save({'status': 'complete', 'cases': []}, path)
        return {'cases': len(cases)}
    monkeypatch.setattr(retention_run, 'raw_generate', fake_generate)
    import argparse
    return argparse.Namespace(arm='P1', part='gsm8k', threads=16, batch=16)


def test_alarm_stops_work_in_progress_at_the_deadline(tmp_path, monkeypatch):
    def slow_load():
        time.sleep(30)
    args = generation_fixture(tmp_path, monkeypatch, 1 / 3600, slow_load)  # one-second part bound
    start = time.time()
    with pytest.raises(TimeoutError):
        ug.stage_generate(args)
    assert time.time() - start < 5
    assert not (tmp_path / 'generations/P1-gsm8k.meta.json').exists()


def test_only_an_in_time_completion_is_stamped(tmp_path, monkeypatch):
    args = generation_fixture(tmp_path, monkeypatch, 1., lambda: object())
    ug.stage_generate(args)
    meta = json.loads((tmp_path / 'generations/P1-gsm8k.meta.json').read_text())
    assert meta['completed_at'] <= meta['deadline']
    import signal
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
