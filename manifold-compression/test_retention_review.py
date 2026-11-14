import json

import pytest

import retention_review as review


def test_numeric_grading_uses_stated_final_answer_without_expected_value_hints():
    case = dict(id='new', category='primary', family='multiply', check={'kind': 'numeric', 'expected': '12'})
    assert review.grade_answer('3 x 4 = 12\n\n12', case)['grade']['correct']
    assert not review.grade_answer('3 x 4 = 12\n\n13', case)['grade']['correct']


def test_test_outputs_are_not_read_before_dev_review(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'load_data', lambda: {'test': []})
    with pytest.raises(FileNotFoundError):
        review.inputs(tmp_path, 'test')


def test_review_requires_complete_packet_identity_and_boolean_judgments(tmp_path):
    packet = tmp_path / 'packet.json'
    packet.write_text('{}')
    path = tmp_path / 'review.json'
    data = {'reviewer': 'claude', 'packet_sha256': review.sha(packet),
            'grades': {'one': {'correct': True, 'reason': 'All required facts'}}}
    path.write_text(json.dumps(data))
    assert review.read_review(path, packet, {'one'}, 'claude') == data
    with pytest.raises(ValueError, match='incomplete'):
        review.read_review(path, packet, {'one', 'two'}, 'claude')
    data['grades']['one']['correct'] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Boolean'):
        review.read_review(path, packet, {'one'}, 'claude')


def test_historical_interval_prefix_preserves_input_mutation_veto():
    case = dict(id='regression-code-intervals', category='program', family='historical:code-intervals',
                check={'kind': 'program', 'tests': [[{'intervals': [[2, 3], [0, 1]]}, [[0, 1], [2, 3]]]]})
    safe = 'def main_solution(intervals):\n return sorted(intervals)'
    mutation = 'def main_solution(intervals):\n intervals.sort()\n return intervals'
    assert review.grade_answer(safe, case)['grade']['correct']
    assert not review.grade_answer(mutation, case)['grade']['correct']
