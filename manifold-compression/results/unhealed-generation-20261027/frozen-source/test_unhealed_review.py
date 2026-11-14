import copy
import json
import sys
from types import SimpleNamespace

import pytest
import unhealed_review as review


def item(id_='blind'):
    return {'id': id_, 'rubric_kind': 'new', 'rubric': {'essential': ['a', 'b']}}


def rating(id_='blind', correct=True):
    return {'id': id_, 'correct': correct, 'essential': [True, correct],
            'core_contradiction': False, 'optional_errors': [], 'reason': 'Fixture reasoning.'}


def test_science_essentials_cannot_be_overridden_by_correct_label():
    row = rating()
    row['essential'][1] = False
    with pytest.raises(ValueError, match='disagrees'):
        review.validate_rating(row, item())
    row['correct'] = False
    review.validate_rating(row, item())
    row = rating()
    row['optional_errors'] = ['Unrequested side-detail is inaccurate.']
    assert review.validate_rating(row, item())['correct']  # Optional errors alone do not fail.
    row['core_contradiction'] = True
    with pytest.raises(ValueError, match='disagrees'):
        review.validate_rating(row, item())
    row = rating()
    row['correct'] = 1
    with pytest.raises(ValueError, match='boolean'):
        review.validate_rating(row, item())


def test_duplicate_json_keys_and_rating_ids_are_rejected(tmp_path):
    p = tmp_path / 'bad.json'
    p.write_text('{"correct": false, "correct": true}')
    with pytest.raises(ValueError, match='Duplicate JSON'):
        review.read(p)
    with pytest.raises(ValueError, match='Duplicate ids'):
        review.index([rating(), rating()], 'ratings')


def test_adjudication_binds_both_reviews_and_exact_disagreements():
    packet = {'blind': item()}
    reviews = {'claude': {'blind': rating()}, 'codex': {'blind': rating(correct=False)}}
    hashes = {'claude': 'a', 'codex': 'b'}
    diff = {'review_sha256': hashes, 'items': [{'id': 'blind'}]}
    consensus = {'review_sha256': hashes, 'approved_by': ['claude', 'codex'], 'items': [rating(correct=False)]}
    assert not review.resolve(packet, reviews, hashes, diff, consensus)['blind']['correct']
    for broken in [None, consensus | {'approved_by': ['claude']}, consensus | {'items': []},
                   consensus | {'review_sha256': hashes | {'codex': 'changed'}}]:
        with pytest.raises(ValueError):
            review.resolve(packet, reviews, hashes, diff, broken)
    # Even equal pass/fail labels can hide clause disagreements that need review.
    reviews['claude']['blind'] = rating(correct=False) | {'essential': [False, False]}
    assert review.disagreement_ids(reviews) == ['blind']
    reviews['claude']['blind'] = rating(correct=False)
    empty = {'review_sha256': hashes, 'items': []}
    with pytest.raises(ValueError, match='Adjudication'):
        review.resolve(packet, reviews, hashes, empty)


def test_blind_workflow_refuses_one_reviewer_and_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'OUT', tmp_path)
    records = {}
    rules = {'science': {'reviewers': ['claude', 'codex']}}
    for arm in review.ARMS:
        records[arm] = {}
        for name, part, rubric in [('new', 'science-new', {'essential': ['one', 'two']}),
                                   ('legacy', 'track-a', {'required': ['one'], 'disqualifying': ['wrong']})]:
            c = {'id': name, 'category': 'science', 'family': name, 'group': name,
                 'messages': [{'role': 'user', 'content': 'Fixture question'}],
                 'reference_answer': 'Fixture reference', 'source_url': 'https://example.org',
                 'manual_rubric': rubric, 'rehearsal_exposed': False}
            records[arm][name] = {'case': c, 'part': part, 'answer': 'Fixture answer',
                                  'graded': {'ended': True, 'loop': False, 'grade': None}}
    snapshot = ({'rules': rules}, {}, records, {'input': 'original'})
    monkeypatch.setattr(review, 'snapshot', lambda: copy.deepcopy(snapshot))
    review.prepare()
    folder = tmp_path / 'review'
    packet = review.read(folder / 'packet.json')
    assert len(packet['items']) == 6
    assert all('arm' not in p for p in packet['items'])
    sha = review.digest(folder / 'packet.json')
    ratings = []
    for p in packet['items']:
        ratings.append(rating(p['id']) if p['rubric_kind'] == 'new'
                       else {'id': p['id'], 'correct': True, 'reason': 'Original rubric met.'})
    review.write_new(folder / 'reviewer-claude.json', {'reviewer': 'claude', 'packet_sha256': sha, 'items': ratings})
    with pytest.raises(ValueError, match='Both independent'):
        review.disagreements()
    assert not (folder / 'disagreements.json').exists()
    review.write_new(folder / 'reviewer-codex.json', {'reviewer': 'codex', 'packet_sha256': sha, 'items': ratings})
    review.disagreements()
    with pytest.raises(FileNotFoundError):
        review.finish()  # No reveal/decision until explicit final sign-off, even for zero disputes.
    diff = review.read(folder / 'disagreements.json')
    review.write_new(folder / 'adjudication.json', {'packet_sha256': sha, 'review_sha256': diff['review_sha256'],
                                                 'approved_by': ['claude', 'codex'], 'items': []})
    result = review.finish()
    assert result == review.finish(write=False)
    assert all(r['correct'] for rows in result['arms'].values() for r in rows)
    snapshot[3]['input'] = 'changed'
    with pytest.raises(ValueError, match='inputs changed'):
        review.finish(write=False)


def test_snapshot_rejects_generation_changed_after_grading(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'OUT', tmp_path)
    suite = {'parts': {'track-a': [{'id': 'a', 'family': 'f', 'category': 'primary', 'group': 'a'}]}}
    protocol = {'arms': list(review.ARMS), 'input_sha256': {}, 'rules': {}}
    review.write_new(tmp_path / 'suite.json', suite)
    review.write_new(tmp_path / 'protocol.json', protocol)
    monkeypatch.setitem(sys.modules, 'unhealed_generation', SimpleNamespace(
        OUT=tmp_path, require_frozen=lambda: protocol, check_meta=lambda *args: None))
    monkeypatch.setattr(review, 'validate_suite', lambda *args: {'a': suite['parts']['track-a'][0]})
    source = tmp_path / 'generations/stock-track-a.json'
    review.write_new(source, {'status': 'complete', 'cases': [suite['parts']['track-a'][0] | {'answer': '4'}]})
    review.write_new(source.with_suffix('.meta.json'), {})
    review.write_new(tmp_path / 'grades/stock.json', {'arm': 'stock', 'suite_sha256': review.digest(tmp_path / 'suite.json'),
                                                    'protocol_sha256': review.digest(tmp_path / 'protocol.json'),
                                                    'generation_sha256': {p: review.digest(source) for p in review.PARTS},
                                                    'cases': {'a': {}}})
    source.write_text(source.read_text().replace('"4"', '"5"'))
    with pytest.raises(ValueError, match='changed since grading'):
        review.snapshot()
