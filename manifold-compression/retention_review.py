"""Local isolated grading and two-reviewer science packets for frozen endpoints.

Test outputs cannot be opened by this script until development science has two
reviews and the frozen development gates have been applied. No model promotion.
"""
import argparse
from collections import Counter
import copy
from fractions import Fraction
import json
from pathlib import Path
import random
import uuid

from curriculum_numeric_audit import final_numeric
from horizon_data import run_reference
from local_projection_probe import grade_case, grade_program
from retention_healing import SelectionPolicy, compare_cases
from retention_prepare import OUT, ROOT, save, sha

ARMS = ('A0', 'A1', 'A2', 'S0', 'S1', 'S2')


def load_data():
    frozen = json.loads((OUT / 'frozen.json').read_text())
    if sha(OUT / 'data.json') != frozen['data_sha256']:
        raise ValueError('Frozen evaluation data changed')
    return json.loads((OUT / 'data.json').read_text())


def grade_answer(answer, case):
    if case['category'] == 'science':
        return None
    spec = copy.deepcopy(case)
    # The original interval test also checks that inputs were not mutated.
    if spec['id'].startswith('regression-'):
        spec['id'] = spec['id'][len('regression-'):]
    result = {'grade': grade_case(answer, spec)}
    if spec['check']['kind'] == 'numeric':
        result['original_grade'] = copy.deepcopy(result['grade'])
        try:
            number, method = final_numeric(answer)
            result['grade'] = {'correct': number == Fraction(str(spec['check']['expected'])),
                               'parsed': True, 'format_pass': answer.strip() == str(spec['check']['expected'])}
            result['numeric_audit'] = {'value': str(number), 'method': method}
        except (ValueError, ZeroDivisionError, TypeError):
            result['grade'] = {'correct': False, 'parsed': False, 'format_pass': False}
    if 'supplied_check' in spec:
        result['supplied_grade'] = grade_program(answer, spec['supplied_check']['tests'])
    return result


def inputs(run, split):
    data = load_data()
    review = run / 'review'
    if split == 'test':
        gate = json.loads((review / 'dev-review.json').read_text())
        if gate['status'] != 'development_review_complete' or gate['data_sha256'] != sha(OUT / 'data.json'):
            raise ValueError('Development review must finish before test unsealing')
    cases = {c['id']: c for c in data[split]}
    outputs, sources = {}, {}
    for arm in ARMS:
        path = run / 'cuda' / f'{arm}-{split}{"-sealed" if split == "test" else ""}.json'
        raw = json.loads(path.read_text())
        if raw['status'] != 'complete':
            raise ValueError('Partial endpoint is not a completed suite: ' + arm)
        records = {r['id']: r for r in raw['cases']}
        if len(records) != len(raw['cases']) or records.keys() != cases.keys():
            raise ValueError('Missing, duplicate or unexpected case IDs')
        for id_, record in records.items():
            if any(record[key] != cases[id_][key] for key in ('family', 'category', 'group')):
                raise ValueError('Generated case identity mismatch')
        outputs[arm] = records
        sources[arm] = sha(path)
    return data, cases, outputs, sources


def prepare(run, split):
    data, cases, outputs, sources = inputs(run, split)
    review = run / 'review'
    review.mkdir(exist_ok=True)
    destination = review / (split + '-automatic.json')
    if destination.exists() or (review / (split + '-science-blind.json')).exists():
        raise ValueError('Review packet already exists')
    if run_reference('def main_solution(x):\n return x+1', {'x': 3}) != {'ok': True, 'result': 4}:
        raise RuntimeError('Networkless isolated verifier unavailable')
    automatic, blind, key = {}, [], {}
    for arm in ARMS:
        automatic[arm] = {}
        for id_, row in outputs[arm].items():
            case = cases[id_]
            if case['category'] != 'science':
                automatic[arm][id_] = grade_answer(row['answer'], case)
                continue
            blind_id = uuid.uuid4().hex[:12]
            key[blind_id] = {'arm': arm, 'case': id_}
            blind.append({'id': blind_id, 'case': id_, 'messages': case['messages'],
                          'answer': row['answer'], 'rubric': case['manual_rubric'],
                          'reference_answer': case['reference_answer'], 'source_url': case['source_url']})
    random.SystemRandom().shuffle(blind)
    if len(key) != len(blind):
        raise ValueError('Blind identifier collision')
    save({'split': split, 'data_sha256': sha(OUT / 'data.json'), 'raw_sha256': sources,
          'grader_code_sha256': sha(__file__), 'arms': automatic}, destination)
    save({'split': split, 'instructions': 'Blind to arm. All required facts, no disqualifying factual errors. Grade factual content independently of repetition/termination.',
          'cases': blind}, review / (split + '-science-blind.json'))
    save(key, review / (split + '-science-key.json'))
    print(json.dumps({'split': split, 'automatic_cases': sum(map(len, automatic.values())),
                      'blind_cases': len(blind), 'packet': str(review / (split + '-science-blind.json'))}), flush=True)


def read_review(path, packet, ids, reviewer):
    result = json.loads(path.read_text())
    if result['reviewer'] != reviewer or result['packet_sha256'] != sha(packet):
        raise ValueError('Reviewer or packet provenance mismatch')
    if result['grades'].keys() != ids:
        raise ValueError('Science judgments incomplete')
    for record in result['grades'].values():
        if type(record['correct']) is not bool or not record.get('reason'):
            raise ValueError('Science judgment needs Boolean correctness and reason')
    return result


def finish(run, split):
    data, cases, outputs, sources = inputs(run, split)
    review = run / 'review'
    packet = review / (split + '-science-blind.json')
    key = json.loads((review / (split + '-science-key.json')).read_text())
    ids = {r['id'] for r in json.loads(packet.read_text())['cases']}
    if key.keys() != ids:
        raise ValueError('Blind key differs from packet')
    judgments = {name: read_review(review / f'{split}-science-{name}.json', packet, ids, name)
                 for name in ('codex', 'claude')}
    disagreement = [i for i in ids if judgments['codex']['grades'][i]['correct'] != judgments['claude']['grades'][i]['correct']]
    adjudicated = {}
    if disagreement:
        path = review / f'{split}-science-consensus.json'
        if not path.exists():
            save({'unresolved': disagreement, 'packet_sha256': sha(packet)}, review / (split + '-science-disagreements.json'))
            raise ValueError('Science disagreements require joint adjudication')
        consensus = json.loads(path.read_text())
        if consensus['packet_sha256'] != sha(packet) or set(consensus['approved_by']) != {'claude', 'codex'}:
            raise ValueError('Consensus is not approved by both reviewers')
        adjudicated = consensus['grades']
        if adjudicated.keys() != set(disagreement):
            raise ValueError('Consensus must cover exactly the disputed cases')
        if any(type(r['correct']) is not bool or not r.get('reason') for r in adjudicated.values()):
            raise ValueError('Consensus judgment invalid')
    science = {arm: {} for arm in ARMS}
    for blind_id, mapping in key.items():
        decision = adjudicated.get(blind_id, judgments['codex']['grades'][blind_id])
        science[mapping['arm']][mapping['case']] = decision
    automatic = json.loads((review / (split + '-automatic.json')).read_text())
    if automatic['raw_sha256'] != sources or automatic['data_sha256'] != sha(OUT / 'data.json'):
        raise ValueError('Inputs changed since automatic grading')
    normalized, details, summary = {}, {}, {}
    for arm, records in outputs.items():
        normalized[arm], details[arm] = [], []
        for id_, row in records.items():
            case = cases[id_]
            grade = {'grade': science[arm][id_]} if case['category'] == 'science' else automatic['arms'][arm][id_]
            normalized[arm].append({k: case[k] for k in ('id', 'category', 'family', 'group')} |
                {'correct': grade['grade']['correct'], 'loop': row['combined_loop_flag'], 'ended': row['ended']})
            details[arm].append(row | grade)
        summary[arm] = {category: {'correct': sum(r['correct'] for r in normalized[arm] if r['category'] == category),
                                  'n': sum(r['category'] == category for r in normalized[arm])}
                        for category in ('primary', 'retention', 'program', 'science')}
        summary[arm].update(loops=sum(r['loop'] for r in normalized[arm]), ended=sum(r['ended'] for r in normalized[arm]))
    policy = SelectionPolicy(**data['protocol']['selection_policy'])
    comparisons = {arm: compare_cases(normalized[arm[0] + '0'], normalized[arm], policy)
                   for arm in ('A1', 'A2', 'S1', 'S2')}
    objective_contrasts = {}
    for before, after in (('A1', 'A2'), ('S1', 'S2')):
        contrast = compare_cases(normalized[before], normalized[after], policy)
        contrast.pop('eligible')
        contrast.pop('reasons')
        objective_contrasts[before + '-vs-' + after] = contrast | {
            'scope': 'Descriptive paired effect of adding replay KL; not a checkpoint-selection gate'}
    result = {'status': 'development_review_complete' if split == 'dev' else 'test_review_complete',
              'data_sha256': sha(OUT / 'data.json'), 'raw_sha256': sources,
              'science_packet_sha256': sha(packet), 'reviewers': ['codex', 'claude'],
              'science_disagreements_adjudicated': disagreement, 'summary': summary,
              'comparisons': comparisons, 'objective_contrasts': objective_contrasts,
              'normalized': normalized, 'no_auto_promotion': True,
              'blinding_limit': 'Arm labels hidden during science grading; response style may still suggest model identity',
              'scope': 'Development eligibility' if split == 'dev' else 'Descriptive fixed-endpoint test comparisons; not checkpoint selection',
              'promotion': 'Parent default; requires all planned numerical, broad-retention and historical-veto evidence plus joint review'}
    save(details, review / (split + '-graded-answers.json'))
    save(result, review / (split + '-review.json'))
    print(json.dumps({'split': split, 'summary': summary,
                      'gates': {k: {'eligible': v['eligible'], 'reasons': v['reasons']} for k, v in comparisons.items()}}), flush=True)


def regression(run):
    data = load_data()
    if run_reference('def main_solution(x):\n return x+1', {'x': 3}) != {'ok': True, 'result': 4}:
        raise RuntimeError('Networkless isolated verifier unavailable')
    cases = {c['id']: c for c in data['regression']}
    graded, missing = {}, []
    for arm in ARMS:
        path = run / 'cuda' / (arm + '-regression.json')
        if not path.exists() or json.loads(path.read_text())['status'] != 'complete':
            missing.append(arm)
            continue
        rows = json.loads(path.read_text())['cases']
        if {r['id'] for r in rows} != cases.keys() or len(rows) != len(cases):
            raise ValueError('Historical regression IDs differ')
        graded[arm] = {r['id']: r | grade_answer(r['answer'], cases[r['id']]) for r in rows}
    losses = {}
    for arm in ('A1', 'A2', 'S1', 'S2'):
        base = arm[0] + '0'
        if arm not in graded or base not in graded:
            losses[arm] = {'passed': False, 'reason': 'missing evidence'}
            continue
        ids = [id_ for id_ in cases if graded[base][id_]['grade']['correct'] and not graded[arm][id_]['grade']['correct']]
        loops = [id_ for id_ in cases if not graded[base][id_]['combined_loop_flag'] and graded[arm][id_]['combined_loop_flag']]
        ended = [id_ for id_ in cases if graded[base][id_]['ended'] and not graded[arm][id_]['ended']]
        losses[arm] = {'passed': not ids and not loops and len(ended) <= 1,
                       'correctness_lost': ids, 'new_loops': loops, 'lost_completions': ended}
    result = {'missing': missing, 'veto': losses, 'arms': graded, 'gains_do_not_count_as_primary': True}
    save(result, run / 'review/regression-review.json')
    print(json.dumps({'missing': missing, 'veto': losses}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'finish', 'regression'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    args = parser.parse_args()
    regression(args.run) if args.stage == 'regression' else globals()[args.stage](args.run, args.split)
