"""Blinded, independently rated science review for the frozen G1 screen.

CLI: prepare | disagreements | finish. Both reviewers save reviewer-<name>.json
before running disagreements. Never open arm-map.json until both reviews and
any adjudication are saved. The packet contains no arm labels or output scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/unhealed-generation-20261027'
REVIEWERS = ('claude', 'codex')
ARMS = ('stock', 'P0', 'P1')
PARTS = ('track-a', 'gsm8k', 'humaneval', 'science-new')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    # Duplicate JSON keys must not silently replace a recorded grade or identity.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x') as stream:
        stream.write(text)


def index(rows, label):
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate ids in {label}')
    return dict(zip(ids, rows, strict=True))


def boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f'{label} must be a JSON boolean')
    return value


def category(case, part):
    if part == 'track-a':
        return {'primary': 'arithmetic', 'retention': 'replay', 'program': 'horizon',
                'science': 'science_legacy'}[case['category']]
    if part == 'humaneval':
        if case['primary_mask']:
            return 'humaneval_primary'
        if case['training_overlap']:
            return 'humaneval_overlap'
        if case['rehearsal_exposed']:
            return 'humaneval_quarantine'
        raise ValueError('Unclassified HumanEval item')
    return {'gsm8k': 'gsm8k', 'science-new': 'science_new'}[part]


def validate_suite(suite, rules):
    if set(suite['parts']) != set(PARTS):
        raise ValueError('Missing or unexpected suite parts')
    cases = index([c for rows in suite['parts'].values() for c in rows], 'suite')
    excluded = set(rules['quarantine']['excluded_ids'])
    eligible = Counter()
    for part, rows in suite['parts'].items():
        for c in rows:
            exposed = boolean(c['rehearsal_exposed'], 'rehearsal_exposed')
            if exposed != (c['id'] in excluded):
                raise ValueError(f"Quarantine mismatch: {c['id']}")
            if part == 'gsm8k' and exposed:
                raise ValueError('Exposed GSM8K item was not replaced')
            if part == 'humaneval':
                overlap = boolean(c['training_overlap'], 'training_overlap')
                if boolean(c['primary_mask'], 'primary_mask') != (not overlap and not exposed):
                    raise ValueError('HumanEval primary mask differs from rules')
            if not exposed:
                eligible[category(c, part)] += 1
    if set(cases) & excluded != excluded - {'gsm8k-1', 'gsm8k-1199'}:
        raise ValueError('The expected rehearsal diagnostics are missing')
    expected = rules['reporting']['expected_generated_counts']
    if {p: len(v) for p, v in suite['parts'].items()} != expected:
        raise ValueError('Generated suite counts differ from rules')
    if dict(eligible) != rules['reporting']['expected_eligible_counts']:
        raise ValueError('Eligible category counts differ from rules')
    science = suite['parts']['science-new']
    if Counter(Counter(c['group'] for c in science).values()) != {2: 20}:
        raise ValueError('New science must have twenty two-item concept groups')
    return cases


def snapshot():
    """Require the entire fixed screen, validate provenance, and hash all review inputs."""
    import unhealed_generation as generation
    if generation.OUT != OUT:
        raise ValueError('Runner and reviewer output directories differ')
    protocol = generation.require_frozen()
    if set(protocol['arms']) != set(ARMS) or len(protocol['arms']) != 3:
        raise ValueError('G1 requires exactly stock, P0 and P1')
    for name, expected in protocol['input_sha256'].items():
        if digest(ROOT.parent / name) != expected:
            raise ValueError(f'Changed model input: {name}')
    suite = read(OUT / 'suite.json')
    cases = validate_suite(suite, protocol['rules'])
    sources = {name: digest(OUT / name) for name in ('suite.json', 'protocol.json')}
    result = {}
    for arm in ARMS:
        grade_path = OUT / 'grades' / f'{arm}.json'
        grades = read(grade_path)
        if (grades['arm'] != arm or grades['suite_sha256'] != sources['suite.json']
                or grades['protocol_sha256'] != sources['protocol.json']
                or set(grades['cases']) != set(cases)
                or set(grades['generation_sha256']) != set(PARTS)):
            raise ValueError(f'{arm}: grade provenance/coverage mismatch')
        sources[str(grade_path.relative_to(OUT))] = digest(grade_path)
        result[arm] = {}
        for part, items in suite['parts'].items():
            path = OUT / 'generations' / f'{arm}-{part}.json'
            meta = path.with_suffix('.meta.json')
            generation.check_meta(read(meta), arm, part, protocol)
            value = read(path)
            rows = index(value['cases'], str(path))
            if value['status'] != 'complete' or set(rows) != {c['id'] for c in items}:
                raise ValueError(f'{path}: incomplete or mismatched cases')
            if digest(path) != grades['generation_sha256'][part]:
                raise ValueError(f'{path}: changed since grading')
            sources[str(path.relative_to(OUT))] = digest(path)
            sources[str(meta.relative_to(OUT))] = digest(meta)
            for c in items:
                row, g = rows[c['id']], grades['cases'][c['id']]
                if any(row[k] != c[k] or g[k] != c[k] for k in ('family', 'category', 'group')):
                    raise ValueError(f"{c['id']}: identity mismatch")
                if (g['part'] != part or g['rehearsal_exposed'] != c['rehearsal_exposed']
                        or g['primary_mask'] != c.get('primary_mask')):
                    raise ValueError(f"{c['id']}: grade mask mismatch")
                for key, raw in [('ended', 'ended'), ('loop', 'combined_loop_flag')]:
                    if boolean(g[key], key) != boolean(row[raw], raw):
                        raise ValueError(f"{c['id']}: grade event mismatch")
                if c['category'] == 'science':
                    if g['grade'] is not None:
                        raise ValueError('Science must remain ungraded before independent review')
                else:
                    boolean(g['grade']['correct'], 'correct')
                result[arm][c['id']] = {'case': c, 'part': part, 'answer': row['answer'], 'graded': g}
    return protocol, suite, result, sources


def prepare():
    protocol, _, records, sources = snapshot()
    folder = OUT / 'review'
    if any((folder / p).exists() for p in ('packet.json', 'arm-map.json', 'manifest.json')):
        raise ValueError('Review packet already exists; no reshuffle')
    packet, mapping = [], {}
    for arm in ARMS:
        for case_id, record in records[arm].items():
            c = record['case']
            if c['category'] != 'science':
                continue
            blind_id = uuid.uuid4().hex
            packet.append({'id': blind_id, 'case': case_id, 'messages': c['messages'],
                           'answer': record['answer'], 'reference_answer': c['reference_answer'],
                           'source_url': c['source_url'], 'rubric': c['manual_rubric'],
                           'rubric_kind': 'new' if record['part'] == 'science-new' else 'legacy'})
            mapping[blind_id] = {'arm': arm, 'case': case_id}
    random.SystemRandom().shuffle(packet)
    write_new(folder / 'packet.json', {'policy': protocol['rules']['science'], 'items': packet})
    write_new(folder / 'arm-map.json', mapping)
    write_new(folder / 'manifest.json', {'sources': sources, 'packet_sha256': digest(folder / 'packet.json'),
                                        'map_sha256': digest(folder / 'arm-map.json')})
    print(f'Blinded science packet: {len(packet)} answers; no arm map printed.')


def validate_rating(row, item):
    boolean(row['correct'], 'correct')
    if not isinstance(row.get('reason'), str) or not row['reason'].strip():
        raise ValueError('Each rating needs a nonempty reason')
    if item['rubric_kind'] == 'new':
        marks = row.get('essential')
        if not isinstance(marks, list) or len(marks) != len(item['rubric']['essential']):
            raise ValueError('Rate every essential clause in order')
        for mark in marks:
            boolean(mark, 'essential mark')
        contradiction = boolean(row.get('core_contradiction'), 'core_contradiction')
        if row['correct'] != (all(marks) and not contradiction):
            raise ValueError('Correctness disagrees with essential clauses/core contradiction')
        errors = row.get('optional_errors')
        if not isinstance(errors, list) or any(not isinstance(e, str) for e in errors):
            raise ValueError('Record optional errors as a list (empty if none)')
    return row


def load_rating(path, reviewer, packet, packet_sha):
    rating = read(path)
    if rating.get('reviewer') != reviewer or rating.get('packet_sha256') != packet_sha:
        raise ValueError('Wrong reviewer or packet hash')
    rows = index(rating['items'], str(path))
    if set(rows) != set(packet):
        raise ValueError('Rating must cover each blind id exactly once')
    return {k: validate_rating(row, packet[k]) for k, row in rows.items()}


def disagreement_ids(reviews):
    def decision(r):
        return r['correct'], r.get('essential'), r.get('core_contradiction')
    return sorted(k for k in reviews['codex'] if decision(reviews['codex'][k]) != decision(reviews['claude'][k]))


def load_blind_reviews():
    folder = OUT / 'review'
    manifest = read(folder / 'manifest.json')
    if digest(folder / 'packet.json') != manifest['packet_sha256']:
        raise ValueError('Packet changed after preparation')
    packet = index(read(folder / 'packet.json')['items'], 'packet')
    # Both files must exist before either is read or any rating is revealed.
    if not all((folder / f'reviewer-{name}.json').exists() for name in REVIEWERS):
        raise ValueError('Both independent reviews must be saved first')
    reviews = {name: load_rating(folder / f'reviewer-{name}.json', name, packet, manifest['packet_sha256'])
               for name in REVIEWERS}
    hashes = {name: digest(folder / f'reviewer-{name}.json') for name in REVIEWERS}
    return manifest, packet, reviews, hashes


def disagreements():
    manifest, packet, reviews, hashes = load_blind_reviews()
    items = [{'id': k, 'packet_item': packet[k], 'reviews': {r: reviews[r][k] for r in REVIEWERS}}
             for k in disagreement_ids(reviews)]
    write_new(OUT / 'review/disagreements.json', {'packet_sha256': manifest['packet_sha256'],
                                                'review_sha256': hashes, 'items': items})
    print(f'{len(items)} blinded disagreements; arm map remains unopened.')


def resolve(packet, reviews, hashes, disagreement_file, consensus=None):
    disputed = disagreement_ids(reviews)
    if disagreement_file['review_sha256'] != hashes or [i['id'] for i in disagreement_file['items']] != disputed:
        raise ValueError('Disagreement record no longer matches independent reviews')
    if consensus is None or consensus.get('review_sha256') != hashes:
        raise ValueError('Adjudication must bind both independent reviews (empty items if no disputes)')
    if sorted(consensus.get('approved_by', [])) != list(REVIEWERS):
        raise ValueError('Both reviewers must approve adjudication')
    choices = index(consensus['items'], 'adjudication')
    if set(choices) != set(disputed):
        raise ValueError('Adjudicate exactly the disputed blind ids')
    choices = {k: validate_rating(v, packet[k]) for k, v in choices.items()}
    return {k: choices.get(k, reviews['codex'][k]) for k in packet}


def finish(write=True):
    folder = OUT / 'review'
    manifest, packet, reviews, hashes = load_blind_reviews()
    disagreement_file = read(folder / 'disagreements.json')
    if disagreement_file['packet_sha256'] != manifest['packet_sha256']:
        raise ValueError('Disagreement packet hash mismatch')
    consensus_path = folder / 'adjudication.json'
    consensus = read(consensus_path)
    if consensus.get('packet_sha256') != manifest['packet_sha256']:
        raise ValueError('Adjudication packet hash mismatch')
    final = resolve(packet, reviews, hashes, disagreement_file, consensus)
    # Only after both ratings/adjudication have been validated may the map be opened.
    protocol, _, records, sources = snapshot()
    if sources != manifest['sources'] or digest(folder / 'arm-map.json') != manifest['map_sha256']:
        raise ValueError('Review inputs changed')
    mapping = read(folder / 'arm-map.json')
    if set(mapping) != set(packet):
        raise ValueError('Blind map coverage mismatch')
    manual = {}
    for key, identity in mapping.items():
        address = (identity['arm'], identity['case'])
        if address in manual or packet[key]['case'] != identity['case']:
            raise ValueError('Blind map duplicates or changes a case')
        manual[address] = final[key]
    expected = {(arm, k) for arm in ARMS for k, v in records[arm].items() if v['case']['category'] == 'science'}
    if set(manual) != expected:
        raise ValueError('Blind map does not cover all science arm/case pairs')
    normalized = {}
    for arm in ARMS:
        normalized[arm] = []
        for case_id, record in records[arm].items():
            c, g = record['case'], record['graded']
            grade = manual[(arm, case_id)] if c['category'] == 'science' else g['grade']
            normalized[arm].append({'id': case_id, 'part': record['part'], 'family': c['family'],
                                    'group': c['group'], 'category': category(c, record['part']),
                                    'rehearsal_exposed': c['rehearsal_exposed'],
                                    'training_overlap': c.get('training_overlap'), 'tf_scored': c.get('tf_scored'),
                                    'primary_mask': c.get('primary_mask'), 'correct': grade['correct'],
                                    'loop': g['loop'], 'ended': g['ended'], 'grade': grade})
    for name in ['packet.json', 'arm-map.json', 'manifest.json', 'disagreements.json'] + [f'reviewer-{r}.json' for r in REVIEWERS]:
        sources['review/' + name] = digest(folder / name)
    if consensus is not None:
        sources['review/adjudication.json'] = digest(consensus_path)
    result = {'status': 'complete', 'sources': sources, 'rules': protocol['rules'], 'arms': normalized,
              'science_review': {'reviewers': list(REVIEWERS), 'disagreements': len(disagreement_ids(reviews)),
                                 'adjudicated': True}}
    if write:
        write_new(OUT / 'normalized.json', result)
        print('All three arms normalized with independently reviewed science and bound input hashes.')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'disagreements', 'finish'))
    args = ap.parse_args()
    {'prepare': prepare, 'disagreements': disagreements, 'finish': finish}[args.stage]()


if __name__ == '__main__':
    main()
