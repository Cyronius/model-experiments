"""Prepare the paired retention pilot without loading model weights or renting compute.

Run in the existing environment with the networkless bwrap verifier available.
Science cases are independently authored/reviewed in retention_science_eval.json.
The output is immutable input data, not authorization to launch an experiment.
"""
import argparse
import ast
from collections import Counter
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re

from correctness_eval import synthetic_cases
from correctness_prepare import pure_reference
from curriculum_tasks import FAMILIES, math_example, training_algorithms
from healing_data import encode_complete
from horizon_data import arithmetic, equivalent, run_reference

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'results/retention-objective-20261027'
PARENT = ROOT / 'results/curriculum-healing-20261026'
SOURCE = REPO / 'models/manifold-audit/Qwen3.5-2B'
SOURCE_ID = {'repo': 'Qwen/Qwen3.5-2B', 'revision': '15852e8c16360a2fea060d615a32b45270f8a8fc'}
HISTORY = [
    'local-projection-20261025/data.json',
    'local-projection-20261025/extended-calibration-source.json',
    'broad-more-20261025/data.json',
    'correctness-healing-20261025/data.json',
    'broad-healing-20261025/data.json',
    'curriculum-healing-20261026/data.json',
    'capacity-projection-20261026/data.json',
]
MIX = {'math': .35, 'retain': .30, 'verified_code': .30, 'science': .05}
REJECTED_PROGRAM_GROUPS = {
    'horizon:ed40d6ed4f41f8381661fb44b8a69489a77bbda7c550678d4ad2bd4efea9c0e8':
        'Peer review: underspecified Smolyak rule; declared int output conflicts with reference dict output.'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def math_group(family, values):
    values = list(values)
    if family == 'multiply':
        values = sorted(values)
    elif family == 'add_subtract':
        values = sorted(values[:2]) + values[2:]
    elif family == 'fraction':
        values = sorted([str(Fraction(values[0], values[1])), str(Fraction(values[2], values[3]))])
    return 'math:' + family + ':' + canonical(values)


def prompt_messages(messages):
    return messages[:-1] if messages and messages[-1]['role'] == 'assistant' else messages


def historical_exclusions(paths):
    groups, prompts, ids = set(), set(), set()

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('messages'), list) and all(
                    isinstance(m, dict) and isinstance(m.get('content'), str) and 'role' in m
                    for m in value['messages']):
                messages = prompt_messages(value['messages'])
                prompts.add(canonical(messages))
                for message in messages:
                    for a, b in re.findall(r'(\d+) (?:multiplied by|times) (\d+)', message['content']):
                        groups.add(math_group('multiply', [int(a), int(b)]))
            if value.get('family') in FAMILIES and 'values' in value:
                groups.add(math_group(value['family'], value['values']))
            if 'a' in value and 'b' in value and 'verification' in value:
                if type(value['a']) is int and type(value['b']) is int:
                    groups.add(math_group('multiply', [value['a'], value['b']]))
            if value.get('problem_group'):
                groups.add('horizon:' + value['problem_group'])
            if isinstance(value.get('id'), str):
                ids.add(value['id'])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        visit(json.loads(Path(path).read_text()))
    return groups, prompts, ids


def numeric_splits(groups, prompts, dev_n=8, test_n=16):
    """Fresh instances of known families, excluding known canonical operands."""
    used_groups, used_prompts = set(groups), set(prompts)
    result = {}
    for split, n, seed in [('dev', dev_n, 26102711), ('test', test_n, 26102712)]:
        rng = random.Random(seed)
        rows = []
        for family in FAMILIES:
            count = 0
            for _ in range(20000):
                raw = math_example(rng, family)
                group = math_group(family, raw['values'])
                messages = [{'role': 'user', 'content': raw['transfer'] + ' Return only the final number or reduced fraction.'}]
                if group in used_groups or canonical(messages) in used_prompts:
                    continue
                if arithmetic(raw['expression']) != Fraction(raw['expected']):
                    raise ValueError('Arithmetic oracle disagreement')
                used_groups.add(group)
                used_prompts.add(canonical(messages))
                rows.append({'id': f'rh-{split}-math-{family}-{count}', 'family': family,
                             'category': 'primary', 'group': group, 'messages': messages,
                             'check': {'kind': 'numeric', 'expected': raw['expected']},
                             'max_new_tokens': 256, 'oracle': raw})
                count += 1
                if count == n:
                    break
            if count != n:
                raise ValueError('Exhausted arithmetic instances for ' + family)
        counts = Counter()
        for attempt in range(20000):
            if len(counts) == 7 and all(v == n for v in counts.values()):
                break
            for case in synthetic_cases(seed * 100000 + attempt, 1):
                family = case['family']
                if counts[family] == n:
                    continue
                if family == 'units':
                    tenths = rng.randint(101, 9999)
                    case['messages'] = [{'role': 'user', 'content': f'Convert {tenths / 10:g} liters to milliliters. Return only the number.'}]
                    case['check']['expected'] = str(tenths * 100)
                if family == 'arithmetic':
                    pair = re.search(r'(\d+) multiplied by (\d+)', case['messages'][0]['content'])
                    group = math_group('multiply', list(map(int, pair.groups())))
                else:
                    group = 'retention:' + family + ':' + digest(case['messages'])
                prompt = canonical(case['messages'])
                if group in used_groups or prompt in used_prompts:
                    continue
                used_groups.add(group)
                used_prompts.add(prompt)
                case.update(id=f'rh-{split}-retain-{family}-{counts[family]}',
                            category='retention', group=group, replay_family=True)
                rows.append(case)
                counts[family] += 1
        if len(counts) != 7 or any(v != n for v in counts.values()):
            raise ValueError('Exhausted retention instance sampler')
        result[split] = rows
    return result


def conservative_mutations(inputs):
    """Candidates for DIAGNOSTIC reference agreement, not proven valid domains.

    Preserve shapes, integer signs, order, duplicates and distances by applying
    a common translation to a scalar or flat integer list. Natural-language
    domain review is still required before any such case can affect a grade.
    """
    for key, value in inputs.items():
        numbers = [value] if type(value) is int else value
        if not isinstance(numbers, list) or not numbers or any(type(x) is not int for x in numbers):
            continue
        for shift in (-1, 1):
            changed = [x + shift for x in numbers]
            sign = lambda x: (x > 0) - (x < 0)
            if any(sign(a) != sign(b) for a, b in zip(numbers, changed)):
                continue
            candidate = copy.deepcopy(inputs)
            candidate[key] = changed[0] if type(value) is int else changed
            yield candidate


def heldout_programs(prepared, groups, prompts, ids, counts=(6, 12)):
    result, audits = {}, []
    used_groups, used_prompts = set(groups), set(prompts)
    for split, source_split, n in [('dev', 'validation', counts[0]), ('test', 'test', counts[1])]:
        candidates = [json.loads(line) for line in (prepared / f'{source_split}.jsonl').open()]
        candidates.sort(key=lambda r: digest(['retention-program-v1', r['id']]))
        rows = []
        for raw in candidates:
            if raw['kind'] != 'code' or raw['id'] in ids:
                continue
            group = 'horizon:' + raw['problem_group']
            if group in used_groups or group in REJECTED_PROGRAM_GROUPS:
                continue
            code = raw['reference_code']
            prompt = raw['messages'][0]['content'].split('\n\nGiven the following')[0]
            if len(code) > 1400 or len(prompt) > 1800 or not pure_reference(code):
                continue
            function = next((x for x in ast.parse(code).body if isinstance(x, ast.FunctionDef) and x.name == 'main_solution'), None)
            if function is None:
                continue
            inputs, expected = raw['check']['inputs'], raw['check']['expected_output']
            first, second = run_reference(code, inputs), run_reference(code, inputs)
            if not first or first != second or not equivalent(first['result'], expected):
                continue
            signature = 'main_solution(' + ast.unparse(function.args) + ')'
            messages = [{'role': 'user', 'content': prompt + '\nUse exactly the function signature ' + signature + '. Return only complete Python code.'}]
            if canonical(messages) in used_prompts:
                continue
            used_groups.add(group)
            used_prompts.add(canonical(messages))
            diagnostic = []
            for candidate in conservative_mutations(inputs):
                a, b = run_reference(code, candidate), run_reference(code, candidate)
                if a and a == b:
                    diagnostic.append([candidate, a['result']])
                if len(diagnostic) == 4:
                    break
            case = {'id': f'rh-{split}-program-{len(rows)}', 'family': 'horizon_program',
                    'category': 'program', 'group': group, 'messages': messages,
                    'check': {'kind': 'program', 'tests': [[inputs, expected]]},
                    'max_new_tokens': 512, 'source_id': raw['id'],
                    'grading_scope': 'supplied I/O; not proof of full specification',
                    'differential_diagnostics': diagnostic,
                    'differential_domain_review': 'pending; not included in primary grade'}
            rows.append(case)
            audits.append({'id': case['id'], 'group': group, 'source_id': raw['id'],
                           'prompt': messages[0]['content'], 'reference_code': code,
                           'supplied_tests': case['check']['tests'],
                           'differential_diagnostics': diagnostic,
                           'domain_review_required': True})
            if len(rows) == n:
                break
        if len(rows) != n:
            raise ValueError(f'Only {len(rows)} independently held-out program groups for {split}')
        result[split] = rows
    return result, audits


def make_train(tok, old):
    train = []
    for index, row in enumerate(old['train']):
        kind = row['kind']
        if kind not in MIX:
            continue
        messages = copy.deepcopy(row['messages'])
        if kind == 'math':
            raw = row['provenance']
            wording = raw['transfer'] if index % 2 else raw['prompt']
            worked = index % 4 < 2
            prompt = wording + (' Show a brief calculation and the final answer.' if worked else ' Return only the final number or reduced fraction.')
            answer = raw['work'] + '\n\nAnswer: ' + raw['expected'] if worked else raw['expected']
            messages = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}]
            group = math_group(raw['family'], raw['values'])
        elif kind == 'verified_code':
            group = 'authored-code:' + row['provenance']['algorithm']
        elif kind == 'science':
            group = 'science-train:' + row['provenance']['topic']
        else:
            group = 'retention-train:' + digest(prompt_messages(messages))
        encoded = encode_complete(tok, messages, 1024)
        if encoded is None:
            raise ValueError('Revised complete training response exceeds 1024 tokens')
        train.append({'id': f'rh-train-{index}', 'kind': kind, 'group': group,
                      'replay': kind == 'retain', 'messages': messages,
                      'provenance': row['provenance'], **encoded})
    return train


def schedule(train, steps=16, tokens_per_update=4096):
    pools = {kind: [i for i, row in enumerate(train) if row['kind'] == kind] for kind in MIX}
    if any(not rows for rows in pools.values()):
        raise ValueError('Missing curriculum kind')
    for kind, rows in pools.items():
        random.Random('retention-schedule:' + kind).shuffle(rows)
    tokens, positions = Counter(), Counter()
    updates = []
    for _ in range(steps):
        update, count = [], 0
        while count < tokens_per_update:
            kind = min(MIX, key=lambda k: tokens[k] / MIX[k])
            i = pools[kind][positions[kind] % len(pools[kind])]
            n = len(train[i]['targets'])
            update.append(i)
            count += n
            tokens[kind] += n
            positions[kind] += 1
        updates.append(update)
    repetitions = Counter(i for update in updates for i in update)
    return updates, {'target_mix': MIX, 'tokens_by_kind': dict(tokens),
                     'presentations_by_kind': dict(positions), 'unique_rows': len(repetitions),
                     'pool_counts': {k: len(v) for k, v in pools.items()},
                     'presentations_per_pool_row': {k: positions[k] / len(v) for k, v in pools.items()},
                     'maximum_row_presentations': max(repetitions.values()),
                     'answer_tokens': sum(tokens.values())}


def prepare(out=OUT, science_path=ROOT / 'retention_science_eval.json'):
    if (out / 'data.json').exists():
        raise ValueError('Prepared data already exists; refuse overwrite')
    if run_reference('def main_solution(x):\n return x + 1', {'x': 3}) != {'ok': True, 'result': 4}:
        raise RuntimeError('Networkless bwrap verifier unavailable; do not weaken isolation')
    science = json.loads(science_path.read_text())
    if 'items' in science:
        science = {split: [row for row in science['items'] if row['split'] == split]
                   for split in ('dev', 'test')}
    from scale_data import load_tokenizer
    tok = load_tokenizer(SOURCE)
    paths = [ROOT / 'results' / name for name in HISTORY]
    groups, prompts, ids = historical_exclusions(paths)
    old = json.loads((PARENT / 'data.json').read_text())
    train = make_train(tok, old)
    groups.update(r['group'] for r in train)
    prompts.update(canonical(prompt_messages(r['messages'])) for r in train)
    splits = numeric_splits(groups, prompts)
    groups.update(r['group'] for s in splits.values() for r in s)
    prompts.update(canonical(r['messages']) for s in splits.values() for r in s)
    programs, audits = heldout_programs(REPO / 'target/manifold-healing/horizon-prepared', groups, prompts, ids)
    for split, count in [('dev', 4), ('test', 8)]:
        if len(science[split]) != count:
            raise ValueError('Science case count differs from agreed protocol')
        splits[split] += programs[split]
        for case in science[split]:
            case = copy.deepcopy(case)
            if 'rubric' in case:
                case['manual_rubric'] = case.pop('rubric')
            if canonical(case['messages']) in prompts or not case.get('manual_rubric'):
                raise ValueError('Science overlap or missing factual rubric')
            case.update(category='science', group=case.get('group', 'science-eval:' + case['id']),
                        family=case.get('family', 'science'), check=None, max_new_tokens=512)
            prompts.add(canonical(case['messages']))
            splits[split].append(case)
    regression = []
    seen = set()
    for case in old['dev'] + old['test'] + old['legacy']:
        if not case.get('check') or case['check']['kind'] != 'program':
            continue
        key = canonical(case['messages'])
        if key in seen:
            continue
        seen.add(key)
        case = copy.deepcopy(case)
        case.update(id='regression-' + case['id'], category='program',
                    group='historical:' + digest(case['messages']),
                    family='historical:' + case['id'], veto_only=True)
        regression.append(case)
    # Re-execute independently authored multi-input training references.
    code_verification = {}
    for name, signature, task, code, tests in training_algorithms():
        for inputs, expected in tests:
            actual = run_reference(code, inputs)
            if not actual or not equivalent(actual['result'], expected):
                raise ValueError('Authored code oracle disagreement: ' + name)
        code_verification[name] = len(tests)
    updates, mix = schedule(train)
    partition = {s: {r['group'] for r in rows} for s, rows in [('train', train), *splits.items()]}
    if partition['train'] & partition['dev'] or partition['train'] & partition['test'] or partition['dev'] & partition['test']:
        raise ValueError('Train/dev/test instance or source groups overlap')
    all_eval = splits['dev'] + splits['test']
    if len({canonical(c['messages']) for c in all_eval}) != len(all_eval):
        raise ValueError('Duplicate evaluation prompts')
    policy = {'min_primary_gain': .05, 'max_retention_loss_fraction': .05,
              'max_program_net_loss': 0, 'max_science_net_loss': 0,
              'max_new_loops': 0, 'max_lost_completions': 1}
    protocol = {
        'status': 'data_prepared_pending_peer_review', 'source': SOURCE_ID,
        'arms': ['A0', 'A1', 'A2', 'S0', 'S1', 'S2'],
        'parent': str(PARENT / 'selected'), 'parent_step': 24,
        'steps': 16, 'rank': 64, 'alpha': 128, 'lr': 3e-5, 'warmup_steps': 2,
        'optimizer': 'fresh AdamW beta=(0.9,0.95), weight_decay=0, clip_norm=1',
        'loss': 'answer CE plus replay-only exact forward KL for A2/S2; T=lambda=1; shared total-answer-token denominator',
        'endpoint': 'Fixed update16; all six preregistered endpoints receive one final test. No test-driven checkpoint selection or auto-promotion. Parent default.',
        'test_blinding': 'Development science reviewed and gates applied before sealed test summaries are opened',
        'precision': 'FP32 stored weights; BF16 autocast training; FP32 evaluation; TF32 disabled',
        'kernels': 'One pinned torch fallback for all arms; record runtime versions and control results',
        'generation_batch_size': 16, 'training_batch': 'one row per forward, accumulate >=4096 answer tokens per update',
        'seed': 26102719, 'selection_policy': policy,
        'code_guardrail': 'Historical authored code cases veto losses only; gains do not count as primary improvement',
        'science_guardrail': 'One item per science family in each split: any individual science loss vetoes promotion',
        'program_extras': 'Type-preserving reference-differential inputs are diagnostics pending domain review; supplied I/O is primary',
        'split_counts': {s: dict(Counter(r['category'] for r in rows)) for s, rows in splits.items()},
        'mix': mix,
        'limitations': [
            'Fresh numeric instances share task families/templates with training/replay; this is not novel-family generalization.',
            'Program source groups are disjoint; single supplied I/O is weaker than authored edge/property checks.',
            'Several held-out programs are simple interfaces, formatting or arithmetic; program gains do not establish reasoning recovery.',
            'Science applications and prompts differ, but taught concepts can overlap.',
            'No claim of absence from source-model pretraining. Known historical groups/prompts are excluded where metadata permits.',
            'Stock arms start fresh; projected arms continue a trained parent. Their interaction is not a clean projection-only causal effect.',
            'Learning rate, curriculum and dose differ from prior studies; only A1/A2 and S1/S2 isolate the added KL term.'
            ' Horizon I/O-prediction training was removed in favor of independently verified program-writing examples.'
        ]}
    data = {'source': SOURCE_ID, 'train': train, 'updates': updates, **splits,
            'regression': regression, 'protocol': protocol}
    save(data, out / 'data.json')
    save(audits, out / 'program-domain-review.json')
    save({'status': 'prepared', 'policy': policy, 'data_sha256': sha(out / 'data.json'),
          'science_rubrics_sha256': sha(science_path), 'no_auto_promotion': True}, out / 'generation-gates.json')
    save({'files': {str(p.relative_to(REPO)): sha(p) for p in paths + [science_path]},
          'data_sha256': sha(out / 'data.json'), 'program_audit_sha256': sha(out / 'program-domain-review.json'),
          'known_historical_groups': len(groups), 'known_prompt_count': len(prompts),
          'authored_code_verification': code_verification,
          'rejected_program_groups': REJECTED_PROGRAM_GROUPS,
          'split_groups': {s: sorted(v) for s, v in partition.items()},
          'group_scope': 'Canonical numeric instances; exact Horizon source groups; authored-code IDs; science prompt/application IDs. Semantic task families are not disjoint.'}, out / 'data-manifest.json')
    save(protocol, out / 'protocol.json')
    print(json.dumps({'split_counts': protocol['split_counts'], 'mix': mix,
                      'historical_programs': len(regression), 'out': str(out)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--science', type=Path, default=ROOT / 'retention_science_eval.json')
    args = parser.parse_args()
    prepare(args.out, args.science)
