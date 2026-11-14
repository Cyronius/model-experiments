"""Unhealed generation screen: stock, P0 and P1 answer one sealed suite on CPU.

PAIR_PLAN_20261026 next step 1. The teacher-forced screen found P1 closer to
stock than P0; this checks whether that holds for generated answers before any
healing. Stages:

  prepare   build the suite: Track A test plus dev programs/science, an opened
            HumanEval-derived chat diagnostic, a GSM8K sample screened against
            every calibration/training corpus and the teacher-forced manifest,
            and new science items with essential/optional rubrics
  freeze    bind the reviewed decision rules (Codex), the suite, every code
            dependency and every arm input by SHA-256
  generate  one arm and one suite part, greedy, under a hard deadline; refuses
            a draft protocol, changed inputs (checked before loading) and any
            existing output (no silent resume)
  grade     automatic grades for every non-science case, after validating each
            generation's metadata; science goes to the blind two-reviewer packet
"""
import argparse
import ast
import copy
import json
import random
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'results/unhealed-generation-20261027'
MANIFEST = ROOT / 'results/tf-metrics-20261027/manifest.jsonl'
TRACK_A_DATA = ROOT / 'results/retention-objective-20261027/data.json'
GSM8K = REPO / 'bench/data/gsm8k_raw.jsonl'
HUMANEVAL = ('humaneval', 'openai_humaneval/test-00000-of-00001.parquet')
SOURCE = REPO / 'models/manifold-audit/Qwen3.5-2B'
SOURCE_FILES = ['config.json', 'model.safetensors-00001-of-00001.safetensors', 'tokenizer.json',
                'tokenizer_config.json', 'chat_template.jinja']
RECIPES = {'P0': ['manifold-compression/results/local-projection-20261025/shared-broad-recipe.pt'],
           'P1': ['manifold-compression/results/allocation-pilot-20261027/p1-recipe.pt'],
           'P1-norefit': ['manifold-compression/results/allocation-pilot-20261027/p1-recipe.pt']}
PARTS = ('track-a', 'humaneval', 'gsm8k', 'science-new')
# Decision-relevant arms first, so a stop limit cannot cost the P0/P1 comparison.
ORDER = ['P0', 'P1', 'stock']
PART_ORDER = ['track-a', 'gsm8k', 'humaneval', 'science-new']
SEED = 26102751
GSM8K_COUNT = 200
MAX_NEW = 512
HUMANEVAL_SUFFIX = '\nReturn only complete Python code.'
GSM8K_SUFFIX = '\n\nSolve this step by step. Give the final answer as a number alone on the last line.'
# The rules file (Codex) must decide all of these before anything is generated.
RULE_KEYS = ('arms', 'runtime', 'bootstrap', 'progression', 'stock_floors', 'science', 'horizon', 'reporting')
# Real candidate cases that a pre-freeze rehearsal ran on stock and P1 (96-token cap),
# whose grades were seen and some outputs read; one prompted the unclosed-fence fix.
# GSM8K ones are removed before sampling; the rest are generated and reported but
# excluded from every gate, floor and event rate. See OUT/rehearsal/record.json.
REHEARSAL_EXPOSED = ['rh-test-retain-arithmetic-0', 'rh-test-program-3', 'HumanEval-0', 'HumanEval-23',
                     'gsm8k-1', 'gsm8k-1199']
# Codex's review and analysis code and inputs, bound by the same freeze.
PEER_FILES = ['unhealed_rules.json', 'unhealed_science.json', 'unhealed_review.py', 'unhealed_analysis.py']
DRIVER = 'run_unhealed_generation.sh'
SCIENCE = 'unhealed_science.json'


def sha(path):
    from allocation_pilot import sha as digest
    return digest(path)


def save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_stock():
    from local_projection_probe import load_source
    return load_source()


def load_p0():
    from allocation_pilot import load_p0 as load
    return load()


def load_p1():
    from allocation_pilot import load_p1 as load
    return load()


def prune_without_refit(model, recipe):
    """Keep P1's neurons but use the projected model's own down columns for them."""
    from allocation_pilot import prune_mlp
    for layer, keep in zip(model.model.layers, recipe['keep'], strict=True):
        prune_mlp(layer.mlp, keep, layer.mlp.down_proj.weight[:, keep].clone())
    model.config.intermediate_size = recipe['mlp_width']
    return model


def load_p1_norefit():
    """Refit control (next step 2): P1 with its refit down weights replaced by the originals."""
    import torch
    from allocation_pilot import OUT as PILOT
    from local_projection_models import build, recipe_bases
    from local_projection_probe import load_source
    recipe = torch.load(PILOT / 'p1-recipe.pt', map_location='cpu', weights_only=True)
    projection = recipe['projection']
    return prune_without_refit(build(load_source(), recipe_bases(projection), projection['scales']), recipe)


ARMS = {'stock': load_stock, 'P0': load_p0, 'P1': load_p1, 'P1-norefit': load_p1_norefit}


def arm_inputs(arm):
    """Repository-relative files whose bytes decide an arm's weights, tokenizer and prompts."""
    source = [str((SOURCE / name).relative_to(REPO)) for name in SOURCE_FILES]
    return source + RECIPES.get(arm, [])


def dependencies():
    """Every module in this directory that the stages import, found by importing them."""
    import allocation_pilot, curriculum_numeric_audit, eval_data, healing_data  # noqa: E401,F401
    import local_projection_models, local_projection_probe, retention_review, retention_run  # noqa: E401,F401
    import summarize_scale_probe, tf_metrics  # noqa: E401,F401
    # torch.ops and torch.classes report bare relative names; only real absolute paths count.
    files = {Path(m.__file__).resolve() for m in list(sys.modules.values())
             if getattr(m, '__file__', None) and Path(m.__file__).is_absolute()}
    files = {f.name for f in files if f.parent == ROOT and not f.name.startswith(('test_', 'conftest'))}
    return sorted(files | {Path(__file__).name})


# ---------------------------------------------------------------- suite

def track_a_cases():
    """Every Track A test case, plus the development programs and science.

    Nothing is selected on these here, so reusing the opened test split is a
    fixed comparison of unhealed constructions, not a held-out claim.
    """
    from retention_review import load_data
    data = load_data()
    cases = data['test'] + [c for c in data['dev'] if c['category'] in ('program', 'science')]
    return [copy.deepcopy(c) | {'rehearsal_exposed': c['id'] in REHEARSAL_EXPOSED} for c in cases]


def tf_scored(source):
    return {r['id'] for r in map(json.loads, MANIFEST.read_text().splitlines()) if r['source'] == source}


def humaneval_cases(blocked=lambda text: False):
    """An opened diagnostic: custom chat prompt and token cap, not leaderboard pass@1."""
    from eval_data import table
    scored = tf_scored('humaneval')
    cases = []
    for r in table(*HUMANEVAL):  # every item is generated; primary_mask selects the gated subset
        id_ = r['task_id'].replace('/', '-')
        content = 'Complete this Python function.\n\n```python\n' + r['prompt'] + '```' + HUMANEVAL_SUFFIX
        cases.append({'id': id_, 'family': 'humaneval', 'category': 'code', 'group': id_,
                      'messages': [{'role': 'user', 'content': content}], 'max_new_tokens': MAX_NEW,
                      'tf_scored': 'humaneval-' + id_ in scored,
                      'training_overlap': blocked(r['prompt'] + r['canonical_solution']),
                      'rehearsal_exposed': id_ in REHEARSAL_EXPOSED,
                      'check': {'kind': 'humaneval', 'prompt': r['prompt'], 'test': r['test'],
                                'entry_point': r['entry_point']}})
        cases[-1]['primary_mask'] = not (cases[-1]['training_overlap'] or cases[-1]['rehearsal_exposed'])
    return cases


def gsm8k_answer(text):
    value = text.split('####')[-1].strip().replace(',', '')
    return str(Fraction(value))


def gsm8k_cases(count=GSM8K_COUNT, seed=SEED, blocked=lambda text: False):
    """A fixed sample, drawn only from problems absent from the teacher-forced
    manifest, the rehearsal and every screened corpus. The pool is filtered before sampling."""
    used = {int(id_.split('-')[1]) for id_ in tf_scored('gsm8k')}
    exposed = {int(id_.split('-')[1]) for id_ in REHEARSAL_EXPOSED if id_.startswith('gsm8k-')}
    problems = [json.loads(line) for line in GSM8K.read_text().splitlines()]
    pool = [i for i in range(len(problems)) if i not in used | exposed
            and not blocked(problems[i]['question'] + '\n' + problems[i]['answer'])]
    chosen = sorted(random.Random(seed).sample(pool, count))
    cases = [{'id': f'gsm8k-{i}', 'family': 'gsm8k', 'category': 'gsm8k', 'group': f'gsm8k-{i}',
              'messages': [{'role': 'user', 'content': problems[i]['question'] + GSM8K_SUFFIX}],
              'max_new_tokens': MAX_NEW, 'check': {'kind': 'gsm8k', 'expected': gsm8k_answer(problems[i]['answer'])}}
             for i in chosen]
    for case in cases:
        case['rehearsal_exposed'] = False
    return cases, {'problems': len(problems), 'tf_scored': len(used), 'rehearsal_exposed': len(exposed),
                   'pool': len(pool)}


def science_cases(path, blocked=lambda text: False):
    """New science items, in the Track A science schema plus essential/optional clauses."""
    items = json.loads(Path(path).read_text())['items']
    required = {'id', 'family', 'messages', 'reference_answer', 'source_url', 'manual_rubric'}
    for item in items:
        if missing := required - item.keys():
            raise ValueError(f"{item.get('id')}: missing {sorted(missing)}")
        if not item['manual_rubric'].get('essential'):
            raise ValueError(f"{item['id']}: no essential clauses")
        if blocked(' '.join(m['content'] for m in item['messages']) + '\n' + item['reference_answer']):
            raise ValueError(f"{item['id']}: overlaps a calibration or training corpus")
    return [{**copy.deepcopy(item), 'category': 'science', 'group': 'science-new:' + item['family'],
             'check': None, 'max_new_tokens': MAX_NEW, 'rehearsal_exposed': False} for item in items]


def corpus_screen(tok):
    """The teacher-forced manifest's 13-gram screen, over its corpora plus Track A's training data."""
    from tf_metrics import EXCLUDE, NGRAM, RESULTS, grams, training_grams
    corpora = [(RESULTS / p, k) for p, k in EXCLUDE] + [(TRACK_A_DATA, ('train',))]
    seen, report = training_grams(tok, corpora, NGRAM)
    return (lambda text: bool(grams(tok(text, add_special_tokens=False).input_ids, NGRAM) & seen)), report


def order(cases, tok):
    """Fix batch composition before generation: prompt length, then id."""
    def length(c):
        text = tok.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return len(tok(text, add_special_tokens=False).input_ids)
    return sorted(cases, key=lambda c: (c['max_new_tokens'], length(c), c['id']))


def stage_prepare(args):
    from transformers import AutoTokenizer
    path = OUT / 'suite.json'
    if path.exists():
        raise ValueError(f'{path} exists; the suite is sealed once written')
    tok = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    blocked, corpora = corpus_screen(tok)
    gsm8k, pool = gsm8k_cases(blocked=blocked)
    parts = {'track-a': track_a_cases(), 'humaneval': humaneval_cases(blocked), 'gsm8k': gsm8k,
             'science-new': science_cases(args.science, blocked)}
    ids = [c['id'] for cases in parts.values() for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case ids across parts')
    if missing := set(REHEARSAL_EXPOSED) - set(ids) - {i for i in REHEARSAL_EXPOSED if i.startswith('gsm8k-')}:
        raise ValueError(f'Rehearsal-exposed ids not in the suite: {sorted(missing)}')
    parts = {name: order(cases, tok) for name, cases in parts.items()}
    humaneval = parts['humaneval']
    suite = {'parts': parts, 'seed': SEED, 'gsm8k_count': GSM8K_COUNT, 'gsm8k_pool': pool,
             'counts': {name: len(cases) for name, cases in parts.items()},
             'humaneval_subsets': {'tf_scored': sum(c['tf_scored'] for c in humaneval),
                                   'training_overlap': sum(c['training_overlap'] for c in humaneval),
                                   'rehearsal_exposed': sum(c['rehearsal_exposed'] for c in humaneval),
                                   'primary_mask': sum(c['primary_mask'] for c in humaneval)},
             'rehearsal_exposed': sorted(c['id'] for cases in parts.values() for c in cases if c['rehearsal_exposed']),
             'screen': {'ngram': 13, 'corpora': corpora,
                        'limitations': 'Token 13-gram overlap only; paraphrases and shared upstream sources can remain.'},
             'labels': {'humaneval': 'HumanEval-derived chat diagnostic (custom prompt, 512-token cap), opened: '
                                     'not leaderboard pass@1',
                        'track-a': 'Opened Track A suite; fixed comparison of unhealed constructions'},
             'inputs': {'track_a_data': sha(TRACK_A_DATA), 'gsm8k': sha(GSM8K), 'tf_manifest': sha(MANIFEST),
                        'science_new': sha(args.science)},
             'decoding': 'greedy, thinking disabled, repetition penalty 1, FP32, CPU torch fallback'}
    save(suite, path)
    print(json.dumps(suite['counts']), json.dumps(suite['humaneval_subsets']), json.dumps(pool), sha(path))


# ------------------------------------------------------------ protocol

def check_suite(suite, rules):
    """The sealed suite must still match its inputs, the frozen science file and the rules' counts."""
    from unhealed_review import validate_suite
    current = {'track_a_data': sha(TRACK_A_DATA), 'gsm8k': sha(GSM8K), 'tf_manifest': sha(MANIFEST),
               'science_new': sha(ROOT / SCIENCE)}
    if suite['inputs'] != current:
        raise ValueError('Suite inputs differ from the files being frozen')
    validate_suite(suite, rules)


def stage_freeze(args):
    """Bind the reviewed rules, suite, code and arm inputs; refuses an incomplete rules file.

    Runtime settings come only from the rules file, so the protocol cannot
    disagree with what was reviewed.
    """
    import torch
    path = OUT / 'protocol.json'
    if path.exists():
        raise ValueError(f'{path} exists; a frozen protocol is not rewritten')
    rules = json.loads(Path(args.rules).read_text())
    if missing := [k for k in RULE_KEYS if k not in rules]:
        raise ValueError(f'Rules file lacks {missing}')
    if sorted(rules['arms']) != sorted(ORDER):
        raise ValueError(f'G1 freezes exactly {ORDER}')
    if sorted(args.reviewers) != ['claude', 'codex']:
        raise ValueError('Both reviewers must sign the freeze')
    runtime = rules['runtime']
    if (runtime.get('device'), runtime.get('precision')) != ('cpu', 'float32'):
        raise ValueError('G1 runs on CPU in float32')
    if runtime.get('arm_order') != ORDER or runtime.get('part_order') != PART_ORDER:
        raise ValueError('Rules run order differs from the generation driver')
    hours = {arm: runtime.get('maximum_hours_per_part', {}).get(arm) for arm in rules['arms']}
    if not all(isinstance(h, (int, float)) and 0 < h <= 3 for h in hours.values()):
        raise ValueError('Every arm needs a positive per-part deadline of at most 3 hours')
    total = runtime.get('maximum_total_hours')
    if not (isinstance(total, (int, float)) and 0 < total <= 8):
        raise ValueError('The shared generation limit must be positive and at most 8 hours')
    if not all(isinstance(runtime.get(k), int) and runtime[k] > 0 for k in ('threads', 'batch')):
        raise ValueError('Threads and batch must be positive integers')
    check_suite(json.loads((OUT / 'suite.json').read_text()), rules)
    protocol = {'status': 'frozen', 'reviewers': sorted(args.reviewers), 'rules': rules,
                'rules_sha256': sha(args.rules), 'suite_sha256': sha(OUT / 'suite.json'),
                'code_sha256': {name: sha(ROOT / name) for name in sorted(set(dependencies()) | set(PEER_FILES) | {DRIVER})},
                'input_sha256': {name: sha(REPO / name) for arm in rules['arms'] for name in arm_inputs(arm)},
                'arms': rules['arms'], 'threads': runtime['threads'], 'batch': runtime['batch'],
                'max_hours_per_part': hours, 'max_total_hours': total, 'torch': torch.__version__,
                'run_order': {'arms': ORDER, 'parts': PART_ORDER}}
    save(protocol, path)
    print(sha(path))


def require_frozen(arm=None):
    """Refuse a draft protocol, changed suite or code, or (before loading) changed arm inputs."""
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if protocol.get('status') != 'frozen':
        raise RuntimeError('Protocol is not frozen')
    if sha(OUT / 'suite.json') != protocol['suite_sha256']:
        raise RuntimeError('Suite changed after freezing')
    for name, digest in protocol['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'{name} changed after freezing')
    if arm is not None:
        if arm not in protocol['arms']:
            raise RuntimeError(f'{arm} is not a frozen arm')
        for name in arm_inputs(arm):
            if sha(REPO / name) != protocol['input_sha256'].get(name):
                raise RuntimeError(f'{name} changed after freezing')
    return protocol


# ----------------------------------------------------------- generation

def shared_deadline(protocol):
    """One clock for the whole screen, started by the first generate command.

    Created exclusively and bound to the protocol, so re-running a part cannot reset it.
    """
    path, digest = OUT / 'clock.json', sha(OUT / 'protocol.json')
    start = time.time()
    clock = {'protocol_sha256': digest, 'start': start, 'deadline': start + protocol['max_total_hours'] * 3600}
    try:
        with path.open('x') as stream:
            stream.write(json.dumps(clock, indent=2) + '\n')
    except FileExistsError:
        clock = json.loads(path.read_text())
        if clock['protocol_sha256'] != digest:
            raise RuntimeError('The shared clock belongs to another protocol')
    if time.time() >= clock['deadline']:
        raise TimeoutError('The shared generation limit has passed')
    return clock['deadline']


def stage_generate(args):
    import torch
    from transformers import AutoTokenizer
    from retention_run import configure, raw_generate
    protocol = require_frozen(args.arm)
    if (args.threads, args.batch, torch.__version__) != (protocol['threads'], protocol['batch'], protocol['torch']):
        raise ValueError('Threads, batch and torch must match the protocol')
    path = OUT / 'generations' / f'{args.arm}-{args.part}.json'
    if path.exists():
        raise ValueError(f'{path} exists; no silent resume or overwrite')
    deadline = min(time.time() + protocol['max_hours_per_part'][args.arm] * 3600, shared_deadline(protocol))
    # raw_generate checks its deadline only between batches; this alarm stops a
    # batch (or model load) in progress, leaving the output marked running.
    def expired(signum, frame):
        raise TimeoutError('G1 generation deadline reached')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(deadline - time.time(), 1e-3))
    try:
        cases = json.loads((OUT / 'suite.json').read_text())['parts'][args.part]
        configure(args.threads)
        model = ARMS[args.arm]()
        tok = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
        meta = {'arm': args.arm, 'part': args.part, 'protocol_sha256': sha(OUT / 'protocol.json'),
                'input_sha256': {name: protocol['input_sha256'][name] for name in arm_inputs(args.arm)},
                'torch': torch.__version__, 'threads': args.threads, 'batch': args.batch, 'deadline': deadline}
        save(meta, path.with_suffix('.meta.json'))
        summary = raw_generate(model, tok, cases, path, deadline=deadline, batch_size=args.batch)
        finished = time.time()
        if finished > deadline:
            raise TimeoutError('G1 generation finished after its deadline')
        # Only an in-time completion is stamped; grading requires the stamp.
        save(meta | {'completed_at': finished}, path.with_suffix('.meta.json'))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(summary))


# -------------------------------------------------------------- grading

def code_blocks(answer):
    """Fenced code, else the whole answer; keeps the first line's indentation for body-only answers.

    A final fence left open when the model stops still counts as code: answers
    that end before the closing ``` are otherwise complete.
    """
    blocks = re.findall(r'```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)', answer, re.S | re.I)
    return '\n\n'.join(blocks) if blocks else answer.lstrip('\n').rstrip()


def fences_closed(answer):
    return answer.count('```') % 2 == 0


def humaneval_program(answer, check):
    """The prompt (imports, helpers, stub) followed by the answer's code.

    A full redefinition replaces the stub; an indented body continues it.
    """
    program = check['prompt'] + '\n\n' + code_blocks(answer)
    try:
        ast.parse(program)
    except SyntaxError:
        return None
    return program + '\n\n' + check['test'] + f"\n\ncheck({check['entry_point']})\n"


# Same isolation as horizon_data.run_reference: no network, home, project files
# or credentials; candidate output is discarded. Success is a fresh nonce
# printed after the tests, so an early exit or a forged "pass" cannot count.
SANDBOX = r'''
import contextlib, json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024*1024, 1024*1024))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
class Sink:
    def write(self, value): return len(value)
    def flush(self): pass
payload = json.load(sys.stdin)
try:
    with contextlib.redirect_stdout(Sink()), contextlib.redirect_stderr(Sink()):
        exec(compile(payload['program'], '<candidate>', 'exec'), {'__name__': '__candidate__'})
    sys.stdout.write('\n' + payload['nonce'] + '\n')
except BaseException as error:
    sys.stdout.write('\nFAIL ' + type(error).__name__ + '\n')
'''


def run_program(program, timeout=15):
    nonce = secrets.token_hex(16)
    cmd = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--clearenv',
           '--setenv', 'PATH', '/usr/bin:/bin', '--setenv', 'LANG', 'C.UTF-8',
           '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64',
           '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--chdir', '/tmp',
           '/usr/bin/python3', '-I', '-c', SANDBOX]
    try:
        with tempfile.TemporaryFile() as out:
            r = subprocess.run(cmd, input=json.dumps({'program': program, 'nonce': nonce}).encode(),
                               stdout=out, stderr=subprocess.DEVNULL, timeout=timeout)
            out.seek(0)
            lines = out.read(1 << 20).decode(errors='replace').strip().splitlines()
    except (subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0 and bool(lines) and lines[-1] == nonce


def grade_humaneval(answer, check):
    program = humaneval_program(answer, check)
    if program is None:
        return {'correct': False, 'syntax_pass': False, 'fences_closed': fences_closed(answer)}
    return {'correct': run_program(program), 'syntax_pass': True, 'fences_closed': fences_closed(answer)}


# Whole numeric tokens, so '3/2', '.5' and '1e3' are read as one value each.
NUMBER = re.compile(r'[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?(?:\s*/\s*\d+)?')
BARE = re.compile(r'[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+)?')


def number(token):
    numerator, _, denominator = token.replace(',', '').replace(' ', '').partition('/')
    try:
        return Fraction(numerator) / Fraction(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return None


def grade_gsm8k(answer, check):
    """Primary: final-answer content from the unmodified Track A parser, which
    accepts a final number line or an explicit answer/boxed form. Recorded
    separately: whether the last line is the bare number the prompt asked for,
    and (secondary) the last numeric token anywhere in the answer."""
    from curriculum_numeric_audit import final_numeric
    expected = Fraction(check['expected'])
    try:
        content, method = final_numeric(answer)
    except (ValueError, ZeroDivisionError, TypeError):
        content, method = None, None
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    tokens = NUMBER.findall(answer)
    flexible = number(tokens[-1]) if tokens else None
    return {'correct': content == expected, 'final_answer_value': None if content is None else str(content),
            'final_answer_method': method, 'bare_final_line': bool(lines) and bool(BARE.fullmatch(lines[-1])),
            'flexible_correct': flexible == expected, 'flexible_value': None if flexible is None else str(flexible)}


def grade_case(answer, case):
    if case['category'] == 'science':
        return None
    kind = case['check']['kind']
    if kind == 'humaneval':
        return grade_humaneval(answer, case['check'])
    if kind == 'gsm8k':
        return grade_gsm8k(answer, case['check'])
    from retention_review import grade_answer
    result = grade_answer(answer, case)
    if case['category'] != 'program':
        return {'correct': result['grade']['correct'], **result}
    # Horizon: the unchanged Track A grade is reported; the gate uses the same
    # grader and tests on the code read with the unclosed-final-fence rule.
    tolerant = grade_answer('```python\n' + code_blocks(answer) + '\n```', case)
    return {'correct': tolerant['grade']['correct'], 'fences_closed': fences_closed(answer),
            'track_a_grade': result, 'fence_tolerant_grade': tolerant}


def check_meta(meta, arm, part, protocol):
    expected = {'arm': arm, 'part': part, 'protocol_sha256': sha(OUT / 'protocol.json'),
                'input_sha256': {name: protocol['input_sha256'][name] for name in arm_inputs(arm)},
                'torch': protocol['torch'], 'threads': protocol['threads'], 'batch': protocol['batch']}
    if wrong := [k for k, v in expected.items() if meta.get(k) != v]:
        raise ValueError(f'{arm}-{part} metadata differs from the protocol: {wrong}')
    if not isinstance(meta.get('completed_at'), float) or not meta['completed_at'] <= meta.get('deadline', 0):
        raise ValueError(f'{arm}-{part} did not complete before its deadline')


def stage_grade(args):
    protocol = require_frozen(args.arm)
    suite = json.loads((OUT / 'suite.json').read_text())
    path = OUT / 'grades' / f'{args.arm}.json'
    if path.exists():
        raise ValueError(f'{path} exists')
    if run_program('def f():\n    return 1\n\n\ndef check(g):\n    assert g() == 1\n\ncheck(f)\n') is not True:
        raise RuntimeError('Isolated verifier unavailable')
    grades, sources = {}, {}
    for part, cases in suite['parts'].items():
        source = OUT / 'generations' / f'{args.arm}-{part}.json'
        check_meta(json.loads(source.with_suffix('.meta.json').read_text()), args.arm, part, protocol)
        generated = json.loads(source.read_text())
        if generated['status'] != 'complete':
            raise ValueError(f'{source} is incomplete')
        ids = [r['id'] for r in generated['cases']]
        if len(ids) != len(set(ids)) or sorted(ids) != sorted(c['id'] for c in cases):
            raise ValueError(f'{source} does not match the suite one-to-one')
        rows = {r['id']: r for r in generated['cases']}
        for case in cases:
            row = rows[case['id']]
            if any(row[k] != case[k] for k in ('family', 'category', 'group')):
                raise ValueError(f"{case['id']}: identity mismatch")
            grades[case['id']] = {'part': part, 'category': case['category'], 'family': case['family'],
                                  'group': case['group'], 'rehearsal_exposed': case['rehearsal_exposed'],
                                  'primary_mask': case.get('primary_mask'), 'ended': row['ended'],
                                  'loop': row['combined_loop_flag'], 'grade': grade_case(row['answer'], case)}
        sources[part] = sha(source)
    save({'arm': args.arm, 'protocol_sha256': sha(OUT / 'protocol.json'), 'suite_sha256': sha(OUT / 'suite.json'),
          'generation_sha256': sources, 'cases': grades}, path)
    auto = [g for g in grades.values() if g['grade'] is not None]
    print(args.arm, sum(g['grade']['correct'] for g in auto), '/', len(auto))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--science', type=Path, required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--rules', type=Path, required=True)
    f.add_argument('--reviewers', nargs='+', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--arm', choices=ARMS, required=True)
    g.add_argument('--part', choices=PARTS, required=True)
    g.add_argument('--threads', type=int, default=16)
    g.add_argument('--batch', type=int, default=16)
    r = sub.add_parser('grade')
    r.add_argument('--arm', choices=ARMS, required=True)
    args = ap.parse_args()
    {'prepare': stage_prepare, 'freeze': stage_freeze, 'generate': stage_generate,
     'grade': stage_grade}[args.stage](args)


if __name__ == '__main__':
    main()
