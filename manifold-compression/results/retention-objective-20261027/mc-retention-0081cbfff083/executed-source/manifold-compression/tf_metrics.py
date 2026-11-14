"""Teacher-forced diagnostics for Qwen3.5-2B variants (PAIR_PLAN_20261026.md, 0b).

Secondary measurements only: the generation suite, not these numbers, selects
checkpoints. Text rows compare the pinned stock model's next-token distribution
with the candidate's at fixed positions: exact full-vocabulary forward
KL(teacher || candidate), top-1 agreement, and candidate cross-entropy on the
reference text. Multiple-choice rows score each choice's summed continuation
log-probability; picks by sum, per-token and per-character normalization are all
recorded. Arm differences are paired and bootstrapped over item groups.

Stages: prepare (frozen manifest plus training-overlap audit), score (one arm),
compare (two arms). Prepare uses local files only; other sources are passed in
as JSONL with --chat / --mc after they have been fetched and reviewed.
"""
import argparse
import hashlib
import importlib
import json
import random
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE = REPO / 'models/manifold-audit/Qwen3.5-2B'
RESULTS = ROOT / 'results'
OUT = RESULTS / 'tf-metrics-20261027'
NGRAM = 13
RULES = ('sum', 'per_token', 'per_char')
# Every corpus any current arm was calibrated or trained on. Add new training data with --exclude.
EXCLUDE = [('local-projection-20261025/data.json', ('small', 'large', 'broad')),
           ('local-projection-20261025/extended-calibration-source.json', ('calibration',)),
           ('correctness-healing-20261025/data.json', ('train',)),
           ('broad-healing-20261025/data.json', ('train',)),
           ('curriculum-healing-20261026/data.json', ('train',)),
           ('capacity-projection-20261026/data.json', ('train',)),
           ('broad-more-20261025/data.json', ('calibration',))]
ARMS = {'stock': 'local_projection_probe:load_source',
        'shared-broad': 'allocation_pilot:load_p0',
        'p1': 'allocation_pilot:load_p1',
        'broad-healed': 'load_broad_healing:load'}


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ---------- manifest ----------

def strings(obj, minimum=32):
    if isinstance(obj, str):
        if len(obj) >= minimum:
            yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v, minimum)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v, minimum)


def grams(ids, n=NGRAM):
    return {hash(tuple(ids[i:i + n])) for i in range(len(ids) - n + 1)}


def training_grams(tok, sources, n=NGRAM):
    seen, report = set(), []
    for path, keys in sources:
        data = json.loads(Path(path).read_text())
        texts = [s for k in keys for s in strings(data[k])]
        for text in texts:
            seen |= grams(tok(text, add_special_tokens=False).input_ids, n)
        report.append({'path': str(Path(path).relative_to(REPO)), 'keys': list(keys),
                       'sha256': sha(path), 'strings': len(texts)})
    return seen, report


def letter_prompt(question, choices):
    lines = [question.strip()] + [f'{"ABCDEFGH"[i]}. {c}' for i, c in enumerate(choices)]
    return '\n'.join(lines) + '\nAnswer:', [f' {"ABCDEFGH"[i]}' for i in range(len(choices))]


def cloze_prompt(question, choices):
    return f'Question: {question.strip()}\nAnswer:', [f' {c.strip()}' for c in choices]


def continue_prompt(text, choices):
    # Sentence completion (HellaSwag): the choice continues the text directly.
    return text.strip(), [f' {c.strip()}' for c in choices]


STYLES = {'letter': letter_prompt, 'cloze': cloze_prompt, 'continue': continue_prompt}


def mc_row(tok, item, source, index, style):
    """Raw-text scoring, no chat template: context and each continuation are tokenized
    separately and concatenated, so every choice is scored on identical context ids."""
    context, continuations = STYLES[style](item['question'], item['choices'])
    return {'id': f'{source}-{index}', 'group': f"{source}/{item.get('subject', source)}", 'source': source,
            'kind': 'mc', 'style': style, 'context_ids': tok(context, add_special_tokens=False).input_ids,
            'choice_ids': [tok(c, add_special_tokens=False).input_ids for c in continuations],
            'choice_chars': [len(c) for c in continuations], 'answer': int(item['answer'])}


def text_row(tok, messages, source, rid, max_length):
    from healing_data import encode_complete
    enc = encode_complete(tok, messages, max_length)
    if enc is None:
        return None
    # positions[i] predicts token positions[i] + 1: the final assistant answer and its EOS.
    return {'id': f'{source}-{rid}', 'group': f'{source}/{rid}', 'source': source, 'kind': 'text',
            'input_ids': enc['input_ids'], 'positions': enc['positions']}


def document_rows(tok, doc, source, length, per_doc):
    """Raw prose, no chat template: consecutive token chunks, every next token scored."""
    ids = tok(doc['text'], add_special_tokens=False).input_ids
    return [{'id': f"{source}-{doc['id']}-{j}", 'group': f"{source}/{doc['id']}", 'source': source, 'kind': 'text',
             'input_ids': ids[j * length:(j + 1) * length], 'positions': list(range(length - 1))}
            for j in range(min(per_doc, len(ids) // length))]


def local_sources(tok, max_length, gsm8k=None):
    rows = []
    probe = json.loads((RESULTS / 'scale-probe-20261024/data.json').read_text())
    for r in probe['validation']:
        rows.append(text_row(tok, r['messages'], 'ultrachat-heldout', r['id'], max_length))
    problems = list(enumerate(read_jsonl(REPO / 'bench/data/gsm8k_raw.jsonl')))
    if gsm8k is not None:
        problems = sorted(random.Random(26102702).sample(problems, gsm8k))
    for i, r in problems:
        messages = [{'role': 'user', 'content': r['question']}, {'role': 'assistant', 'content': r['answer']}]
        rows.append(text_row(tok, messages, 'gsm8k', i, max_length))
    return rows


def overlapping(row, seen, n=NGRAM):
    if row['kind'] == 'text':
        return bool(grams(row['input_ids'], n) & seen)
    return bool(grams(row['context_ids'] + row['choice_ids'][row['answer']], n) & seen)


def stage_prepare(args):
    from transformers import AutoTokenizer
    path = args.out / 'manifest.jsonl'
    if path.exists():
        raise ValueError('Manifest exists; it is frozen')
    tok = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    rows = local_sources(tok, args.max_length, args.gsm8k)
    for spec in args.text:
        file, source = spec.split(':', 1)
        rows += [r for doc in read_jsonl(file) for r in document_rows(tok, doc, source, args.chunk, args.chunks_per_doc)]
    for spec in args.chat:
        file, source = spec.split(':', 1)
        rows += [text_row(tok, r['messages'], source, r.get('id', i), args.max_length)
                 for i, r in enumerate(read_jsonl(file))]
    for spec in args.mc:
        file, source, style = spec.split(':', 2)
        rows += [mc_row(tok, item, source, i, style) for i, item in enumerate(read_jsonl(file))]
    too_long = sum(r is None for r in rows)
    rows = [r for r in rows if r is not None]
    excluded = [(RESULTS / p, k) for p, k in EXCLUDE] + [
        (Path(s.split(':', 1)[0]), tuple(s.split(':', 1)[1].split(','))) for s in args.exclude]
    seen, sources = training_grams(tok, excluded, args.ngram)
    kept, dropped = [], {}
    for r in rows:
        if overlapping(r, seen, args.ngram):
            dropped[r['source']] = dropped.get(r['source'], 0) + 1
        else:
            kept.append(r)
    if len({r['id'] for r in kept}) != len(kept):
        raise ValueError('Duplicate row ids')
    path.write_text(''.join(json.dumps(r) + '\n' for r in kept))
    counts = {}
    for r in kept:
        c = counts.setdefault(r['source'], {'rows': 0, 'groups': set(), 'scored_tokens': 0})
        c['rows'] += 1
        c['groups'].add(r['group'])
        c['scored_tokens'] += len(r['positions']) if r['kind'] == 'text' else 0
    save({'manifest_sha256': sha(path), 'tokenizer': str(SOURCE.relative_to(REPO)), 'ngram': args.ngram,
          'max_length': args.max_length, 'chunk': args.chunk, 'chunks_per_doc': args.chunks_per_doc,
          'gsm8k_sample': args.gsm8k, 'inputs': {s: sha(s.split(':')[0]) for s in args.chat + args.text + args.mc},
          'too_long_dropped': too_long, 'overlap_dropped': dropped,
          'excluded_corpora': sources, 'training_ngrams': len(seen),
          'sources': {k: dict(v, groups=len(v['groups'])) for k, v in counts.items()},
          'mc_prompt': 'raw text, no chat template; letter style lists choices and scores " A".." D"; '
                       'cloze style scores " <choice>" after "Question: ...\\nAnswer:"',
          'limitations': f'{args.ngram}-gram token overlap screen against the listed corpora only; it is not '
                         'group-level disjointness (paraphrases or shared upstream sources can remain). '
                         'A diagnostic pool, not the fresh selection suite.'},
         args.out / 'prepare-report.json')
    log(f'Manifest: {len(kept)} rows kept, {sum(dropped.values())} dropped for training overlap')


# ---------- scoring ----------

def device_of(model):
    return next(model.parameters()).device


def final_states(model, ids, positions):
    device = device_of(model)
    h = model.model(input_ids=torch.tensor([ids], device=device), use_cache=False).last_hidden_state
    return h[0, positions.to(device)]


def text_metrics(teacher, model, row, chunk=64):
    device = device_of(teacher)
    positions = torch.tensor(row['positions'], device=device)
    targets = torch.tensor(row['input_ids'], device=device)[positions + 1]
    ht = final_states(teacher, row['input_ids'], positions)
    hs = ht if model is teacher else final_states(model, row['input_ids'], positions)
    kl = top1 = ce = teacher_ce = 0.
    for a, b, t in zip(ht.split(chunk), hs.split(chunk), targets.split(chunk)):
        lp = teacher.lm_head(a).double().log_softmax(-1)
        lq = lp if model is teacher else model.lm_head(b).double().log_softmax(-1)
        kl += (lp.exp() * (lp - lq)).sum().item()
        top1 += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
        ce -= lq.gather(-1, t[:, None]).sum().item()
        teacher_ce -= lp.gather(-1, t[:, None]).sum().item()
    return {'tokens': len(positions), 'kl': kl, 'top1': top1, 'ce': ce, 'teacher_ce': teacher_ce}


def choice_logprobs(model, context, choices):
    """Summed log-probability of each continuation after the shared context.

    Single-token choices are all read from one context forward. Otherwise the
    choices run as one right-padded batch; the model is causal, so padding after
    a sequence cannot change its logits.
    """
    device = device_of(model)
    if all(len(c) == 1 for c in choices):
        h = final_states(model, context, torch.tensor([len(context) - 1]))
        lq = model.lm_head(h).double().log_softmax(-1)[0]
        return [lq[c[0]].item() for c in choices]
    width = len(context) + max(map(len, choices))
    ids = torch.tensor([context + c + [0] * (width - len(context) - len(c)) for c in choices], device=device)
    h = model.model(input_ids=ids, use_cache=False).last_hidden_state
    out = []
    for i, c in enumerate(choices):
        positions = torch.arange(len(context) - 1, len(context) + len(c) - 1, device=device)
        lq = model.lm_head(h[i, positions]).double().log_softmax(-1)
        out.append(lq.gather(-1, torch.tensor(c, device=device)[:, None]).sum().item())
    return out


def choice_metrics(model, row):
    logprobs = choice_logprobs(model, row['context_ids'], row['choice_ids'])
    scores = {'sum': logprobs,
              'per_token': [lp / len(c) for lp, c in zip(logprobs, row['choice_ids'])],
              'per_char': [lp / n for lp, n in zip(logprobs, row['choice_chars'])]}
    picks = {rule: max(range(len(s)), key=s.__getitem__) for rule, s in scores.items()}
    return {'logprobs': logprobs, 'picks': picks, 'correct': {r: int(p == row['answer']) for r, p in picks.items()}}


@torch.no_grad()
def score_row(teacher, model, row):
    out = {k: row[k] for k in ('id', 'group', 'source', 'kind')}
    out.update(text_metrics(teacher, model, row) if row['kind'] == 'text' else choice_metrics(model, row))
    return out


def resolve(spec):
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)


def stage_score(args):
    name = args.name or args.arm
    loader = args.loader or ARMS[args.arm]
    if name in ARMS and loader != ARMS[name]:
        raise ValueError(f'{name} is a registered arm with a different loader')
    path, partial = args.out / f'scores-{name}.json', args.out / f'scores-{name}.partial.json'
    # No resume: a restarted run could mix rows from different checkpoints or code.
    if path.exists() or partial.exists():
        raise ValueError('Scores or a partial run exist for this arm; remove the partial file deliberately')
    manifest = args.out / 'manifest.jsonl'
    rows = read_jsonl(manifest)
    result = {'arm': name, 'loader': loader, 'manifest_sha256': sha(manifest), 'code_sha256': sha(__file__),
              'torch': torch.__version__, 'threads': torch.get_num_threads(), 'device': args.device,
              'precision': 'FP32, TF32 disabled', 'rows': []}
    teacher = resolve(ARMS['stock'])().to(args.device).eval()
    model = teacher if loader == ARMS['stock'] else resolve(loader)().to(args.device).eval()
    start = time.perf_counter()
    for i, row in enumerate(rows):
        result['rows'].append(score_row(teacher, model, row))
        if (i + 1) % 32 == 0:
            save(result, partial)
            log(f'{name}: {i + 1}/{len(rows)}')
    result['seconds'] = time.perf_counter() - start
    result['status'] = 'complete'
    save(result, path)
    partial.unlink(missing_ok=True)
    log(f"{name}: complete in {result['seconds']:.0f}s")


# ---------- comparison ----------

def paired_bootstrap(groups, base, arm, weights=None, draws=10000, seed=0):
    """Weighted mean of arm - base, resampling whole groups with replacement.

    base and arm are per-item sums; weights are their counts (tokens for text
    rows, 1 for choices), so token metrics are per-token ratio estimators.
    """
    keys = {g: i for i, g in enumerate(sorted(set(groups)))}
    index = torch.tensor([keys[g] for g in groups])
    base = torch.tensor(base, dtype=torch.float64)
    arm = torch.tensor(arm, dtype=torch.float64)
    w = torch.ones(len(base), dtype=torch.float64) if weights is None else torch.tensor(weights, dtype=torch.float64)
    n = len(keys)
    diff = torch.zeros(n, dtype=torch.float64).index_add_(0, index, arm - base)
    size = torch.zeros(n, dtype=torch.float64).index_add_(0, index, w)
    gen = torch.Generator().manual_seed(seed)
    counts = torch.zeros(draws, n, dtype=torch.float64)
    counts.scatter_add_(1, torch.randint(0, n, (draws, n), generator=gen), torch.ones(draws, n, dtype=torch.float64))
    estimates = (counts @ diff) / (counts @ size)
    lo, hi = torch.quantile(estimates, torch.tensor([.025, .975], dtype=torch.float64)).tolist()
    return {'base': (base.sum() / w.sum()).item(), 'arm': (arm.sum() / w.sum()).item(),
            'difference': (diff.sum() / size.sum()).item(), 'ci95': [lo, hi],
            'groups': n, 'items': len(base), 'draws': draws, 'seed': seed}


def compare_scores(base, arm, draws=10000, seed=0):
    if base['manifest_sha256'] != arm['manifest_sha256']:
        raise ValueError('Scores come from different manifests')
    if base.get('device') != arm.get('device'):
        raise ValueError('Compare arms scored on the same device and precision')
    b = {r['id']: r for r in base['rows']}
    a = {r['id']: r for r in arm['rows']}
    if len(b) != len(base['rows']) or len(a) != len(arm['rows']):
        raise ValueError('Duplicate row ids')
    if b.keys() != a.keys():
        raise ValueError('Arms scored different rows')
    for i, x in b.items():
        y = a[i]
        if any(x[k] != y[k] for k in ('kind', 'source', 'group')) or \
                (x['kind'] == 'text' and x['tokens'] != y['tokens']) or \
                (x['kind'] == 'mc' and len(x['logprobs']) != len(y['logprobs'])):
            raise ValueError(f'Row {i} differs in identity between arms')
    out = {'base': base['arm'], 'arm': arm['arm'], 'manifest_sha256': base['manifest_sha256'], 'metrics': {}}
    sources = sorted({(r['kind'], r['source']) for r in b.values()})
    for kind, source in sources:
        ids = sorted(i for i in b if b[i]['source'] == source and b[i]['kind'] == kind)
        groups = [b[i]['group'] for i in ids]
        if kind == 'text':
            tokens = [b[i]['tokens'] for i in ids]
            for metric in ('kl', 'top1', 'ce'):
                out['metrics'][f'{source}/{metric}'] = paired_bootstrap(
                    groups, [b[i][metric] for i in ids], [a[i][metric] for i in ids], tokens, draws, seed)
            out['metrics'][f'{source}/teacher_ce'] = sum(b[i]['teacher_ce'] for i in ids) / sum(tokens)
            continue
        for rule in RULES:
            x = [b[i]['correct'][rule] for i in ids]
            y = [a[i]['correct'][rule] for i in ids]
            stats = paired_bootstrap(groups, x, y, None, draws, seed)
            stats['base_only_correct'] = sum(p and not q for p, q in zip(x, y))
            stats['arm_only_correct'] = sum(q and not p for p, q in zip(x, y))
            stats['same_pick'] = sum(b[i]['picks'][rule] == a[i]['picks'][rule] for i in ids) / len(ids)
            out['metrics'][f'{source}/accuracy_{rule}'] = stats
    return out


def stage_compare(args):
    base = json.loads((args.out / f'scores-{args.base}.json').read_text())
    arm = json.loads((args.out / f'scores-{args.name or args.arm}.json').read_text())
    if base.get('status') != 'complete' or arm.get('status') != 'complete':
        raise ValueError('Both arms must be completely scored')
    result = compare_scores(base, arm, args.draws, args.seed)
    save(result, args.out / f"compare-{args.base}-vs-{args.name or args.arm}.json")
    for key, value in result['metrics'].items():
        if isinstance(value, dict):
            log(f"{key}: {value['difference']:+.4f} [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['prepare', 'score', 'compare'])
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='cpu', help='compare only arms scored on the same device')
    ap.add_argument('--max-length', type=int, default=1024)
    ap.add_argument('--ngram', type=int, default=NGRAM)
    ap.add_argument('--chat', action='append', default=[], help='JSONL of {id, messages}, as FILE:SOURCE')
    ap.add_argument('--text', action='append', default=[], help='JSONL of {id, text} documents, as FILE:SOURCE')
    ap.add_argument('--chunk', type=int, default=512, help='tokens per document chunk')
    ap.add_argument('--chunks-per-doc', type=int, default=2)
    ap.add_argument('--gsm8k', type=int, help='deterministic sample of local GSM8K problems (default: all)')
    ap.add_argument('--mc', action='append', default=[],
                    help='JSONL of {question, choices, answer, subject?}, as FILE:SOURCE:letter|cloze|continue')
    ap.add_argument('--exclude', action='append', default=[], help='extra training corpus, as FILE:key1,key2')
    ap.add_argument('--arm', choices=sorted(ARMS), help='registered arm to score or compare')
    ap.add_argument('--loader', help='module:function returning an FP32 model, for unregistered arms')
    ap.add_argument('--name', help='arm name when --loader is used')
    ap.add_argument('--base', default='stock')
    ap.add_argument('--draws', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.loader and not args.name:
        ap.error('--loader needs --name')
    if args.stage in ('score', 'compare') and not (args.arm or args.name):
        ap.error('choose --arm or --loader with --name')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=True)
    globals()[f'stage_{args.stage}'](args)


if __name__ == '__main__':
    main()
