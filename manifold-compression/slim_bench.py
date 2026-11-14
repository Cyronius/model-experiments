"""A quick multiple-choice benchmark: how much of the dense Qwen3.5-9B's knowledge and reasoning the
healed 25% student keeps.

Four public sets, zero-shot, from the sample eval_data.py converted (tf_metrics.py scored Qwen3.5-2B
on it): MMLU (20 questions per subject), ARC-Challenge, ARC-Easy and HellaSwag. Raw text, no chat
template, tf_metrics' prompts: MMLU lists the choices and scores the letters " A" to " D"; ARC scores
each answer's text after "Question: ...\\nAnswer:"; HellaSwag scores each ending after the text. The
pick is the choice with the highest log-probability, summed or per character (the set's rule).

Every sequence is <|endoftext|>, the context and a choice in the 64K tokenizer, as the drift benchmark
reads text, scored over the 64K kept rows, renormalized: so the dense, healed, distilled and
every-neuron arms differ only in the model body. The dense_full arm is the dense model as shipped,
with the full tokenizer and output head, and so are the rules' references, smaller Qwen3.5 models with
the same tokenizer, which place the student on a size scale. Items whose context and right answer
share a 13-token window with the healing data (which section 17's distillation also drew its text
from) are dropped. An arm named <arm>_k<n> is that arm with its learned routers picking n experts
instead of the carve's K. A later run can score only new arms and name an earlier run's folder
(earlier, the same rows), whose arms the report then compares with.

Settings, predictions and how to read the result are in slim_bench_rules.json; outputs go to its
folder beside the carve's. Stages:

  prepare  (local) the rows: each item in both tokenizers, after the overlap audit.
  score    the rebuilt student's reproduction check, then the rules' arms in order, one file each.
  report   (local) accuracy per set and arm, and the rules' pairs of arms compared item by item.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import heal_probe as hp
import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
import slim_opd as so
from h0_mlp_probe import deadline, log, save, sha
from tf_metrics import STYLES

ROOT = mc.ROOT
RULES = ROOT / 'slim_bench_rules.json'
CODE = ['slim_bench.py', 'slim_opd.py', 'heal_probe.py', 'slim_heal.py', 'slim_batch.py', 'moe_carve.py',
        'h0_mlp_probe.py', 'tf_metrics.py', 'test_slim_bench.py']
DATA = mc.REPO / 'models/manifold-compression/eval-data/converted'
# The run's folder beside the carve's (the rules' folder; main sets it).
FOLDER = 'bench'


def out(name):
    return mc.OUT / FOLDER / name


def fresh(name):
    if out(name).exists():
        raise ValueError(f'{name} exists; refusing to overwrite')
    out(name).parent.mkdir(parents=True, exist_ok=True)
    return out(name)


def provenance():
    return {'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'rules_sha256': sha(RULES),
            'heal_rules_sha256': sha(sh.RULES), 'carve_rules_sha256': sha(sb.CARVE_RULES),
            'slim_rules_sha256': sha(sb.RULES), 'adapter_sha256': sha(sh.out('adapter.pt'))}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- rows

def encoded(context, choices, encode, eot):
    return {'context': [eot] + encode(context), 'choices': [encode(c) for c in choices]}


def item_rows(spec, items, short, full, eot):
    """One row per item: its group (MMLU's subject; otherwise the item), answer, choice lengths in
    characters and ids in both tokenizers."""
    rows = []
    for n, item in enumerate(items):
        context, choices = STYLES[spec['style']](item['question'], item['choices'])
        rows.append({'id': f"{spec['name']}-{n}", 'set': spec['name'],
                     'group': item['subject'] if spec['group'] == 'subject' else str(n),
                     'answer': int(item['answer']), 'chars': [len(c) for c in choices],
                     '64k': encoded(context, choices, short.encode, eot),
                     'full': encoded(context, choices, lambda t: full.encode(t, add_special_tokens=False).ids, eot)})
        if spec['style'] == 'letter' and any(len(c) != 1 for v in ('64k', 'full') for c in rows[-1][v]['choices']):
            raise ValueError(f'{rows[-1]["id"]}: a letter is not one token')
    return rows


def windows(seqs, n):
    """The sorted, distinct hashes of every n-token window of seqs."""
    return np.unique(np.concatenate([sh.window_hashes(seqs[i:i + 256], n).ravel() for i in range(0, len(seqs), 256)]))


def overlapping(rows, ref, n):
    """Per row, whether its context and right answer (64K ids) share an n-token window with ref's."""
    found = []
    for r in rows:
        h = sh.window_hashes(r['64k']['context'][1:] + r['64k']['choices'][r['answer']], n)
        at = np.searchsorted(ref, h).clip(max=len(ref) - 1)
        found.append(bool((ref[at] == h).any()))
    return found


def stage_prepare(args):
    rules = json.loads(RULES.read_text())
    sb.setup()
    full, short, _ = sb.load_tokenizers()
    eot = full.token_to_id('<|endoftext|>')
    record = json.loads(sh.out('data.json').read_text())
    if sha(sh.out('data.npz')) != record['data_sha256']:
        raise ValueError('The healing data changed after the healing run')
    ref = windows(np.load(sh.out('data.npz'))['ids'], rules['overlap_tokens'])
    rows, sets = [], {}
    for spec in rules['sets']:
        path = DATA / spec['file']
        if sha(path) != spec['sha256']:
            raise ValueError(f'{spec["file"]} is not the converted sample')
        found = item_rows(spec, read_jsonl(path), short, full, eot)
        shared = overlapping(found, ref, rules['overlap_tokens'])
        rows += [r for r, s in zip(found, shared) if not s]
        sets[spec['name']] = {'items': len(found), 'dropped': [r['id'] for r, s in zip(found, shared) if s],
                              'tokens': {v: sum(len(s[2]) for s in sequences(found, v)) for v in ('64k', 'full')}}
        log(f"{spec['name']}: {len(found)} items, {len(sets[spec['name']]['dropped'])} share a window with the "
            f"healing data, {sets[spec['name']]['tokens']['64k']} tokens")
    fresh('rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    save(dict(provenance(), healing_data_sha256=record['data_sha256'], rows=len(rows), rows_sha256=sha(out('rows.jsonl')),
              sets=sets), fresh('prepare.json'))


# ---------------------------------------------------------------- scoring

def sequences(rows, which):
    """(row, choice, input ids, first scored position) of every pass: a row whose choices are all
    single tokens reads them from one pass over its context (choice None); otherwise one pass per
    choice, without its last token."""
    seqs = []
    for n, r in enumerate(rows):
        context, choices = r[which]['context'], r[which]['choices']
        if all(len(c) == 1 for c in choices):
            seqs.append((n, None, context, len(context) - 1))
        else:
            seqs += [(n, j, context + c[:-1], len(context) - 1) for j, c in enumerate(choices)]
    return seqs


def batches(lengths, budget):
    """Indices, longest first, in batches of at most budget padded tokens (or one sequence)."""
    batch = []
    for k in sorted(range(len(lengths)), key=lambda k: -lengths[k]):
        if batch and lengths[batch[0]] * (len(batch) + 1) > budget:
            yield batch
            batch = []
        batch.append(k)
    if batch:
        yield batch


@torch.no_grad()
def choice_logprobs(model, rows, which, head, index, pad, budget):
    """Per row, each choice's summed log-probability after the context, over head's rows
    (renormalized; index maps token ids to them), in fp32 from the final hidden states. Batches
    are right-padded: the model is causal, so padding after a sequence cannot change it."""
    seqs = sequences(rows, which)
    logprobs = [[None] * len(r['chars']) for r in rows]
    for batch in batches([len(s[2]) for s in seqs], budget):
        x = torch.full((len(batch), len(seqs[batch[0]][2])), pad, dtype=torch.long)
        for b, k in enumerate(batch):
            x[b, :len(seqs[k][2])] = torch.tensor(seqs[k][2])
        h = model.model(input_ids=x.to(mc.DEVICE), use_cache=False).last_hidden_state
        for b, k in enumerate(batch):
            n, j, _, start = seqs[k]
            choices = rows[n][which]['choices']
            span = 1 if j is None else len(choices[j])
            lq = F.linear(h[b, start:start + span].float(), head).log_softmax(-1)
            if j is None:
                for c, ids in enumerate(choices):
                    logprobs[n][c] = float(lq[0, index[ids[0]]])
            else:
                target = index[torch.tensor(choices[j], device=lq.device)]
                logprobs[n][j] = float(lq.gather(-1, target[:, None]).sum())
    return logprobs


def picks(row, logprobs, rule):
    scores = logprobs if rule == 'sum' else [lp / n for lp, n in zip(logprobs, row['chars'])]
    return max(range(len(scores)), key=scores.__getitem__)


def accuracy(rows, logprobs, rules):
    """Each set's share right under its rule."""
    rule = {s['name']: s['rule'] for s in rules['sets']}
    right = {}
    for r, lp in zip(rows, logprobs):
        right.setdefault(r['set'], []).append(picks(r, lp, rule[r['set']]) == r['answer'])
    return {s: float(np.mean(v)) for s, v in right.items()}


def variant(arm):
    """An arm's base arm and how many experts its routers pick (None: the carve's K)."""
    match = re.fullmatch(r'(.+)_k(\d+)', arm)
    return (match[1], int(match[2])) if match else (arm, None)


def use_arm(slim, arm, distilled):
    """Give the student the arm's adapter, routing and expert count; returns whether the student runs."""
    if arm == 'dense_full':
        return False
    base, k = variant(arm)
    student, routing = so.ARMS[base]
    if base == 'distilled':
        slim.load(distilled)
    else:
        hp.heal(slim, True)
    hp.route(slim, routing)
    if k is not None:
        so.experts(slim, k)
    return student


def reference(path, threads):
    """A smaller model of the family, as shipped; it must share the dense model's tokenizer."""
    if sha(mc.REPO / path / 'tokenizer.json') != sha(mc.MODEL / 'tokenizer.json'):
        raise ValueError(f'{path} has another tokenizer')
    saved, mc.MODEL = mc.MODEL, mc.REPO / path
    try:
        return mc.load_model(threads)
    finally:
        mc.MODEL = saved


def stage_score(args):
    rules = json.loads(RULES.read_text())
    heal_rules, docs = sh.setup()
    prepared = json.loads(out('prepare.json').read_text())
    if sha(out('rows.jsonl')) != prepared['rows_sha256']:
        raise ValueError('rows.jsonl is not the prepare stage\'s')
    rows = read_jsonl(out('rows.jsonl'))
    t0 = time.time()
    distilled = so.finished_adapter(rules['adapter'])
    model, slim, vocab, same = hp.load_healed(heal_rules, docs)
    full_index = torch.arange(model.config.vocab_size, device=mc.DEVICE)
    heads = {'64k': (vocab.head.float(), vocab.index), 'full': (model.lm_head.weight.float(), full_index)}
    arms, carved = {}, slim.s
    with deadline(rules['hours']):
        healed, check = so.reproduce(model, slim, vocab, docs, heal_rules, rules)
        for arm in rules['arms']:
            t = time.time()
            if arm in rules['references']:
                other = reference(rules['references'][arm], heal_rules['threads'])
                logprobs = choice_logprobs(other, rows, 'full', other.lm_head.weight.float(),
                                           torch.arange(other.config.vocab_size, device=mc.DEVICE), vocab.eot,
                                           rules['batch_tokens'])
                del other
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                which = 'full' if arm == 'dense_full' else '64k'
                try:
                    slim.on = use_arm(slim, arm, distilled)
                    logprobs = choice_logprobs(model, rows, which, *heads[which], vocab.eot, rules['batch_tokens'])
                finally:
                    slim.on, slim.s = False, carved
                    hp.route(slim, 'learned')
                    hp.heal(slim, True)
            fresh(f'scores_{arm}.jsonl').write_text(''.join(json.dumps({'id': r['id'], 'logprobs': lp}) + '\n'
                                                            for r, lp in zip(rows, logprobs)))
            arms[arm] = {'seconds': time.time() - t, 'accuracy': accuracy(rows, logprobs, rules)}
            log(f'{arm}: ' + ', '.join(f'{s} {a:.3f}' for s, a in arms[arm]['accuracy'].items()) +
                f', {arms[arm]["seconds"]:.0f} s')
    save(dict(provenance(), factors_identical=same, reproduces_healed=check, healed=healed, arms=arms,
              seconds=time.time() - t0), fresh('score.json'))


# ---------------------------------------------------------------- report

def mcnemar(base, arm):
    """Exact two-sided McNemar p from the items only one of two arms gets right."""
    b, a = int((base & ~arm).sum()), int((arm & ~base).sum())
    m = a + b
    return min(1.0, 2 * sum(math.comb(m, x) for x in range(min(a, b) + 1)) / 2 ** m) if m else 1.0


def resampled(groups, draws, seed):
    """(draws, groups) counts of each group in a resample of whole groups with replacement, and each
    item's group."""
    labels, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=draws), inverse


def compare(groups, chance, base, arm, draws, seed):
    """arm's accuracy minus base's, and arm's share of base's accuracy above chance, each with a 95%
    interval from resampling whole groups; McNemar's exact p. base and arm are per-item right (bool)."""
    w, inverse = resampled(groups, draws, seed)
    total = lambda v: w @ np.bincount(inverse, v.astype(float), w.shape[1])
    n, c, b, a = total(np.ones(len(base))), total(chance), total(base), total(arm)
    differences, shares = (a - b) / n, (a - c) / (b - c)
    point = lambda v: float(v.mean())
    return {'base': point(base), 'arm': point(arm), 'difference': point(arm) - point(base),
            'difference_ci95': np.quantile(differences, [.025, .975]).tolist(),
            'share': (point(arm) - point(chance)) / (point(base) - point(chance)),
            'share_ci95': np.quantile(shares, [.025, .975]).tolist(),
            'base_only': int((base & ~arm).sum()), 'arm_only': int((arm & ~base).sum()),
            'mcnemar_p': mcnemar(base, arm), 'items': len(base)}, shares


def stage_report(args):
    rules = json.loads(RULES.read_text())
    mc.configure(sb.CARVE_RULES)
    rows = read_jsonl(out('rows.jsonl'))
    score = json.loads(out('score.json').read_text()) if out('score.json').exists() else None
    rule = {s['name']: s['rule'] for s in rules['sets']}
    folders = [mc.OUT / FOLDER] + ([mc.OUT / rules['earlier']] if 'earlier' in rules else [])
    if any(sha(f / 'rows.jsonl') != sha(out('rows.jsonl')) for f in folders):
        raise ValueError('The earlier run scored other rows')
    right = {}
    for arm in dict.fromkeys(rules['arms'] + [a for pair in rules['pairs'] for a in pair]):
        found = [f / f'scores_{arm}.jsonl' for f in folders if (f / f'scores_{arm}.jsonl').exists()]
        if not found:
            continue
        scores = {r['id']: r['logprobs'] for r in read_jsonl(found[0])}
        right[arm] = np.array([picks(r, scores[r['id']], rule[r['set']]) == r['answer'] for r in rows])
    sets = {}
    for spec in rules['sets']:
        mask = np.array([r['set'] == spec['name'] for r in rows])
        chosen = [r for r in rows if r['set'] == spec['name']]
        chance = np.array([1 / len(r['chars']) for r in chosen])
        groups = [r['group'] for r in chosen]
        sets[spec['name']] = {'items': len(chosen), 'chance': float(chance.mean()), 'rule': spec['rule'],
                              'accuracy': {arm: float(v[mask].mean()) for arm, v in right.items()}, 'pairs': {}}
        for base, arm in rules['pairs']:
            if base in right and arm in right:
                stats, shares = compare(groups, chance, right[base][mask], right[arm][mask], rules['draws'], rules['seed'])
                sets[spec['name']]['pairs'][f'{arm}_vs_{base}'] = stats
    # Across sets: the mean of the sets' shares, its interval from the same draws, set by set.
    overall = {}
    for base, arm in rules['pairs']:
        if base in right and arm in right:
            draws = [compare([r['group'] for r in rows if r['set'] == s['name']],
                             np.array([1 / len(r['chars']) for r in rows if r['set'] == s['name']]),
                             right[base][[r['set'] == s['name'] for r in rows]],
                             right[arm][[r['set'] == s['name'] for r in rows]], rules['draws'], rules['seed'] + k)[1]
                     for k, s in enumerate(rules['sets'])]
            overall[f'{arm}_vs_{base}'] = {
                'share': float(np.mean([sets[s['name']]['pairs'][f'{arm}_vs_{base}']['share'] for s in rules['sets']])),
                'share_ci95': np.quantile(np.mean(draws, 0), [.025, .975]).tolist(),
                'difference': float(np.mean([sets[s['name']]['pairs'][f'{arm}_vs_{base}']['difference']
                                             for s in rules['sets']]))}
    result = {'rules_sha256': sha(RULES), 'rows_sha256': sha(out('rows.jsonl')),
              'reproduces_healed': score and score['reproduces_healed'], 'sets': sets, 'overall': overall,
              'seconds': score and {arm: a['seconds'] for arm, a in score['arms'].items()}}
    save(result, out('report.json'))
    for name, s in sets.items():
        log(f'{name} ({s["items"]}, chance {s["chance"]:.2f}): ' +
            ', '.join(f'{arm} {a:.3f}' for arm, a in s['accuracy'].items()))
        for pair, p in s['pairs'].items():
            log(f'  {pair}: {p["difference"]:+.3f} [{p["difference_ci95"][0]:+.3f}, {p["difference_ci95"][1]:+.3f}], '
                f'share {p["share"]:.2f} [{p["share_ci95"][0]:.2f}, {p["share_ci95"][1]:.2f}], p {p["mcnemar_p"]:.2g}')
    for pair, p in overall.items():
        log(f'all sets, {pair}: share {p["share"]:.2f} [{p["share_ci95"][0]:.2f}, {p["share_ci95"][1]:.2f}], '
            f'difference {p["difference"]:+.3f}')


def main():
    global RULES, FOLDER
    stages = {'prepare': stage_prepare, 'score': stage_score, 'report': stage_report}
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=stages)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--model', type=Path, default=mc.MODEL)
    ap.add_argument('--rules', type=Path, default=RULES)
    ap.add_argument('--heal-rules', type=Path, default=sh.RULES)
    ap.add_argument('--slim-rules', type=Path, default=sb.RULES)
    ap.add_argument('--carve-rules', type=Path, default=sb.CARVE_RULES)
    args = ap.parse_args()
    mc.DEVICE, mc.MODEL = args.device, args.model
    RULES, sh.RULES = args.rules.resolve(), args.heal_rules.resolve()
    sb.RULES, sb.CARVE_RULES = args.slim_rules.resolve(), args.carve_rules.resolve()
    # slim_opd's reproduction check writes into the same folder.
    FOLDER = so.FOLDER = json.loads(RULES.read_text()).get('folder', FOLDER)
    stages[args.stage](args)


if __name__ == '__main__':
    main()
