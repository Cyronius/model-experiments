"""The loop breaker's second branch (loop_breaker2_rules.json): what to write once the thinking is closed.

loop_breaker.py's two neurons end a re-check loop inside <think> by raising </think> at a line end
inside a verbatim repeat. Their generation test (2026-11-27: greedy loops 28 -> 12 of 96, 17 fixed,
1 new, 16 of the 17 after the neurons fired) showed what the program lacks: the answer after the
forced </think> can repeat as well, the neurons fire again, and they write </think> a second time;
the model then answers with nothing but </think> lines (10 of the 12 loops left). The missing branch:

    IF repeating AND at a line end:   IF the thinking is open THEN </think>   ELSE <|im_end|>

It needs a third read, "the thinking is closed", and answers that repeat after a close to fit and
screen it on. Stages:

    loop_breaker2.py prompts              # CPU, network: the fresh test prompts (GSM8K test rows)
    loop_breaker2.py fit --device cuda    # the model with the first neurons answers the fitting
                                          # prompts; the unedited model is captured over those answers
    loop_breaker2.py build                # CPU: the closed read, the second pair, the screen -> breaker2.pt
    loop_breaker2.py test --arm unedited|breaker|breaker2 --prompts check|fresh [--sampled] --device cuda
    loop_breaker2.py test --arm dense --prompts check|fresh   # the model the carve was cut from, same build
    loop_breaker2.py outcome              # CPU: the predictions' counts from the saved answers -> outcome.json

The second pair (rules' design): its gate is the first pair's with the line-end read moved into it
(repeating AND at a line end), its up projection the closed read m.x, and its write the unit
<|im_end|> write times a dose minus the first pair's write. So where the first pair fires, the two
together write (1 - m.x) of the </think> write and m.x of the <|im_end|> write: the closed read is
loose (it is a number, not a bit), and as a scale on the writes its error costs a late or doubled
close, where as a gate it would move the repeat probe's threshold.
"""
import argparse
import json
import time

import numpy as np
import torch

import heal_block as hb
import heal_block_check as hbc
import heal_probe as hp
import loop_breaker as lb
import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
import slim_opd as so

RULES = mc.ROOT / 'loop_breaker2_rules.json'
log = lb.log


def answers_and_captures(model, slim, block, tok, prompts, first, saved, out):
    """gen_fit.jsonl: the greedy answers of the model with the first neurons in. capture/: the
    unedited model (the neurons out again) teacher-forced over those answers, as lb.capture saves it:
    its logits there are what any other neurons would add their write to."""
    cfg, path = first['fit'], out / 'gen_fit.jsonl'
    piece = dict(block.prefix)[first['layer']]
    if path.exists():
        rows = [json.loads(line) for line in path.open()]
    else:
        t0 = time.time()
        lb.install(block, first, saved, 'breaker')
        try:
            gen = lb.greedy(model, slim, block, lb.vocab_of(model, tok), prompts, cfg['generate'], cfg['batch'],
                            cfg['generate']['seed'])
        finally:
            del piece.mlp.bridge
        rows = lb.answer_rows(gen, prompts, tok, 'breaker')
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        log(f'generated in {time.time() - t0:.0f} s')
    if piece.mlp._modules.get('bridge') is not None:
        raise RuntimeError('The neurons are still in: the capture would not be the unedited model')
    lb.capture(model, block, tok, rows, prompts, first, out)
    return rows


def stage_fit(args, rules, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    first = json.loads(lb.RULES.read_text())
    prompts = lb.fit_prompts(first, tok)
    model, slim, block, record = lb.build_model(first)
    saved = torch.load(mc.ROOT / first['out'] / 'breaker.pt', weights_only=True)
    rows = answers_and_captures(model, slim, block, tok, prompts, first, saved, out)
    s = hp.generations(rows, first['loop'])
    think = tok.convert_tokens_to_ids(lb.STOPS[0])
    closed = sum(think in r['tokens'] for r in rows)
    twice = sum(r['tokens'].count(think) > 1 for r in rows)
    log(f"fitting prompts with the first neurons: {s['correct']}/{len(rows)} right, {s['finished']} finished, "
        f"{s['loops']} loops; {closed} answers close their thinking, {twice} write </think> more than once")
    (out / 'fit.json').write_text(json.dumps(dict(record, summary={k: v for k, v in s.items() if k != 'looping'},
                                                  closed=closed, more_than_one_close=twice), indent=1))


# ---------------------------------------------------------------- the second pair (CPU)

def closed_after(tokens, think):
    """Per captured position 0..len(tokens) (position t predicts token t): whether a </think> is
    among the tokens before it, and whether two are."""
    seen = np.concatenate([[0], np.cumsum(np.asarray(tokens) == think)])
    return torch.from_numpy(seen >= 1), torch.from_numpy(seen >= 2)


def end_write(q, rms):
    """v for a unit dose of <|im_end|>: in the span of the two stop rows q (</think>, <|im_end|>),
    raising <|im_end|> by 1 and </think> by 0 where the final residual's RMS is rms."""
    coef = torch.linalg.solve(q @ q.T, torch.tensor([0.0, 1.0], dtype=q.dtype))
    return rms * coef @ q


def four_neurons(saved, m, v_end1, dose, line):
    """gate (4, H), up (4, H), down (H, 4): the first pair as it is, then the second pair: gate
    k (w - b c - line (c - n)), up m, write dose v_end1 - v."""
    k, clamp, c = saved['k'], saved['clamp'], saved['c']
    gate = k * (saved['w'] - saved['b'] * c - line * (c - saved['up'][0]))
    d = dose * v_end1 - saved['v']
    return (torch.cat([saved['gate'], torch.stack([gate, gate - clamp * c])]),
            torch.cat([saved['up'], torch.stack([m, m])]),
            torch.cat([saved['down'], torch.stack([d / clamp, -d / clamp], 1)], 1))


def activations(t, saved, line):
    """Per position: the first pair's activation (a multiple of its write v) and the second pair's
    (a multiple of dose v_end1 - v), from the reads in t."""
    b, k, clamp = saved['b'], saved['k'], saved['clamp']
    return (lb.activation(t['s'], t['u'], b, k, clamp, t['nx']),
            lb.activation(t['s'] - line * (t['u'] - t['nx']), t['u'], b, k, clamp, t['mx']))


def terms(z, head, index, saved, m, v_end1):
    """Per position: the four reads, and the scalars that give the argmax over the unedited top five
    and the two stop tokens for any activations and dose (logits x residual RMS), as lb.row_terms."""
    x, r = torch.from_numpy(z['x'].astype(np.float32)), torch.from_numpy(z['r'])
    top = torch.from_numpy(z['top'].astype(np.int64))
    stops = torch.tensor(head['stops'])
    q = head['rows'][index[top]] * head['scale']
    qs = head['rows'][index[stops]] * head['scale']
    gain = lambda v: torch.cat([q @ v, (qs @ v).expand(len(r), 2)], 1)
    return {'s': x @ saved['w'], 'u': x @ saved['c'], 'nx': x @ saved['up'][0], 'mx': x @ m,
            'base': torch.cat([torch.einsum('tkh,th->tk', q, r), r @ qs.T], 1),
            'think': gain(saved['v']), 'end': gain(v_end1), 'tokens': torch.cat([top, stops.expand(len(r), 2)], 1)}


def choice(t, a1=None, a2=None, dose=0.0):
    """The argmax token per position with the first pair's activation a1 and the second's a2."""
    logits = t['base']
    if a1 is not None:
        logits = logits + a1[:, None] * t['think']
    if a2 is not None:
        logits = logits + a2[:, None] * (dose * t['end'] - t['think'])
    return t['tokens'].gather(1, logits.argmax(1, keepdim=True))[:, 0]


def first_difference(old, new):
    hits = (old != new).nonzero()[:, 0]
    return (int(hits[0]), int(new[hits[0]])) if len(hits) else (None, None)


def judge(source, kind, t, a1, a2, dose, closed, second, stops):
    """One answer under the four neurons. source 'unedited': the answer is the unedited model's, and
    the four are compared with the first pair alone on it. source 'first': the answer is the first
    pair's own, and the four leave it where their argmax first differs from the first pair's.
    Returns (population, outcome, position, token):
      think   the first pair alone closes the thinking here (its first change is </think>, thinking
              open): 'kept' if the four make the same first change, else what they do instead.
      closed  the first pair fires after a close (before any second </think>): 'ended' if the four
              first differ with <|im_end|> at a closed position, else 'none' or what they do.
      done    another finished answer: 'none' if the four change nothing.
      rest    anything else (cut at the cap, loops no pair touches), for the record.
    An <|im_end|> while the thinking is open is 'end_in_think' in every population."""
    think, end = stops
    base, one, four = choice(t), choice(t, a1), choice(t, a1, a2, dose)
    fires_closed = bool(((a1 > 0.5) & closed & ~second).any())
    at1, token1 = first_difference(base, one)
    if source == 'unedited':
        at, token = first_difference(base, four)
        opens = at1 is not None and token1 == think and not bool(closed[at1])
        population = 'closed' if fires_closed else 'think' if opens else 'done' if kind == 'done' else 'rest'
    else:
        at, token = first_difference(one, four)
        population = 'closed' if fires_closed else 'done' if kind == 'done' else 'rest'
    if at is None:
        return population, 'none', at, token
    if token == end and not bool(closed[at]):
        return population, 'end_in_think', at, token
    if population == 'think':
        return population, ('kept' if (at, token) == (at1, token1) else 'think_elsewhere' if token == think else 'other'), at, token
    if population == 'closed':
        return population, ('ended' if token == end else 'second_close' if token == think else 'other'), at, token
    return population, ('ended' if token == end else 'changed'), at, token


def tally(answers, chosen, saved, line, dose, stops):
    """Counts {population: {outcome: n}} and each answer's record, over the chosen (set, row) pairs."""
    counts, each = {}, []
    for key in chosen:
        a = answers[key]
        a1, a2 = activations(a['terms'], saved, line)
        population, outcome, at, token = judge(key[0], a['kind'], a['terms'], a1, a2, dose, a['closed'], a['second'], stops)
        counts.setdefault(population, {}).setdefault(outcome, 0)
        counts[population][outcome] += 1
        each.append({'set': key[0], 'row': key[1], 'id': a['id'], 'kind': a['kind'], 'population': population,
                     'outcome': outcome, 'at': at, 'token': token})
    return counts, each


def allowed(counts, share):
    """Whether a setting harms little enough: no <|im_end|> inside the thinking, at most max(1,
    share) of the other finished answers changed, at least 90% of the first pair's closes kept."""
    bad = sum(c.get('end_in_think', 0) for c in counts.values())
    done, think = counts.get('done', {}), counts.get('think', {})
    changed = sum(v for k, v in done.items() if k != 'none')
    return bad == 0 and changed <= max(1, int(share * sum(done.values()))) and think.get('kept', 0) >= 0.9 * sum(think.values())


def select(ok, doses):
    """Of the allowed settings: the smallest dose that ends the most 'closed' answers, then the next
    dose of the grid if it is allowed too (a step of margin: the closed read is lowest in the hard cases)."""
    best = max(ok, key=lambda g: (g['counts'].get('closed', {}).get('ended', 0), -g['dose']))
    above = [d for d in sorted(doses) if d > best['dose']]
    return next((g for g in ok if above and g['dose'] == above[0]), best)


def stage_build(args, rules, out):
    torch.set_num_threads(args.threads)
    first = json.loads(lb.RULES.read_text())
    build(first, mc.ROOT / first['out'], rules, out)


def build(first, one, rules, out):
    """From the first neurons' folder `one` (breaker.pt, tokens.json, the unedited answers' captures)
    and this folder's captures (the first neurons' answers): build.json, and breaker2.pt if a dose is allowed."""
    saved = torch.load(one / 'breaker.pt', weights_only=True)
    tokens = json.loads((one / 'tokens.json').read_text())
    cfg = rules['neurons']
    line = cfg['line']
    sets = {}
    for name, folder in (('unedited', one), ('first', out)):
        rows = [json.loads(row) for row in (folder / 'gen_fit.jsonl').open()]
        head = torch.load(folder / 'head.pt', weights_only=True)
        index = torch.full((head['vocab'],), -1, dtype=torch.long)
        index[head['ids']] = torch.arange(len(head['ids']))
        sets[name] = {'rows': rows, 'head': head, 'index': index, 'folder': folder, 'kinds': lb.kinds(rows, first)}
    stops = sets['unedited']['head']['stops']
    think = stops[0]
    fitting = lb.split(sets['unedited']['kinds'])                 # by prompt, as the first probe's split
    keys = [(name, k) for name in sets for k in range(len(sets[name]['rows']))]
    load = lambda key: np.load(sets[key[0]]['folder'] / 'capture' / f'{key[1]}.npz')
    state = lambda key: closed_after(sets[key[0]]['rows'][key[1]]['tokens'], think)

    def batches():                                                # the closed read: every position up to a second </think>
        for key in keys:
            if fitting[key[1]]:
                closed, second = state(key)
                yield torch.from_numpy(load(key)['x'].astype(np.float32))[~second], closed[~second].float()

    m = lb.fit_readout(batches(), rules['closed']['ridge'])
    v_end1 = end_write(saved['q'], saved['rms'])
    line_end = torch.zeros(sets['unedited']['head']['vocab'], dtype=torch.bool)
    line_end[tokens['line_ends']] = True
    answers = {}
    for key in keys:
        s, k = sets[key[0]], key[1]
        closed, second = state(key)
        answers[key] = {'id': s['rows'][k]['id'], 'kind': s['kinds'][k][0], 'closed': closed, 'second': second,
                        'current': line_end[torch.tensor([tokens['prompt_last'][k]] + s['rows'][k]['tokens'])],
                        'terms': terms(load(key), s['head'], s['index'], saved, m, v_end1)}
    report = {'answers': {name: {kind: sum(k == kind for k, _ in s['kinds']) for kind in ('loop', 'drift', 'done', 'cut')}
                          for name, s in sets.items()}}
    log(f"answers: {report['answers']}")

    # The closed read where it is used: at the positions where the first pair fires, held-out prompts.
    read = {}
    for part, want in (('fitting', True), ('held_out', False)):
        got = {'open': [], 'closed': [], 'open_line_ends': [], 'closed_line_ends': []}
        for key in keys:
            if fitting[key[1]] == want:
                a = answers[key]
                a1, _ = activations(a['terms'], saved, line)
                mx, keep = a['terms']['mx'], ~a['second']
                got['open'].append(mx[(a1 > 0.5) & ~a['closed']])
                got['closed'].append(mx[(a1 > 0.5) & a['closed'] & keep])
                got['open_line_ends'].append(mx[a['current'] & ~a['closed']])
                got['closed_line_ends'].append(mx[a['current'] & a['closed'] & keep])
        got = {k: torch.cat(v) for k, v in got.items()}
        read[part] = {k: {'positions': len(v), 'quantiles': np.quantile(v.numpy(), [0, .01, .1, .5, .9, .99, 1]).tolist() if len(v) else None}
                      for k, v in got.items()}
        read[part]['auroc_where_it_fires'] = lb.auroc(got['closed'], got['open']) if len(got['closed']) and len(got['open']) else None
        read[part]['auroc_line_ends'] = lb.auroc(got['closed_line_ends'], got['open_line_ends'])
    report['closed_read'] = read
    for part in read:
        fmt = lambda name: None if read[part][name]['quantiles'] is None else np.round(read[part][name]['quantiles'], 2).tolist()
        log(f"closed read, {part}, where the first pair fires (min, 1%, 10%, median, 90%, 99%, max): open {fmt('open')} "
            f"({read[part]['open']['positions']}), closed {fmt('closed')} ({read[part]['closed']['positions']})")

    fit_keys = [key for key in keys if fitting[key[1]]]
    held_keys = [key for key in keys if not fitting[key[1]]]
    grid = []
    for dose in rules['grid']['doses']:
        counts, _ = tally(answers, fit_keys, saved, line, dose, stops)
        grid.append({'dose': dose, 'counts': counts, 'allowed': allowed(counts, 0.02)})
    report['grid'] = grid
    ok = [g for g in grid if g['allowed']]
    if not ok:
        report['selected'] = None
        (out / 'build.json').write_text(json.dumps(report, indent=1))
        log('no dose is allowed on the fitting prompts: no second pair')
        return
    best = select(ok, rules['grid']['doses'])
    counts, each = tally(answers, held_keys, saved, line, best['dose'], stops)
    closed, done, opened = counts.get('closed', {}), counts.get('done', {}), counts.get('think', {})
    ended, bad = closed.get('ended', 0), sum(c.get('end_in_think', 0) for c in counts.values())
    changed = sum(v for k, v in done.items() if k != 'none')
    gate = (ended >= 0.6 * sum(closed.values()) and bad == 0 and changed <= 0.05 * sum(done.values())
            and opened.get('kept', 0) >= 0.9 * sum(opened.values()))
    report.update(selected=best, held_out={'counts': counts, 'answers': each}, gate=bool(gate))
    log(f"selected dose {best['dose']} (fitting: {best['counts']})")
    log(f"held out: {counts}; gate {'holds' if gate else 'fails'}")
    gate_w, up_w, down_w = four_neurons(saved, m, v_end1, best['dose'], line)
    torch.save(dict(saved, gate=gate_w, up=up_w, down=down_w, m=m, v_end1=v_end1, end_dose=best['dose'], line=line),
               out / 'breaker2.pt')
    (out / 'build.json').write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------- generation with the neurons in

def sampled_answers(model, slim, block, vocab, prompts, cfg, batch):
    """The prompts sampled at the check's settings, with heal_block_check gsm8k --sampled's seeds
    and flags: an answer the neurons never touch is that check's answer token for token."""
    gen = []
    block.swap(True)
    try:
        with torch.autocast(mc.DEVICE, dtype=torch.bfloat16):
            for b in range(0, len(prompts), batch):
                part = [p['ids'] for p in prompts[b:b + batch]]
                gen += so.generate(model, slim, vocab, part, cfg, True, [False] * len(part), cfg['sampling'], cfg['seed'] + 1 + b)
                log(f'{len(gen)}/{len(prompts)} answered')
    finally:
        block.swap(False)
    return gen


def dense_answers(model, slim, tok, prompts, cfg, batch=None):
    """The model the carve was cut from on the same prompts, greedy: no pieces swapped in, the
    router off (slim_opd.generate's dense side), the same state, cap and seed as the other arms."""
    gen, batch = [], batch or len(prompts)
    with torch.autocast(mc.DEVICE, dtype=torch.bfloat16):
        for b in range(0, len(prompts), batch):
            gen += so.generate(model, slim, lb.vocab_of(model, tok), [p['ids'] for p in prompts[b:b + batch]], cfg, False, True,
                               seed=cfg['seed'] + b)
            log(f'{len(gen)}/{len(prompts)} answered')
    return lb.answer_rows(gen, prompts, tok, 'dense')


def answers_of(model, slim, block, tok, prompts, cfg, first, saved, arm, batch=None, sampled=False):
    """An arm's answers to the prompts, greedy or sampled (lb.answer_rows), and, with neurons in,
    their activations at each step's last position: (pairs, steps, prompts), or None if the steps
    do not line up. saved None: the unedited model."""
    neurons = None
    piece = dict(block.prefix)[first['layer']]
    if saved is not None:
        neurons = lb.install(block, first, saved, 'breaker')
        neurons.record = []
    try:
        if sampled:
            gen = sampled_answers(model, slim, block, lb.vocab_of(model, tok), prompts, cfg, batch or len(prompts))
        else:
            gen = lb.greedy(model, slim, block, lb.vocab_of(model, tok), prompts, cfg, batch or len(prompts), cfg['seed'])
    finally:
        if neurons is not None:
            del piece.mlp.bridge
    acts = None
    if neurons is not None and len(neurons.record) and all(len(f) == len(prompts) for f in neurons.record):
        f = torch.stack(neurons.record).float()                       # (steps, prompts, neurons)
        acts = torch.stack([(f[..., j] - f[..., j + 1]) / saved['clamp'] for j in range(0, f.shape[-1], 2)]).numpy()
    return lb.answer_rows(gen, prompts, tok, arm), acts


@torch.no_grad()
def verify(model, block, tok, first, one, saved, out, count=6):
    """The four neurons in the model against the screen's forecast: on held-out answers of the first
    pair where it fires after a close, where the model's argmax (whole vocabulary) with the four
    first differs from its argmax with the first pair alone."""
    report = json.loads((out / 'build.json').read_text())
    rows = [json.loads(line) for line in (out / 'gen_fit.jsonl').open()]
    prompts = lb.fit_prompts(first, tok)
    piece = dict(block.prefix)[first['layer']]
    checked = []
    for a in [a for a in report['held_out']['answers'] if a['set'] == 'first' and a['population'] == 'closed'][:count]:
        k = a['row']
        ids = prompts[k]['ids'] + rows[k]['tokens']
        start = len(prompts[k]['ids']) - 1
        picks = []
        for weights in (one, saved):
            lb.install(block, first, weights, 'breaker')
            try:
                with hbc.Student(block):
                    picks.append(model.lm_head(lb.student_pass(model, ids)[start:]).argmax(-1))
            finally:
                del piece.mlp.bridge
        hits = (picks[0] != picks[1]).nonzero()[:, 0]
        at = int(hits[0]) if len(hits) else None
        checked.append({'id': a['id'], 'forecast': a['at'], 'model': at, 'forecast_token': a['token'],
                        'model_token': int(picks[1][at]) if at is not None else None})
        log(f"{a['id']}: forecast {a['outcome']} at {a['at']}, the model differs at {at}")
    return checked


def test_prompts(tok, rules, cfg, name):
    file = cfg['prompts'] if name == 'check' else rules['fresh']['file']
    prompts = json.loads((mc.ROOT / file).read_text())['prompts'][:cfg['count']]
    for p in prompts:
        p['ids'] = tok.encode(so.template(tok, p['question'], cfg.get('enable_thinking')), add_special_tokens=False)
    return prompts


def sampled_check(one):
    """The unedited model's sampled answers to the check prompts (heal_block_check gsm8k --sampled)."""
    return one.parent / 'checks' / 'gen_whole-pieces4_sampled.jsonl'


def stage_test(args, rules, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    first = json.loads(lb.RULES.read_text())
    one = mc.ROOT / first['out']
    check = json.loads(hbc.GSM_RULES.read_text())
    cfg = dict(check['eval'], compact=None)
    prompts = test_prompts(tok, rules, cfg, args.prompts)
    model, slim, block, record = lb.build_model(first)
    decoding = 'sampled' if args.sampled else 'greedy'
    record.update(arm=args.arm, prompts=args.prompts, decoding=decoding)
    name = f'{args.arm}_{args.prompts}' + ('_sampled' if args.sampled else '')
    saved = None
    if args.arm == 'dense' and args.sampled:
        raise ValueError('the dense arm is greedy only')
    if args.arm not in ('unedited', 'dense'):
        pair = torch.load(one / 'breaker.pt', weights_only=True)
        saved = pair if args.arm == 'breaker' else torch.load(out / 'breaker2.pt', weights_only=True)
        if args.arm == 'breaker2':
            record['verify'] = verify(model, block, tok, first, pair, saved, out)
        neurons = lb.install(block, first, saved, 'breaker')
        try:
            record['kl_arm'] = hb.end_to_end(block, block.docs)
        finally:
            del dict(block.prefix)[first['layer']].mlp.bridge
        log(f"end-to-end KL with the neurons {record['kl_arm']['kl']:.4f} (without {record['kl']['kl']:.4f})")
    t0 = time.time()
    if args.arm == 'dense':
        rows, acts = dense_answers(model, slim, tok, prompts, cfg, args.batch), None
    else:
        rows, acts = answers_of(model, slim, block, tok, prompts, cfg, first, saved, args.arm, args.batch, args.sampled)
    (out / f'gen_{args.arm}_{args.prompts}_{decoding}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    if acts is not None:
        np.save(out / f'acts_{name}.npy', acts.astype(np.float16))
    s = hp.generations(rows, check['loop'])
    record.update(summary={k: v for k, v in s.items() if k != 'looping'}, looping=s['looping'], seconds=time.time() - t0)
    log(f"{name}: {s['correct']}/{len(rows)} right, {s['finished']} finished, {s['loops']} loops")
    # The same prompts' answers by the unedited model and by the first pair alone, where they exist.
    earlier = {arm: out / f'gen_{arm}_{args.prompts}_{decoding}.jsonl' for arm in ('unedited', 'breaker')}
    if args.prompts == 'check':                                   # the check prompts' earlier answers live elsewhere
        earlier = {'unedited': sampled_check(one), 'breaker': out / 'gen_breaker_check_sampled.jsonl'} if args.sampled else \
            {'unedited': one / 'gen_unedited_greedy.jsonl', 'breaker': one / 'gen_breaker_greedy.jsonl'}
    for arm, key in (('unedited', 'against_unedited'), ('breaker', 'against_first')):
        if args.arm not in ('unedited', 'dense', arm) and earlier[arm].exists():
            log(f'against {arm}:')
            record[key] = lb.against([json.loads(line) for line in earlier[arm].open()], rows,
                                     None if acts is None else np.abs(acts).max(0), check['loop'])
    (out / f'test_{name}.json').write_text(json.dumps(record, indent=1))


def marks(rows, think, loop):
    """What the predictions and criteria count in one arm's answers: right, right in a finished answer, finished, loops, answers
    with three or more </think>, and answers that end (or run to the cap) with their thinking open."""
    s = hp.generations(rows, loop)
    closes = [r['tokens'].count(think) for r in rows]
    return {'right': s['correct'], 'right_and_finished': sum(bool(r['correct'] and r['finished']) for r in rows),
            'finished': s['finished'], 'loops': s['loops'],
            'three_or_more_closes': sum(c >= 3 for c in closes),
            'ended_inside_thinking': sum(bool(r['finished']) and c == 0 for r, c in zip(rows, closes)),
            'cut_inside_thinking': sum(not r['finished'] and c == 0 for r, c in zip(rows, closes))}


def outcome(files, think, loop):
    """files: {prompt set: {arm: answers file}}. Each arm's marks and, for the arms that exist, the
    paired comparisons the criteria name (later arm against earlier: loops fixed and new, sign tests)."""
    report = {}
    for name, arms in files.items():
        rows = {arm: [json.loads(line) for line in path.open()] for arm, path in arms.items() if path.exists()}
        entry = {'marks': {arm: marks(r, think, loop) for arm, r in rows.items()}, 'pairs': {}}
        for arm, m in entry['marks'].items():
            log(f"{name} {arm}: " + ', '.join(f"{k.replace('_', ' ')} {v}" for k, v in m.items()))
        for base, arm in (('unedited', 'breaker'), ('unedited', 'breaker2'), ('breaker', 'breaker2'), ('dense', 'breaker2')):
            if base in rows and arm in rows:
                log(f'{name}: {arm} against {base}')
                entry['pairs'][f'{arm}_against_{base}'] = lb.against(rows[base], rows[arm], None, loop)
        report[name] = entry
    return report


def stage_outcome(args, rules, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    first = json.loads(lb.RULES.read_text())
    one = mc.ROOT / first['out']
    check = json.loads(hbc.GSM_RULES.read_text())
    files = {'check': {'unedited': one / 'gen_unedited_greedy.jsonl', 'breaker': one / 'gen_breaker_greedy.jsonl',
                       'breaker2': out / 'gen_breaker2_check_greedy.jsonl', 'dense': out / 'gen_dense_check_greedy.jsonl'},
             'fresh': {arm: out / f'gen_{arm}_fresh_greedy.jsonl' for arm in ('unedited', 'breaker', 'breaker2', 'dense')},
             'check_sampled': {'unedited': sampled_check(one), 'breaker': out / 'gen_breaker_check_sampled.jsonl',
                               'breaker2': out / 'gen_breaker2_check_sampled.jsonl'}}
    report = outcome(files, tok.convert_tokens_to_ids(lb.STOPS[0]), check['loop'])
    for name, entry in report.items():                              # the arms' KL where their test wrote one
        for arm in entry['marks']:
            path = out / f'test_{arm}_{name}.json' if (name, arm) != ('check', 'breaker') else one / 'test_breaker.json'
            if arm not in ('unedited', 'dense') and path.exists():
                t = json.loads(path.read_text())
                entry['marks'][arm]['kl_rise'] = t['kl_arm']['kl'] - t['kl']['kl']
    (out / 'outcome.json').write_text(json.dumps(report, indent=1))


def stage_prompts(args, rules, out):
    """The fresh test prompts: GSM8K test rows after the check's (probe_prompts_96.json is rows 16-111)."""
    cfg = rules['fresh']
    path = mc.ROOT / cfg['file']
    if path.exists():
        raise ValueError(f'{path.name} exists; refusing to overwrite')
    a, b = cfg['rows']
    prompts = [{'id': f'gsm8k-test-{n}', 'question': r['question'],
                'answer': float(r['answer'].rpartition('####')[2].strip().replace(',', ''))}
               for n, r in sh.paged('openai/gsm8k', 'main', [[a, b - a]], split='test')]
    if len(prompts) != b - a:
        raise ValueError(f'{len(prompts)} rows')
    path.write_text(json.dumps({'source': f'openai/gsm8k main test, rows {a}-{b - 1}', 'prompts': prompts}, indent=1))
    log(f'{len(prompts)} prompts in {path.name}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['prompts', 'fit', 'build', 'test', 'outcome'])
    ap.add_argument('--arm', default='breaker2', choices=['unedited', 'breaker', 'breaker2', 'dense'],
                    help='test: dense is the model the carve was cut from (greedy only)')
    ap.add_argument('--prompts', default='check', choices=['check', 'fresh'], help='test: the 96 check prompts or the fresh ones')
    ap.add_argument('--sampled', action='store_true', help="test: sample at the model card's settings instead of greedy")
    ap.add_argument('--batch', type=int, default=None, help='test: prompts generated together (default all)')
    ap.add_argument('--threads', type=int, default=8, help='build: CPU threads')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    rules = json.loads(RULES.read_text())
    mc.DEVICE = 'cpu' if args.stage in ('prompts', 'build', 'outcome') else args.device
    mc.configure(sb.CARVE_RULES)
    out = mc.ROOT / rules['out']
    out.mkdir(parents=True, exist_ok=True)
    {'prompts': stage_prompts, 'fit': stage_fit, 'build': stage_build, 'test': stage_test,
     'outcome': stage_outcome}[args.stage](args, rules, out)


if __name__ == '__main__':
    main()
