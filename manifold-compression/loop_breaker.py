"""A loop breaker written by hand into a model healed in pieces (loop_breaker_rules.json).

The user, 2026-11-27: "can you do direct neuron surgery as per some of the latest 'weight space
programming' concepts to fix the looping in this model". The model's greedy loops are re-check
cycles inside <think> (it states an answer, writes "Wait, let's check the question again",
re-derives the same lines verbatim, and repeats). The edit is a small program in two always-on
SwiGLU neurons beside the last carved MLP: IF the residual says the answer is repeating its own
text AND the current token ends a line THEN raise the logit of </think>. Both reads are closed-form
linear probes on the last MLP's input (the gate reads the repeat, the up projection the line end:
a SwiGLU neuron multiplies the two), the write is </think>'s unembedding row through the final
norm's scale; no gradient descent. Every step is screened by teacher forcing before any generation.
(The first design wrote </think> and <|im_end|> wherever the repeat probe fired: the screen showed
it would end answers with <|im_end|> in mid-sentence; see the rules' amendment.)

    loop_breaker.py fit --device cuda     # the student's greedy answers to the fitting prompts, captured
    loop_breaker.py build                 # CPU: probe, neurons, screen on held-out answers -> breaker.pt
    loop_breaker.py tokens                # CPU: tokens.json for build, if fit ran before it wrote one
    loop_breaker.py test --arm unedited|breaker|breaker-random-write --device cuda
    loop_breaker.py compare --arm breaker # CPU: an arm's saved answers against unedited, if it ran first
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import heal_block as hb
import heal_block_check as hbc
import heal_probe as hp
import loop_edit as le
import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
import slim_opd as so

RULES = mc.ROOT / 'loop_breaker_rules.json'
STOPS = ('</think>', '<|im_end|>')
log = hbc.log


# ---------------------------------------------------------------- the written neurons

def activation(s, u, b, k, clamp, up=None):
    """What the two neurons write, as a multiple of v: about min(max(k (s - b), 0), clamp) / clamp
    times up, from the probe score s = w.x and the constant u = c.x (about 1). up is what the up
    projection reads: the constant, or a second condition (a SwiGLU neuron multiplies its gate by
    its up projection, so a 0-or-1 read there makes the neuron an AND of the two)."""
    z = k * (s - b * u)
    return (F.silu(z) - F.silu(z - clamp * u)) * (u if up is None else up) / clamp


def neuron_weights(w, c, v, b, k, clamp, up=None):
    """The neurons' gate (2, H), up (2, H) and down (H, 2) weights for activation() times v."""
    gate, up = k * (w - b * c), c if up is None else up
    return torch.stack([gate, gate - clamp * c]), torch.stack([up, up]), torch.stack([v / clamp, -v / clamp], 1)


class Breaker(hb.SwiGLUBridge):
    """Hand-written always-on SwiGLU neurons beside a carved MLP, computed in fp32 whatever the
    autocast (their threshold is a difference of two large dot products). record, if a list, takes
    each call's last-position neuron outputs."""
    record = None

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            f = self.features(x.float())
            if self.record is not None:
                self.record.append(f[:, -1].detach().cpu())
            return self.down(f)


def stop_write(q, rms):
    """v for a unit dose: in the span of the written tokens' scaled unembedding rows q (tokens, H),
    raising each of their logits by 1 where the final residual's RMS is rms."""
    coef = torch.linalg.solve(q @ q.T, torch.ones(len(q), dtype=q.dtype))
    return rms * coef @ q


def random_write(v, q, seed=0):
    """A direction of v's norm orthogonal to the stop rows q: the control's write."""
    r = torch.randn(v.shape, generator=torch.Generator().manual_seed(seed), dtype=v.dtype)
    basis = torch.linalg.qr(q.T).Q
    r = r - basis @ (basis.T @ r)
    return r * v.norm() / r.norm()


# ---------------------------------------------------------------- answers, cycles, labels

def cycle(tokens, cfg):
    """(period, start) of the cycle an answer ends in, or None: the smallest period whose last two
    copies agree, and the first index from which a copy agrees with the next one."""
    t = np.asarray(tokens)
    n = len(t)
    for p in range(cfg['min_period'], min(cfg['max_period'], n // 2) + 1):
        if (t[n - p:] == t[n - 2 * p:n - p]).mean() >= cfg['agree']:
            break
    else:
        return None
    eq = t[:n - p] == t[p:]
    share = np.convolve(eq, np.ones(p) / p, 'valid')         # share[i]: agreement over [i, i + p)
    hits = np.nonzero((share >= cfg['agree'] - 1e-9) & eq[:len(share)])[0]
    return (p, int(hits[0])) if len(hits) else None


def kinds(rows, rules):
    """Per answer: ('loop', (period, start)), ('drift', None) for a loop without a cycle, ('done',
    None) for a finished answer, ('cut', None) for one cut at the cap that is not a loop."""
    looping = set(hp.generations(rows, rules['loop'])['looping'])
    found = []
    for r in rows:
        if r['id'] in looping:
            c = cycle(r['tokens'], rules['cycle'])
            found.append(('loop', c) if c else ('drift', None))
        else:
            found.append(('done' if r['finished'] else 'cut', None))
    return found


def split(found):
    """Per answer True for fitting, False for held out: alternating within each kind, in order."""
    count, fitting = {}, []
    for kind, _ in found:
        fitting.append(count.get(kind, 0) % 2 == 0)
        count[kind] = count.get(kind, 0) + 1
    return fitting


def repeating(tokens, window):
    """Per position t in 0..len(tokens): whether the window tokens before t occurred earlier in the answer."""
    seen, inside = set(), [False] * (len(tokens) + 1)
    for t in range(window, len(tokens) + 1):
        span = tuple(tokens[t - window:t])
        inside[t] = span in seen
        seen.add(span)
    return inside


def line_ends(tok):
    """The ids of the tokens whose text holds a line break."""
    return sorted(i for t, i in tok.get_vocab().items() if '\n' in tok.convert_tokens_to_string([t]))


def classes(kind, cyc, length, margin):
    """(positive, negative) boolean masks over an answer's length + 1 captured positions (position
    t predicts generated token t, and its last generated token is t - 1)."""
    t = np.arange(length + 1)
    if kind == 'loop':
        p, s = cyc
        return t - 1 >= s + p + margin, t < s - p
    if kind == 'done':
        return np.zeros(length + 1, bool), np.ones(length + 1, bool)
    return np.zeros(length + 1, bool), np.zeros(length + 1, bool)


def fit_prompts(rules, tok):
    """The fitting prompts: GSM8K train rows that the benchmark corpus does not use, in row order."""
    docs = json.loads((mc.OUT / 'corpus.json').read_text())['docs']
    used = {x for d in docs for x in d.get('items', [])}
    cfg, prompts = rules['fit'], []
    a, b = cfg['rows']
    for n, r in sh.paged('openai/gsm8k', 'main', [[a, b - a]]):
        if f'gsm8k-{n}' in used:
            continue
        answer = float(r['answer'].rpartition('####')[2].strip().replace(',', ''))
        prompts.append({'id': f'gsm8k-train-{n}', 'question': r['question'], 'answer': answer,
                        'ids': tok.encode(so.template(tok, r['question']), add_special_tokens=False)})
        if len(prompts) == cfg['count']:
            break
    return prompts


def answer_rows(gen, prompts, tok, arm):
    rows = []
    for row, p in zip(gen, prompts, strict=True):
        answer = sb.final_number(tok.decode(row['tokens']))
        rows.append({'id': p['id'], 'arm': arm, 'tokens': row['tokens'], 'length': len(row['tokens']),
                     'finished': row['stop'] is not None, 'answer': answer,
                     'correct': answer is not None and abs(answer - p['answer']) < 1e-6,
                     'nonfinite_at': row.get('nonfinite_at')})
    return rows


# ---------------------------------------------------------------- the model (GPU stages)

def build_model(rules):
    folder = mc.ROOT / rules['model']['folder']
    return hbc.build(argparse.Namespace(pieces=[folder / p / 'block.pt' for p in rules['model']['pieces']]))


def vocab_of(model, tok):
    class Vocab:      # the full vocabulary: nothing removed
        stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]
        removed = torch.zeros(model.config.vocab_size, device=mc.DEVICE)
    return Vocab


def greedy(model, slim, block, vocab, prompts, cfg, batch, seed):
    """The student's greedy answers (slim_opd.generate's rows), the pieces swapped in."""
    gen = []
    block.swap(True)
    try:
        with torch.autocast(mc.DEVICE, dtype=torch.bfloat16):
            for b in range(0, len(prompts), batch):
                gen += so.generate(model, slim, vocab, [p['ids'] for p in prompts[b:b + batch]], cfg, True, True,
                                   seed=seed + b)
                log(f'{len(gen)}/{len(prompts)} answered')
    finally:
        block.swap(False)
    return gen


@torch.no_grad()
def student_pass(model, ids):
    """The student's final hidden states over one sequence (inside hbc.Student)."""
    return model.model(input_ids=torch.tensor([ids], device=mc.DEVICE), use_cache=False).last_hidden_state[0]


@torch.no_grad()
def capture(model, block, tok, rows, prompts, rules, out):
    """Teacher-forced pass of the student over each of its answers. Per position from the prompt's
    last on (position t predicts generated token t): the last layer's MLP input, the final residual,
    the top five tokens and the stop tokens' logits. head.pt: the unembedding rows those need."""
    folder = out / 'capture'
    folder.mkdir(exist_ok=True)
    piece, norm = dict(block.prefix)[rules['layer']], model.model.norm
    got = {}
    hooks = [piece.mlp.register_forward_pre_hook(lambda m, a: got.__setitem__('x', a[0])),
             norm.register_forward_pre_hook(lambda m, a: got.__setitem__('r', a[0]))]
    stops = [tok.convert_tokens_to_ids(t) for t in STOPS]
    seen, agree, total = set(stops), 0, 0
    try:
        with hbc.Student(block):
            for k, (row, p) in enumerate(zip(rows, prompts, strict=True)):
                h = student_pass(model, p['ids'] + row['tokens'])
                a = len(p['ids']) - 1
                x, r = got['x'][0, a:], got['r'][0, a:]      # taken before the check below calls the norm again
                if k == 0 and not torch.allclose(norm(r).float(), h[a:].float(), atol=1e-2, rtol=1e-2):
                    raise RuntimeError('model.model.norm is not the final norm')
                ids, values, stopped = [], [], []
                for part in h[a:].split(512):                # the whole vocabulary, a few positions at a time
                    logits = model.lm_head(part).float()
                    top = logits.topk(5, -1)
                    ids.append(top.indices.cpu())
                    values.append(top.values.cpu())
                    stopped.append(logits[:, stops].cpu())
                ids = torch.cat(ids)
                seen.update(ids.unique().tolist())
                agree += int((ids[:-1, 0] == torch.tensor(row['tokens'])).sum())
                total += len(row['tokens'])
                np.savez(folder / f'{k}.npz', x=x.float().cpu().numpy().astype(np.float16),
                         r=r.float().cpu().numpy(), top=ids.numpy().astype(np.int32),
                         top_logits=torch.cat(values).numpy(), stop_logits=torch.cat(stopped).numpy())
    finally:
        for hook in hooks:
            hook.remove()
    kept = sorted(seen)
    torch.save({'ids': torch.tensor(kept), 'rows': model.lm_head.weight[kept].float().cpu(),
                'scale': (1 + norm.weight.float()).cpu(), 'eps': norm.eps, 'stops': stops,
                'vocab': model.config.vocab_size}, out / 'head.pt')
    log(f'captured {len(rows)} answers; the teacher-forced argmax is the generated token at {agree / total:.4f} of positions')


def stage_fit(args, rules, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    prompts = fit_prompts(rules, tok)
    model, slim, block, record = build_model(rules)
    cfg, path = rules['fit'], out / 'gen_fit.jsonl'
    if path.exists():
        rows = [json.loads(line) for line in path.open()]
    else:
        t0 = time.time()
        gen = greedy(model, slim, block, vocab_of(model, tok), prompts, cfg['generate'], cfg['batch'], cfg['generate']['seed'])
        rows = answer_rows(gen, prompts, tok, 'fit')
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        log(f'generated in {time.time() - t0:.0f} s')
    s = hp.generations(rows, rules['loop'])
    found = kinds(rows, rules)
    log(f"fitting prompts: {s['correct']}/{len(rows)} right, {s['finished']} finished, {s['loops']} loops "
        f"({sum(k == 'loop' for k, _ in found)} with a cycle)")
    capture(model, block, tok, rows, prompts, rules, out)
    stage_tokens(args, rules, out, tok, prompts)
    (out / 'fit.json').write_text(json.dumps(dict(record, summary={k: v for k, v in s.items() if k != 'looping'}), indent=1))


def stage_tokens(args, rules, out, tok=None, prompts=None):
    """tokens.json, what the build needs from the tokenizer: the line-end tokens, </think>, and
    each fitting prompt's last token (the current token at an answer's first captured position)."""
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
        prompts = fit_prompts(rules, tok)
    (out / 'tokens.json').write_text(json.dumps({'line_ends': line_ends(tok), 'think': tok.convert_tokens_to_ids(STOPS[0]),
                                                 'written': [tok.convert_tokens_to_ids(t) for t in rules['neurons'].get('tokens', STOPS)],
                                                 'prompt_last': [p['ids'][-1] for p in prompts]}))


# ---------------------------------------------------------------- the probe and the screen (CPU)

def fit_probe(batches, ridge):
    """w, c and the positives' residual RMS from (x, positive mask, negative mask, rms) batches:
    w = (S_w + ridge I)^-1 (mean+ - mean-), unit within-class deviation; c: mean c.x = 1, least variance."""
    gram, total, n, rms = {}, {}, {}, []
    for x, pos, neg, r in batches:
        for name, mask in (('pos', pos), ('neg', neg)):
            part = x[mask]
            if len(part):
                gram[name] = gram.get(name, 0) + (part.T @ part).double()
                total[name] = total.get(name, 0) + part.double().sum(0)
                n[name] = n.get(name, 0) + len(part)
        rms.append(r[pos])
    mean = {k: total[k] / n[k] for k in n}
    h = len(mean['pos'])
    within = sum(gram[k] - n[k] * torch.outer(mean[k], mean[k]) for k in n) / (sum(n.values()) - 2)
    eye = torch.eye(h, dtype=torch.float64)
    w = torch.linalg.solve(within + ridge * within.trace() / h * eye, mean['pos'] - mean['neg'])
    w = w / (w @ within @ w).sqrt()
    count = sum(n.values())
    m = sum(total.values()) / count
    cov = sum(gram.values()) / count - torch.outer(m, m)
    c = torch.linalg.solve(cov + ridge * cov.trace() / h * eye, m)
    c = c / (m @ c)
    return w.float(), c.float(), float(torch.cat(rms).median()), n


def fit_readout(batches, ridge):
    """n with n.x closest to y in the least-squares sense (ridge), from (x, y) batches."""
    gram, xty = 0, 0
    for x, y in batches:
        gram = gram + (x.T @ x).double()
        xty = xty + (x.T @ y).double()
    h = len(xty)
    return torch.linalg.solve(gram + ridge * gram.trace() / h * torch.eye(h, dtype=torch.float64), xty).float()


def row_terms(z, head, index, w, c, v1, n=None):
    """Per position: the probe score, the constant (and the up projection's read, if it has its
    own), and the scalars that give the argmax over the unedited top five and the stop tokens for
    any activation and dose (logits x residual RMS)."""
    x, r = torch.from_numpy(z['x'].astype(np.float32)), torch.from_numpy(z['r'])
    top = torch.from_numpy(z['top'].astype(np.int64))
    stops = torch.tensor(head['stops'])
    q = head['rows'][index[top]] * head['scale']                       # (T, 5, H)
    qs = head['rows'][index[stops]] * head['scale']                    # (2, H)
    return {'s': x @ w, 'u': x @ c, 'up': None if n is None else x @ n, 'rms': (r * r).mean(-1).sqrt(),
            'base': torch.cat([torch.einsum('tkh,th->tk', q, r), r @ qs.T], 1),
            'gain': torch.cat([q @ v1, (qs @ v1).expand(len(r), 2)], 1),
            'tokens': torch.cat([top, stops.expand(len(r), 2)], 1)}


def first_change(terms, act, dose):
    """(position, new token) where the edited argmax first differs from the unedited one; (None, None)."""
    old = terms['tokens'].gather(1, terms['base'].argmax(1, keepdim=True))[:, 0]
    new = terms['tokens'].gather(1, (terms['base'] + (act * dose)[:, None] * terms['gain']).argmax(1, keepdim=True))[:, 0]
    hits = (old != new).nonzero()[:, 0]
    return (int(hits[0]), int(new[hits[0]])) if len(hits) else (None, None)


def outcome(kind, cyc, row, terms, act, dose, tokens):
    """Where the edited answer would leave the unedited text, as the rules' screen names it.
    tokens: the tokenizer's (tokens.json); terms['inside']: whether each position is in a repeat."""
    t, token = first_change(terms, act, dose)
    if t is None:
        return 'none', t, token
    inside = bool(terms['inside'][t])
    if kind == 'loop':
        p, s = cyc
        if not inside and t < s - p:
            return 'collateral', t, token
        if token not in tokens['written']:
            return 'other', t, token
        return ('broken' if t < s + 3 * p else 'late'), t, token
    think = tokens['think']
    own = row['tokens'].index(think) if think in row['tokens'] else None
    if token == think and own is not None and 0 <= own - t <= 64:
        return 'benign', t, token
    return ('collateral_repeating' if inside else 'collateral'), t, token


def auroc(pos, neg):
    order = torch.cat([pos, neg]).argsort()
    ranks = torch.empty(len(order), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def collateral(counts):
    return counts.get('collateral', 0) + counts.get('collateral_repeating', 0)


def tally(rows, found, chosen, terms, b, dose, cfg, tokens):
    """Screen outcomes of one setting over the chosen answers: counts by kind, and each answer's.
    Answers cut at the cap and drift loops are named as finished answers are, for the record only."""
    counts, each = {}, []
    for k in chosen:
        kind, cyc = found[k]
        act = activation(terms[k]['s'], terms[k]['u'], b, cfg['k'], cfg['clamp'], terms[k]['up'])
        name, t, token = outcome(kind, cyc, rows[k], terms[k], act, dose, tokens)
        counts.setdefault(kind, {}).setdefault(name, 0)
        counts[kind][name] += 1
        each.append({'row': k, 'id': rows[k]['id'], 'kind': kind, 'cycle': cyc, 'outcome': name, 'at': t, 'token': token})
    return counts, each


def stage_build(args, rules, out):
    torch.set_num_threads(args.threads)
    rows = [json.loads(line) for line in (out / 'gen_fit.jsonl').open()]
    head = torch.load(out / 'head.pt', weights_only=True)
    tokens = json.loads((out / 'tokens.json').read_text())
    index = torch.full((head['vocab'],), -1, dtype=torch.long)
    index[head['ids']] = torch.arange(len(head['ids']))
    stops, margin, cfg = head['stops'], rules['labels']['margin'], rules['neurons']
    found = kinds(rows, rules)
    fitting = split(found)
    load = lambda k: np.load(out / 'capture' / f'{k}.npz')
    answers = {kind: sum(k == kind for k, _ in found) for kind in ('loop', 'drift', 'done', 'cut')}
    log(f'answers: {answers}')
    used = [k for k in range(len(rows)) if fitting[k] and found[k][0] in ('loop', 'done')]

    def batches():
        for k in used:
            z = load(k)
            pos, neg = classes(*found[k], rows[k]['length'], margin)
            r = torch.from_numpy(z['r'])
            yield torch.from_numpy(z['x'].astype(np.float32)), torch.from_numpy(pos), torch.from_numpy(neg), (r * r).mean(-1).sqrt()

    w, c, rms, n = fit_probe(batches(), rules['probe']['ridge'])
    q = head['rows'][index[torch.tensor(stops)]] * head['scale']
    written = [stops.index(i) for i in tokens['written']]
    v1 = stop_write(q[written], rms)
    log(f"probe from {n['pos']} repeating and {n['neg']} ordinary positions; residual RMS {rms:.2f}, |v| per unit dose {float(v1.norm()):.2f}")
    report = {'answers': answers, 'positions': n, 'rms': rms}
    # The up projection's own read: 1 where the current token ends a line, 0 elsewhere.
    line_end = torch.zeros(head['vocab'], dtype=torch.bool)
    line_end[tokens['line_ends']] = True
    current = lambda k: line_end[torch.tensor([tokens['prompt_last'][k]] + rows[k]['tokens'])]
    up = None
    if cfg.get('up') == 'line_end':
        up = fit_readout(((torch.from_numpy(load(k)['x'].astype(np.float32)), current(k).float()) for k in used), rules['probe']['ridge'])
    terms = {k: row_terms(load(k), head, index, w, c, v1, up) for k in range(len(rows))}
    for k, row in enumerate(rows):
        terms[k]['inside'] = torch.tensor(repeating(row['tokens'], rules['screen']['window']))
    if up is not None:
        held = [k for k in range(len(rows)) if not fitting[k] and found[k][0] in ('loop', 'done')]
        at_end = torch.cat([terms[k]['up'][current(k)] for k in held])
        elsewhere = torch.cat([terms[k]['up'][~current(k)] for k in held])
        report['line_end_read'] = {'at_line_ends': np.quantile(at_end.numpy(), [.001, .01, .5]).tolist(),
                                   'elsewhere': np.quantile(elsewhere.numpy(), [.5, .99, .999, 1]).tolist(),
                                   'auroc': auroc(at_end, elsewhere)}
        log(f"line-end read, held out: at line ends {report['line_end_read']['at_line_ends']} (0.1%, 1%, median), "
            f"elsewhere {report['line_end_read']['elsewhere']} (median, 99%, 99.9%, max)")

    def scores(want_fitting):
        pos, neg, const = [], [], []
        for k, row in enumerate(rows):
            if fitting[k] == want_fitting and found[k][0] in ('loop', 'done'):
                a, m = classes(*found[k], row['length'], margin)
                pos.append(terms[k]['s'][torch.from_numpy(a)])
                neg.append(terms[k]['s'][torch.from_numpy(m)])
                const.append(terms[k]['u'])
        return torch.cat(pos), torch.cat(neg), torch.cat(const)

    pos_fit, neg_fit, const = scores(True)
    pos_held, neg_held, _ = scores(False)
    report.update(constant={'mean': float(const.mean()), 'std': float(const.std())},
                  auroc={'fitting': auroc(pos_fit, neg_fit), 'held_out': auroc(pos_held, neg_held)},
                  score={'positive_fit': float(pos_fit.mean()), 'negative_fit': float(neg_fit.mean()),
                         'positive_held': float(pos_held.mean()), 'negative_held': float(neg_held.mean())})
    log(f"AUROC {report['auroc']['fitting']:.4f} fitting, {report['auroc']['held_out']:.4f} held out; "
        f"c.x {report['constant']['mean']:.3f} +- {report['constant']['std']:.3f}")
    # How far the written tokens are from winning inside the second copy, in logits: the least gap
    # per loop, anywhere and at line ends.
    gaps, at_ends = [], []
    for k, (kind, cyc) in enumerate(found):
        if kind == 'loop':
            period, s = cyc
            logit = terms[k]['base'] / terms[k]['rms'][:, None]
            gap = logit[:, :5].max(1).values - logit[:, [5 + j for j in written]].max(1).values
            ends = current(k)[s + period:s + 2 * period]
            gaps.append(float(gap[s + period:s + 2 * period].min()))
            if ends.any():
                at_ends.append(float(gap[s + period:s + 2 * period][ends].min()))
    report['least_gap_in_second_copy'] = {'anywhere': np.quantile(gaps, [.25, .5, .75, 1]).tolist(),
                                          'at_line_ends': np.quantile(at_ends, [.25, .5, .75, 1]).tolist()}
    log(f"least gap to a written token in the second copy (quartiles, max): anywhere {np.round(report['least_gap_in_second_copy']['anywhere'], 2).tolist()}, "
        f"at line ends {np.round(report['least_gap_in_second_copy']['at_line_ends'], 2).tolist()}")

    grid = []
    fit_rows = [k for k in range(len(rows)) if fitting[k]]
    held_rows = [k for k in range(len(rows)) if not fitting[k]]
    kind_count = lambda chosen, kind: sum(found[k][0] == kind for k in chosen)
    limit = {kind: max(1, int(0.02 * kind_count(fit_rows, kind))) for kind in ('loop', 'done')}
    for rate in rules['grid']['false_positive_rates']:
        b = float(np.quantile(neg_fit.numpy(), 1 - rate))
        for dose in rules['grid']['doses']:
            counts, _ = tally(rows, found, fit_rows, terms, b, dose, cfg, tokens)
            loop, done = counts.get('loop', {}), counts.get('done', {})
            grid.append({'rate': rate, 'b': b, 'dose': dose, 'loop': loop, 'done': done,
                         'allowed': collateral(loop) <= limit['loop'] and collateral(done) <= limit['done']})
    allowed = [g for g in grid if g['allowed']]
    report['grid'] = grid
    if not allowed:
        report['selected'] = None
        (out / 'build.json').write_text(json.dumps(report, indent=1))
        log('no setting keeps the collateral within the limit: no breaker')
        return
    best = max(allowed, key=lambda g: (g['loop'].get('broken', 0), -g['dose'], -g['rate']))
    counts, each = tally(rows, found, held_rows, terms, best['b'], best['dose'], cfg, tokens)
    loops, done = kind_count(held_rows, 'loop'), kind_count(held_rows, 'done')
    broken, spoiled = counts.get('loop', {}).get('broken', 0), collateral(counts.get('done', {}))
    gate = broken >= 0.6 * loops and spoiled <= 0.05 * done
    report.update(selected=best, held_out={'loops': loops, 'finished': done, 'counts': counts, 'answers': each}, gate=bool(gate))
    log(f"selected: false-positive rate {best['rate']}, dose {best['dose']} (fitting: {best['loop']}, {best['done']})")
    log(f"held out: {broken}/{loops} loops broken ({counts.get('loop', {})}); finished answers {counts.get('done', {})}; "
        f"cut {counts.get('cut', {})}, drift {counts.get('drift', {})}; gate {'holds' if gate else 'fails'}")
    v = best['dose'] * v1
    gate_w, up_w, down_w = neuron_weights(w, c, v, best['b'], cfg['k'], cfg['clamp'], up)
    torch.save({'gate': gate_w, 'up': up_w, 'down': down_w, 'q': q, 'v': v, 'w': w, 'c': c, 'b': best['b'],
                'dose': best['dose'], 'k': cfg['k'], 'clamp': cfg['clamp'], 'rms': rms}, out / 'breaker.pt')
    (out / 'build.json').write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------- generation with the neurons in

def install(block, rules, saved, arm):
    piece = dict(block.prefix)[rules['layer']]
    neurons = Breaker(saved['gate'].shape[1], saved['gate'].shape[0]).to(mc.DEVICE)
    down = saved['down']
    if arm == 'breaker-random-write':
        r = random_write(saved['v'], saved['q'])
        down = torch.stack([r / saved['clamp'], -r / saved['clamp']], 1)
    with torch.no_grad():
        neurons.gate.weight.copy_(saved['gate'])
        neurons.up.weight.copy_(saved['up'])
        neurons.down.weight.copy_(down)
        neurons.down.bias.zero_()
    piece.mlp.bridge = neurons.requires_grad_(False)
    return neurons


@torch.no_grad()
def verify(model, block, neurons, tok, rules, out, count=6):
    """The neurons in the model against the screen's forecast: on held-out answers, where the
    model's argmax (whole vocabulary) first leaves the answer's own teacher-forced argmax."""
    report = json.loads((out / 'build.json').read_text())
    rows = [json.loads(line) for line in (out / 'gen_fit.jsonl').open()]
    prompts = fit_prompts(rules, tok)
    piece = dict(block.prefix)[rules['layer']]
    checked = []
    for a in [a for a in report['held_out']['answers'] if a['kind'] == 'loop'][:count]:
        k = a['row']
        ids = prompts[k]['ids'] + rows[k]['tokens']
        start = len(prompts[k]['ids']) - 1
        with hbc.Student(block):
            new = model.lm_head(student_pass(model, ids)[start:]).argmax(-1)
            del piece.mlp.bridge
            try:
                old = model.lm_head(student_pass(model, ids)[start:]).argmax(-1)
            finally:
                piece.mlp.bridge = neurons
        hits = (old != new).nonzero()[:, 0]
        at = int(hits[0]) if len(hits) else None
        checked.append({'id': a['id'], 'forecast': a['at'], 'model': at, 'forecast_token': a['token'],
                        'model_token': int(new[at]) if at is not None else None})
        log(f"{a['id']}: forecast leaves at {a['at']} ({a['outcome']}), the model at {at}")
    return checked


def against(base, rows, acts, loop):
    """An arm's answers paired by prompt with the unedited model's: loops fixed and new, right gained
    and lost (one-sided exact sign tests), and for each answer where it left the unedited text and
    whether the neurons fired (activation above 0.5) at that token or within the 8 before it."""
    bs, s = hp.generations(base, loop), hp.generations(rows, loop)
    was, now = set(bs['looping']), set(s['looping'])
    right0, right1 = {r['id'] for r in base if r['correct']}, {r['id'] for r in rows if r['correct']}
    fixed, new = was - now, now - was
    pair = {'unedited': {k: v for k, v in bs.items() if k != 'looping'}, 'loops_fixed': len(fixed), 'loops_new': len(new),
            'p_fewer_loops': le.sign_p(len(fixed), len(fixed) + len(new)),
            'right_gained': len(right1 - right0), 'right_lost': len(right0 - right1),
            'p_more_right': le.sign_p(len(right1 - right0), len(right1 - right0) + len(right0 - right1))}
    if acts is not None:
        fired = []
        for j, (b, r) in enumerate(zip(base, rows, strict=True)):
            both = min(len(b['tokens']), len(r['tokens']))
            left = next((t for t in range(both) if b['tokens'][t] != r['tokens'][t]), both if len(b['tokens']) != len(r['tokens']) else None)
            on = left is not None and left < len(acts) and bool((acts[max(0, left - 8):left + 1, j] > 0.5).any())
            fired.append({'id': r['id'], 'left_at': left, 'fired': on, 'was_loop': r['id'] in was, 'is_loop': r['id'] in now})
        pair['answers'] = fired
        pair['fixed_after_firing'] = sum(f['fired'] for f in fired if f['id'] in fixed)
        pair['changed_without_firing'] = sum(f['left_at'] is not None and not f['fired'] for f in fired)
    log(f"against unedited ({bs['correct']} right, {bs['finished']} finished, {bs['loops']} loops): loops fixed {len(fixed)}, "
        f"new {len(new)} (p {pair['p_fewer_loops']:.3g}); right +{pair['right_gained']} -{pair['right_lost']}"
        + (f"; {pair['fixed_after_firing']} of the fixed left their text after the neurons fired" if acts is not None else ''))
    return pair


def stage_test(args, rules, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    check = json.loads(hbc.GSM_RULES.read_text())
    cfg = dict(check['eval'], compact=None)
    prompts = json.loads((mc.ROOT / cfg['prompts']).read_text())['prompts'][:cfg['count']]
    for p in prompts:
        p['ids'] = tok.encode(so.template(tok, p['question'], cfg.get('enable_thinking')), add_special_tokens=False)
    model, slim, block, record = build_model(rules)
    record['arm'] = args.arm
    neurons = None
    if args.arm != 'unedited':
        saved = torch.load(out / 'breaker.pt', weights_only=True)
        neurons = install(block, rules, saved, args.arm)
        if args.arm == 'breaker':
            record['verify'] = verify(model, block, neurons, tok, rules, out)
        record['kl_arm'] = hb.end_to_end(block, block.docs)
        log(f"end-to-end KL with the neurons {record['kl_arm']['kl']:.4f} (without {record['kl']['kl']:.4f})")
        neurons.record = []
    t0 = time.time()
    gen = greedy(model, slim, block, vocab_of(model, tok), prompts, cfg, args.batch or len(prompts), cfg['seed'])
    rows = answer_rows(gen, prompts, tok, args.arm)
    (out / f'gen_{args.arm}_greedy.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    s = hp.generations(rows, check['loop'])
    record.update(summary={k: v for k, v in s.items() if k != 'looping'}, looping=s['looping'], seconds=time.time() - t0)
    log(f"{args.arm}: {s['correct']}/{len(rows)} right, {s['finished']} finished, {s['loops']} loops")
    acts = None
    if neurons is not None and len(neurons.record) and all(len(f) == len(prompts) for f in neurons.record):
        f = torch.stack(neurons.record).float()                       # (steps, prompts, 2)
        acts = ((f[..., 0] - f[..., 1]) / saved['clamp']).numpy()
        np.save(out / f'acts_{args.arm}.npy', acts.astype(np.float16))
    base_path = out / 'gen_unedited_greedy.jsonl'
    if args.arm != 'unedited' and base_path.exists():
        record['against_unedited'] = against([json.loads(line) for line in base_path.open()], rows, acts, check['loop'])
    (out / f'test_{args.arm}.json').write_text(json.dumps(record, indent=1))


def stage_compare(args, rules, out):
    """An arm's saved answers against the unedited model's, for an arm that ran before it."""
    check = json.loads(hbc.GSM_RULES.read_text())
    read = lambda arm: [json.loads(line) for line in (out / f'gen_{arm}_greedy.jsonl').open()]
    path = out / f'acts_{args.arm}.npy'
    record = json.loads((out / f'test_{args.arm}.json').read_text())
    record['against_unedited'] = against(read('unedited'), read(args.arm),
                                         np.load(path).astype(np.float32) if path.exists() else None, check['loop'])
    (out / f'test_{args.arm}.json').write_text(json.dumps(record, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['fit', 'tokens', 'build', 'test', 'compare'])
    ap.add_argument('--arm', default='breaker', choices=['unedited', 'breaker', 'breaker-random-write'])
    ap.add_argument('--batch', type=int, default=None, help='test: prompts generated together (default all)')
    ap.add_argument('--threads', type=int, default=8, help='build: CPU threads')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    rules = json.loads(RULES.read_text())
    mc.DEVICE = 'cpu' if args.stage in ('build', 'tokens', 'compare') else args.device
    mc.configure(sb.CARVE_RULES)
    out = mc.ROOT / rules['out']
    out.mkdir(parents=True, exist_ok=True)
    {'fit': stage_fit, 'tokens': stage_tokens, 'build': stage_build, 'test': stage_test, 'compare': stage_compare}[args.stage](args, rules, out)


if __name__ == '__main__':
    main()
