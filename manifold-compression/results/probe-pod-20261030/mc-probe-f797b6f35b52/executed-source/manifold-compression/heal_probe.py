"""Two checks on the healed student of slim_heal.py, before deciding how to heal further.

1. Where the remaining drift sits. The healed student is evaluated as at its healing milestones
   with its learned routers (which must reproduce the last milestone), with the perfect router
   (each token's K experts with the most contribution, from the student's own activations, as
   moe_carve's cluster_token arm) and with every MLP neuron running.
2. Why its long generations loop. The DeltaNet state check is split into its two parts: fp32 token
   by token against the chunked kernel, and int16 against fp32 token by token. The 16 GSM8K
   generations are repeated with fp32 state, once with the learned and once with the perfect router.

The healing pod kept its low-rank factors, so they are rebuilt from the calibration documents. The
adapter is the healing run's adapter.pt: its bf16 LoRA factors are what the student's bf16 forward
used, and its routers are fp32.

Settings and predictions are in heal_probe_rules.json; outputs go to the probe/ folder beside the
carve's. Stages:

  evaluate  learned, perfect and all-neuron routing on the eval documents.
  state     the state check in parts.
  generate  --arm fp32 | perfect: the GSM8K generations.
  report    (local) one summary, with the healing run's and the dense model's generations scored
            the same way.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
from h0_mlp_probe import deadline, group_mean, log, paired_interval, save, sha, token_kl

ROOT = mc.ROOT
RULES = ROOT / 'heal_probe_rules.json'
CODE = ['heal_probe.py', 'slim_heal.py', 'slim_batch.py', 'moe_carve.py', 'h0_mlp_probe.py', 'test_heal_probe.py']


def out(name):
    return mc.OUT / 'probe' / name


def fresh(name):
    if out(name).exists():
        raise ValueError(f'{name} exists; refusing to overwrite')
    out(name).parent.mkdir(parents=True, exist_ok=True)
    return out(name)


def provenance():
    return {'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'rules_sha256': sha(RULES),
            'heal_rules_sha256': sha(sh.RULES), 'carve_rules_sha256': sha(sb.CARVE_RULES),
            'slim_rules_sha256': sha(sb.RULES), 'adapter_sha256': sha(sh.out('adapter.pt'))}


def setup():
    heal_rules, docs = sh.setup()
    return json.loads(RULES.read_text()), heal_rules, docs


def load_healed(heal_rules, docs):
    """The healed student: the low-rank factors (rebuilt once, then kept beside the run folder) and
    the healing run's adapter. Also returns whether the factors match the healing run's bit for bit."""
    healed = json.loads(sh.out('heal.json').read_text())
    if sha(sh.out('adapter.pt')) != healed['adapter_sha256']:
        raise ValueError('adapter.pt is not the healing run\'s adapter')
    model = mc.load_model(heal_rules['threads'])
    if not sh.factors_path().exists():
        factors = sh.make_factors(model, docs)
        torch.save({i: {n: (u.cpu(), v.cpu()) for n, (u, v) in f.items()} for i, f in factors.items()},
                   sh.factors_path())
    factors = torch.load(sh.factors_path(), weights_only=True, map_location=mc.DEVICE)
    slim = sh.build(model, heal_rules, factors)
    slim.load(torch.load(sh.out('adapter.pt'), weights_only=True, map_location=mc.DEVICE))
    model.eval()
    return model, slim, sh.Vocab(model), sha(sh.factors_path()) == healed['factors_sha256']


def route(slim, mode):
    """Set how the student picks its experts: 'learned' (its routers), 'perfect' (each token's K
    experts with the most contribution, from its own activations) or 'all' (every neuron). Recall
    is recorded as 1 for the last two, which run every perfect expert."""
    vars(slim).pop('select', None)
    if mode == 'learned':
        return
    if mode not in ('perfect', 'all'):
        raise ValueError(mode)

    def select(i, a, x):
        b, t, n = a.shape
        if mode == 'all':
            keep = torch.ones_like(a, dtype=torch.bool)
        else:
            mass = mc.expert_mass(slim.s, i, (a.float().abs() * slim.wnorm[i]).flatten(0, 1))
            keep = mc.expand(slim.s, i, mc.topmask(mass, mc.at(slim.s, 'K', i))).view(b, t, n)
            if t > 1:
                keep[:, 0] = True
        if slim.recall is not None and t > 1:
            slim.recall.append(1.0)
        return keep

    slim.select = select


def repeated(tokens, window, last):
    """Share of the window-token spans in the last tokens that repeat an earlier span."""
    tail = tokens[-last:]
    spans = [tuple(tail[i:i + window]) for i in range(len(tail) - window + 1)]
    return 1 - len(set(spans)) / len(spans) if spans else 0.0


def stage_evaluate(args):
    rules, heal_rules, docs = setup()
    cfg = rules['evaluate']
    if cfg['arms'][0] != 'learned':
        raise ValueError('The learned arm runs first, to check the rebuilt student')
    last = json.loads(sh.out('heal.json').read_text())['milestones'][-1]
    t0 = time.time()
    model, slim, vocab, same = load_healed(heal_rules, docs)
    log(f'Student loaded; factors {"identical to" if same else "differ from"} the healing run\'s')
    evals = sb.docs_of(docs, 'eval')
    rows, arms = {}, {}
    with deadline(cfg['max_hours']):
        for mode in cfg['arms']:
            route(slim, mode)
            rows[mode] = sh.evaluate(model, slim, vocab, evals)
            with fresh(f'eval_{mode}.jsonl').open('w') as stream:
                stream.write(''.join(json.dumps(r) + '\n' for r in rows[mode]))
            arms[mode] = sh.summarize(rows[mode], heal_rules['data']['held_out'])
            log(f'{mode}: KL {arms[mode]["kl"]:.3f} (in-mix {arms[mode]["in_mix"]["kl"]:.3f}, held-out '
                f'{arms[mode]["held_out"]["kl"]:.3f}), 64K bits per byte {arms[mode]["bpb_student"]:.4f}, '
                f'agreement {arms[mode]["agree"]:.3f}')
            if mode == 'learned':
                check = {'kl': arms[mode]['kl'] - last['kl'], 'bpb': arms[mode]['bpb_student'] / last['bpb_student'] - 1}
                if abs(check['kl']) > cfg['check']['kl'] or abs(check['bpb']) > cfg['check']['bpb']:
                    raise RuntimeError(f'The rebuilt student does not reproduce the healed one: {check}')
    route(slim, 'learned')
    report = json.loads(mc.RULES.read_text())['report']
    text = {m: [r for r in rs if r['domain'] != 'copy'] for m, rs in rows.items()}
    differences = {}
    for new, base in zip(cfg['arms'][1:], cfg['arms']):
        diff, low, high = paired_interval([r['kl']['student'] for r in text[new]], [r['kl']['student'] for r in text[base]],
                                          1.0, report['draws'], report['seed'], [r['domain'] for r in text[new]])
        differences[f'{new}-{base}'] = {'difference': diff, 'low': low, 'high': high}
    save(dict(provenance(), factors_identical=same, reproduces_healed=check, arms=arms, differences=differences,
              seconds=time.time() - t0), fresh('evaluate.json'))


@torch.no_grad()
def stage_state(args):
    rules, heal_rules, docs = setup()
    cfg = rules['state']
    t0 = time.time()
    model, slim, _, _ = load_healed(heal_rules, docs)
    rows = []
    kl = lambda p, q: float(token_kl(p, q)[1:].mean())
    with deadline(cfg['max_hours']):
        for doc in sb.docs_of(docs, 'eval', cfg['docs_per_domain']):
            ids = doc['input_ids']
            slim.on = False
            dense = model.lm_head(sb.hidden(model, [ids])[0]).float()
            slim.on = True
            logits = {'chunked': model.lm_head(sb.hidden(model, [ids])[0]).float()}
            for variant in cfg['variants']:
                with sb.StoredState(model, [variant], cfg['seed']):
                    logits[variant] = model.lm_head(sb.hidden(model, [ids])[0]).float()
            slim.on = False
            rows.append({'id': doc['id'], 'domain': doc['domain'],
                         'from_dense': {k: kl(dense, v) for k, v in logits.items()},
                         'between': {f'{a}|{b}': kl(logits[a], logits[b]) for a, b in cfg['pairs']}})
            del dense, logits
    fresh('state.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    groups = [r['domain'] for r in rows]
    mean = lambda part: {k: group_mean([r[part][k] for r in rows], groups) for k in rows[0][part]}
    summary = {'docs': len(rows), 'from_dense': mean('from_dense'), 'between': mean('between')}
    log('from dense: ' + ', '.join(f'{k} {v:.4f}' for k, v in summary['from_dense'].items()) +
        '; between: ' + ', '.join(f'{k} {v:.2e}' for k, v in summary['between'].items()))
    save(dict(provenance(), **summary, seconds=time.time() - t0), fresh('state.json'))


def stage_generate(args):
    from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
    rules, heal_rules, docs = setup()
    arm, gen = rules['generate']['arms'][args.arm], heal_rules['generate']
    prompts = json.loads((ROOT / gen['prompts']).read_text())['prompts'][:gen['count']]
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    model, slim, vocab, _ = load_healed(heal_rules, docs)
    route(slim, arm['router'])
    stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]
    removed = torch.full((model.config.vocab_size,), -torch.inf, device=mc.DEVICE)
    removed[vocab.kept] = 0

    class KeptRows(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return scores + removed

    path = out(f'generate_{args.arm}.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['id'] for line in path.open()} if path.exists() else set()
    slim.on = True
    with deadline(rules['generate']['max_hours']):
        for p in prompts:
            if p['id'] in done:
                continue
            text = tok.apply_chat_template([{'role': 'user', 'content': p['question']}], tokenize=False,
                                           add_generation_prompt=True)
            ids = vocab.short.encode(text)
            t0 = time.time()
            with torch.no_grad(), sb.StoredState(model, [arm['state']], gen['seed']):
                x = sb.tensor([ids])
                result = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=gen['max_new_tokens'],
                                        do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=stop,
                                        pad_token_id=stop[1], logits_processor=LogitsProcessorList([KeptRows()]))
            tokens = result[0, len(ids):].tolist()
            end = next((i for i, t in enumerate(tokens) if t in stop), None)
            tokens = tokens[:end] if end is not None else tokens
            answer = sb.final_number(tok.decode(tokens))
            row = {'id': p['id'], 'router': arm['router'], 'state': arm['state'], 'seconds': time.time() - t0,
                   'tokens': tokens, 'length': len(tokens), 'finished': end is not None, 'answer': answer,
                   'correct': answer is not None and abs(answer - p['answer']) < 1e-6}
            with path.open('a') as f:
                f.write(json.dumps(row) + '\n')
            log(f'{p["id"]}: {row["length"]} tokens{"" if row["finished"] else " (cap)"}, '
                f'{"right" if row["correct"] else "wrong"}, {row["seconds"]:.0f} s')
    slim.on = False


def generations(rows, loop):
    """Counts for one set of generations; a loop is a run that hit the cap repeating itself."""
    stuck = [r['id'] for r in rows
             if not r['finished'] and repeated(r['tokens'], loop['window'], loop['last']) >= loop['threshold']]
    return {'prompts': len(rows), 'correct': sum(r['correct'] for r in rows), 'finished': sum(r['finished'] for r in rows),
            'mean_length': float(np.mean([r['length'] for r in rows])), 'loops': len(stuck), 'looping': stuck}


def stage_report(args):
    rules, heal_rules, _ = setup()
    read = lambda path: json.loads(path.read_text())
    lines = lambda path: [json.loads(line) for line in path.open()]
    report = {}
    if out('evaluate.json').exists():
        e = read(out('evaluate.json'))
        report['evaluate'] = {k: e[k] for k in ('factors_identical', 'reproduces_healed', 'arms', 'differences')}
    if out('state.json').exists():
        report['state'] = {k: v for k, v in read(out('state.json')).items() if k in ('docs', 'from_dense', 'between')}
    sets = {'dense, fp32 state': [dict(r['outputs']['fp32'], id=r['id']) for r in lines(sb.out('generate.jsonl'))],
            'healed, learned router, int16 state': lines(sh.out('generate.jsonl'))}
    for name in rules['generate']['arms']:
        if out(f'generate_{name}.jsonl').exists():
            rows = lines(out(f'generate_{name}.jsonl'))
            sets[f'healed, {rows[0]["router"]} router, {rows[0]["state"]} state'] = rows
    count = heal_rules['generate']['count']
    report['generate'] = {name: generations(rows[:count], rules['loop']) for name, rows in sets.items()}
    save(dict(provenance(), **report), fresh('report.json'))
    for name, g in report['generate'].items():
        log(f'{name}: {g["correct"]}/{g["prompts"]} right, {g["finished"]} finished, {g["loops"]} loops, '
            f'mean length {g["mean_length"]:.0f}')


def main():
    global RULES
    stages = {'evaluate': stage_evaluate, 'state': stage_state, 'generate': stage_generate, 'report': stage_report}
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=stages)
    ap.add_argument('--arm', choices=['fp32', 'perfect'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--model', type=Path, default=mc.MODEL)
    ap.add_argument('--rules', type=Path, default=RULES)
    ap.add_argument('--heal-rules', type=Path, default=sh.RULES)
    ap.add_argument('--slim-rules', type=Path, default=sb.RULES)
    ap.add_argument('--carve-rules', type=Path, default=sb.CARVE_RULES)
    args = ap.parse_args()
    if args.stage == 'generate' and args.arm is None:
        ap.error('generate needs --arm')
    mc.DEVICE, mc.MODEL = args.device, args.model
    RULES, sh.RULES = args.rules.resolve(), args.heal_rules.resolve()
    sb.RULES, sb.CARVE_RULES = args.slim_rules.resolve(), args.carve_rules.resolve()
    stages[args.stage](args)


if __name__ == '__main__':
    main()
