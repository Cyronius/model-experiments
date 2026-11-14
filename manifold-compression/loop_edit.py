"""Loop-direction edit of a locally healed student (loop_edit_rules.json).

find: teacher-forced on the student's own looping greedy answers, each layer's mean MLP output inside
the loop minus before it gives a direction (fit prompts only); held-out looping rows and a sampled
control measure how well it separates looping from position. edit: removes the direction from a
few layers' MLP output, or from the down columns that write most of it, then measures KL against
the dense model and greedy loops on the test prompts.

    loop_edit.py find --rules loop_edit_rules.json --device cuda
    loop_edit.py edit --rules loop_edit_rules.json --device cuda
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import heal_geometry as hg
import heal_probe as hp
import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
import slim_opd as so

ROOT = Path(__file__).resolve().parent


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def onset(tokens, cfg):
    """First generated position from which cfg['share'] of the next cfg['span'] positions repeat an
    earlier cfg['window']-token span; None if the answer never settles into a loop."""
    w, seen, rep = cfg['window'], set(), []
    for t in range(len(tokens) - w + 1):
        span = tuple(tokens[t:t + w])
        rep.append(span in seen)
        seen.add(span)
    rep = np.array(rep, dtype=float)
    if len(rep) < cfg['span']:
        return None
    share = np.convolve(rep, np.ones(cfg['span']) / cfg['span'], 'valid')
    hits = np.nonzero(share >= cfg['share'])[0]
    return int(hits[0]) if len(hits) else None


def load(rules):
    geometry = json.loads((ROOT / 'heal_geometry_rules.json').read_text())
    folder = hg.install(geometry, rules['run'])
    sh.RULES = folder / 'rules.json'
    heal_rules, docs = sh.setup()
    model, slim, vocab, _ = hp.load_healed(heal_rules, docs)
    tok = so.generation_tools(model, vocab)
    check = json.loads((ROOT / 'geometry_check_rules.json').read_text())
    prompts = so.check_prompts(check, tok, vocab)
    return folder, heal_rules, docs, model, slim, vocab, tok, prompts


def rows_of(folder, source):
    path = folder.parent / source.split('/', 1)[1]
    return {r['id']: r for r in map(json.loads, path.open())}


class Capture:
    """Forward hooks on every layer's down projection: fn(layer, MLP output, kept activations) for the
    student's pass over one unpadded sequence, as (T, hidden) and (T, neurons) floats."""

    def __init__(self, model, slim):
        self.slim, self.fn, self.hooks = slim, None, []
        for i, layer in enumerate(model.model.layers):
            self.hooks.append(layer.mlp.down_proj.register_forward_hook(self.hook(i)))

    def hook(self, i):
        def f(module, inputs, output):
            if self.slim.on and self.fn is not None:
                self.fn(i, output[0].float(), inputs[0][0].float())
        return f

    def close(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def forward(model, slim, ids):
    slim.on = True
    try:
        model(input_ids=torch.tensor([ids], device=mc.DEVICE), use_cache=False, logits_to_keep=1)
    finally:
        slim.on = False


def stage_find(args, rules):
    folder, heal_rules, docs, model, slim, vocab, tok, prompts = load(rules)
    out = folder / rules.get('out', 'loop_edit')
    out.mkdir(exist_ok=True)
    greedy = rows_of(folder, rules['source'])
    sampled = rows_of(folder, rules['control']['source'])
    cfg, ccfg = rules['onset'], rules['control']
    edge = rules['split']['boundary']
    layers = len(model.model.layers)
    cap = Capture(model, slim)
    t0 = time.time()

    def regions(p):
        r = greedy[p['id']]
        if r['finished']:
            return None
        o = onset(r['tokens'], cfg)
        if o is None or o < cfg['min_pre']:
            return None
        end = min(len(r['tokens']), o + cfg['loop_skip'] + cfg['loop_tokens'])
        if end - (o + cfg['loop_skip']) < 100:
            return None
        n = len(p['ids'])
        return p['ids'] + r['tokens'][:end], (n, n + o), (n + o + cfg['loop_skip'], n + end), o

    # Pass 1, fit prompts: mean MLP output and mean kept activation before and in the loop.
    down = model.model.layers[0].mlp.down_proj.base
    diff = torch.zeros(layers, down.out_features, device=mc.DEVICE)
    act = torch.zeros(layers, down.in_features, device=mc.DEVICE)
    norm = torch.zeros(layers, device=mc.DEVICE)
    fit, limit = [], rules['split'].get('limit', len(prompts))
    for p in prompts[edge:]:
        if len(fit) >= limit:
            break
        got = regions(p)
        if got is None:
            continue
        ids, (a, b), (c, d), o = got

        def fit_fn(i, y, x):
            diff[i] += y[c:d].mean(0) - y[a:b].mean(0)
            act[i] += x[c:d].mean(0) - x[a:b].mean(0)
            norm[i] += y[a:d].pow(2).sum(-1).mean().sqrt()
        cap.fn = fit_fn
        forward(model, slim, ids)
        fit.append({'id': p['id'], 'onset': o, 'length': len(ids) - len(p['ids'])})
    diff, act, norm = diff / len(fit), act / len(fit), norm / len(fit)
    direction = diff / diff.norm(dim=-1, keepdim=True)
    log(f'fit: {len(fit)} looping rows, median onset {np.median([f["onset"] for f in fit]):.0f}, {time.time() - t0:.0f} s')

    # Pass 2, test prompts: the test rows' own difference, and per-row projections.
    tdiff = torch.zeros_like(diff)
    proj_rows, test = [], []
    for p in prompts[:edge]:
        if len(test) >= limit:
            break
        got = regions(p)
        if got is None:
            continue
        ids, (a, b), (c, d), o = got
        row = []

        def test_fn(i, y, x):
            tdiff[i] += y[c:d].mean(0) - y[a:b].mean(0)
            s = y @ direction[i]
            row.append((float(s[c:d].mean()), float(s[a:b].mean()), float(s[a:b].std())))
        cap.fn = test_fn
        forward(model, slim, ids)
        proj_rows.append(row)
        test.append({'id': p['id'], 'onset': o})
    tdiff /= len(test)
    # Control: sampled answers that ran long without looping, early against late positions.
    control = []
    for p in prompts[:edge]:
        if len(control) >= limit:
            break
        r = sampled[p['id']]
        if not r['finished'] or r['length'] < ccfg['min_length']:
            continue
        n, rise = len(p['ids']), []

        def control_fn(i, y, x):
            s = y @ direction[i]
            rise.append(float(s[n + ccfg['late'][0]:n + ccfg['late'][1]].mean()
                              - s[n + ccfg['early'][0]:n + ccfg['early'][1]].mean()))
        cap.fn = control_fn
        forward(model, slim, p['ids'] + r['tokens'][:ccfg['late'][1]])
        control.append(rise)
    cap.close()
    proj = np.array(proj_rows)  # (rows, layers, [loop, pre, pre std])
    per_layer = []
    for i in range(layers):
        effect = proj[:, i, 0] - proj[:, i, 1]
        per_layer.append({
            'layer': i,
            'fit_size': float(diff[i].norm()), 'output_rms': float(norm[i]),
            'relative': float(diff[i].norm() / norm[i]),
            'cosine_test': float(torch.nn.functional.cosine_similarity(diff[i], tdiff[i], dim=0)),
            'test_higher': float(np.mean(effect > 0)),
            'test_effect': float(effect.mean()),
            'test_effect_sd': float(effect.mean() / proj[:, i, 2].mean()),
            'control_rise': float(np.mean([c[i] for c in control])) if control else None,
        })
    for row in per_layer:
        row['control_share'] = (row['control_rise'] / row['test_effect']
                                if row['control_rise'] is not None and row['test_effect'] else None)
    ranked = sorted(per_layer, key=lambda r: -r['test_effect_sd'])
    for r in ranked[:8]:
        log(f"layer {r['layer']:2d}: effect {r['test_effect_sd']:.2f} sd, cosine {r['cosine_test']:.2f}, "
            f"higher {r['test_higher']:.2f}, relative {r['relative']:.3f}, control share {r['control_share']}")
    torch.save({'direction': direction.cpu(), 'activation_diff': act.cpu()}, out / 'direction.pt')
    (out / 'find.json').write_text(json.dumps({
        'fit': fit, 'test': test, 'control_rows': len(control), 'layers': per_layer,
        'ranked': [r['layer'] for r in ranked], 'seconds': time.time() - t0}, indent=1))


class Edit:
    """Removes each chosen layer's direction from the student's MLP output (mode all), or the part
    of it the chosen down columns write (mode cols, as zeroing their component along it)."""

    def __init__(self, model, slim, direction, plan):
        self.slim, self.hooks = slim, []
        for i, cols in plan.items():
            d = direction[i].to(mc.DEVICE)
            m = model.model.layers[i].mlp.down_proj
            if cols is None:
                self.hooks.append(m.register_forward_hook(self.all(d)))
            else:
                c = torch.zeros(m.base.in_features, device=mc.DEVICE)
                c[cols] = d @ hp.down_weight(m)[:, cols]
                self.hooks.append(m.register_forward_hook(self.cols(d, c)))

    def all(self, d):
        def f(module, inputs, output):
            if not self.slim.on:
                return output
            y = output.float()
            return (y - (y @ d)[..., None] * d).to(output.dtype)
        return f

    def cols(self, d, c):
        def f(module, inputs, output):
            if not self.slim.on:
                return output
            return (output.float() - (inputs[0].float() @ c)[..., None] * d).to(output.dtype)
        return f

    def close(self):
        for h in self.hooks:
            h.remove()


def plan_of(edit, found, saved, model):
    if edit['name'] == 'none':
        return {}
    layers = found['ranked'] if edit['layers'] == 'all' else found['ranked'][:edit['layers']]
    if edit['mode'] == 'all':
        return {i: None for i in layers}
    plan = {}
    for i in layers:
        w = hp.down_weight(model.model.layers[i].mlp.down_proj)
        writes = saved['direction'][i].to(mc.DEVICE) @ w
        contribution = saved['activation_diff'][i].to(mc.DEVICE) * writes
        plan[i] = contribution.topk(edit['columns']).indices
    return plan


def sign_p(k, n):
    return sum(math.comb(n, x) for x in range(k, n + 1)) / 2 ** n if n else 1.0


def stage_edit(args, rules):
    folder, heal_rules, docs, model, slim, vocab, tok, prompts = load(rules)
    out = folder / rules.get('out', 'loop_edit')
    found = json.loads((out / 'find.json').read_text())
    saved = torch.load(out / 'direction.pt', weights_only=True)
    gcfg = rules['generate']
    test = prompts[:args.prompts or gcfg['prompts']]
    base_rows = [rows_of(folder, rules['source'])[p['id']] for p in test]
    base = so.truncated(base_rows, test, tok, gcfg['max_new_tokens'])
    base_sum = hp.generations(base, rules['loop'])
    base_loops = set(base_sum['looping'])
    log(f"baseline at {gcfg['max_new_tokens']}: {base_sum['correct']}/{len(test)} right, {base_sum['loops']} loops")
    results = {'baseline': {k: v for k, v in base_sum.items() if k != 'looping'}}
    if args.only:
        rules['edits'] = [e for e in rules['edits'] if e['name'] in args.only]
    for e in rules['edits']:
        t0 = time.time()
        plan = plan_of(e, found, saved, model)
        edit = Edit(model, slim, saved['direction'], plan)
        try:
            rows = sh.evaluate(model, slim, vocab, sb.docs_of(docs, 'eval', args.docs) if args.docs else sb.docs_of(docs, 'eval'))
            kl = sh.summarize(rows, heal_rules['data']['held_out'])
            res = {'layers': sorted(plan), 'kl': kl['kl'], 'kl_held_out': kl['held_out']['kl'], 'agree': kl['agree']}
            log(f"{e['name']}: layers {sorted(plan)}, KL {kl['kl']:.4f} (held-out {kl['held_out']['kl']:.4f})")
            if e.get('generate', True):
                gen = so.generate(model, slim, vocab, [p['ids'] for p in test],
                                  {'state': gcfg['state'], 'max_new_tokens': args.tokens or gcfg['max_new_tokens'],
                                   'compact': gcfg['compact']}, True, True, None, gcfg['seed'])
                gen = so.check_rows(gen, test, tok, 'healed', False)
                (out / f"gen_{e['name']}.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in gen))
                s = hp.generations(gen, rules['loop'])
                loops = set(s['looping'])
                right = {r['id'] for r in gen if r['correct']}
                was_right = {r['id'] for r in base if r['correct']}
                res.update({k: v for k, v in s.items() if k != 'looping'})
                res['loops_fixed'], res['loops_new'] = len(base_loops - loops), len(loops - base_loops)
                res['p_fewer_loops'] = sign_p(res['loops_fixed'], res['loops_fixed'] + res['loops_new'])
                res['right_gained'], res['right_lost'] = len(right - was_right), len(was_right - right)
                res['p_more_right'] = sign_p(res['right_gained'], res['right_gained'] + res['right_lost'])
                log(f"{e['name']}: {s['correct']}/{len(test)} right, {s['finished']} finished, {s['loops']} loops "
                    f"(fixed {res['loops_fixed']}, new {res['loops_new']}, p {res['p_fewer_loops']:.3f})")
        finally:
            edit.close()
        res['seconds'] = time.time() - t0
        results[e['name']] = res
        (out / 'edit.json').write_text(json.dumps(results, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['find', 'edit'])
    p.add_argument('--rules', type=Path, default=ROOT / 'loop_edit_rules.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--only', nargs='*', help='edit: run only these edits')
    p.add_argument('--docs', type=int, help='edit: eval documents per domain (smoke test)')
    p.add_argument('--prompts', type=int, help='edit: test prompts (smoke test)')
    p.add_argument('--limit', type=int, help='find: at most this many rows per pass (smoke test)')
    p.add_argument('--tokens', type=int, help='edit: generation cap (smoke test; writes edit_smoke.json)')
    args = p.parse_args()
    mc.DEVICE = args.device
    rules = json.loads(args.rules.resolve().read_text())
    if args.limit:
        rules['split']['limit'], rules['out'] = args.limit, 'loop_edit_smoke'
    if args.tokens:
        rules['out'] = 'loop_edit_smoke'
    {'find': stage_find, 'edit': stage_edit}[args.stage](args, rules)


if __name__ == '__main__':
    main()
