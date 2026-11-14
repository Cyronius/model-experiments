"""Second loop-direction edit (loop_edit2_rules.json), following the recipe in the weight-space
programming logs (RikiyaT/lm_understand-logs, exp8_math/runs/p50c): a residual-stream direction from
matched productive and repetitive branches of one prefix, an alpha-scaled edit of the down columns
that write it (W' = W0 - alpha u (mask * u^T W0)), a teacher-forced screen of the branches'
log-probabilities, then greedy GSM8K generations.

    loop_edit2.py all --device cuda        # fit, screen, edit in one process
    loop_edit2.py fit|screen|edit --device cuda

fit: the student answers GSM8K train prompts greedily; each looping answer gives a prefix (up to the
loop onset), the student's own repetitive branch and the dense model's productive branch; the
student is teacher-forced on both and the means of its residual stream, MLP output and kept
activations over each branch are saved. screen: each edit's change to the branches' log-probs on
held-out pairs. edit: KL and greedy generations on the 96 check prompts per arm.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import heal_probe as hp
import loop_edit as le
import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
import slim_opd as so

ROOT = Path(__file__).resolve().parent
log = le.log


def fit_prompts(rules, tok, vocab, docs):
    used = {x for d in docs for x in d['items']}
    cfg, prompts = rules['fit'], []
    a, b = cfg['rows']
    for n, r in sh.paged('openai/gsm8k', 'main', [[a, b - a]]):
        if f'gsm8k-{n}' in used:
            continue
        answer = float(r['answer'].rpartition('####')[2].strip().replace(',', ''))
        prompts.append({'id': f'gsm8k-train-{n}', 'question': r['question'], 'answer': answer,
                        'ids': vocab.short.encode(so.template(tok, r['question']))})
        if len(prompts) == cfg['count']:
            break
    return prompts


def repeats(tokens, start, window):
    """Share of the 20-token windows that begin at or after start and appeared earlier in tokens."""
    seen, hits, total = set(), 0, 0
    for t in range(len(tokens) - window + 1):
        span = tuple(tokens[t:t + window])
        if t >= start:
            total += 1
            hits += span in seen
        seen.add(span)
    return hits / total if total else 0.0


class Means:
    """Hooks: per layer, the mean over positions [a, b) of the residual after the layer, the MLP
    output and the kept activations, for the student's pass over one sequence."""

    def __init__(self, model, slim):
        self.slim, self.span, self.hooks, self.got = slim, None, [], {}
        for i, layer in enumerate(model.model.layers):
            self.hooks.append(layer.register_forward_hook(self.residual(i)))
            self.hooks.append(layer.mlp.down_proj.register_forward_hook(self.mlp(i)))

    def residual(self, i):
        def f(module, inputs, output):
            if self.slim.on and self.span is not None:
                h = output[0] if isinstance(output, tuple) else output
                a, b = self.span
                self.got[('res', i)] = h[0, a:b].float().mean(0).cpu()
        return f

    def mlp(self, i):
        def f(module, inputs, output):
            if self.slim.on and self.span is not None:
                a, b = self.span
                self.got[('mlp', i)] = output[0, a:b].float().mean(0).cpu()
                self.got[('act', i)] = inputs[0][0, a:b].float().mean(0).cpu()
        return f

    def close(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def branch_logp(model, slim, vocab, ids, start):
    """The student's mean log-probability per token of ids[start:] (64K head)."""
    slim.on = True
    try:
        h = sb.hidden(model, [ids])[0, start - 1:len(ids) - 1]
    finally:
        slim.on = False
    logp = F.linear(h, vocab.head).float().log_softmax(-1)
    target = vocab.index[sb.tensor(ids[start:])]
    return float(logp.gather(-1, target[:, None]).mean())


def stage_fit(ctx, rules):
    folder, heal_rules, docs, model, slim, vocab, tok, _ = ctx
    out = folder / rules['out']
    out.mkdir(exist_ok=True)
    t0 = time.time()
    prompts = fit_prompts(rules, tok, vocab, docs)
    g = rules['fit']['generate']
    path = out / 'gen_fit.jsonl'
    if path.exists():
        rows = [json.loads(l) for l in path.open()]
    else:
        rows = []
        for b in range(0, len(prompts), 80):
            part = prompts[b:b + 80]
            gen = so.generate(model, slim, vocab, [p['ids'] for p in part], g, True, True, None, g['seed'])
            rows += so.check_rows(gen, part, tok, 'healed', False)
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    s = hp.generations(rows, rules['loop'])
    log(f"fit prompts: {s['correct']}/{len(rows)} right, {s['finished']} finished, {s['loops']} loops, {time.time() - t0:.0f} s")
    looping, ocfg, pcfg = set(s['looping']), rules['onset'], rules['pairs']
    L, w = pcfg['branch_tokens'], ocfg['window']
    cands = []
    for r, p in zip(rows, prompts, strict=True):
        if r['id'] not in looping:
            continue
        t = r['tokens']
        o = le.onset(t, ocfg)
        if o is None:
            continue
        # The branch point: where the span the loop first repeats first appeared, so the prefix holds
        # no copy of the loop body; the repetitive branch is the first period and L tokens of repeats.
        span = tuple(t[o:o + w])
        f = next(k for k in range(o + 1) if tuple(t[k:k + w]) == span)
        length = min(o - f + L, pcfg['max_branch'])
        if f < ocfg['min_pre'] or len(t) < f + length:
            continue
        cands.append({'id': r['id'], 'onset': o, 'branch_at': f, 'prefix': p['ids'] + t[:f], 'rep': t[f:f + length]})
    gen = so.generate(model, slim, vocab, [c['prefix'] for c in cands],
                      {'state': g['state'], 'max_new_tokens': pcfg['max_branch'], 'compact': True}, False, True, None, 0)
    pairs, dropped = [], {'short': 0, 'repeats': 0}
    for c, d in zip(cands, gen, strict=True):
        prod = d['tokens'][:len(c['rep'])]
        if len(prod) < 32:
            dropped['short'] += 1
            continue
        if repeats(c['prefix'] + prod, len(c['prefix']), ocfg['window']) > 0.5:
            dropped['repeats'] += 1
            continue
        pairs.append(dict(c, prod=prod, prod_stopped=d['stop'] is not None))
    log(f'pairs: {len(pairs)} of {len(cands)} looping answers with an onset (dropped {dropped})')
    means = Means(model, slim)
    caps = []
    try:
        for pr in pairs:
            n, one = len(pr['prefix']), {}
            for kind in ('rep', 'prod'):
                means.span, means.got = (n, n + len(pr[kind])), {}
                le.forward(model, slim, pr['prefix'] + pr[kind])
                one[kind] = dict(means.got)
            caps.append(one)
    finally:
        means.close()
    layers = len(model.model.layers)

    def stack(idx, kind, what):
        return torch.stack([torch.stack([caps[k][kind][(what, i)] for i in range(layers)]) for k in idx])

    direction_idx, screen_idx = list(range(0, len(pairs), 2)), list(range(1, len(pairs), 2))
    res_d = stack(direction_idx, 'rep', 'res') - stack(direction_idx, 'prod', 'res')  # (pairs, layers, H)
    u = res_d.mean(0)
    u = u / u.norm(dim=-1, keepdim=True)
    res_s = stack(screen_idx, 'rep', 'res') - stack(screen_idx, 'prod', 'res')
    mlp_s = stack(screen_idx, 'rep', 'mlp') - stack(screen_idx, 'prod', 'mlp')
    act_d = (stack(direction_idx, 'rep', 'act') - stack(direction_idx, 'prod', 'act')).mean(0)  # (layers, N)
    per_layer = []
    for i in range(layers):
        along = res_s[:, i] @ u[i]
        written = mlp_s[:, i] @ u[i]
        per_layer.append({'layer': i, 'higher': float((along > 0).float().mean()),
                          'cosine_screen': float(F.cosine_similarity(res_s[:, i].mean(0), u[i], dim=0)),
                          'residual_along': float(along.mean()), 'mlp_along': float(written.mean()),
                          'mlp_share': float(written.mean() / along.mean())})
    edge = rules['layer_margin']
    ranked = sorted([r for r in per_layer if edge <= r['layer'] < layers - edge], key=lambda r: -r['mlp_share'])
    for r in ranked[:8]:
        log(f"layer {r['layer']:2d}: MLP writes {r['mlp_share']:.3f} of the residual difference along u, "
            f"higher {r['higher']:.2f}, cosine {r['cosine_screen']:.2f}")
    chosen = [r['layer'] for r in ranked[:2]]
    columns = {}
    for i in chosen:
        w = hp.down_weight(model.model.layers[i].mlp.down_proj)
        writes = u[i].to(mc.DEVICE) @ w
        columns[i] = (act_d[i].to(mc.DEVICE) * writes).topk(64).indices.cpu()
        share = float((act_d[i].to(mc.DEVICE) * writes)[columns[i].to(mc.DEVICE)].sum() / (act_d[i].to(mc.DEVICE) * writes).sum())
        log(f'layer {i}: top 64 columns carry {share:.2f} of the writing contrast along u')
    torch.save({'u': u, 'chosen': chosen, 'columns': columns}, out / 'direction.pt')
    (out / 'pairs.json').write_text(json.dumps({
        'fit_summary': {k: v for k, v in s.items() if k != 'looping'}, 'candidates': len(cands), 'dropped': dropped,
        'pairs': [{k: v for k, v in pr.items() if k != 'prefix'} | {'prefix_length': len(pr['prefix'])} for pr in pairs],
        'direction_pairs': direction_idx, 'screen_pairs': screen_idx, 'layers': per_layer,
        'ranked': [r['layer'] for r in ranked], 'chosen': chosen, 'seconds': time.time() - t0}, indent=1))
    torch.save([pr['prefix'] for pr in pairs], out / 'prefixes.pt')


class Edit:
    """W' = W0 - alpha u (mask * u^T W0) on chosen layers' down projections, as hooks on the student."""

    def __init__(self, model, slim, plan):
        self.slim, self.hooks = slim, []
        for i, (u, c, alpha) in plan.items():
            m = model.model.layers[i].mlp.down_proj
            self.hooks.append(m.register_forward_hook(self.hook(u.to(mc.DEVICE), None if c is None else c.to(mc.DEVICE), alpha)))

    def hook(self, u, c, alpha):
        def f(module, inputs, output):
            if not self.slim.on:
                return output
            y = output.float()
            coef = y @ u if c is None else inputs[0].float() @ c
            return (y - alpha * coef[..., None] * u).to(output.dtype)
        return f

    def close(self):
        for h in self.hooks:
            h.remove()


def plan_of(e, saved, model):
    if e['name'] == 'none':
        return {}
    plan = {}
    for i in saved['chosen']:
        u = saved['u'][i].to(mc.DEVICE)
        w = hp.down_weight(model.model.layers[i].mlp.down_proj)
        if e['columns'] == 'all':
            plan[i] = (u, None, e['alpha'])
            continue
        mask = torch.zeros(w.shape[1], device=mc.DEVICE)
        mask[saved['columns'][i].to(mc.DEVICE)] = 1
        writes = mask * (u @ w)
        if not e.get('random'):
            plan[i] = (u, writes, e['alpha'])
            continue
        g = torch.Generator(device='cpu').manual_seed(i)
        r = torch.randn(u.shape[0], generator=g).to(mc.DEVICE)
        r = r - (r @ u) * u
        r = r / r.norm()
        rwrites = mask * (r @ w)
        # Same Frobenius norm of the change: alpha ||writes|| = beta ||rwrites||.
        plan[i] = (r, rwrites, e['alpha'] * float(writes.norm() / rwrites.norm()))
    return plan


def stage_screen(ctx, rules):
    folder, heal_rules, docs, model, slim, vocab, tok, _ = ctx
    out = folder / rules['out']
    found = json.loads((out / 'pairs.json').read_text())
    saved = torch.load(out / 'direction.pt', weights_only=True)
    prefixes = torch.load(out / 'prefixes.pt')
    pairs = [dict(found['pairs'][k], prefix=prefixes[k]) for k in found['screen_pairs']]
    results = {}
    for e in [{'name': 'none'}] + rules['screen']['edits']:
        edit = Edit(model, slim, plan_of(e, saved, model))
        try:
            got = {kind: [branch_logp(model, slim, vocab, p['prefix'] + p[kind], len(p['prefix'])) for p in pairs]
                   for kind in ('rep', 'prod')}
        finally:
            edit.close()
        results[e['name']] = got
    base = results['none']
    summary = {}
    for name, got in results.items():
        if name == 'none':
            continue
        d = {k: float(np.mean(np.array(got[k]) - np.array(base[k]))) for k in ('rep', 'prod')}
        d['selectivity'] = d['prod'] - d['rep']
        summary[name] = d
        log(f"screen {name}: repetitive {d['rep']:+.4f}, productive {d['prod']:+.4f} nats/token")
    (out / 'screen.json').write_text(json.dumps({'pairs': len(pairs), 'summary': summary, 'raw': results}, indent=1))


def stage_edit(ctx, rules):
    folder, heal_rules, docs, model, slim, vocab, tok, prompts = ctx
    out = folder / rules['out']
    saved = torch.load(out / 'direction.pt', weights_only=True)
    screen = json.loads((out / 'screen.json').read_text())['summary']
    gcfg = rules['generate']
    edits = {e['name']: e for e in rules['screen']['edits']}
    arms = [a for a in gcfg['arms'] if a != 'best']
    best = max((n for n in screen if n not in arms and not edits[n].get('random')), key=lambda n: screen[n]['selectivity'])
    if best != 'full_a1':
        arms.append(best)
    log(f'arms: {arms} (best screen edit {best})')
    test = prompts[:gcfg['prompts']]
    checked = {r['id']: r for r in map(json.loads, (folder / 'check' / 'gen_healed_greedy.jsonl').open())}
    base_rows = [checked[p['id']] for p in test]
    base = so.truncated(base_rows, test, tok, gcfg['max_new_tokens'])
    bs = hp.generations(base, rules['loop'])
    base_loops, was_right = set(bs['looping']), {r['id'] for r in base if r['correct']}
    log(f"baseline at {gcfg['max_new_tokens']}: {bs['correct']}/{len(test)} right, {bs['loops']} loops")
    results = {'baseline': {k: v for k, v in bs.items() if k != 'looping'}, 'arms': arms}
    for name in arms:
        t0 = time.time()
        plan = plan_of(edits[name], saved, model)
        edit = Edit(model, slim, plan)
        try:
            kl = sh.summarize(sh.evaluate(model, slim, vocab, sb.docs_of(docs, 'eval')), heal_rules['data']['held_out'])
            res = {'kl': kl['kl'], 'kl_held_out': kl['held_out']['kl'], 'agree': kl['agree'], 'screen': screen[name]}
            log(f"{name}: KL {kl['kl']:.4f} (held-out {kl['held_out']['kl']:.4f})")
            gen = so.generate(model, slim, vocab, [p['ids'] for p in test],
                              {'state': gcfg['state'], 'max_new_tokens': gcfg['max_new_tokens'], 'compact': gcfg['compact']},
                              True, True, None, gcfg['seed'])
            gen = so.check_rows(gen, test, tok, 'healed', False)
            (out / f'gen_{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in gen))
            s = hp.generations(gen, rules['loop'])
            loops, right = set(s['looping']), {r['id'] for r in gen if r['correct']}
            res.update({k: v for k, v in s.items() if k != 'looping'})
            res['loops_fixed'], res['loops_new'] = len(base_loops - loops), len(loops - base_loops)
            res['p_fewer_loops'] = le.sign_p(res['loops_fixed'], res['loops_fixed'] + res['loops_new'])
            res['right_gained'], res['right_lost'] = len(right - was_right), len(was_right - right)
            res['p_more_right'] = le.sign_p(res['right_gained'], res['right_gained'] + res['right_lost'])
            log(f"{name}: {s['correct']}/{len(test)} right (+{res['right_gained']} -{res['right_lost']}), "
                f"{s['finished']} finished, {s['loops']} loops (fixed {res['loops_fixed']}, new {res['loops_new']}, "
                f"p {res['p_fewer_loops']:.3g})")
        finally:
            edit.close()
        res['seconds'] = time.time() - t0
        results[name] = res
        (out / 'edit.json').write_text(json.dumps(results, indent=1))
    # The sampled check, for the arm with the fewest loops if it qualifies.
    ok = [n for n in arms if results[n]['p_fewer_loops'] < 0.05 and abs(results[n]['kl'] - 0.623) <= 0.03
          and not edits[n].get('random')]
    if ok:
        name = min(ok, key=lambda n: results[n]['loops'])
        ecfg = json.loads((ROOT / 'geometry_check_rules.json').read_text())['eval']
        edit = Edit(model, slim, plan_of(edits[name], saved, model))
        try:
            gen = so.generate(model, slim, vocab, [p['ids'] for p in test],
                              {'state': ecfg['state'], 'max_new_tokens': ecfg['max_new_tokens'], 'compact': True},
                              True, [False] * len(test), ecfg['sampling'], ecfg['seed'] + 1)
        finally:
            edit.close()
        gen = so.check_rows(gen, test, tok, 'healed', True)
        (out / f'gen_{name}_sampled.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in gen))
        s = hp.generations(gen, rules['loop'])
        base_s = [r for r in map(json.loads, (folder / 'check' / 'gen_healed_sampled.jsonl').open())]
        bright = {r['id'] for r in base_s if r['correct']}
        right = {r['id'] for r in gen if r['correct']}
        results['sampled'] = dict({k: v for k, v in s.items() if k != 'looping'}, arm=name,
                                  right_gained=len(right - bright), right_lost=len(bright - right))
        log(f"{name} sampled at {ecfg['max_new_tokens']}: {s['correct']}/{len(test)} right "
            f"(+{len(right - bright)} -{len(bright - right)}), {s['finished']} finished, {s['loops']} loops")
        (out / 'edit.json').write_text(json.dumps(results, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['fit', 'screen', 'edit', 'all'])
    p.add_argument('--rules', type=Path, default=ROOT / 'loop_edit2_rules.json')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    mc.DEVICE = args.device
    rules = json.loads(args.rules.resolve().read_text())
    ctx = le.load(rules)
    stages = {'fit': [stage_fit], 'screen': [stage_screen], 'edit': [stage_edit],
              'all': [stage_fit, stage_screen, stage_edit]}[args.stage]
    for stage in stages:
        stage(ctx, rules)


if __name__ == '__main__':
    main()
