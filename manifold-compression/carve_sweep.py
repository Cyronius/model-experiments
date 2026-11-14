"""Sweep of the co-firing carve's shape at the 25% budget: shared expert size S, routed expert
size c, expert count E, experts per token K. Perfect router; reuses moe_carve.py's statistics,
corpus and evaluation. Rules: carve_sweep_rules.json.

Stages: build (one config per call, --key; clusters on the CPU), evaluate (all configs as arms of
one forward pass per held-out document), report (domain-equal KL and paired comparisons)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import moe_carve as mc
from h0_mlp_probe import paired_interval, save, sha

ROOT = Path(__file__).resolve().parent
RULES = ROOT / 'carve_sweep_rules.json'


def setup():
    rules = json.loads(RULES.read_text())
    mc.configure(ROOT / rules['rules'])
    base = json.loads((ROOT / rules['rules']).read_text())
    out = ROOT / rules['out']
    out.mkdir(parents=True, exist_ok=True)
    return rules, base, out


def build_one(stats, base, cfg):
    tokens = stats['tokens'].double()
    rate = stats['active'].double() / tokens[:, None]
    layers, n_neurons = rate.shape[0], rate.shape[-1]
    assert cfg['S'] + cfg['E'] * cfg['c'] == n_neurons
    shared = (mc.topmask(rate.mean(1), cfg['S']) if cfg['S'] else torch.zeros(layers, n_neurons, dtype=torch.bool))
    cluster = mc.cluster_rest(stats, shared, cfg['E'], base)
    return {'experts': cfg['E'], 'c': cfg['c'], 'S': cfg['S'], 'K': cfg['K'],
            'n': cfg['S'] + cfg['K'] * cfg['c'], 'shared': shared, 'cluster': cluster}


def stage_build(args):
    rules, base, out = setup()
    torch.set_num_threads(args.threads)
    stats = torch.load(mc.OUT / 'stats.pt', weights_only=True)
    for key in args.key:
        cfg = rules['configs'][key]
        path = out / f'set_{key}.pt'
        if path.exists():
            continue
        t0 = time.time()
        if key == rules['baseline']:
            s = torch.load(mc.OUT / 'sets.pt', weights_only=True)['0.25']
            s = {k: s[k] for k in ('experts', 'c', 'S', 'K', 'n', 'shared', 'cluster')}
        else:
            s = build_one(stats, base, cfg)
        torch.save(s, path)
        print(f'{key}: S {s["S"]} c {s["c"]} E {s["experts"]} K {s["K"]} n {s["n"]}, {time.time() - t0:.0f} s', flush=True)


def arms_of(rules):
    keys = list(rules['configs'])
    return [None] + [('cluster_token', k) for k in keys], ['dense'] + [f'cluster_token@{k}' for k in keys]


def stage_evaluate(args):
    rules, base, out = setup()
    mc.DEVICE = args.device
    sets = {k: torch.load(out / f'set_{k}.pt', weights_only=True, map_location=args.device) for k in rules['configs']}
    corpus = json.loads((mc.OUT / 'corpus.json').read_text())
    model = mc.load_model(args.threads)
    carver = mc.Carver(model, sets)
    arms, names = arms_of(rules)
    path = out / 'evaluation.jsonl'
    done = {json.loads(line)['id'] for line in path.open()} if path.exists() else set()
    docs = [d for d in corpus['docs'] if d['split'] == 'eval' and d['id'] not in done]
    print(f'{len(docs)} documents, {len(arms)} arms each', flush=True)
    for doc in docs:
        t0 = time.time()
        row = mc.evaluate_doc(model, carver, doc, arms, names)
        with path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(f'{doc["id"]}: {time.time() - t0:.0f} s', flush=True)


def stage_report(args):
    rules, base, out = setup()
    rows = [json.loads(line) for line in (out / 'evaluation.jsonl').open()]
    _, names = arms_of(rules)
    arms = mc.arm_summary(rows, names)
    text = [r for r in rows if r['domain'] != 'copy']
    groups = [r['domain'] for r in text]
    b = f'cluster_token@{rules["baseline"]}'
    report = {}
    for key, cfg in rules['configs'].items():
        n = f'cluster_token@{key}'
        diff, low, high = paired_interval([r['kl'][n] for r in text], [r['kl'][b] for r in text], 1.0,
                                          rules['report']['draws'], rules['report']['seed'], groups)
        report[key] = {**cfg, 'kl': arms[n]['kl'], 'copy_p99': arms[n]['copy_p99'], 'delta_nll': arms[n]['delta_nll'],
                       'minus_baseline': {'difference': diff, 'low': low, 'high': high},
                       'kl_by_domain': arms[n]['kl_by_domain']}
        print(f'{key:10s} S {cfg["S"]:5d} E {cfg["E"]:4d} K {cfg["K"]:3d}  KL {arms[n]["kl"]:.3f}  '
              f'p99 {arms[n]["copy_p99"]:.2f}  dNLL {arms[n]["delta_nll"]:+.3f}  vs base {diff:+.3f} [{low:+.3f}, {high:+.3f}]')
    save({'documents': len(text), 'rules_sha256': sha(RULES), 'evaluation_sha256': sha(out / 'evaluation.jsonl'),
          'configs': report}, out / 'report.json')


def stage_route(args):
    """Learned routers for each --key's co-firing experts, trained on the calibration documents and
    scored on the held-out ones against the perfect router."""
    rules, base, out = setup()
    mc.DEVICE = args.device
    corpus = json.loads((mc.OUT / 'corpus.json').read_text())
    model = mc.load_model(args.threads)
    calib = [d for d in corpus['docs'] if d['split'] == 'calib']
    held = [d for d in corpus['docs'] if d['split'] == 'eval']
    for key in args.key:
        path = out / f'route_{key}.json'
        if path.exists():
            continue
        t0 = time.time()
        s = torch.load(out / f'set_{key}.pt', weights_only=True, map_location=args.device)
        carver = mc.Carver(model, {key: s})
        carver.routers, losses = mc.train_routers(model, carver, s, calib, base['route'])
        torch.save([r.state_dict() for r in carver.routers], out / f'routers_{key}.pt')
        arms = [None, ('cluster_token', key), ('learned', key)]
        names = ['dense'] + [f'{p}@{key}' for p, _ in arms[1:]]
        rows, recall = [], []
        for doc in held:
            rows.append(mc.evaluate_doc(model, carver, doc, arms, names))
            recall.append(mc.router_recall(model, carver, s, doc))
        text = [r for r in rows if r['domain'] != 'copy']
        groups = [r['domain'] for r in text]
        summary = mc.arm_summary(rows, names)
        diff, low, high = paired_interval([r['kl'][names[2]] for r in text], [r['kl'][names[1]] for r in text], 1.0,
                                          rules['report']['draws'], rules['report']['seed'], groups)
        save({'key': key, 'config': rules['configs'][key], 'loss_by_step': losses, 'arms': summary,
              'recall_mean': float(np.mean(recall)), 'recall_by_layer': np.mean(recall, 0).tolist(),
              'learned_minus_perfect': {'difference': diff, 'low': low, 'high': high,
                                        'ratio': summary[names[2]]['kl'] / summary[names[1]]['kl']},
              'seconds': time.time() - t0}, path)
        print(f'{key}: perfect {summary[names[1]]["kl"]:.3f} learned {summary[names[2]]["kl"]:.3f} '
              f'(p99 {summary[names[2]]["copy_p99"]:.2f}) recall {np.mean(recall):.2f}, {time.time() - t0:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'evaluate', 'report', 'route'])
    ap.add_argument('--key', nargs='*', default=[])
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    {'build': stage_build, 'evaluate': stage_evaluate, 'report': stage_report, 'route': stage_route}[args.stage](args)


if __name__ == '__main__':
    main()
