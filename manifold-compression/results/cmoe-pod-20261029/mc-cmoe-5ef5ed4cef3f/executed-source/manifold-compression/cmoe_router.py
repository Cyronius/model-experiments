"""CMoE's training-free router on our co-firing experts, against our learned linear routers and the
perfect router, at the primary budget, before any healing.

CMoE (Pei et al. 2025, arXiv 2502.04416; code github.com/JarvisPei/CMoE, construct_moe in
CMoE_utils.py and Router in CMoE_model.py) picks one representative neuron per routed expert: the
member closest in L1 distance to the mean of the expert's activation markers. Its router scores
expert e as |silu(x . g_e) * (x . u_e)|, where g_e and u_e are the representative's gate and up
rows scaled to unit length, and runs the top K. Its softmax and its load-balancing bias (zero
until fine-tuning moves it) do not change the top K. Our experts were clustered on our own markers
(each sampled calibration token's top 25% of neurons by contribution), so the representatives
are picked from those markers.

Stages:
  pick      (local) the representative neurons, from the carve's calibration statistics.
  evaluate  KL from dense on the eval documents with perfect, learned and CMoE routing, and each
            router's recall of the perfect experts.

Outputs go to the cmoe/ folder beside the carve's.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import moe_carve as mc
from h0_mlp_probe import deadline, log, paired_interval, save, sha

ROOT = mc.ROOT
CODE = ['cmoe_router.py', 'moe_carve.py', 'h0_mlp_probe.py', 'test_cmoe_router.py']
MAX_HOURS = 0.5


def out(name):
    return mc.OUT / 'cmoe' / name


def provenance():
    return {'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'carve_rules_sha256': sha(mc.RULES)}


def representatives(markers, cluster, experts):
    """(experts,) neuron index per routed expert: the member closest in L1 distance to the mean
    marker vector of its expert. markers is (tokens, neurons), cluster (neurons,) with -1 shared."""
    picked = []
    for e in range(experts):
        members = (cluster == e).nonzero().squeeze(1)
        x = markers[:, members].float()
        picked.append(int(members[(x - x.mean(1, keepdim=True)).abs().sum(0).argmin()]))
    return torch.tensor(picked)


class CMoECarver(mc.Carver):
    """mc.Carver with one more policy, 'cmoe': CMoE's router from representative neurons."""

    def __init__(self, model, sets, reps):
        super().__init__(model, sets)
        self.cmoe = [(F.normalize(m.gate_proj.weight[r].float(), dim=1), F.normalize(m.up_proj.weight[r].float(), dim=1))
                     for m, r in zip(self.mlps, reps.to(mc.DEVICE))]

    def cmoe_scores(self, i, x):
        gate, up = self.cmoe[i]
        x = x.float()
        return (F.silu(x @ gate.T) * (x @ up.T)).abs()

    def keep(self, arm, i, a, x):
        policy, budget = arm
        if policy != 'cmoe':
            return super().keep(arm, i, a, x)
        s = self.sets[budget]
        return mc.expand(s, i, mc.topmask(self.cmoe_scores(i, x), mc.at(s, 'K', i)))


def stage_pick(args):
    rules = json.loads(mc.RULES.read_text())
    budget = rules['route']['budget']
    if out('representatives.pt').exists():
        raise ValueError('Representatives exist; refusing to overwrite')
    stats = torch.load(mc.OUT / 'stats.pt', weights_only=True)
    s = torch.load(mc.OUT / 'sets.pt', weights_only=True)[budget]
    n = s['cluster'].shape[1]
    reps = torch.stack([representatives(torch.from_numpy(np.unpackbits(bits.numpy(), axis=1, count=n)),
                                        s['cluster'][layer], s['experts'])
                        for layer, bits in enumerate(stats['bits'])])
    out('representatives.pt').parent.mkdir(parents=True, exist_ok=True)
    torch.save(reps, out('representatives.pt'))
    save(dict(provenance(), budget=budget, stats_sha256=sha(mc.OUT / 'stats.pt'), sets_sha256=sha(mc.OUT / 'sets.pt'),
              representatives_sha256=sha(out('representatives.pt'))), out('pick.json'))
    log(f'{reps.shape[1]} representatives in each of {reps.shape[0]} layers')


def stage_evaluate(args):
    rules = json.loads(mc.RULES.read_text())
    budget = rules['route']['budget']
    if out('evaluate.json').exists():
        raise ValueError('Evaluation exists; refusing to overwrite')
    corpus = json.loads((mc.OUT / 'corpus.json').read_text())
    sets = torch.load(mc.OUT / 'sets.pt', weights_only=True, map_location=mc.DEVICE)
    s = sets[budget]
    model = mc.load_model(rules['evaluate']['threads'])
    carver = CMoECarver(model, sets, torch.load(out('representatives.pt'), weights_only=True))
    carver.routers = []
    for state in torch.load(mc.OUT / 'routers.pt', weights_only=True, map_location=mc.DEVICE):
        router = nn.Linear(model.config.hidden_size, s['experts']).to(mc.DEVICE)
        router.load_state_dict(state)
        carver.routers.append(router.eval().requires_grad_(False))
    arms = [None, ('cluster_token', budget), ('learned', budget), ('cmoe', budget)]
    names = ['dense'] + [f'{p}@{budget}' for p, _ in arms[1:]]
    rows, recall = [], {'learned': [], 'cmoe': []}
    t0 = time.time()
    with deadline(MAX_HOURS), torch.no_grad():
        for doc in [d for d in corpus['docs'] if d['split'] == 'eval']:
            rows.append(mc.evaluate_doc(model, carver, doc, arms, names))
            per_layer = {k: [] for k in recall}

            def record(layer, a, x):
                k = mc.at(s, 'K', layer)
                best = mc.topmask(mc.expert_mass(s, layer, carver.score(layer, a[0, 1:])), k)
                for name, scores in (('learned', carver.routers[layer](x[0, 1:].float())),
                                     ('cmoe', carver.cmoe_scores(layer, x[0, 1:]))):
                    per_layer[name].append(float((best & mc.topmask(scores, k)).sum() / (best.shape[0] * k)))

            carver.record = record
            try:
                model.model(input_ids=torch.tensor([doc['input_ids']], device=mc.DEVICE), use_cache=False)
            finally:
                carver.record = None
            for k in recall:
                recall[k].append(per_layer[k])
    out('evaluation.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    text = [r for r in rows if r['domain'] != 'copy']
    groups = [r['domain'] for r in text]
    report = rules['report']
    diff, low, high = paired_interval([r['kl'][names[3]] for r in text], [r['kl'][names[2]] for r in text], 1.0,
                                      report['draws'], report['seed'], groups)
    summary = mc.arm_summary(rows, names)
    save(dict(provenance(), corpus_sha256=sha(mc.OUT / 'corpus.json'), sets_sha256=sha(mc.OUT / 'sets.pt'),
              routers_sha256=sha(mc.OUT / 'routers.pt'), representatives_sha256=sha(out('representatives.pt')),
              evaluation_sha256=sha(out('evaluation.jsonl')), arms=summary,
              recall={k: float(np.mean(v)) for k, v in recall.items()},
              recall_by_layer={k: np.mean(v, 0).tolist() for k, v in recall.items()},
              cmoe_minus_learned={'difference': diff, 'low': low, 'high': high},
              seconds=time.time() - t0), out('evaluate.json'))
    log(', '.join(f'{n} KL {summary[n]["kl"]:.3f}' for n in names[1:]) +
        f'; recall learned {np.mean(recall["learned"]):.3f}, cmoe {np.mean(recall["cmoe"]):.3f}')


def main():
    stages = {'pick': stage_pick, 'evaluate': stage_evaluate}
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=stages)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--model', type=Path, default=mc.MODEL)
    ap.add_argument('--rules', type=Path, default=ROOT / 'moe_carve_rules_v2.json')
    args = ap.parse_args()
    mc.configure(args.rules)
    mc.DEVICE, mc.MODEL = args.device, args.model
    stages[args.stage](args)


if __name__ == '__main__':
    main()
