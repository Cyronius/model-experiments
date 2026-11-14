"""Two checks on the healed student of slim_heal.py, before deciding how to heal further.

1. Where the remaining drift sits. The healed student is evaluated as at its healing milestones
   with its learned routers (which must reproduce the last milestone), with the perfect router
   (each token's K experts with the most contribution, from the student's own activations, as
   moe_carve's cluster_token arm) and with every MLP neuron running.
2. Why its long generations loop. The DeltaNet state check is split into its two parts: fp32 token
   by token against the chunked kernel, and int16 against fp32 token by token. The 16 GSM8K
   generations are repeated with fp32 state, once with the learned and once with the perfect router.
3. Whether the loops come from the 25% sparsity or from the healing: the generations with every
   neuron running, with and without the healing adapter. Without it, the student is the dense
   model with only the low-rank DeltaNet cut and the 64K vocabulary.
4. Whether a soft edge on the learned routers stops the loops: each chosen expert is weighted by how
   far its router score clears the best expert left out, so experts fade in and out instead of
   switching. The state check then measures, for each way of routing, how far a tiny numerical
   change (the chunked kernel against fp32 token by token) moves the student.
5. Whether folding the skipped neurons into the ones that run stops the loops, with no training: the
   output weights of the neurons that run gain what the skipped ones would have added, as far as
   the neurons that run predict it (see fit_layer). Two weaker folds of the same fit scale whole
   experts instead: one gain per expert, and one on the shared expert alone.
6. Whether subtracting the fold instead of adding it helps: the same fit, scaled from -1 to +1, so
   the drift is traced along the fold's direction on both sides of the healed student.
7. Whether a fold can stop the loops, in parallel pods on 48 new GSM8K prompts: the learned
   student alone; the fold at finer strengths; the fold fitted layer by layer in order; a fold that
   tracks the student with every neuron (see fit_folds); and both students with the model card's
   recommended sampling instead of greedy decoding.

The healing pod kept its low-rank factors, so they are rebuilt from the calibration documents. The
adapter is the healing run's adapter.pt: its bf16 LoRA factors are what the student's bf16 forward
used, and its routers are fp32.

Settings and predictions are in heal_probe_rules.json (checks 1 and 2, folder probe/),
heal_probe_all_rules.json (check 3, folder probe_all/), heal_probe_soft_rules.json (check 4,
folder probe_soft/), heal_probe_fold_rules.json (check 5, folder probe_fold/) and
heal_probe_minus_rules.json (check 6, folder probe_minus/) and heal_probe_loops_*_rules.json (check
7, one folder probe_loops_* per pod, reported together by heal_probe_loops_rules.json); the folders
sit beside the carve's. Stages:

  evaluate  the rules' arms on the eval documents.
  state     the state check in parts, for each of the rules' ways of routing.
  generate  --arm (one of the rules' generate arms): the GSM8K generations.
  report    (local) one summary, with the healing run's and the dense model's generations scored
            the same way.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
from h0_mlp_probe import deadline, group_mean, log, paired_interval, save, sha, token_kl

ROOT = mc.ROOT
RULES = ROOT / 'heal_probe_rules.json'
FOLDER = 'probe'
CODE = ['heal_probe.py', 'slim_heal.py', 'slim_batch.py', 'moe_carve.py', 'h0_mlp_probe.py', 'test_heal_probe.py']


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
    # A copy: on the CPU, state() shares the live parameters' storage.
    slim.adapters = {False: copy.deepcopy(slim.state()),
                     True: torch.load(sh.out('adapter.pt'), weights_only=True, map_location='cpu')}
    heal(slim, True)
    model.eval()
    return model, slim, sh.Vocab(model), sha(sh.factors_path()) == healed['factors_sha256']


def heal(slim, healed):
    """Give the student the healing run's adapter (True) or its state before healing (False): LoRA
    updates of zero and the carve's routers."""
    slim.load(slim.adapters[healed])


def arm(spec):
    """An evaluate arm: a router name for the healed student, or {name, router, healed, fold}."""
    return {'name': spec, 'router': spec, 'healed': True} if isinstance(spec, str) else spec


def use_fold(slim, spec):
    """Fold the student as spec says (a fold variant, or none), times spec's scale (default 1; -1
    subtracts the fold). The folds were fitted for the learned routers."""
    variant, scale = spec.get('fold'), spec.get('scale', 1.0)
    if variant is not None and (spec['router'] != 'learned' or variant not in getattr(slim, 'folds', ())):
        raise ValueError(f'No {variant} fold for the {spec["router"]} router')
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f'Fold scale {scale!r} is not a number')
    slim.folded, slim.fold_scale = variant, float(scale)


def route(slim, mode):
    """Set how the student picks its experts: 'learned' (its routers), 'perfect' (each token's K
    experts with the most contribution, from its own activations), 'all' (every neuron) or 'soft'
    (the learned routers with a soft edge, see soft). Recall is recorded as 1 for perfect and all,
    which run every perfect expert."""
    vars(slim).pop('select', None)
    if mode == 'learned':
        return
    if mode not in ('perfect', 'all', 'soft'):
        raise ValueError(mode)
    if mode == 'soft' and getattr(slim, 'width', None) is None:
        raise ValueError('The soft edge needs its widths (edge)')

    def select(i, a, x):
        b, t, n = a.shape
        if mode == 'soft':
            return soft(slim, i, a, x)
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


def soft(slim, i, a, x):
    """(B, T, N) neuron weights with a soft edge: the learned routers' K experts, each weighted by
    min(1, (its score - the best score left out) / the layer's width); shared neurons weigh 1. Two
    experts swapping places at the edge both weigh about 0, so a tiny change no longer switches a
    whole expert on or off. Recall is recorded as the learned router's."""
    b, t, n = a.shape
    k, cluster = mc.at(slim.s, 'K', i), slim.s['cluster'][i]
    scores = slim.routers[i](x.flatten(0, 1).float())
    edge = scores.topk(k + 1, -1).values[:, -1:]
    weight = ((scores - edge) / slim.width[i]).clamp(0, 1)
    keep = torch.where(cluster < 0, 1.0, weight[:, cluster.clamp_min(0)]).view(b, t, n)
    if t > 1:
        keep[:, 0] = 1
        if slim.recall is not None:
            mass = mc.expert_mass(slim.s, i, (a[:, 1:].float().abs() * slim.wnorm[i]).flatten(0, 1))
            picked = (weight > 0).view(b, t, -1)[:, 1:].flatten(0, 1)
            slim.recall.append(float((mc.topmask(mass, k) & picked).sum() / (mass.shape[0] * k)))
    return keep.to(a.dtype)


@torch.no_grad()
def edge_widths(model, slim, docs, cfg):
    """Each layer's soft-edge width: the cfg['quantile'] quantile, on cfg['docs_per_domain']
    calibration documents per domain, of the gap between the learned routers' K-th and (K+1)-th
    score."""
    gaps = [[] for _ in model.model.layers]

    def select(i, a, x):
        top = slim.routers[i](x[:, 1:].flatten(0, 1).float()).topk(mc.at(slim.s, 'K', i) + 1, -1).values
        gaps[i].append((top[:, -2] - top[:, -1]).cpu())
        return sh.Slim.select(slim, i, a, x)

    slim.select, slim.on = select, True
    for doc in sb.docs_of(docs, 'calib', cfg['docs_per_domain']):
        sb.hidden(model, [doc['input_ids']])
    slim.on = False
    vars(slim).pop('select')
    return [float(torch.cat(g).quantile(cfg['quantile'])) for g in gaps]


def edge(model, slim, docs, rules):
    """Give the student its soft-edge widths when the rules have a soft edge: worked out by the first
    stage that needs them, then read from soft_edge.json."""
    if 'soft' not in rules:
        return
    if not out('soft_edge.json').exists():
        widths = edge_widths(model, slim, docs, rules['soft'])
        save(dict(rules['soft'], width=widths), fresh('soft_edge.json'))
        log('Soft-edge widths by layer: ' + ', '.join(f'{w:.3f}' for w in widths))
    slim.width = json.loads(out('soft_edge.json').read_text())['width']


FOLDS = {'fold': 'the fold', 'gains': 'expert gains', 'gain_shared': 'a shared-expert gain'}
FITS = {'together': '', 'in_order': ' fitted in order', 'track': ' tracking every neuron'}


class Fold(torch.nn.Module):
    """A student's MLP down projection with a fold of the skipped neurons into the neurons that run.
    Variant 'fold' adds delta (out, N) times the neurons that run: each neuron's output weights gain a
    mix of the others'. The gain variants scale each neuron's output by its expert's gain, extra[v]
    being the gain minus 1 by neuron ('gains': the shared expert and every routed expert,
    'gain_shared': the shared expert only). Only the routed student folds (owner.on, with
    owner.folded naming a variant), and not at the first position of a multi-token call, where every
    neuron runs. owner.fold_scale multiplies what the fold adds: -1 subtracts it, and for the gains
    it scales each gain's distance from 1."""

    def __init__(self, inner, owner, delta, extra):
        super().__init__()
        self.inner, self.owner, self.delta, self.extra = inner, owner, delta, extra

    def forward(self, a):
        y = self.inner(a)
        variant = getattr(self.owner, 'folded', None) if self.owner.on else None
        if variant is None:
            return y
        z = F.linear(a, self.delta.to(a.dtype)) if variant == 'fold' else self.inner(a * self.extra[variant].to(a.dtype))
        z = z * getattr(self.owner, 'fold_scale', 1.0)
        if a.dim() == 3 and a.shape[1] > 1:
            z[:, 0] = 0
        return y + z


class Moments:
    """Sums over tokens for one layer: the Gram matrix of the neurons that run (N, N), their products
    with the lost output (N, out), the squared lost and every-neuron outputs, and the tokens."""

    def __init__(self, n, d, device):
        self.g = torch.zeros(n, n, device=device)
        self.c = torch.zeros(n, d, device=device)
        self.lost = self.full = 0.0
        self.tokens = 0

    def add(self, run, lost, full):
        self.g.addmm_(run.T, run)
        self.c.addmm_(run.T, lost)
        self.lost += float(lost.square().sum())
        self.full += float(full.square().sum())
        self.tokens += run.shape[0]


class _Stop(Exception):
    """Ends a forward pass once its last layer of interest is recorded."""


def inner_down(model, i):
    """Layer i's student down projection (with its LoRA update), without a fold."""
    down = model.model.layers[i].mlp.down_proj
    return down.inner if isinstance(down, Fold) else down


def until_stop(model, ids):
    try:
        sb.hidden(model, ids)
    except _Stop:
        pass


@torch.no_grad()
def fold_moments(model, slim, docs, layers, batch, folded=None):
    """{layer: Moments} on docs (equal lengths, batch at a time) from the routed student's own MLP
    inputs, past the first position: the activations of the neurons the learned routers run, and the
    output of the student's down projection (with its LoRA update) from the skipped ones (lost) and
    from all of them (full). The student is unfolded, or folded as slim.folds go (folded='fold')."""
    n = slim.s['cluster'][0].numel()
    moments = {i: Moments(n, model.config.hidden_size, mc.DEVICE) for i in layers}

    def select(i, a, x):
        keep = sh.Slim.select(slim, i, a, x)
        if i in moments:
            down = inner_down(model, i)
            a1 = a[:, 1:].flatten(0, 1)
            run = a1 * keep[:, 1:].flatten(0, 1)
            moments[i].add(run.float(), down(a1 - run).float(), down(a1).float())
            if i == max(layers):
                raise _Stop
        return keep

    slim.select, slim.on, slim.folded, slim.fold_scale = select, True, folded, 1.0
    try:
        for k in range(0, len(docs), batch):
            until_stop(model, [doc['input_ids'] for doc in docs[k:k + batch]])
    finally:
        slim.on, slim.folded = False, None
        vars(slim).pop('select')
    return moments


@torch.no_grad()
def track_moments(model, slim, docs, layers, batch, folded='fold'):
    """{layer: Moments} like fold_moments, toward the healed student with every neuron instead: the
    lost output is the gap, at the layer's output, between that student and the routed one (folded
    as slim.folds go), so it includes the drift from the layers before; full is the every-neuron
    student's MLP output."""
    n = slim.s['cluster'][0].numel()
    moments = {i: Moments(n, model.config.hidden_size, mc.DEVICE) for i in layers}
    captured = {}

    def output(i):
        def hook(module, args, out):
            captured[i] = (out[0] if isinstance(out, tuple) else out)[:, 1:].flatten(0, 1).float()
            if i == max(layers):
                raise _Stop
        return hook

    def every(i, a, x):
        if i in moments:
            captured['full', i] = inner_down(model, i)(a[:, 1:].flatten(0, 1)).float()
        return torch.ones_like(a, dtype=torch.bool)

    def routed(i, a, x):
        keep = sh.Slim.select(slim, i, a, x)
        if i in moments:
            captured['run', i] = (a[:, 1:].flatten(0, 1) * keep[:, 1:].flatten(0, 1)).float()
        return keep

    hooks = [model.model.layers[i].register_forward_hook(output(i)) for i in layers]
    slim.on, slim.fold_scale = True, 1.0
    try:
        for k in range(0, len(docs), batch):
            ids = [doc['input_ids'] for doc in docs[k:k + batch]]
            slim.select, slim.folded = every, None
            until_stop(model, ids)
            target = {i: captured.pop(i) for i in layers}
            slim.select, slim.folded = routed, folded
            until_stop(model, ids)
            for i in layers:
                moments[i].add(captured.pop(('run', i)), target.pop(i) - captured.pop(i), captured.pop(('full', i)))
    finally:
        for hook in hooks:
            hook.remove()
        slim.on, slim.folded = False, None
        vars(slim).pop('select', None)
    return moments


@torch.no_grad()
def down_weight(down):
    """The student's (out, N) down weight: the frozen one plus its LoRA update."""
    if down.u is not None:
        raise ValueError('A factored down projection')
    w = down.base.weight.float()
    return w if down.a is None else w + down.b.float() @ down.a.float() * down.scale


@torch.no_grad()
def fit_layer(fit, held, w, group, rhos):
    """One layer's folds, from its moments on the fit and held-out tokens.

    The fold is the change delta (out, N) to the down weight w (out, N) that best predicts the lost
    output from the neurons that run: ridge regression toward no change, so each neuron keeps its own
    output weights unless the data say otherwise. The penalty is a share rho of the Gram matrix's mean
    diagonal. The gain variants fit the same loss with one number per expert (group (N,) is each
    neuron's expert: 0 shared, 1.. routed) or for the shared expert alone. Each variant keeps the rho
    that leaves the least lost output on the held-out moments, or no change when none leaves less.

    Returns ({'fold': delta, 'gains': extra (N,), 'gain_shared': extra (N,)}, report)."""
    def error(s, g, c):
        """Held-out squared error of the lost output after adding s.T to the down weight."""
        return held.lost - 2 * float((s * c).sum()) + float((s * (g @ s)).sum())

    def best(candidates, err):
        chosen = (held.lost, None, None)
        for rho, x in candidates:
            e = err(x)
            if e < chosen[0]:
                chosen = (e, rho, x)
        return chosen

    # The Gram matrix has no negative eigenvalues; rounding can give it some.
    e, q = torch.linalg.eigh(fit.g)
    e = e.clamp_min(0)
    qc = q.T @ fit.c
    scale = float(fit.g.diagonal().mean())
    solves = ((rho, q @ (qc / (e + rho * scale)[:, None])) for rho in rhos)
    full_err, full_rho, s = best(solves, lambda s: error(s, held.g, held.c))
    delta = torch.zeros_like(w) if s is None else s.T.contiguous()
    del e, q, qc, s

    # With one gain per expert, delta = w diag(p gamma) for the experts' one-hot p (N, E): the loss
    # is quadratic in gamma with these (E, E) and (E,) moments.
    p = F.one_hot(group, int(group.max()) + 1).float()
    h = w.T @ w
    gram = lambda m: p.T @ (m.g * h) @ p
    cross = lambda m: p.T @ (w.T * m.c).sum(1)
    mf, bf, mh, bh = gram(fit), cross(fit), gram(held), cross(held)
    del h
    err_gamma = lambda gamma: held.lost - 2 * float(gamma @ bh) + float(gamma @ mh @ gamma)

    def gains(only_shared):
        m, b = (mf[:1, :1], bf[:1]) if only_shared else (mf, bf)
        eye = torch.eye(len(b), device=b.device)
        for rho in rhos:
            gamma = torch.linalg.solve(m + rho * float(m.diagonal().mean()) * eye, b)
            yield rho, F.pad(gamma, (0, len(bf) - len(gamma)))

    folds, report = {'fold': delta}, {'lost_share': held.lost / held.full, 'tokens': fit.tokens,
                                      'held_tokens': held.tokens}
    report['fold'] = {'rho': full_rho, 'recovered': 1 - full_err / held.lost,
                      'change': float(delta.norm() / w.norm()),
                      'column_change_median': float((delta.norm(dim=0) / w.norm(dim=0).clamp_min(1e-12)).median())}
    for name, only_shared in (('gains', False), ('gain_shared', True)):
        err, rho, gamma = best(gains(only_shared), err_gamma)
        gamma = torch.zeros_like(bf) if gamma is None else gamma
        folds[name] = gamma[group]
        routed = 1 + gamma[1:]
        report[name] = {'rho': rho, 'recovered': 1 - err / held.lost, 'shared_gain': 1 + float(gamma[0])}
        if not only_shared:
            report[name].update(routed_gain_median=float(routed.median()), routed_gain_min=float(routed.min()),
                                routed_gain_max=float(routed.max()))
    return folds, report


# Kept on the pod beside the run folder, not synced: 3.2 GB, and refitted in minutes.
def fold_path():
    return mc.OUT / f'{FOLDER}_fold.pt'


def install(model, slim, weights):
    """Wrap every layer's down projection in a Fold with the given weights; the student starts unfolded."""
    for i, layer in enumerate(model.model.layers):
        m = layer.mlp
        inner = m.down_proj.inner if isinstance(m.down_proj, Fold) else m.down_proj
        extra = {k: weights[k][i].to(mc.DEVICE) for k in FOLDS if k != 'fold'}
        m.down_proj = Fold(inner, slim, weights['fold'][i].to(mc.DEVICE), extra)
    slim.folds, slim.folded, slim.fold_scale = tuple(FOLDS), None, 1.0


def fit_folds(model, slim, docs, cfg):
    """Every layer's folds and report: fitted on cfg['fit_per_domain'] calibration documents per
    domain, each penalty chosen on the next cfg['held_per_domain'], cfg['group'] layers per pass.
    cfg['fit'] (FITS) says what each layer is fitted on:
      together  (default) the unfolded student's inputs, so no fold is fitted for another;
      in_order  the student's inputs with every earlier group already folded;
      track     the same, one layer at a time, toward the healed student with every neuron
                (track_moments): each fold also makes up for the drift the layers before leave."""
    kind = cfg.get('fit', 'together')
    if kind not in FITS or (kind == 'track' and cfg['group'] != 1):
        raise ValueError(f'No {kind} fit in groups of {cfg["group"]}')
    fit_docs = sb.docs_of(docs, 'calib', cfg['fit_per_domain'])
    held_docs = sb.docs_of(docs, 'calib', cfg['held_per_domain'], start=cfg['fit_per_domain'])
    weights, report = {k: [] for k in FOLDS}, []
    layers = list(range(len(model.model.layers)))
    if kind != 'together':
        # Zero folds, filled in as each group is fitted, fold the student for the groups after it.
        n, d = slim.s['cluster'][0].numel(), model.config.hidden_size
        install(model, slim, {'fold': [torch.zeros(d, n, dtype=torch.bfloat16) for _ in layers],
                              **{k: [torch.zeros(n) for _ in layers] for k in FOLDS if k != 'fold'}})
    moments = lambda chosen, group: (track_moments if kind == 'track' else fold_moments)(
        model, slim, chosen, group, cfg['batch'], folded=None if kind == 'together' else 'fold')
    for start in range(0, len(layers), cfg['group']):
        group = layers[start:start + cfg['group']]
        fit, held = moments(fit_docs, group), moments(held_docs, group)
        for i in group:
            w = down_weight(inner_down(model, i))
            folds, row = fit_layer(fit.pop(i), held.pop(i), w, slim.s['cluster'][i] + 1, cfg['rhos'])
            for k in FOLDS:
                weights[k].append(folds[k].to(torch.bfloat16 if k == 'fold' else torch.float32).cpu())
            if kind != 'together':
                model.model.layers[i].mlp.down_proj.delta = weights['fold'][-1].to(mc.DEVICE)
            report.append(dict(row, layer=i))
            log(f'layer {i}: {"gap to every neuron" if kind == "track" else "lost"} {row["lost_share"]:.3f} of the output; recovered by the fold '
                f'{row["fold"]["recovered"]:.3f} (rho {row["fold"]["rho"]}), by expert gains '
                f'{row["gains"]["recovered"]:.3f}, by a shared gain {row["gain_shared"]["recovered"]:.3f} '
                f'(gain {row["gain_shared"]["shared_gain"]:.3f})')
            del w, folds
        del fit, held
        if torch.cuda.is_available():
            log(f'GPU memory: {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, '
                f'{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB at most so far')
    return weights, report


def fold(model, slim, docs, rules):
    """Give the student its folds when the rules have them: fitted by the first stage that needs them
    (fit_folds), then read from fold_path()."""
    if 'fold' not in rules:
        return
    cfg = rules['fold']
    if not out('fold.json').exists():
        t0 = time.time()
        with deadline(cfg['max_hours']):
            weights, report = fit_folds(model, slim, docs, cfg)
        torch.save(weights, fold_path())
        save(dict(cfg, layers=report, weights_sha256=sha(fold_path()), seconds=time.time() - t0), fresh('fold.json'))
        del weights
    if sha(fold_path()) != json.loads(out('fold.json').read_text())['weights_sha256']:
        raise ValueError(f'{fold_path().name} is not the fitted fold')
    install(model, slim, torch.load(fold_path(), weights_only=True, map_location='cpu'))


def repeated(tokens, window, last):
    """Share of the window-token spans in the last tokens that repeat an earlier span."""
    tail = tokens[-last:]
    spans = [tuple(tail[i:i + window]) for i in range(len(tail) - window + 1)]
    return 1 - len(set(spans)) / len(spans) if spans else 0.0


def stage_evaluate(args):
    rules, heal_rules, docs = setup()
    cfg = rules['evaluate']
    specs = [arm(a) for a in cfg['arms']]
    if (specs[0]['router'], specs[0]['healed']) != ('learned', True):
        raise ValueError('The healed learned arm runs first, to check the rebuilt student')
    last = json.loads(sh.out('heal.json').read_text())['milestones'][-1]
    t0 = time.time()
    model, slim, vocab, same = load_healed(heal_rules, docs)
    log(f'Student loaded; factors {"identical to" if same else "differ from"} the healing run\'s')
    edge(model, slim, docs, rules)
    fold(model, slim, docs, rules)
    evals = sb.docs_of(docs, 'eval')
    rows, arms = {}, {}
    with deadline(cfg['max_hours']):
        for spec in specs:
            mode = spec['name']
            route(slim, spec['router'])
            heal(slim, spec['healed'])
            use_fold(slim, spec)
            rows[mode] = sh.evaluate(model, slim, vocab, evals)
            with fresh(f'eval_{mode}.jsonl').open('w') as stream:
                stream.write(''.join(json.dumps(r) + '\n' for r in rows[mode]))
            arms[mode] = sh.summarize(rows[mode], heal_rules['data']['held_out'])
            log(f'{mode}: KL {arms[mode]["kl"]:.3f} (in-mix {arms[mode]["in_mix"]["kl"]:.3f}, held-out '
                f'{arms[mode]["held_out"]["kl"]:.3f}), 64K bits per byte {arms[mode]["bpb_student"]:.4f}, '
                f'agreement {arms[mode]["agree"]:.3f}')
            if spec is specs[0]:
                check = {'kl': arms[mode]['kl'] - last['kl'], 'bpb': arms[mode]['bpb_student'] / last['bpb_student'] - 1}
                if abs(check['kl']) > cfg['check']['kl'] or abs(check['bpb']) > cfg['check']['bpb']:
                    raise RuntimeError(f'The rebuilt student does not reproduce the healed one: {check}')
    route(slim, 'learned')
    heal(slim, True)
    use_fold(slim, {'router': 'learned'})
    report = json.loads(mc.RULES.read_text())['report']
    text = {m: [r for r in rs if r['domain'] != 'copy'] for m, rs in rows.items()}
    differences = {}
    names = [s['name'] for s in specs]
    # Each arm against the one before it, or every arm against the rules' baseline arm.
    base = cfg.get('baseline')
    pairs = [(n, base) for n in names if n != base] if base else list(zip(names[1:], names))
    for new, base in pairs:
        diff, low, high = paired_interval([r['kl']['student'] for r in text[new]], [r['kl']['student'] for r in text[base]],
                                          1.0, report['draws'], report['seed'], [r['domain'] for r in text[new]])
        differences[f'{new}-{base}'] = {'difference': diff, 'low': low, 'high': high}
    save(dict(provenance(), factors_identical=same, reproduces_healed=check, arms=arms, differences=differences,
              seconds=time.time() - t0), fresh('evaluate.json'))


@torch.no_grad()
def stage_state(args):
    rules, heal_rules, docs = setup()
    cfg = rules['state']
    routers = cfg.get('arms', ['learned'])
    t0 = time.time()
    model, slim, _, _ = load_healed(heal_rules, docs)
    edge(model, slim, docs, rules)
    rows = []
    kl = lambda p, q: float(token_kl(p, q)[1:].mean())
    with deadline(cfg['max_hours']):
        for doc in sb.docs_of(docs, 'eval', cfg['docs_per_domain']):
            ids = doc['input_ids']
            slim.on = False
            dense = model.lm_head(sb.hidden(model, [ids])[0]).float()
            row = {'id': doc['id'], 'domain': doc['domain'], 'arms': {}}
            for name in routers:
                route(slim, name)
                slim.on = True
                logits = {'chunked': model.lm_head(sb.hidden(model, [ids])[0]).float()}
                for variant in cfg['variants']:
                    with sb.StoredState(model, [variant], cfg['seed']):
                        logits[variant] = model.lm_head(sb.hidden(model, [ids])[0]).float()
                slim.on = False
                row['arms'][name] = {'from_dense': {k: kl(dense, v) for k, v in logits.items()},
                                     'between': {f'{a}|{b}': kl(logits[a], logits[b]) for a, b in cfg['pairs']}}
                del logits
            route(slim, 'learned')
            rows.append(row)
            del dense
    fresh('state.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    groups = [r['domain'] for r in rows]
    mean = lambda name, part: {k: group_mean([r['arms'][name][part][k] for r in rows], groups)
                               for k in rows[0]['arms'][name][part]}
    summary = {'docs': len(rows), 'arms': {n: {p: mean(n, p) for p in ('from_dense', 'between')} for n in routers}}
    for name, s in summary['arms'].items():
        log(f'{name}: from dense ' + ', '.join(f'{k} {v:.4f}' for k, v in s['from_dense'].items()) +
            '; between ' + ', '.join(f'{k} {v:.2e}' for k, v in s['between'].items()))
    save(dict(provenance(), **summary, seconds=time.time() - t0), fresh('state.json'))


def best_scale(rules, name):
    """For a generate arm whose scale is 'best': the scale of the evaluate arm with its fold that
    drifted least, if that is at least the rules' margin below the healed learned arm without a fold
    (any, with a margin of None) and no other generate arm already has it; None otherwise."""
    spec, arms = rules['generate']['arms'][name], json.loads(out('evaluate.json').read_text())['arms']
    specs = [arm(a) for a in rules['evaluate']['arms']]
    base = next(s['name'] for s in specs if (s['router'], s['healed'], s.get('fold')) == ('learned', True, None))
    pick = min((s for s in specs if s['router'] == spec['router'] and s['healed'] and s.get('fold') == spec['fold']),
               key=lambda s: arms[s['name']]['kl'])
    scale, kl = pick.get('scale', 1.0), arms[pick['name']]['kl']
    margin = rules['generate']['best']['margin']
    if margin is not None and kl > arms[base]['kl'] - margin:
        log(f'{name}: no fold scale drifts less than {base} by the margin (best {pick["name"]}, KL {kl:.3f})')
        return None
    if any(n != name and (s.get('fold'), s.get('scale', 1.0), s['state']) == (spec['fold'], scale, spec['state'])
           for n, s in rules['generate']['arms'].items()):
        log(f'{name}: the best scale, {scale:g}, is another generate arm\'s')
        return None
    log(f'{name}: scale {scale:g} ({pick["name"]}, KL {kl:.3f})')
    return scale


def recommended(scores, generated, cfg):
    """Scores (B, V) filtered for sampling as the model card recommends (cfg: presence_penalty,
    temperature, top_k, top_p): every token already generated loses presence_penalty, then only the
    top_k best are kept, and of those the fewest whose probabilities sum to top_p."""
    if generated.shape[1]:
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
        scores = scores - cfg['presence_penalty'] * seen
    scores = scores / cfg['temperature']
    scores = scores.masked_fill(scores < scores.topk(cfg['top_k'], -1).values[:, -1:], -torch.inf)
    order = scores.sort(-1, descending=True)
    probs = order.values.softmax(-1)
    return scores.scatter(1, order.indices, order.values.masked_fill(probs.cumsum(-1) - probs > cfg['top_p'], -torch.inf))


def prompt_set(rules, heal_rules):
    """The rules' GSM8K prompts file and count, or the healing run's."""
    cfg, gen = rules['generate'], heal_rules['generate']
    return cfg.get('prompts', gen['prompts']), cfg.get('count', gen['count'])


def stage_generate(args):
    from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
    rules, heal_rules, docs = setup()
    spec, gen = dict(rules['generate']['arms'][args.arm]), heal_rules['generate']
    if spec.get('scale') == 'best':
        spec['scale'] = best_scale(rules, args.arm)
        if spec['scale'] is None:
            return
    healed = spec.get('healed', True)
    sampling = rules['generate']['sampling'] if spec.get('sample') else None
    source, count = prompt_set(rules, heal_rules)
    prompts = json.loads((ROOT / source).read_text())['prompts'][:count]
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    model, slim, vocab, _ = load_healed(heal_rules, docs)
    edge(model, slim, docs, rules)
    fold(model, slim, docs, rules)
    route(slim, spec['router'])
    heal(slim, healed)
    use_fold(slim, spec)
    stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]
    removed = torch.full((model.config.vocab_size,), -torch.inf, device=mc.DEVICE)
    removed[vocab.kept] = 0

    class KeptRows(LogitsProcessor):
        """The 64K kept rows, and with sampling the recommended filter on the tokens after start."""
        start = 0

        def __call__(self, input_ids, scores):
            scores = scores + removed
            return scores if sampling is None else recommended(scores, input_ids[:, self.start:], sampling)

    kept = KeptRows()
    # With sampling the filter does all the work, so generate samples from what it leaves.
    decode = (dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0) if sampling else
              dict(do_sample=False, temperature=None, top_p=None, top_k=None))

    path = out(f'generate_{args.arm}.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['id'] for line in path.open()} if path.exists() else set()
    slim.on = True
    with deadline(rules['generate']['max_hours']):
        for index, p in enumerate(prompts):
            if p['id'] in done:
                continue
            text = tok.apply_chat_template([{'role': 'user', 'content': p['question']}], tokenize=False,
                                           add_generation_prompt=True)
            ids = vocab.short.encode(text)
            kept.start = len(ids)
            torch.manual_seed(gen['seed'] + index)
            t0 = time.time()
            with torch.no_grad(), sb.StoredState(model, [spec['state']], gen['seed']):
                x = sb.tensor([ids])
                result = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=gen['max_new_tokens'],
                                        **decode, eos_token_id=stop, pad_token_id=stop[1],
                                        logits_processor=LogitsProcessorList([kept]))
            tokens = result[0, len(ids):].tolist()
            end = next((i for i, t in enumerate(tokens) if t in stop), None)
            tokens = tokens[:end] if end is not None else tokens
            answer = sb.final_number(tok.decode(tokens))
            row = {'id': p['id'], 'router': spec['router'], 'state': spec['state'], 'healed': healed,
                   'fold': spec.get('fold'), 'scale': spec.get('scale', 1.0),
                   'fit': rules['fold'].get('fit', 'together') if spec.get('fold') else None,
                   'sample': sampling is not None, 'prompts': source, 'seconds': time.time() - t0,
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
    finished = [r['length'] for r in rows if r['finished']]
    return {'prompts': len(rows), 'correct': sum(r['correct'] for r in rows), 'finished': len(finished),
            'mean_length': float(np.mean([r['length'] for r in rows])), 'loops': len(stuck), 'looping': stuck,
            'finished_median_length': float(np.median(finished)) if finished else None}


def label(row):
    routing = {'learned': 'learned router', 'perfect': 'perfect router', 'all': 'every neuron',
               'soft': 'learned router with a soft edge'}[row['router']]
    if row.get('fold'):
        scale = row.get('scale', 1.0)
        routing += (f' with {FOLDS[row["fold"]]}{FITS[row.get("fit") or "together"]}' +
                    ('' if scale == 1 else ' subtracted' if scale == -1 else f' × {scale:g}'))
    return (f'{"healed" if row.get("healed", True) else "unhealed"}, {routing}, {row["state"]} state' +
            (', sampled' if row.get('sample') else ''))


def stage_report(args):
    rules, heal_rules, _ = setup()
    read = lambda path: json.loads(path.read_text())
    lines = lambda path: [json.loads(line) for line in path.open()]
    report = {}
    if out('evaluate.json').exists():
        e = read(out('evaluate.json'))
        report['evaluate'] = {k: e[k] for k in ('factors_identical', 'reproduces_healed', 'arms', 'differences')}
    if out('state.json').exists():
        report['state'] = {k: v for k, v in read(out('state.json')).items() if k in ('docs', 'from_dense', 'between', 'arms')}
    if out('soft_edge.json').exists():
        report['soft_edge'] = read(out('soft_edge.json'))
    if out('fold.json').exists():
        f = read(out('fold.json'))
        mean = lambda variant, key: float(np.mean([row[variant][key] for row in f['layers']]))
        report['fold'] = {'lost_share': float(np.mean([row['lost_share'] for row in f['layers']])),
                          'recovered': {k: mean(k, 'recovered') for k in FOLDS},
                          'shared_gain': mean('gain_shared', 'shared_gain'), 'layers': f['layers']}
    # The dense model's and the healing run's own generations are on the healing run's prompts.
    sets = {} if 'prompts' in rules['generate'] else {
        'dense, fp32 state': [dict(r['outputs']['fp32'], id=r['id']) for r in lines(sb.out('generate.jsonl'))],
        'healed, learned router, int16 state': lines(sh.out('generate.jsonl'))}
    # Earlier probe folders' generations first, then this rules file's arms.
    paths = [p for folder in rules.get('earlier', []) for p in sorted((mc.OUT / folder).glob('generate_*.jsonl'))]
    for path in paths + [out(f'generate_{name}.jsonl') for name in rules['generate']['arms']]:
        if path.exists():
            rows = lines(path)
            if label(rows[0]) in sets:
                raise ValueError(f'Two sets of generations are both {label(rows[0])}')
            sets[label(rows[0])] = rows
    count = prompt_set(rules, heal_rules)[1]
    report['generate'] = {name: generations(rows[:count], rules['loop']) for name, rows in sets.items()}
    save(dict(provenance(), **report), fresh('report.json'))
    for name, g in report['generate'].items():
        log(f'{name}: {g["correct"]}/{g["prompts"]} right, {g["finished"]} finished, {g["loops"]} loops, '
            f'mean length {g["mean_length"]:.0f}')


def main():
    global RULES, FOLDER
    stages = {'evaluate': stage_evaluate, 'state': stage_state, 'generate': stage_generate, 'report': stage_report}
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=stages)
    ap.add_argument('--arm')
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
    rules = json.loads(RULES.read_text())
    FOLDER = rules.get('folder', FOLDER)
    if args.stage == 'generate' and args.arm not in rules['generate']['arms']:
        ap.error('generate needs --arm, one of ' + ', '.join(rules['generate']['arms']))
    stages[args.stage](args)


if __name__ == '__main__':
    main()
