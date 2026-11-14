"""H0 follow-up: is the narrow-chart floor a property of the width or of the chart?

H0b found that squeezing SmolLM2's residual through a 768-wide PCA chart at one layer,
with the true MLP output, costs more downstream KL than deleting the MLP. This probe
compares linear k-wide charts on that floor alone, with no block fits:

  shared  the H0 basis (eigenvectors of the mean trace-normalised second moment, all layers);
  site    PCA of this site's own second moment;
  fisher  the reduced-rank chart minimising E[(h - D E h)^T F (h - D E h)], where F is the
          Fisher information of the teacher's outputs with respect to the state at this site.

Every chart is strict: encode E (k x d), decode D (d x k), E D = I. Fisher is estimated on
a stratified quarter of the H0 train docs; floors are measured on the H0 dev docs, leaving
the H0 test split alone. Exploratory; no gate.

Stages: fisher, floors.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import h0_mlp_probe as h0

OUT = h0.ROOT / 'results/h0-floor-20261028'
CONFIG = {'seed': 26102881, 'fisher_fraction': 0.25, 'widths': [768, 1536], 'dampings': [1e-3, 1e-1],
          'threads': 16, 'eval_batch': 4}


def tail_grad(model, start, hidden, kwargs):
    """h0.tail with gradients enabled, so the state at the site can be differentiated."""
    for layer, kw in zip(model.model.layers[start:], kwargs[start:]):
        hidden = layer(hidden, **kw)
    return model.lm_head(model.model.norm(hidden))


def fisher_sum(model, ids, sites, generator):
    """Sum over positions 1.. of g g^T, g = d log p(y) / d h at each site, y sampled from the teacher.

    Position 0 keeps the teacher state in every replay, so it is excluded on both sides.
    """
    _, captured, kwargs, _ = h0.capture(model, ids, sites)
    result = {}
    for s in sites:
        h = captured[s][1].detach().clone().requires_grad_(True)
        logp = F.log_softmax(tail_grad(model, s + 1, h[None], kwargs)[0].float(), -1)[1:]
        y = torch.multinomial(logp.detach().exp(), 1, generator=generator)
        logp.gather(-1, y).sum().backward()
        g = h.grad[1:].double()
        result[s] = g.T @ g
    return result


def pca_chart(basis, k):
    p = basis[:, :k]
    return p.T.contiguous(), p.contiguous()


def site_basis(moment):
    values, vectors = torch.linalg.eigh(moment.double())
    return vectors[:, values.argsort(descending=True)].float()


def fisher_chart(moment, fisher, k, damping):
    """Encode E (k, d) and decode D (d, k) minimising E||F^1/2 (h - D E h)||^2, with E D = I.

    With M = F^1/2 C F^1/2 and U its top-k eigenvectors: E = U^T F^1/2, D = F^-1/2 U.
    damping adds damping * mean eigenvalue of F to every eigenvalue.
    """
    d = fisher.shape[0]
    f = fisher.double() + damping * fisher.trace() / d * torch.eye(d, dtype=torch.float64)
    w, q = torch.linalg.eigh(f)
    root, inv_root = (q * w.sqrt()) @ q.T, (q * w.rsqrt()) @ q.T
    values, u = torch.linalg.eigh(root @ moment.double() @ root)
    u = u[:, values.argsort(descending=True)[:k]]
    return (u.T @ root).float(), (inv_root @ u).float()


def weighted_residual(moment, fisher, encode, decode):
    """Second-order share of output sensitivity the chart discards: tr(F R C R^T) / tr(F C), R = I - D E."""
    c, f = moment.double(), fisher.double()
    r = torch.eye(c.shape[0], dtype=torch.float64) - decode.double() @ encode.double()
    return ((f @ r @ c @ r.T).trace() / (f @ c).trace()).item()


def charts_for(site, moments, shared, fisher):
    """{name: (encode, decode)} for one site; moments[i] is the train second moment at norm input i."""
    moment = moments[2 * (site + 1)]  # the MLP output of layer s is the input of layer s + 1's first norm
    own = site_basis(moment)
    charts = {}
    for k in CONFIG['widths']:
        charts[f'shared/{k}'] = pca_chart(shared, k)
        charts[f'site/{k}'] = pca_chart(own, k)
        for damping in CONFIG['dampings']:
            charts[f'fisher{damping:g}/{k}'] = fisher_chart(moment, fisher, k, damping)
    return moment, charts


def stage_fisher(args):
    rules = json.loads(h0.RULES.read_text())
    corpus = json.loads((h0.OUT / 'corpus.json').read_text())
    path = OUT / 'fisher.pt'
    if path.exists():
        raise ValueError('Fisher estimate exists; refusing to overwrite')
    docs = h0.split_docs(corpus, 'train')
    chosen = h0.subset(docs, CONFIG['fisher_fraction'])
    model = h0.load_teacher(CONFIG['threads'])
    d, sites = model.config.hidden_size, rules['sites']
    total = {s: torch.zeros(d, d, dtype=torch.float64) for s in sites}
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    tokens, start = 0, time.monotonic()
    for j, i in enumerate(chosen):
        for s, value in fisher_sum(model, docs[i]['input_ids'], sites, generator).items():
            total[s] += value
        tokens += len(docs[i]['input_ids']) - 1
        if (j + 1) % 25 == 0:
            h0.log(f'fisher: {j + 1}/{len(chosen)} docs, {tokens} tokens, {time.monotonic() - start:.0f} s')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({s: v / tokens for s, v in total.items()}, path)
    h0.save({'config': CONFIG, 'docs': len(chosen), 'tokens': tokens, 'corpus_sha256': h0.sha(h0.OUT / 'corpus.json'),
             'fisher_sha256': h0.sha(path)}, OUT / 'fisher.json')
    h0.log('Fisher estimate complete')


def stage_floors(args):
    rules = json.loads(h0.RULES.read_text())
    corpus = json.loads((h0.OUT / 'corpus.json').read_text())
    path = OUT / 'floors.jsonl'
    if path.exists():
        raise ValueError('Floors exist; refusing to overwrite')
    fisher = torch.load(OUT / 'fisher.pt')
    moments = torch.load(h0.OUT / 'states/train-moments.pt')
    shared = h0.shared_basis(moments)
    sites = rules['sites']
    charts, predicted = {}, {}
    for s in sites:
        moment, charts[s] = charts_for(s, moments, shared, fisher[s])
        predicted[s] = {name: weighted_residual(moment, fisher[s], e, dec) for name, (e, dec) in charts[s].items()}
    del moments
    model = h0.load_teacher(CONFIG['threads'])
    docs = h0.split_docs(corpus, 'dev')
    start = time.monotonic()
    with path.open('w') as f:
        for j, doc in enumerate(docs):
            logits, captured, kwargs, _ = h0.capture(model, doc['input_ids'], sites)
            row = {'id': doc['id'], 'domain': doc['domain'], 'kl': {}}
            for s in sites:
                h_in, h_out = captured[s]
                preds = {'drop': h_in} | {n: h_out @ e.T @ dec.T for n, (e, dec) in charts[s].items()}
                names = list(preds)
                for i in range(0, len(names), CONFIG['eval_batch']):
                    group = names[i:i + CONFIG['eval_batch']]
                    hidden = torch.stack([torch.cat([h_out[:1], preds[n][1:]]) for n in group])
                    kl = h0.token_kl(logits[None], h0.tail(model, s + 1, hidden, kwargs))[:, 1:]
                    for n, values in zip(group, kl):
                        row['kl'][f'L{s}/{n}'] = values.mean().item()
            f.write(json.dumps(row) + '\n')
            if (j + 1) % 25 == 0:
                h0.log(f'floors: {j + 1}/{len(docs)} docs, {time.monotonic() - start:.0f} s')
    rows = [json.loads(line) for line in path.open()]
    groups = [r['domain'] for r in rows]
    summary = {}
    for s in sites:
        summary[f'L{s}'] = {}
        for n in ['drop'] + list(charts[s]):
            values = [r['kl'][f'L{s}/{n}'] for r in rows]
            summary[f'L{s}'][n] = {'mean_kl': h0.group_mean(values, groups),
                                   'by_domain': {d: float(np.mean([v for v, g in zip(values, groups) if g == d]))
                                                 for d in h0.DOMAINS},
                                   'predicted_weighted_residual': predicted[s].get(n)}
    h0.save({'config': CONFIG, 'dev_docs': len(rows), 'floors_sha256': h0.sha(path),
             'fisher_sha256': h0.sha(OUT / 'fisher.pt'), 'summary': summary,
             'label': 'Exploratory chart-floor diagnostic on H0 dev docs; no gate.'}, OUT / 'floors.json')
    h0.log('Floors complete')


STACK = {'site': [512, 768, 1024, 1280, 1536, 1792], 'shared': [768, 1536]}


def stack_bases(moments, shared):
    """Per-stream-point PCA bases: the embedding output, then each layer's output (moments index 2(l+1))."""
    points = [0] + [2 * (l + 1) for l in range((len(moments) - 1) // 2)]
    return {'site': [site_basis(moments[i]) for i in points], 'shared': [shared] * len(points)}


@torch.no_grad()
def squeezed_logits(model, ids, bases, k):
    """Forward with the residual squeezed through a k-wide chart at every point of the stream.

    bases[0] squeezes the embedding output, bases[l + 1] the output of layer l. Position 0
    keeps the teacher state, as in every H0 replay.
    """
    def squeeze(h, p):
        return torch.cat([h[:, :1], h[:, 1:] @ p @ p.T], 1)
    def squeeze_input(module, a, kw):
        if a:
            return (squeeze(a[0], p[0]),) + a[1:], kw
        return a, kw | {'hidden_states': squeeze(kw['hidden_states'], p[0])}
    layers = model.model.layers
    p = [b[:, :k] for b in bases]
    handles = [layers[0].register_forward_pre_hook(squeeze_input, with_kwargs=True)]
    for l, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(lambda m, a, out, q=p[l + 1]: squeeze(out, q)))
    try:
        return model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
    finally:
        for h in handles:
            h.remove()


def stage_stack(args):
    """Whole-stack floor: every layer's output (and the embedding) squeezed at once, teacher blocks intact."""
    corpus = json.loads((h0.OUT / 'corpus.json').read_text())
    path = OUT / 'stack.jsonl'
    if path.exists():
        raise ValueError('Stack floors exist; refusing to overwrite')
    moments = torch.load(h0.OUT / 'states/train-moments.pt')
    bases = stack_bases(moments, h0.shared_basis(moments))
    del moments
    model = h0.load_teacher(CONFIG['threads'])
    docs = h0.split_docs(corpus, 'dev')
    start = time.monotonic()
    with path.open('w') as f:
        for j, doc in enumerate(docs):
            clean = model(input_ids=torch.tensor([doc['input_ids']]), use_cache=False).logits[0]
            row = {'id': doc['id'], 'domain': doc['domain'], 'kl': {}}
            for chart, widths in STACK.items():
                for k in widths:
                    logits = squeezed_logits(model, doc['input_ids'], bases[chart], k)
                    row['kl'][f'{chart}/{k}'] = h0.token_kl(clean, logits)[1:].mean().item()
            f.write(json.dumps(row) + '\n')
            if (j + 1) % 25 == 0:
                h0.log(f'stack: {j + 1}/{len(docs)} docs, {time.monotonic() - start:.0f} s')
    rows = [json.loads(line) for line in path.open()]
    groups = [r['domain'] for r in rows]
    summary = {}
    for name in rows[0]['kl']:
        values = [r['kl'][name] for r in rows]
        summary[name] = {'mean_kl': h0.group_mean(values, groups),
                         'by_domain': {d: float(np.mean([v for v, g in zip(values, groups) if g == d]))
                                       for d in h0.DOMAINS}}
    h0.save({'widths': STACK, 'dev_docs': len(rows), 'stack_sha256': h0.sha(path), 'summary': summary,
             'label': 'Exploratory whole-stack chart floor on H0 dev docs; teacher blocks intact; no gate.'},
            OUT / 'stack.json')
    h0.log('Stack floors complete')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('fisher', 'floors', 'stack'))
    args = ap.parse_args()
    {'fisher': stage_fisher, 'floors': stage_floors, 'stack': stage_stack}[args.stage](args)


if __name__ == '__main__':
    main()
