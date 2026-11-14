"""H0b: does a compact function fitted on the activation manifold replace an MLP block
better than the teacher's own weights restricted to the same width?

Teacher: SmolLM2-1.7B-Instruct (Llama architecture). At each probe layer the MLP maps
its input residual h_in (the post-attention stream) to its output residual h_out. Every
arm reads a k-wide code of h_in and writes a k-wide code of h_out through a declared chart:

  restricted  shared PCA chart; the teacher MLP folded into it, pruned to m neurons by
              importance, down projection ridge-refit (the P0/P1 line, the baseline);
  pca         the same chart; a native gated MLP of the same shape, trained from restricted;
  learned     a learned chart, frozen before any fit, whose decoder reads only the k-wide
              code; a native MLP of the same shape, trained from restricted;
  full        control that reads the full h_in (same m), writing the PCA code.

The primary metric is oracle replace-and-continue KL: the decoded output replaces the
teacher's state at that layer and the untouched teacher finishes. That is a diagnostic;
a native model never decodes between blocks. Position 0 always keeps the teacher state.

Stages: prepare (corpus), freeze, collect (teacher states), fit, evaluate, gate.
"""
import argparse
import copy
import hashlib
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TEACHER = REPO / 'models/manifold-audit/SmolLM2-1.7B-Instruct'
OUT = ROOT / 'results/h0-native-mlp-20261027'
RULES = ROOT / 'h0_rules.json'
CODE = ['h0_mlp_probe.py', 'test_h0_mlp_probe.py', 'allocation_pilot.py', 'local_projection_models.py',
        'audit_rotation.py']
CHAT = REPO / 'target/manifold-audit/ultrachat/data/train_sft-00000-of-00003-a3ecf92756993583.parquet'
PROSE = REPO / 'ternary-embedder/training-v2/work/projection/texts.json'
HORIZON = ROOT / 'results/correctness-healing-20261025/data.json'
DOMAINS = ('chat', 'prose', 'code', 'math')
SPLITS = ('train', 'dev', 'test')
ARMS = ('restricted', 'pca', 'learned', 'full')
COPY_PROMPT = '\n\nRepeat the passage above exactly.\n\n'


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    tmp.replace(path)


def corpus_rules_sha(rules):
    return hashlib.sha256(json.dumps(rules['corpus'], sort_keys=True).encode()).hexdigest()


def key(salt, domain, ident):
    return hashlib.sha256(f'{salt}\0{domain}\0{ident}'.encode()).hexdigest()


# ---------------------------------------------------------------- corpus

def candidates():
    """(domain, id, messages or text) for every source item, in source order."""
    import pyarrow.parquet as pq
    for row in pq.read_table(CHAT, columns=['prompt_id', 'messages']).to_pylist():
        m = row['messages'][:2]
        if len(m) == 2 and [x['role'] for x in m] == ['user', 'assistant']:
            yield 'chat', row['prompt_id'], m
    for i, item in enumerate(json.loads(PROSE.read_text())):
        text = max(item, key=len) if isinstance(item, list) else item
        yield 'prose', f'prose-{i}', text
    for row in json.loads(HORIZON.read_text())['train']:
        domain = 'code' if row['kind'].startswith('code') else 'math'
        yield domain, row['id'], row['messages'][:2]


def encode(tokenizer, item):
    if isinstance(item, str):
        return tokenizer(item, add_special_tokens=False)['input_ids']
    text = tokenizer.apply_chat_template(item, tokenize=False)
    return tokenizer(text, add_special_tokens=False)['input_ids']


def build_corpus(items, tokenizer, rules):
    """Deterministic document-level splits with per-domain token quotas, plus copy-stress docs.

    Each domain's items are ordered by a salted hash; the hash also picks the split (one in
    ten test, one in ten dev), so no document can land in two splits. Copy docs repeat a
    prose passage that is in no split; copy_span marks the second copy.
    """
    c = rules['corpus']
    salt, low, high = c['salt'], c['min_tokens'], c['max_tokens']
    pools = {d: [] for d in DOMAINS}
    seen = set()
    for domain, ident, item in items:
        digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            pools[domain].append((key(salt, domain, ident), ident, item))
    docs = []
    for domain in DOMAINS:
        quota = c['domains'][domain]
        filled = dict.fromkeys(SPLITS, 0)
        pool = sorted(pools[domain])
        rest = []
        for h, ident, item in pool:
            split = {0: 'test', 1: 'dev'}.get(int(h[:8], 16) % 10, 'train')
            if filled[split] >= quota[split]:
                rest.append((h, ident, item))
                continue
            ids = encode(tokenizer, item)
            if len(ids) < low:
                continue
            docs.append({'id': ident, 'domain': domain, 'split': split, 'input_ids': ids[:high]})
            filled[split] += len(docs[-1]['input_ids'])
        if any(filled[s] < quota[s] for s in SPLITS):
            raise ValueError(f'{domain}: quotas not met {filled}')
        if domain == 'prose':
            spare = rest
    sep = tokenizer(COPY_PROMPT, add_special_tokens=False)['input_ids']
    copies = []
    for h, ident, item in spare:
        if len(copies) == c['copy']['docs']:
            break
        ids = encode(tokenizer, item)
        n = c['copy']['min_passage'] + int(h[8:16], 16) % (c['copy']['max_passage'] - c['copy']['min_passage'] + 1)
        if len(ids) < n:
            continue
        start = n + len(sep)
        copies.append({'id': 'copy-' + ident, 'domain': 'copy', 'split': 'test',
                       'input_ids': ids[:n] + sep + ids[:n], 'copy_span': [start, start + n]})
    if len(copies) < c['copy']['docs']:
        raise ValueError('Not enough prose for copy docs')
    return docs + copies


def stage_prepare(args):
    from transformers import AutoTokenizer
    rules = json.loads(RULES.read_text())
    path = OUT / 'corpus.json'
    if path.exists():
        raise ValueError('Corpus exists; refusing to overwrite')
    tokenizer = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    docs = build_corpus(candidates(), tokenizer, rules)
    sources = {str(p.relative_to(REPO)): sha(p) for p in (CHAT, PROSE, HORIZON, TEACHER / 'tokenizer.json',
                                                         TEACHER / 'tokenizer_config.json')}
    counts = {f'{d["domain"]}/{d["split"]}': 0 for d in docs}
    for d in docs:
        counts[f'{d["domain"]}/{d["split"]}'] += len(d['input_ids'])
    save({'corpus_rules_sha256': corpus_rules_sha(rules), 'sources': sources, 'token_counts': counts, 'docs': docs}, path)
    log(f'Saved {len(docs)} docs: {counts}')


# ---------------------------------------------------------------- freeze and provenance

def teacher_files():
    manifest = json.loads((TEACHER / 'download_manifest.json').read_text())
    return {name: info['sha256'] for name, info in manifest['files'].items()}


def stage_freeze(args):
    import transformers
    path = OUT / 'protocol.json'
    if path.exists():
        raise ValueError('Protocol exists; refusing to overwrite')
    corpus = json.loads((OUT / 'corpus.json').read_text())
    if corpus['corpus_rules_sha256'] != corpus_rules_sha(json.loads(RULES.read_text())):
        raise ValueError('Corpus rules changed after the corpus was built')
    for name, digest in teacher_files().items():
        if sha(TEACHER / name) != digest:
            raise ValueError(f'Teacher file differs from its manifest: {name}')
    save({'status': 'frozen', 'reviewed_by': args.reviewed_by, 'frozen_at': time.time(),
          'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'rules_sha256': sha(RULES),
          'corpus_sha256': sha(OUT / 'corpus.json'), 'teacher_sha256': teacher_files(),
          'torch': torch.__version__, 'transformers': transformers.__version__}, path)
    log('Protocol frozen')


def require_frozen():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    if protocol.get('status') != 'frozen':
        raise RuntimeError('Protocol is not frozen')
    for name, digest in protocol['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'{name} changed after the protocol was frozen')
    if sha(RULES) != protocol['rules_sha256'] or sha(OUT / 'corpus.json') != protocol['corpus_sha256']:
        raise RuntimeError('Rules or corpus changed after the protocol was frozen')
    return protocol, json.loads(RULES.read_text())


class deadline:
    """Hard wall-clock bound: SIGALRM interrupts even a long tensor call between bytecodes."""

    def __init__(self, hours):
        self.seconds = hours * 3600

    def __enter__(self):
        def stop(signum, frame):
            raise TimeoutError(f'Stage exceeded {self.seconds / 3600:.2f} h')
        self.old = signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old)


def load_teacher(threads, check=True):
    from transformers import AutoModelForCausalLM
    if check:
        for name, digest in teacher_files().items():
            if sha(TEACHER / name) != digest:
                raise RuntimeError(f'Teacher file differs from its manifest: {name}')
    torch.set_num_threads(threads)
    model, info = AutoModelForCausalLM.from_pretrained(TEACHER, local_files_only=True, dtype=torch.float32,
                                                       output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Invalid teacher load: {info}')
    return model.eval().requires_grad_(False)


# ---------------------------------------------------------------- teacher forwards

def norms(model):
    """Every RMSNorm in stream order: input and post-attention norms per layer, then the final norm."""
    layers = model.model.layers
    return [n for l in layers for n in (l.input_layernorm, l.post_attention_layernorm)] + [model.model.norm]


@torch.no_grad()
def capture(model, ids, sites, norm_inputs=False):
    """Clean teacher forward recording each probe layer's MLP input and output, and every
    decoder layer's call kwargs so a tail can be replayed from any layer.

    Returns logits (T, V), {site: (h_in, h_out)} as (T, d), kwargs per layer and, if
    asked, the input of every norm as (T, d).
    """
    layers = model.model.layers
    states, kwargs, inputs, handles = {}, [None] * len(layers), [], []
    for i, layer in enumerate(layers):
        def keep_kwargs(module, a, kw, i=i):
            kwargs[i] = kw
        handles.append(layer.register_forward_pre_hook(keep_kwargs, with_kwargs=True))
    for s in sites:
        handles.append(layers[s].post_attention_layernorm.register_forward_pre_hook(
            lambda m, a, s=s: states.__setitem__(('in', s), a[0][0])))
        handles.append(layers[s].register_forward_hook(
            lambda m, a, out, s=s: states.__setitem__(('out', s), out[0])))
    if norm_inputs:
        for n in norms(model):
            handles.append(n.register_forward_pre_hook(lambda m, a: inputs.append(a[0][0])))
    try:
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
    finally:
        for h in handles:
            h.remove()
    return logits, {s: (states[('in', s)], states[('out', s)]) for s in sites}, kwargs, inputs


@torch.no_grad()
def tail(model, start, hidden, kwargs):
    """Finish the teacher from layer start with hidden (B, T, d); kwargs come from a clean
    batch-1 forward of the same sequence and broadcast over B."""
    for layer, kw in zip(model.model.layers[start:], kwargs[start:]):
        hidden = layer(hidden, **kw)
    return model.lm_head(model.model.norm(hidden))


def token_kl(teacher_logits, logits):
    """KL(teacher || arm) per position, (..., T)."""
    t = F.log_softmax(teacher_logits.float(), -1)
    return (t.exp() * (t - F.log_softmax(logits.float(), -1))).sum(-1).clamp_min(0)


# ---------------------------------------------------------------- native blocks and charts

class NativeMLP(nn.Module):
    """Gated SiLU MLP on a code: RMSNorm over its input, gate/up to m neurons, down to the output code."""

    def __init__(self, k_in, m, k_out, eps):
        super().__init__()
        self.norm = nn.Parameter(torch.ones(k_in))
        self.register_buffer('eps', torch.tensor(float(eps)))
        self.gate = nn.Linear(k_in, m, bias=False)
        self.up = nn.Linear(k_in, m, bias=False)
        self.down = nn.Linear(m, k_out, bias=False)

    def hidden(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm
        return F.silu(self.gate(x)) * self.up(x)

    def forward(self, x):
        return self.down(self.hidden(x))

    def scalars(self):
        return sum(p.numel() for p in self.parameters())


def mlp(i, hidden, o):
    net = nn.Sequential(nn.Linear(i, hidden), nn.GELU(), nn.Linear(hidden, o))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class Chart(nn.Module):
    """A k-wide chart: PCA coordinates plus a learned correction.

    decode reads only the code, so the chart is a strict k-wide bottleneck. Both maps are
    positively homogeneous (scale the input, the output scales the same way), as RMSNorm
    readers of a residual stream would need. Zero-initialised corrections start it at PCA.
    """

    def __init__(self, basis, hidden):
        super().__init__()
        d, k = basis.shape
        self.register_buffer('basis', basis.clone())
        self.enc = mlp(d, hidden, k)
        self.dec = mlp(k, hidden, d)

    def encode(self, h):
        r = h.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return h @ self.basis + r * self.enc(h / r)

    def decode(self, c):
        s = (c.pow(2).sum(-1, keepdim=True) / self.basis.shape[0]).sqrt().clamp_min(1e-6)
        return c @ self.basis.T + s * self.dec(c / s)


class PCAChart(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.register_buffer('basis', basis.clone())

    def encode(self, h):
        return h @ self.basis

    def decode(self, c):
        return c @ self.basis.T


def shared_basis(moments):
    """Uncentered shared basis: eigenvectors of the mean trace-normalised second moment, descending."""
    c = sum(m / m.trace() for m in moments) / len(moments)
    values, vectors = torch.linalg.eigh(c.double())
    return vectors[:, values.argsort(descending=True)].float()


def rms_ratio(x, basis):
    """Geometric-mean ratio of the true RMS to the RMS of the projected state."""
    kept = (x @ basis).pow(2).sum(-1) / x.shape[-1]
    return torch.exp((0.5 * (x.pow(2).mean(-1).log() - kept.log())).mean()).item()


@torch.no_grad()
def fold(layer, basis, gamma):
    """The teacher MLP read and written through a PCA chart, all neurons kept.

    With h ~ basis @ c, rms_d(h) = gamma * sqrt(k / d) * rms_k(c), so the norm folds into
    a scalar gain sqrt(d / k) / gamma and eps * d / (k * gamma^2). It is exact at k = d, gamma = 1.
    """
    d, k = basis.shape
    norm, teacher = layer.post_attention_layernorm, layer.mlp
    g = norm.weight.float()
    net = NativeMLP(k, teacher.gate_proj.out_features, k, norm.variance_epsilon * d / (k * gamma ** 2))
    net.norm.fill_(math.sqrt(d / k) / gamma)
    net.gate.weight.copy_((teacher.gate_proj.weight * g) @ basis)
    net.up.weight.copy_((teacher.up_proj.weight * g) @ basis)
    net.down.weight.copy_(basis.T @ teacher.down_proj.weight)
    return net


@torch.no_grad()
def gram(net, codes, chunk=4096):
    n = net.gate.out_features
    total = torch.zeros(n, n, dtype=torch.float64)
    for i in range(0, len(codes), chunk):
        phi = net.hidden(codes[i:i + chunk]).double()
        total.addmm_(phi.T, phi)
    return total


@torch.no_grad()
def prune(net, keep, down):
    out = NativeMLP(net.gate.in_features, len(keep), down.shape[0], net.eps)
    out.norm.copy_(net.norm)
    out.gate.weight.copy_(net.gate.weight[keep])
    out.up.weight.copy_(net.up.weight[keep])
    out.down.weight.copy_(down)
    return out


def restricted(layer, basis, gamma, m, codes, ridge):
    """The baseline: fold, then the P1 importance prune with a ridge refit of the down projection."""
    from allocation_pilot import select_and_refit
    full = fold(layer, basis, gamma)
    keep, down = select_and_refit(gram(full, codes), full.down.weight, m, ridge)
    return prune(full, keep, down), keep


@torch.no_grad()
def full_state_init(layer, keep, down):
    """Control reading the full h_in: the teacher's own kept rows and norm, the restricted refit down."""
    norm, teacher = layer.post_attention_layernorm, layer.mlp
    net = NativeMLP(teacher.gate_proj.in_features, len(keep), down.shape[0], norm.variance_epsilon)
    net.norm.copy_(norm.weight.float())
    net.gate.weight.copy_(teacher.gate_proj.weight[keep])
    net.up.weight.copy_(teacher.up_proj.weight[keep])
    net.down.weight.copy_(down)
    return net


def predictor(arm, net, chart):
    """h_in (.., d) -> predicted h_out (.., d). Coded arms see h_in only through chart.encode."""
    if arm == 'full':
        return lambda h: chart.decode(chart.encode(h) + net(h))

    def predict(h):
        c = chart.encode(h)
        return chart.decode(c + net(c))
    return predict


# ---------------------------------------------------------------- fitting

def nmse(predict, x, y, chunk=8192):
    """sum ||yhat - y||^2 / sum ||y - x||^2: the update error; dropping the block scores 1."""
    num = den = 0.
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            num += (predict(x[i:i + chunk]) - y[i:i + chunk]).pow(2).sum().item()
            den += (y[i:i + chunk] - x[i:i + chunk]).pow(2).sum().item()
    return num / den


def train(module, predict, x, y, dev_x, dev_y, cfg, seed, deadline_at=None):
    """AdamW with cosine decay on the wide-space update error; returns the best-on-dev state.

    Only module's own parameters train; frozen charts are passed in through predict.
    """
    gen = torch.Generator().manual_seed(seed)
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['steps'])
    scale = (y - x).pow(2).sum(-1).mean().item()
    best, best_state, curve = nmse(predict, dev_x, dev_y), copy.deepcopy(module.state_dict()), []
    curve.append({'step': 0, 'dev_nmse': best})
    for step in range(1, cfg['steps'] + 1):
        idx = torch.randint(len(x), (cfg['batch'],), generator=gen)
        loss = (predict(x[idx]) - y[idx]).pow(2).sum(-1).mean() / scale
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            value = nmse(predict, dev_x, dev_y)
            curve.append({'step': step, 'dev_nmse': value, 'train_loss': loss.item()})
            if value < best:
                best, best_state = value, copy.deepcopy(module.state_dict())
            if deadline_at and time.monotonic() > deadline_at:
                raise TimeoutError('Fit exceeded its bound')
    module.load_state_dict(best_state)
    return {'best_dev_nmse': best, 'curve': curve}


def train_chart(chart, samples, dev_samples, cfg, seed):
    """Fit the chart on relative reconstruction error, then freeze it."""
    gen = torch.Generator().manual_seed(seed)
    params = list(chart.enc.parameters()) + list(chart.dec.parameters())
    opt = torch.optim.AdamW(params, lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['steps'])

    def error(h):
        return ((chart.decode(chart.encode(h)) - h).pow(2).sum(-1) / h.pow(2).sum(-1)).mean()

    with torch.no_grad():
        best = error(dev_samples).item()
    best_state, curve = copy.deepcopy(chart.state_dict()), [{'step': 0, 'dev_relative_error': best}]
    for step in range(1, cfg['steps'] + 1):
        loss = error(samples[torch.randint(len(samples), (cfg['batch'],), generator=gen)])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            with torch.no_grad():
                value = error(dev_samples).item()
            curve.append({'step': step, 'dev_relative_error': value})
            if value < best:
                best, best_state = value, copy.deepcopy(chart.state_dict())
    chart.load_state_dict(best_state)
    chart.requires_grad_(False)
    return {'best_dev_relative_error': best, 'curve': curve}


# ---------------------------------------------------------------- gate

def group_mean(values, groups=None):
    """Equal-weight mean of per-group means (a single group when groups is None)."""
    values = np.asarray(values, float)
    groups = np.zeros(len(values), int) if groups is None else np.asarray(groups)
    return float(np.mean([values[groups == g].mean() for g in sorted(set(groups.tolist()))]))


def paired_interval(new, base, margin, draws, seed, groups=None):
    """Document bootstrap of mean(new) - margin * mean(base), percentile 95%.

    With groups, documents are resampled within each group and every group counts equally,
    so many short documents from one domain cannot outweigh the others.
    """
    new, base = np.asarray(new, float), np.asarray(base, float)
    groups = np.zeros(len(new), int) if groups is None else np.asarray(groups)
    labels = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    values = np.zeros(draws)
    for g in labels:
        a, b = new[groups == g], base[groups == g]
        idx = rng.integers(0, len(a), (draws, len(a)))
        values += a[idx].mean(1) - margin * b[idx].mean(1)
    values /= len(labels)
    low, high = np.percentile(values, [2.5, 97.5])
    return group_mean(new, groups) - margin * group_mean(base, groups), float(low), float(high)


def classify_site(new, base, copy_new, copy_base, curve_half, rules, groups=None):
    """PASS / FAIL / INCONCLUSIVE for one site under the frozen gate rules."""
    g = rules['gate']
    point, low, high = paired_interval(new, base, g['margin'], g['draws'], g['seed'], groups)
    tail_new, tail_base = (float(np.quantile(c, g['copy_quantile'])) for c in (copy_new, copy_base))
    result = {'difference': point, 'ci95': [low, high], 'baseline_kl': group_mean(base, groups),
              'new_kl': group_mean(new, groups), 'copy_tail': {'new': tail_new, 'baseline': tail_base}}
    gain = (curve_half - result['new_kl']) / curve_half if curve_half > 0 else 0.
    result['curve_half_to_full_improvement'] = gain
    if result['baseline_kl'] < g['min_baseline_kl']:
        result['verdict'] = 'UNASSESSABLE'
    elif tail_new > tail_base:
        result['verdict'] = 'FAIL'
        result['reason'] = 'copy-tail guard'
    elif high < 0:
        result['verdict'] = 'PASS'
    elif low > 0:
        result['verdict'] = 'FAIL'
        result['reason'] = 'margin not met'
    else:
        result['verdict'] = 'INCONCLUSIVE'
    if result['verdict'] in ('INCONCLUSIVE', 'FAIL') and result.get('reason') != 'copy-tail guard' \
            and gain > g['curve_improvement']:
        result['verdict'] = 'INCONCLUSIVE'
        result['reason'] = 'budget: still improving with data'
    return result


def overall(verdicts, required):
    if sum(v == 'PASS' for v in verdicts) >= required:
        return 'PASS'
    if sum(v == 'FAIL' for v in verdicts) >= required:
        return 'FAIL'
    return 'INCONCLUSIVE'


# ---------------------------------------------------------------- stages that need the frozen protocol

def split_docs(corpus, split, domains=DOMAINS):
    return [d for d in corpus['docs'] if d['split'] == split and d['domain'] in domains]


def stage_collect(args):
    """Teacher states for train and dev docs: all positions after 0 at the probe sites,
    strided samples at every norm input, and train second moments at every norm input."""
    protocol, rules = require_frozen()
    corpus = json.loads((OUT / 'corpus.json').read_text())
    states = OUT / 'states'
    if states.exists():
        raise ValueError('States exist; refusing to overwrite')
    states.mkdir()
    with deadline(rules['runtime']['max_hours']['collect']):
        model = load_teacher(rules['runtime']['threads'])
        d, sites = model.config.hidden_size, rules['sites']
        stride = rules['chart']['samples_per_doc']
        n_norms = len(norms(model))
        moments = torch.zeros(n_norms, d, d, dtype=torch.float64)
        for split in ('train', 'dev'):
            docs = split_docs(corpus, split)
            n = sum(len(doc['input_ids']) - 1 for doc in docs)
            pairs = {s: [np.lib.format.open_memmap(states / f'{split}-L{s}-{w}.npy', 'w+', np.float32, (n, d))
                         for w in ('in', 'out')] for s in sites}
            doc_of, samples, row = np.zeros(n, np.int32), [], 0
            start = time.monotonic()
            for j, doc in enumerate(docs):
                _, captured, _, inputs = capture(model, doc['input_ids'], sites, norm_inputs=True)
                t = len(doc['input_ids']) - 1
                for s in sites:
                    pairs[s][0][row:row + t] = captured[s][0][1:].numpy()
                    pairs[s][1][row:row + t] = captured[s][1][1:].numpy()
                doc_of[row:row + t] = j
                row += t
                positions = torch.linspace(1, t, min(stride, t)).long().unique()
                samples.append(torch.stack([x[positions] for x in inputs], 1))
                if split == 'train':
                    for i, x in enumerate(inputs):
                        moments[i] += (x[1:].T @ x[1:]).double()
                if (j + 1) % 25 == 0:
                    log(f'{split}: {j + 1}/{len(docs)} docs, {row} tokens, {time.monotonic() - start:.0f} s')
            for s in sites:
                for a in pairs[s]:
                    a.flush()
            np.save(states / f'{split}-doc.npy', doc_of)
            torch.save(torch.cat(samples), states / f'{split}-samples.pt')
        torch.save(moments, states / 'train-moments.pt')
    save({'protocol_sha256': sha(OUT / 'protocol.json'), 'completed_at': time.time(),
          'files': {p.name: sha(p) for p in sorted(states.iterdir())}}, OUT / 'collect.json')
    log('Collection complete')


def subset(docs, fraction):
    """Indices of the first fraction of each domain's docs, in corpus order (nested across fractions)."""
    chosen = []
    for domain in DOMAINS:
        idx = [i for i, d in enumerate(docs) if d['domain'] == domain]
        chosen += idx[:max(1, math.ceil(fraction * len(idx)))]
    return sorted(chosen)


def load_pairs(split, site, chosen=None):
    x = torch.from_numpy(np.load(OUT / f'states/{split}-L{site}-in.npy'))
    y = torch.from_numpy(np.load(OUT / f'states/{split}-L{site}-out.npy'))
    if chosen is not None:
        rows = torch.from_numpy(np.isin(np.load(OUT / f'states/{split}-doc.npy'), chosen))
        x, y = x[rows], y[rows]
    return x, y


def require_collected():
    protocol, rules = require_frozen()
    record = json.loads((OUT / 'collect.json').read_text())
    for name, digest in record['files'].items():
        if sha(OUT / 'states' / name) != digest:
            raise RuntimeError(f'Collected file changed: {name}')
    return protocol, rules


def stage_fit(args):
    protocol, rules = require_collected()
    corpus = json.loads((OUT / 'corpus.json').read_text())
    fits = OUT / 'fits'
    if fits.exists():
        raise ValueError('Fits exist; refusing to overwrite')
    fits.mkdir()
    with deadline(rules['runtime']['max_hours']['fit']):
        model = load_teacher(rules['runtime']['threads'])
        basis = shared_basis(torch.load(OUT / 'states/train-moments.pt'))
        samples, dev_samples = (torch.load(OUT / f'states/{s}-samples.pt').flatten(0, 1) for s in ('train', 'dev'))
        report, seed = {'charts': {}, 'arms': {}}, rules['seed']
        for budget in rules['budgets']:
            k = budget['k']
            chart = Chart(basis[:, :k], rules['chart']['hidden'])
            report['charts'][budget['name']] = train_chart(chart, samples, dev_samples, rules['chart'], seed)
            torch.save(chart.state_dict(), fits / f'chart-{budget["name"]}.pt')
            log(f'Chart {budget["name"]}: dev relative error {report["charts"][budget["name"]]["best_dev_relative_error"]:.4f}')
        for site in rules['sites']:
            layer = model.model.layers[site]
            x, y = load_pairs('train', site)
            dev_x, dev_y = load_pairs('dev', site)
            for budget in rules['budgets']:
                name, k, m = budget['name'], budget['k'], budget['m']
                p = basis[:, :k]
                pca = PCAChart(p)
                chart = Chart(p, rules['chart']['hidden'])
                chart.load_state_dict(torch.load(fits / f'chart-{name}.pt'))
                chart.requires_grad_(False)
                gamma = rms_ratio(x, p)
                base, keep = restricted(layer, p, gamma, m, x @ p, rules['restricted']['ridge'])
                nets = {'restricted': base, 'pca': copy.deepcopy(base), 'learned': copy.deepcopy(base),
                        'full': full_state_init(layer, keep, base.down.weight.detach())}
                tag = f'L{site}-{name}'
                report['arms'][tag] = {'gamma': gamma, 'scalars': {a: n.scalars() for a, n in nets.items()},
                                       'restricted_dev_nmse': nmse(predictor('restricted', base, pca), dev_x, dev_y)}
                for arm in ('pca', 'learned', 'full'):
                    used = chart if arm == 'learned' else pca
                    report['arms'][tag][arm] = train(nets[arm], predictor(arm, nets[arm], used), x, y,
                                                     dev_x, dev_y, rules['native'], seed)
                    log(f'{tag} {arm}: dev NMSE {report["arms"][tag][arm]["best_dev_nmse"]:.4f}')
                if name == rules['curve']['budget']:
                    for fraction in rules['curve']['fractions']:
                        sx, sy = load_pairs('train', site, subset(split_docs(corpus, 'train'), fraction))
                        net = copy.deepcopy(base)
                        report['arms'][tag][f'learned@{fraction}'] = train(
                            net, predictor('learned', net, chart), sx, sy, dev_x, dev_y, rules['native'], seed)
                        nets[f'learned@{fraction}'] = net
                torch.save({a: n.state_dict() for a, n in nets.items()}, fits / f'{tag}.pt')
                save(report, OUT / 'fit-report.json')
    report.update({'protocol_sha256': sha(OUT / 'protocol.json'), 'completed_at': time.time(),
                   'files': {p.name: sha(p) for p in sorted(fits.iterdir())}})
    save(report, OUT / 'fit-report.json')
    log('Fits complete')


def load_fits(rules, basis, model):
    """{site: {config: predictor}} for every arm, floor and curve fit."""
    fits, out = OUT / 'fits', {}
    for site in rules['sites']:
        # exact replays the teacher's own state: its KL is the replay noise floor.
        out[site] = {'drop': lambda h_in, h_out: h_in, 'exact': lambda h_in, h_out: h_out}
        for budget in rules['budgets']:
            name, k, m = budget['name'], budget['k'], budget['m']
            p = basis[:, :k]
            pca = PCAChart(p)
            chart = Chart(p, rules['chart']['hidden'])
            chart.load_state_dict(torch.load(fits / f'chart-{name}.pt'))
            states = torch.load(fits / f'L{site}-{name}.pt')
            for config, state in states.items():
                arm = config.split('@')[0]
                k_in = model.config.hidden_size if arm == 'full' else k
                net = NativeMLP(k_in, m, k, 0.)
                net.load_state_dict(state)
                out[site][f'{config}/{name}'] = lambda h_in, h_out, f=predictor(arm, net, chart if arm == 'learned' else pca): f(h_in)
            out[site][f'pca_floor/{name}'] = lambda h_in, h_out, c=pca: c.decode(c.encode(h_out))
            out[site][f'chart_floor/{name}'] = lambda h_in, h_out, c=chart: c.decode(c.encode(h_out))
    return out


def evaluate_doc(model, doc, sites, configs, batch):
    """Per-position KL and update-error sums for every config at every site of one document."""
    logits, captured, kwargs, _ = capture(model, doc['input_ids'], sites)
    result = {}
    for site in sites:
        h_in, h_out = captured[site]
        names = list(configs[site])
        for i in range(0, len(names), batch):
            group = names[i:i + batch]
            with torch.no_grad():
                preds = [configs[site][c](h_in, h_out) for c in group]
            hidden = torch.stack([torch.cat([h_out[:1], p[1:]]) for p in preds])
            arm_logits = tail(model, site + 1, hidden, kwargs)
            kl = token_kl(logits[None], arm_logits)[:, 1:]
            for c, p, values in zip(group, preds, kl):
                result[f'L{site}/{c}'] = {'kl': values.tolist(),
                                          'nmse_num': (p[1:] - h_out[1:]).pow(2).sum().item(),
                                          'nmse_den': (h_out[1:] - h_in[1:]).pow(2).sum().item()}
    return result


def stage_evaluate(args):
    protocol, rules = require_collected()
    record = json.loads((OUT / 'fit-report.json').read_text())
    for name, digest in record['files'].items():
        if sha(OUT / 'fits' / name) != digest:
            raise RuntimeError(f'Fit file changed: {name}')
    path = OUT / 'evaluation.jsonl'
    if path.exists():
        raise ValueError('Evaluation exists; refusing to overwrite')
    corpus = json.loads((OUT / 'corpus.json').read_text())
    with deadline(rules['runtime']['max_hours']['evaluate']):
        model = load_teacher(rules['runtime']['threads'])
        configs = load_fits(rules, shared_basis(torch.load(OUT / 'states/train-moments.pt')), model)
        docs = split_docs(corpus, 'test', DOMAINS + ('copy',))
        start = time.monotonic()
        with path.open('w') as f:
            for j, doc in enumerate(docs):
                row = {'id': doc['id'], 'domain': doc['domain'], 'copy_span': doc.get('copy_span'),
                       'results': evaluate_doc(model, doc, rules['sites'], configs, rules['runtime']['eval_batch'])}
                f.write(json.dumps(row) + '\n')
                if (j + 1) % 10 == 0:
                    log(f'test: {j + 1}/{len(docs)} docs, {time.monotonic() - start:.0f} s')
    save({'protocol_sha256': sha(OUT / 'protocol.json'), 'completed_at': time.time(), 'docs': len(docs),
          'evaluation_sha256': sha(path)}, OUT / 'evaluate.json')
    log('Evaluation complete')


def gate_from_rows(rows, rules):
    """Per-site verdicts for the primary arm against the baseline, plus descriptive means."""
    g = rules['gate']
    natural = [r for r in rows if r['domain'] != 'copy']
    copies = [r for r in rows if r['domain'] == 'copy']

    def doc_means(config, site):
        return [float(np.mean(r['results'][f'L{site}/{config}']['kl'])) for r in natural]

    def copy_tokens(config, site):
        values = []
        for r in copies:
            a, b = r['copy_span']
            # KL index j is position j + 1; positions a - 1 .. b - 2 predict the second copy.
            values += r['results'][f'L{site}/{config}']['kl'][a - 2:b - 2]
        return values

    budget, primary, baseline = g['budget'], g['primary_arm'], g['baseline_arm']
    half = rules['curve']['fractions'][-1]
    groups = [r['domain'] for r in natural] if g['weighting'] == 'domain_equal' else None
    sites = {}
    for site in rules['sites']:
        new, base = f'{primary}/{budget}', f'{baseline}/{budget}'
        result = classify_site(doc_means(new, site), doc_means(base, site), copy_tokens(new, site),
                               copy_tokens(base, site), group_mean(doc_means(f'{primary}@{half}/{budget}', site), groups),
                               rules, groups)
        configs = sorted({k.split('/', 1)[1] for k in natural[0]['results'] if k.startswith(f'L{site}/')})
        result['descriptive'] = {}
        for c in configs:
            entries = [r['results'][f'L{site}/{c}'] for r in natural]
            result['descriptive'][c] = {
                'mean_kl': group_mean(doc_means(c, site), groups),
                'p99_token_kl': float(np.quantile(np.concatenate([e['kl'] for e in entries]), 0.99)),
                'nmse_update': sum(e['nmse_num'] for e in entries) / sum(e['nmse_den'] for e in entries),
                'by_domain': {d: float(np.mean([np.mean(r['results'][f'L{site}/{c}']['kl'])
                                                for r in natural if r['domain'] == d]))
                              for d in DOMAINS if any(r['domain'] == d for r in natural)}}
        sites[site] = result
    return {'sites': sites, 'overall': overall([s['verdict'] for s in sites.values()], g['sites_required'])}


def stage_gate(args):
    protocol, rules = require_frozen()
    record = json.loads((OUT / 'evaluate.json').read_text())
    if sha(OUT / 'evaluation.jsonl') != record['evaluation_sha256']:
        raise RuntimeError('Evaluation file changed')
    rows = [json.loads(line) for line in (OUT / 'evaluation.jsonl').open()]
    result = gate_from_rows(rows, rules)
    result.update({'protocol_sha256': sha(OUT / 'protocol.json'), 'evaluation_sha256': record['evaluation_sha256'],
                   'label': 'Exploratory single-site oracle diagnostic; not a deployed-model claim.'})
    save(result, OUT / 'gate.json')
    log(f"Gate: {result['overall']} ({', '.join(f'L{s}: {v['verdict']}' for s, v in result['sites'].items())})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='stage', required=True)
    sub.add_parser('prepare')
    f = sub.add_parser('freeze')
    f.add_argument('--reviewed-by', nargs='+', required=True)
    for stage in ('collect', 'fit', 'evaluate', 'gate'):
        sub.add_parser(stage)
    args = ap.parse_args()
    {'prepare': stage_prepare, 'freeze': stage_freeze, 'collect': stage_collect, 'fit': stage_fit,
     'evaluate': stage_evaluate, 'gate': stage_gate}[args.stage](args)


if __name__ == '__main__':
    main()
