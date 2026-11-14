"""Separate measurements of the slimming steps planned for Qwen3.5-9B, each against the dense
model on the v2 English-and-code benchmark, then all of them at once.

Corpus, model, expert sets and learned routers come from moe_carve with moe_carve_rules_v2.json;
this file's settings and predictions are in slim_rules.json, and its outputs go to the slim/
folder beside the carve's. Stages:

  prompts   (local) GSM8K test questions for the generate stage.
  trim      (local) the 64K vocabulary: special tokens, the byte alphabet, calibration tokens by
            frequency, then tokens made only of English letters, symbols and code characters in
            merge order, closed under merge ancestry so the trimmed BPE still builds every kept
            token. Checks that both tokenizers round-trip the documents.
  vocab     bits per byte of the eval text with the trimmed tokenizer and the output restricted to
            the kept rows, against the full tokenizer and head.
  cache     a recency cache of output rows (every token admitted as it is seen, the least recently
            used evicted): how often the model's top pick is outside it, the probability outside
            it, and a miss detector trained on calibration documents that sends a position to the
            full head.
  state     the DeltaNet recurrent state stored at lower precision between tokens, math in fp32.
  deltanet  fewer DeltaNet parameters: drop the blocks whose removal costs least, or give their
            projections an activation-aware low rank with the same parameter saving.
  combined  all of it at once: co-firing experts at 25% with the learned router, int16 state, the
            chosen DeltaNet cut, the 16K cache with its detector, and the 64K vocabulary.
  generate  long greedy generations with every state precision: length, where each leaves the
            fp32 text, and answers.
  report    (local) one summary, with parameter counts.

Every KL is from the stock dense model (chunked DeltaNet kernels, fp32 state) on the same tokens.
"""
import argparse
import collections
import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import moe_carve as mc
from h0_mlp_probe import deadline, group_mean, log, paired_interval, save, sha, token_kl

ROOT = mc.ROOT
RULES = ROOT / 'slim_rules.json'
CARVE_RULES = ROOT / 'moe_carve_rules_v2.json'
CODE = ['slim_batch.py', 'moe_carve.py', 'h0_mlp_probe.py', 'test_slim_batch.py']
STORES = ('fp32', 'fp32_jitter', 'bf16', 'fp16', 'fp16_sr', 'int16_row', 'int8_row')


def setup():
    mc.configure(CARVE_RULES)
    return json.loads(RULES.read_text()), json.loads((mc.OUT / 'corpus.json').read_text())['docs']


def out(name):
    return mc.OUT / 'slim' / name


def provenance():
    return {'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'rules_sha256': sha(RULES),
            'carve_rules_sha256': sha(CARVE_RULES), 'corpus_sha256': sha(mc.OUT / 'corpus.json')}


def fresh(name):
    if out(name).exists():
        raise ValueError(f'{name} exists; refusing to overwrite')
    out(name).parent.mkdir(parents=True, exist_ok=True)
    return out(name)


def docs_of(docs, split, per_domain=None, start=0):
    """Documents of one split; with per_domain, that many from each domain from index start on."""
    chosen = [d for d in docs if d['split'] == split]
    if per_domain is None:
        return chosen
    return [d for domain in mc.DOMAINS for d in [x for x in chosen if x['domain'] == domain][start:start + per_domain]]


def tensor(ids):
    return torch.tensor(ids, device=mc.DEVICE)


@torch.no_grad()
def hidden(model, ids):
    """Final normed hidden states (B, T, hidden) for a batch of equal-length id lists."""
    return model.model(input_ids=tensor(ids), use_cache=False).last_hidden_state


def score_arms(logits, ids, doc):
    """Row of KL from logits['dense'], NLL and copy-span KL for every arm, as moe_carve scores them."""
    dense, target = logits['dense'], tensor(ids[1:])
    row = {'id': doc['id'], 'domain': doc['domain'], 'kl': {}, 'nll': {}, 'finite': {}}
    if 'copy_span' in doc:
        row['copy_kl'] = {}
    for name, x in logits.items():
        kl = token_kl(dense, x)
        row['kl'][name] = float(kl[1:].mean())
        row['nll'][name] = float(F.cross_entropy(x[:-1], target))
        row['finite'][name] = bool(torch.isfinite(x).all())
        if 'copy_span' in doc:
            start, end = doc['copy_span']
            row['copy_kl'][name] = kl[start - 1:end - 1].tolist()
    return row


def write_rows(name, rows):
    fresh(name).write_text(''.join(json.dumps(r) + '\n' for r in rows))


# ---------------------------------------------------------------- vocabulary

def byte_decoder():
    """Byte-level BPE's printable stand-in character for every byte, inverted."""
    keep = list(range(ord('!'), ord('~') + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars, extra = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(keep, chars)}


def english(text):
    """True for text made only of ASCII, punctuation, symbols, spaces and Greek letters."""
    return all(ord(c) < 128 or unicodedata.category(c)[0] in 'PSZ' or 0x370 <= ord(c) <= 0x3FF for c in text)


def trim_vocab(tokenizer_json, counts, size):
    """Kept token ids in order of admission and how many came from each source.

    counts maps token id to calibration frequency. Every token brings its merge ancestors in
    first; a token whose ancestry would overflow size is skipped."""
    model = tokenizer_json['model']
    vocab = model['vocab']
    inv = {i: t for t, i in vocab.items()}
    pairs = [tuple(m.split(' ', 1)) if isinstance(m, str) else tuple(m) for m in model['merges']]
    parents = {a + b: (a, b) for a, b in pairs}
    special = sorted({a['id'] for a in tokenizer_json['added_tokens']})
    decode = byte_decoder()
    kept, source = [], collections.Counter()
    seen = set()

    def ancestry(token):
        need, stack = [], [token]
        while stack:
            t = stack.pop()
            if t in seen or t in need:
                continue
            need.append(t)
            stack += parents.get(t, ())
        return need

    def admit(token, label):
        if vocab[token] in special:
            return
        need = ancestry(token)
        if not need or len(seen) + len(need) + len(special) > size:
            return
        for t in need:
            seen.add(t)
            kept.append(vocab[t])
            source[label if t == token else 'ancestry'] += 1

    def readable(token):
        try:
            return english(bytes(decode[c] for c in token).decode('utf-8'))
        except UnicodeDecodeError:
            return False

    for token in (t for t in vocab if len(t) == 1):
        admit(token, 'alphabet')
    for i, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if i in inv:
            admit(inv[i], 'calibration')
    for i in sorted(inv):
        if len(seen) + len(special) >= size:
            break
        if readable(inv[i]):
            admit(inv[i], 'merge_order')
    source['special'] = len(special)
    return kept + special, dict(source)


class TrimmedTokenizer:
    """The tokenizer limited to the kept vocabulary and the merges that build it.

    tokenizers numbers a vocabulary contiguously and puts added tokens right after it, so the
    trimmed tokenizer gets its own ids; encode and decode speak the original ones."""

    def __init__(self, tokenizer_json, kept):
        from tokenizers import Tokenizer
        tj = json.loads(json.dumps(tokenizer_json))
        model = tj['model']
        kept = set(kept)
        special = sorted({a['id'] for a in tj['added_tokens']})
        self.original = sorted(i for i in model['vocab'].values() if i in kept and i not in special) + special
        self.new = {i: n for n, i in enumerate(self.original)}
        keep = {t for t, i in model['vocab'].items() if i in kept}
        pairs = [tuple(m.split(' ', 1)) if isinstance(m, str) else tuple(m) for m in model['merges']]
        model['vocab'] = {t: self.new[i] for t, i in model['vocab'].items() if i in kept}
        model['merges'] = [m for m, (a, b) in zip(model['merges'], pairs) if a + b in keep]
        for a in tj['added_tokens']:
            a['id'] = self.new[a['id']]
        self.tokenizer = Tokenizer.from_str(json.dumps(tj))
        if any(self.tokenizer.token_to_id(a['content']) != a['id'] for a in tj['added_tokens']):
            raise RuntimeError('Added tokens were renumbered')

    def encode(self, text):
        return [self.original[i] for i in self.tokenizer.encode(text, add_special_tokens=False).ids]

    def decode(self, ids):
        return self.tokenizer.decode([self.new[i] for i in ids], skip_special_tokens=False)


def load_tokenizers():
    from tokenizers import Tokenizer
    path = mc.MODEL / 'tokenizer.json'
    tj = json.loads(path.read_text())
    vocab = json.loads(out('vocab64k.json').read_text())
    if vocab['tokenizer_sha256'] != sha(path):
        raise RuntimeError('Tokenizer changed after the trim stage')
    return Tokenizer.from_file(str(path)), TrimmedTokenizer(tj, vocab['kept_ids']), vocab['kept_ids']


def retokenize(full, short, doc):
    """The document's text and its ids under the full and trimmed tokenizers."""
    text = full.decode(doc['input_ids'], skip_special_tokens=False)
    a = full.encode(text, add_special_tokens=False).ids
    b = short.encode(text)
    if full.decode(a, skip_special_tokens=False) != text or short.decode(b) != text:
        raise ValueError(f'{doc["id"]}: tokenizers do not round-trip the text')
    return text, a, b


def stage_trim(args):
    rules, docs = setup()
    cfg = rules['trim']
    path = mc.MODEL / 'tokenizer.json'
    tj = json.loads(path.read_text())
    counts = collections.Counter(t for d in docs_of(docs, 'calib') for t in d['input_ids'])
    kept, source = trim_vocab(tj, counts, cfg['size'])
    if len(set(kept)) != len(kept):
        raise RuntimeError('Duplicate kept ids')
    save({'tokenizer_sha256': sha(path), 'size': len(kept), 'source': source, 'kept_ids': kept},
         fresh('vocab64k.json'))
    full, short, _ = load_tokenizers()
    inflation = collections.defaultdict(lambda: [0, 0])
    for doc in docs:
        _, a, b = retokenize(full, short, doc)
        key = f'{doc["domain"]}/{doc["split"]}'
        inflation[key][0] += len(a)
        inflation[key][1] += len(b)
    report = {k: {'full': a, 'trimmed': b, 'inflation': b / a - 1} for k, (a, b) in sorted(inflation.items())}
    save(dict(provenance(), size=len(kept), source=source, inflation=report), fresh('trim.json'))
    log(f'Kept {len(kept)} ids {source}; eval inflation ' + ', '.join(
        f'{k.split("/")[0]} {v["inflation"]:+.2%}' for k, v in report.items() if k.endswith('/eval')))


@torch.no_grad()
def nll_sum(model, ids, kept=None, index=None):
    """Summed NLL (nats) of ids[1:]; with kept, the output is restricted to those rows."""
    logits = model.lm_head(hidden(model, [ids])[0, :-1]).float()
    target = tensor(ids[1:])
    if kept is not None:
        logits, target = logits[:, kept], index[target]
        if (target < 0).any():
            raise ValueError('Trimmed ids outside the kept rows')
    return float(F.cross_entropy(logits, target, reduction='sum'))


def kept_index(kept, vocab_size):
    kept = tensor(sorted(kept))
    index = torch.full((vocab_size,), -1, dtype=torch.long, device=mc.DEVICE)
    index[kept] = torch.arange(len(kept), device=mc.DEVICE)
    return kept, index


def vocab_row(model, doc, full, short, eot, kept, index):
    text, a, b = retokenize(full, short, doc)
    return {'id': doc['id'], 'domain': doc['domain'], 'bytes': len(text.encode()), 'tokens_full': len(a),
            'tokens_trimmed': len(b), 'nll_full': nll_sum(model, [eot] + a),
            'nll_trimmed': nll_sum(model, [eot] + b, kept, index)}


def vocab_summary(rows, arm='nll_trimmed'):
    text = [r for r in rows if r['domain'] != 'copy']
    groups = [r['domain'] for r in text]
    bpb = {k: [r[k] / np.log(2) / r['bytes'] for r in text] for k in ('nll_full', arm)}
    by_domain = {}
    for d in mc.DOMAINS:
        rs = [r for r in text if r['domain'] == d]
        full = sum(r['nll_full'] for r in rs) / np.log(2) / sum(r['bytes'] for r in rs)
        trim = sum(r[arm] for r in rs) / np.log(2) / sum(r['bytes'] for r in rs)
        by_domain[d] = {'bpb_full': full, 'bpb_trimmed': trim, 'change': trim / full - 1,
                        'inflation': sum(r['tokens_trimmed'] for r in rs) / sum(r['tokens_full'] for r in rs) - 1}
    return {'bpb_full': group_mean(bpb['nll_full'], groups), 'bpb_trimmed': group_mean(bpb[arm], groups),
            'change': float(np.mean([v['change'] for v in by_domain.values()])), 'by_domain': by_domain}


def stage_vocab(args):
    rules, docs = setup()
    model = mc.load_model(rules['threads'])
    full, short, kept = load_tokenizers()
    kept, index = kept_index(kept, model.config.vocab_size)
    eot = full.token_to_id('<|endoftext|>')
    with deadline(rules['vocab']['max_hours']):
        rows = [vocab_row(model, doc, full, short, eot, kept, index) for doc in docs_of(docs, 'eval')]
    write_rows('vocab.jsonl', rows)
    summary = vocab_summary(rows)
    save(dict(provenance(), **summary), fresh('vocab.json'))
    log(f'bits per byte {summary["bpb_full"]:.4f} -> {summary["bpb_trimmed"]:.4f} ({summary["change"]:+.2%})')


# ---------------------------------------------------------------- output-row cache

class RowCache:
    """Least-recently-used set of output rows; mask is a device copy of the membership."""

    def __init__(self, size, vocab_size):
        self.size, self.order = size, collections.OrderedDict()
        self.mask = torch.zeros(vocab_size, dtype=torch.bool, device=mc.DEVICE)

    def admit(self, token):
        if token in self.order:
            self.order.move_to_end(token)
            return
        self.order[token] = None
        self.mask[token] = True
        if len(self.order) > self.size:
            self.mask[self.order.popitem(last=False)[0]] = False

    def stream(self, ids):
        """(T, V) membership before each position's prediction: ids[t] is admitted first."""
        masks = torch.empty(len(ids), self.mask.shape[0], dtype=torch.bool, device=mc.DEVICE)
        for t, token in enumerate(ids):
            self.admit(token)
            masks[t] = self.mask
        return masks


def cache_features(logits, masks, targets):
    """Per position: what the cached rows alone show (detector inputs) and what the full head knows.

    logits (T, V) predict targets (T,); masks (T, V) are the cached rows at each position."""
    cached = logits.masked_fill(~masks, -torch.inf)
    top = cached.topk(2, -1).values
    lse = cached.logsumexp(-1)
    logp = cached - lse[:, None]
    entropy = -torch.where(masks, logp.exp() * logp, torch.zeros_like(logp)).sum(-1)
    full = logits.logsumexp(-1)
    best = logits.argmax(-1)
    return {'max': top[:, 0], 'gap': top[:, 0] - top[:, 1], 'lse': lse, 'entropy': entropy,
            'p_out': (1 - (lse - full).exp()).clamp_min(0), 'lse_full': full,
            'pick_out': ~masks.gather(1, best[:, None])[:, 0], 'true_out': ~masks.gather(1, targets[:, None])[:, 0],
            'cached_pick': cached.argmax(-1), 'pick': best}


DETECTOR_INPUTS = ('max', 'gap', 'lse', 'entropy', 'top_logp', 'lse_vs_predicted')


def detector_inputs(f, predicted_lse):
    """(T, 6) detector features: the cached rows' view plus the gap to a predicted full log-normaliser."""
    return torch.stack([f['max'], f['gap'], f['lse'], f['entropy'], f['max'] - f['lse'], f['lse'] - predicted_lse], 1)


def fit_ridge(xtx, xty, scale):
    """Ridge weights (with bias as the last input) from accumulated moments."""
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
    return torch.linalg.solve(xtx + scale * xtx.diagonal().mean() * eye, xty).float()


def with_bias(h):
    return torch.cat([h.float(), torch.ones(h.shape[0], 1, device=h.device)], 1)


def fit_detector(x, y, recall, seed):
    """Logistic regression of y on standardized x, and the threshold that keeps recall of y."""
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z = (x - mean) / std
    w = torch.zeros(z.shape[1] + 1, device=z.device, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn='strong_wolfe')

    def loss():
        opt.zero_grad()
        value = F.binary_cross_entropy_with_logits(z @ w[:-1] + w[-1], y.float())
        value.backward()
        return value

    opt.step(loss)
    w = w.detach()
    score = z @ w[:-1] + w[-1]
    threshold = (float(torch.quantile(score[y], 1 - recall, interpolation='lower')) if y.any()
                 else float('inf'))
    return {'mean': mean, 'std': std, 'w': w, 'threshold': threshold}


def detect(det, x):
    return ((x - det['mean']) / det['std']) @ det['w'][:-1] + det['w'][-1] >= det['threshold']


def cache_outcome(f, fallback, rows_head):
    """Per-position outcome of serving from the cache unless the detector falls back."""
    size_rows = rows_head['cache']
    final = torch.where(fallback, f['pick'], f['cached_pick'])
    return {'true_hit': float((~f['true_out']).float().mean()), 'pick_hit': float((~f['pick_out']).float().mean()),
            'p_out_mean': float(f['p_out'].mean()), 'p_out_p99': float(torch.quantile(f['p_out'], .99)),
            'fallback': float(fallback.float().mean()),
            'miss_recall': float((fallback & f['pick_out']).sum() / f['pick_out'].sum().clamp_min(1)),
            'pick_kept': float((final == f['pick']).float().mean()),
            'rows_per_token': {k: size_rows + float(fallback.float().mean()) * (v - size_rows)
                               for k, v in rows_head.items() if k != 'cache'}}


def domain_outcomes(per_doc, domains, fn):
    """fn over each domain's concatenated positions, and the domain-equal mean of every number."""
    by = {}
    for d in domains:
        parts = [p for p in per_doc if p[0] == d]
        if parts:
            by[d] = fn(*[torch.cat([p[i] for p in parts]) if isinstance(parts[0][i], torch.Tensor) else
                         {k: torch.cat([p[i][k] for p in parts]) for k in parts[0][i]} for i in range(1, len(parts[0]))])
    text = [v for d, v in by.items() if d != 'copy']
    mean = {k: float(np.mean([v[k] for v in text])) for k in text[0] if not isinstance(text[0][k], dict)}
    mean['rows_per_token'] = {k: float(np.mean([v['rows_per_token'][k] for v in text])) for k in text[0]['rows_per_token']}
    return {'mean': mean, 'by_domain': by}


def stage_cache(args):
    rules, docs = setup()
    cfg = rules['cache']
    model = mc.load_model(rules['threads'])
    vocab_size = model.config.vocab_size
    caches = {n: RowCache(n, vocab_size) for n in cfg['sizes']}
    kept_rows = json.loads(out('vocab64k.json').read_text())['size']
    xtx = torch.zeros(model.config.hidden_size + 1, model.config.hidden_size + 1, dtype=torch.float64, device=mc.DEVICE)
    xty = torch.zeros(model.config.hidden_size + 1, dtype=torch.float64, device=mc.DEVICE)
    train_h, train = [], {n: [] for n in cfg['sizes']}
    seen = 0
    t0 = time.time()
    with deadline(cfg['max_hours']):
        # Calibration documents warm the caches and train the detectors; the first warm_tokens
        # only warm them.
        for doc in docs_of(docs, 'calib'):
            ids = doc['input_ids']
            h = hidden(model, [ids])[0, :-1]
            logits = model.lm_head(h).float()
            targets = tensor(ids[1:])
            x = with_bias(h)
            xtx += (x.T @ x).double()
            xty += (x.T @ logits.logsumexp(-1)).double()
            for n, cache in caches.items():
                f = cache_features(logits, cache.stream(ids)[:-1], targets)
                if seen >= cfg['warm_tokens']:
                    train[n].append({k: f[k].cpu() for k in ('max', 'gap', 'lse', 'entropy', 'pick_out')})
            if seen >= cfg['warm_tokens']:
                train_h.append(h.to(torch.bfloat16).cpu())
            seen += len(ids)
        ridge = fit_ridge(xtx, xty, cfg['ridge'])
        predicted = torch.cat([with_bias(c.to(mc.DEVICE)) @ ridge for c in train_h])
        del train_h
        detectors = {}
        for n in cfg['sizes']:
            f = {k: torch.cat([p[k] for p in train[n]]).to(mc.DEVICE) for k in train[n][0]}
            detectors[n] = fit_detector(detector_inputs(f, predicted), f['pick_out'], cfg['recall'], cfg['seed'])
        del train
        torch.save({'ridge': ridge.cpu(), 'detectors': {n: {k: (v.cpu() if torch.is_tensor(v) else v)
                    for k, v in d.items()} for n, d in detectors.items()}}, fresh('cache_detector.pt'))
        # Eval documents continue the warm caches.
        per_doc = {n: [] for n in cfg['sizes']}
        for doc in docs_of(docs, 'eval'):
            ids = doc['input_ids']
            h = hidden(model, [ids])[0, :-1]
            logits = model.lm_head(h).float()
            predicted = with_bias(h) @ ridge.to(mc.DEVICE)
            for n, cache in caches.items():
                f = cache_features(logits, cache.stream(ids)[:-1], tensor(ids[1:]))
                fallback = detect(detectors[n], detector_inputs(f, predicted))
                per_doc[n].append((doc['domain'], {k: v for k, v in f.items() if k != 'lse_full'}, fallback))
    result = {}
    for n in cfg['sizes']:
        heads = {'cache': n, 'head_64k': kept_rows, 'head_full': vocab_size}
        result[str(n)] = domain_outcomes(per_doc[n], list(mc.DOMAINS) + ['copy'],
                                         lambda f, fallback, heads=heads: cache_outcome(f, fallback, heads))
        log(f'{n} rows: ' + ', '.join(f'{k} {v:.4f}' for k, v in result[str(n)]['mean'].items() if not isinstance(v, dict)))
    save(dict(provenance(), detector_inputs=DETECTOR_INPUTS, detectors={n: {'threshold': d['threshold']}
              for n, d in detectors.items()}, sizes=result, seconds=time.time() - t0), fresh('cache.json'))


# ---------------------------------------------------------------- DeltaNet state precision

def store(name, x, g):
    """x (fp32) after a round trip through storage format name."""
    if name == 'fp32':
        return x
    if name == 'fp32_jitter':
        return x * (1 + 2. ** -23 * torch.randn(x.shape, generator=g, device=x.device))
    if name == 'bf16':
        return x.bfloat16().float()
    if name == 'fp16':
        return x.half().float()
    if name == 'fp16_sr':
        # Stochastic rounding to fp16's 10 mantissa bits: add noise below them, then truncate.
        noise = torch.randint(0, 1 << 13, x.shape, generator=g, device=x.device, dtype=torch.int32)
        return ((x.contiguous().view(torch.int32) + noise) & -(1 << 13)).view(torch.float32).half().float()
    if name in ('int16_row', 'int8_row'):
        top = 32767 if name == 'int16_row' else 127
        scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-30) / top
        return (x / scale).round().clamp(-top, top) * scale
    raise ValueError(name)


class StoredState:
    """Replaces every DeltaNet's chunked and per-token kernels with one fp32 token-by-token gated
    delta rule whose state goes through a storage format after every token, as a per-token kernel
    that keeps the state in memory would; variants[b] is batch element b's format. The output of a
    token uses the fresh state; the next token reads the stored one."""

    def __init__(self, model, variants, seed):
        from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
        self.l2norm = l2norm
        self.layers = [layer.linear_attn for layer in model.model.layers if hasattr(layer, 'linear_attn')]
        self.variants = list(variants)
        self.g = torch.Generator(device=mc.DEVICE).manual_seed(seed)

    def store(self, state):
        if all(v == 'fp32' for v in self.variants):
            return state
        stored = torch.empty_like(state)
        for name in dict.fromkeys(self.variants):
            idx = [b for b, v in enumerate(self.variants) if v == name]
            stored[idx] = store(name, state[idx], self.g)
        return stored

    def rule(self, query, key, value, g, beta, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, **kwargs):
        dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query, key = self.l2norm(query, dim=-1, eps=1e-6), self.l2norm(key, dim=-1, eps=1e-6)
        q, k, v, beta, g = [x.transpose(1, 2).float() for x in (query, key, value, beta, g)]
        q = q * q.shape[-1] ** -0.5
        b, h, t, dk = k.shape
        state = (torch.zeros(b, h, dk, v.shape[-1], device=v.device) if initial_state is None
                 else initial_state.float())
        o = torch.empty(b, h, t, v.shape[-1], device=v.device)
        for i in range(t):
            state = state * g[:, :, i].exp()[..., None, None]
            delta = (v[:, :, i] - (k[:, :, i, None] @ state)[:, :, 0]) * beta[:, :, i, None]
            state = state + k[:, :, i, :, None] * delta[:, :, None]
            o[:, :, i] = (q[:, :, i, None] @ state)[:, :, 0]
            state = self.store(state)
        return o.transpose(1, 2).to(dtype), (state if output_final_state else None)

    def __enter__(self):
        self.saved = [(m.chunk_gated_delta_rule, m.recurrent_gated_delta_rule) for m in self.layers]
        for m in self.layers:
            m.chunk_gated_delta_rule = m.recurrent_gated_delta_rule = self.rule
        return self

    def __exit__(self, *exc):
        for m, (chunk, recurrent) in zip(self.layers, self.saved):
            m.chunk_gated_delta_rule, m.recurrent_gated_delta_rule = chunk, recurrent


def stage_state(args):
    rules, docs = setup()
    cfg = rules['state']
    model = mc.load_model(rules['threads'])
    rows = []
    with deadline(cfg['max_hours']):
        for doc in docs_of(docs, 'eval'):
            ids = doc['input_ids']
            with torch.no_grad():
                logits = {'dense': model.lm_head(hidden(model, [ids])[0]).float()}
                with StoredState(model, STORES, cfg['seed']):
                    h = hidden(model, [ids] * len(STORES))
                logits.update({v: model.lm_head(h[b]).float() for b, v in enumerate(STORES)})
            rows.append(score_arms(logits, ids, doc))
    write_rows('state.jsonl', rows)
    names = ['dense', *STORES]
    summary = mc.arm_summary(rows, names)
    save(dict(provenance(), reference='stock chunked kernels, fp32 state', arms=summary,
              nonfinite_docs={n: sum(not r['finite'][n] for r in rows) for n in names}), fresh('state.json'))
    log(', '.join(f'{n} {summary[n]["kl"]:.2e}' for n in STORES))


def final_number(text):
    """The last number of the answer after any reasoning, preferring a boxed one."""
    tail = text.split('</think>')[-1]
    boxed = re.findall(r'\\boxed\{([^{}]*)\}', tail)
    found = re.findall(r'-?\d[\d,]*(?:\.\d+)?', boxed[-1] if boxed else tail)
    return float(found[-1].replace(',', '')) if found else None


def stage_generate(args):
    from transformers import AutoTokenizer
    rules, _ = setup()
    cfg = rules['generate']
    prompts = json.loads((ROOT / cfg['prompts']).read_text())['prompts'][:cfg['count']]
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    model = mc.load_model(rules['threads'])
    stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]
    path = out('generate.jsonl')
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)['id'] for line in path.open()} if path.exists() else set()
    with deadline(cfg['max_hours']):
        for p in prompts:
            if p['id'] in done:
                continue
            text = tok.apply_chat_template([{'role': 'user', 'content': p['question']}], tokenize=False,
                                           add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False)['input_ids']
            t0 = time.time()
            with torch.no_grad(), StoredState(model, STORES, cfg['seed']):
                batch = tensor([ids] * len(STORES))
                result = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                                        max_new_tokens=cfg['max_new_tokens'], do_sample=False, temperature=None,
                                        top_p=None, top_k=None, eos_token_id=stop, pad_token_id=stop[1])
            new = result[:, len(ids):].tolist()
            outs = {}
            for v, tokens in zip(STORES, new):
                end = next((i for i, t in enumerate(tokens) if t in stop), None)
                tokens = tokens[:end] if end is not None else tokens
                answer = final_number(tok.decode(tokens))
                outs[v] = {'tokens': tokens, 'length': len(tokens), 'finished': end is not None,
                           'answer': answer, 'correct': answer is not None and abs(answer - p['answer']) < 1e-6}
            ref = outs['fp32']['tokens']
            for v in STORES:
                mine = outs[v]['tokens']
                outs[v]['diverges_at'] = next((i for i, (a, b) in enumerate(zip(ref, mine)) if a != b),
                                              None if len(ref) == len(mine) else min(len(ref), len(mine)))
            with path.open('a') as f:
                f.write(json.dumps({'id': p['id'], 'seconds': time.time() - t0, 'outputs': outs}) + '\n')
            log(f'{p["id"]}: ' + ', '.join(f'{v} {o["length"]}{"" if o["correct"] else "x"}' for v, o in outs.items()))


# ---------------------------------------------------------------- fewer DeltaNet parameters

PROJECTIONS = ('in_proj_qkv', 'in_proj_z', 'out_proj')


class DeltaNetCut:
    """Zeroes chosen DeltaNet blocks' token-mixer output per batch element (keep[layer] is a (B,)
    tensor, 1 runs the block), and swaps in replacement projection weights (weights[layer][name])."""

    def __init__(self, model):
        self.blocks = {i: layer.linear_attn for i, layer in enumerate(model.model.layers)
                       if hasattr(layer, 'linear_attn')}
        self.keep, self.weights, self.saved = {}, {}, {}
        for i, m in self.blocks.items():
            m.forward = (lambda *a, i=i, f=m.forward, **k: self.forward(i, f, *a, **k))

    def forward(self, i, f, *args, **kwargs):
        y = f(*args, **kwargs)
        keep = self.keep.get(i)
        return y if keep is None else y * keep[:, None, None].to(y.dtype)

    def swap(self, weights):
        """Install weights ({layer: {projection: tensor}}); swap({}) restores the originals."""
        for i, names in self.saved.items():
            for name, w in names.items():
                getattr(self.blocks[i], name).weight.data = w
        self.saved = {}
        for i, names in weights.items():
            self.saved[i] = {n: getattr(self.blocks[i], n).weight.data for n in names}
            for name, w in names.items():
                getattr(self.blocks[i], name).weight.data = w


def block_parameters(module):
    return sum(p.numel() for p in module.parameters())


def lowrank(w, cov, ranks, eps):
    """For each rank r, the rank-r W' minimising E||(W - W') x||^2 for inputs x with second
    moment cov (SVD-LLM): truncate the SVD of W L, where cov = L L^T, and map back with L^-1."""
    c = cov + eps * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    lower = torch.linalg.cholesky(c).float()
    u, s, vh = torch.linalg.svd(w.float() @ lower, full_matrices=False)
    right = torch.linalg.solve_triangular(lower, vh[:max(ranks)], upper=False, left=False)
    return [(u[:, :r] * s[:r]) @ right[:r] for r in ranks]


def rank_for(w, fraction):
    out_dim, in_dim = w.shape
    return int(fraction * out_dim * in_dim / (out_dim + in_dim))


@torch.no_grad()
def input_moments(model, cut, docs):
    """Per DeltaNet block, the second moments of the in_proj input and the out_proj input."""
    cov, hooks = {}, []
    for i, m in cut.blocks.items():
        for j, name in enumerate(('in_proj_qkv', 'out_proj')):
            d = getattr(m, name).in_features
            cov[i, j] = torch.zeros(d, d, dtype=torch.float64, device=mc.DEVICE)

            def hook(module, inputs, output, key=(i, j)):
                x = inputs[0].flatten(0, -2).float()
                cov[key] += (x.T @ x).double()
            hooks.append(getattr(m, name).register_forward_hook(hook))
    try:
        for doc in docs:
            hidden(model, [doc['input_ids']])
    finally:
        for h in hooks:
            h.remove()
    return cov


def lowrank_weights(cut, cov, fractions, eps):
    """For each parameter fraction, replacement projections ({layer: {name: W'}}) and the
    parameters a factored W' saves."""
    arms = [({}, 0) for _ in fractions]
    for i, m in cut.blocks.items():
        for arm in arms:
            arm[0][i] = {}
        for name in PROJECTIONS:
            w = getattr(m, name).weight
            ranks = [rank_for(w, f) for f in fractions]
            for a, (r, low) in enumerate(zip(ranks, lowrank(w, cov[i, 0 if name != 'out_proj' else 1], ranks, eps))):
                arms[a][0][i][name] = low.to(w.dtype)
                arms[a] = (arms[a][0], arms[a][1] + w.numel() - r * sum(w.shape))
    return arms


def stage_deltanet(args):
    rules, docs = setup()
    cfg = rules['deltanet']
    model = mc.load_model(rules['threads'])
    cut = DeltaNetCut(model)
    blocks = list(cut.blocks)
    select_docs = docs_of(docs, 'calib', cfg['select_per_domain'])
    choose_docs = docs_of(docs, 'calib', cfg['select_per_domain'], cfg['select_per_domain'])
    t0 = time.time()
    with deadline(cfg['max_hours']):
        # Single-block importance: KL when only that block's token mixer is removed.
        kl = np.zeros((len(select_docs), len(blocks)))
        for d, doc in enumerate(select_docs):
            ids = doc['input_ids']
            cut.keep = {i: torch.ones(len(blocks) + 1, device=mc.DEVICE) for i in blocks}
            for j, i in enumerate(blocks):
                cut.keep[i][j + 1] = 0
            h = hidden(model, [ids] * (len(blocks) + 1))
            cut.keep = {}
            dense = model.lm_head(h[0]).float()
            for j in range(len(blocks)):
                kl[d, j] = float(token_kl(dense, model.lm_head(h[j + 1]).float())[1:].mean())
        groups = [doc['domain'] for doc in select_docs]
        importance = [group_mean(kl[:, j], groups) for j in range(len(blocks))]
        order = [blocks[j] for j in np.argsort(importance)]
        log('Least important DeltaNet blocks: ' + ', '.join(map(str, order[:max(cfg['drop'])])))
        block = block_parameters(cut.blocks[blocks[0]])
        projections = sum(getattr(cut.blocks[blocks[0]], n).weight.numel() for n in PROJECTIONS)
        cov = input_moments(model, cut, docs_of(docs, 'calib', cfg['moment_per_domain'],
                                                2 * cfg['select_per_domain']))
        # Each low-rank arm saves as many parameters as dropping k whole blocks.
        fractions = [1 - k * block / (len(blocks) * projections) for k in cfg['drop']]
        arms = {}
        for k, fraction, (weights, saved) in zip(cfg['drop'], fractions,
                                                 lowrank_weights(cut, cov, fractions, cfg['eps'])):
            arms[f'drop{k}'] = {'kind': 'drop', 'blocks': order[:k], 'saved': k * block}
            arms[f'lowrank{k}'] = {'kind': 'lowrank', 'fraction': fraction, 'saved': saved, 'weights': weights}
        del cov
        drops = [n for n, a in arms.items() if a['kind'] == 'drop']

        def run(doc):
            ids = doc['input_ids']
            cut.keep = {i: torch.tensor([1.] + [0. if i in arms[n]['blocks'] else 1. for n in drops],
                                        device=mc.DEVICE) for i in blocks}
            h = hidden(model, [ids] * (len(drops) + 1))
            cut.keep = {}
            logits = {'dense': model.lm_head(h[0]).float()}
            logits.update({n: model.lm_head(h[b + 1]).float() for b, n in enumerate(drops)})
            for n, a in arms.items():
                if a['kind'] == 'lowrank':
                    cut.swap(a['weights'])
                    try:
                        logits[n] = model.lm_head(hidden(model, [ids])[0]).float()
                    finally:
                        cut.swap({})
            return score_arms(logits, ids, doc)

        choose = [run(doc) for doc in choose_docs]
        rows = [run(doc) for doc in docs_of(docs, 'eval')]
    names = ['dense', *arms]
    write_rows('deltanet.jsonl', rows)
    choose_summary = mc.arm_summary(choose, names)
    k = max(cfg['drop'])
    chosen = min((f'drop{k}', f'lowrank{k}'), key=lambda n: choose_summary[n]['kl'])
    summary = mc.arm_summary(rows, names)
    save(dict(provenance(), block_parameters=block, projection_parameters=projections,
              importance=dict(zip(map(str, blocks), importance)), order=order,
              arms={n: {k2: v for k2, v in a.items() if k2 != 'weights'} for n, a in arms.items()},
              eval=summary, choose={n: choose_summary[n]['kl'] for n in names}, chosen=chosen,
              seconds=time.time() - t0), fresh('deltanet.json'))
    log(', '.join(f'{n} {summary[n]["kl"]:.3f}' for n in names[1:]) + f'; chosen {chosen}')


# ---------------------------------------------------------------- everything together

def stage_combined(args):
    rules, docs = setup()
    cfg = rules['combined']
    model = mc.load_model(rules['threads'])
    carve = json.loads(CARVE_RULES.read_text())
    budget = carve['route']['budget']
    sets = torch.load(mc.OUT / 'sets.pt', weights_only=True, map_location=mc.DEVICE)
    carver = mc.Carver(model, sets)
    carver.routers = []
    for state in torch.load(mc.OUT / 'routers.pt', weights_only=True, map_location=mc.DEVICE):
        router = torch.nn.Linear(model.config.hidden_size, sets[budget]['experts']).to(mc.DEVICE)
        router.load_state_dict(state)
        carver.routers.append(router.eval().requires_grad_(False))
    cut = DeltaNetCut(model)
    plan = json.loads(out('deltanet.json').read_text())
    chosen = plan['arms'][plan['chosen']]
    weights = {}
    if chosen['kind'] == 'lowrank':
        dcfg = rules['deltanet']
        cov = input_moments(model, cut, docs_of(docs, 'calib', dcfg['moment_per_domain'],
                                                2 * dcfg['select_per_domain']))
        weights, _ = lowrank_weights(cut, cov, [chosen['fraction']], dcfg['eps'])[0]
        del cov
    detector = torch.load(out('cache_detector.pt'), weights_only=True, map_location=mc.DEVICE)
    size = cfg['cache']
    det, ridge = detector['detectors'][size], detector['ridge']
    full, short, kept = load_tokenizers()
    kept_rows = len(kept)
    kept, index = kept_index(kept, model.config.vocab_size)
    eot = full.token_to_id('<|endoftext|>')
    cache = RowCache(size, model.config.vocab_size)
    for doc in docs_of(docs, 'calib'):
        for token in doc['input_ids']:
            cache.admit(token)

    def body(ids):
        """Final hidden states of the slimmed model."""
        carver.arms = [('learned', budget)]
        if chosen['kind'] == 'drop':
            cut.keep = {i: torch.tensor([0.], device=mc.DEVICE) for i in chosen['blocks']}
        cut.swap(weights)
        try:
            with StoredState(model, [cfg['state']], cfg['seed']):
                return hidden(model, [ids])[0]
        finally:
            carver.arms, cut.keep = None, {}
            cut.swap({})

    rows, vocab_rows, per_doc = [], [], []
    t0 = time.time()
    with deadline(cfg['max_hours']):
        for doc in docs_of(docs, 'eval'):
            ids = doc['input_ids']
            with torch.no_grad():
                dense = model.lm_head(hidden(model, [ids])[0]).float()
                h = body(ids)
                slim = model.lm_head(h).float()
                rows.append(score_arms({'dense': dense, 'combined': slim}, ids, doc))
                f = cache_features(slim[:-1], cache.stream(ids)[:-1], tensor(ids[1:]))
                fallback = detect(det, detector_inputs(f, with_bias(h[:-1]) @ ridge))
                agree = {'no_cache': slim[:-1].argmax(-1) == dense[:-1].argmax(-1),
                         'cache': torch.where(fallback, f['pick'], f['cached_pick']) == dense[:-1].argmax(-1)}
                per_doc.append((doc['domain'], {k: v for k, v in f.items() if k != 'lse_full'}, fallback, agree))
                text, a, b = retokenize(full, short, doc)
                logits = model.lm_head(body([eot] + b)[:-1]).float()[:, kept]
                target = index[tensor(b)]
                vocab_rows.append({'id': doc['id'], 'domain': doc['domain'], 'bytes': len(text.encode()),
                                   'tokens_full': len(a), 'tokens_trimmed': len(b),
                                   'nll_full': nll_sum(model, [eot] + a),
                                   'nll_combined': float(F.cross_entropy(logits, target, reduction='sum'))})
    write_rows('combined.jsonl', rows)
    write_rows('combined_vocab.jsonl', vocab_rows)
    heads = {'cache': size, 'head_64k': kept_rows, 'head_full': model.config.vocab_size}

    def outcome(f, fallback, agree):
        result = cache_outcome(f, fallback, heads)
        result.update({f'agree_{k}': float(v.float().mean()) for k, v in agree.items()})
        return result

    summary = mc.arm_summary(rows, ['dense', 'combined'])
    save(dict(provenance(), budget=budget, state=cfg['state'], deltanet=plan['chosen'], cache_rows=size,
              vocab_rows=kept_rows, body=summary['combined'],
              cache=domain_outcomes(per_doc, list(mc.DOMAINS) + ['copy'], outcome),
              vocab=vocab_summary(vocab_rows, 'nll_combined'), seconds=time.time() - t0), fresh('combined.json'))
    log(f'combined KL {summary["combined"]["kl"]:.3f}')


# ---------------------------------------------------------------- local stages

def stage_prompts(args):
    rules, _ = setup()
    cfg = rules['generate']
    path = ROOT / cfg['prompts']
    if path.exists():
        raise ValueError('Prompts exist; refusing to overwrite')
    rows = mc.fetch_rows('openai/gsm8k', 'main', 0, cfg['count'], split='test')
    prompts = [{'id': f'gsm8k-test-{i}', 'question': r['question'],
                'answer': float(r['answer'].split('####')[-1].strip().replace(',', ''))} for i, r in enumerate(rows)]
    save({'source': 'openai/gsm8k main test, first rows', 'prompts': prompts}, path)


def parameter_counts(config, plan):
    """Stored and per-token parameters, total and without the vocabulary tables, for each step."""
    from transformers import AutoModelForCausalLM
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config)
    groups = collections.Counter()
    for name, p in model.named_parameters():
        part = ('embed' if 'embed_tokens' in name else 'head' if name.startswith('lm_head') else
                'mlp' if '.mlp.' in name else 'deltanet' if 'linear_attn' in name else
                'attention' if 'self_attn' in name else 'other')
        groups[part] += p.numel()
    hidden, vocab, layers = config.hidden_size, config.vocab_size, config.num_hidden_layers
    row = hidden

    def counts(g, active_mlp, active_head):
        body = g['mlp'] + g['deltanet'] + g['attention'] + g['other'] + g.get('router', 0)
        return {'stored_total': body + g['embed'] + g['head'], 'stored_non_embedding': body,
                'active_total': body - g['mlp'] + active_mlp + active_head,
                'active_non_embedding': body - g['mlp'] + active_mlp}

    steps = {'dense': counts(groups, groups['mlp'], groups['head'])}
    g = collections.Counter(groups)
    g['router'] = plan['router']
    mlp = plan['active_neurons'] * 3 * hidden
    steps['moe'] = counts(g, mlp, g['head'])
    g['embed'] = g['head'] = plan['vocab'] * row
    steps['vocab'] = counts(g, mlp, g['head'])
    steps['cache'] = counts(g, mlp, plan['head_rows_per_token'] * row)
    g['deltanet'] -= plan['deltanet_saved']
    steps['deltanet'] = counts(g, mlp, plan['head_rows_per_token'] * row)
    return {'groups': dict(groups), 'steps': steps}


def generate_summary(rows):
    """Per storage format: mean length, share finished, answers right, mean relative length
    change from fp32, and how long each run keeps fp32's exact text."""
    summary = {}
    for v in STORES:
        runs = [(r['outputs'][v], r['outputs']['fp32']) for r in rows]
        summary[v] = {'prompts': len(runs), 'mean_length': float(np.mean([o['length'] for o, _ in runs])),
                      'finished': float(np.mean([o['finished'] for o, _ in runs])),
                      'correct': sum(o['correct'] for o, _ in runs),
                      'length_change': float(np.mean([abs(o['length'] - f['length']) / f['length'] for o, f in runs])),
                      'same_text': sum(o['diverges_at'] is None for o, _ in runs),
                      'median_divergence': float(np.median([o['length'] if o['diverges_at'] is None else o['diverges_at']
                                                            for o, _ in runs]))}
    return summary


def stage_report(args):
    from transformers import AutoConfig
    rules, _ = setup()
    carve = json.loads(CARVE_RULES.read_text())
    read = lambda name: json.loads(out(name).read_text())
    combined, plan = read('combined.json'), read('deltanet.json')
    sets = torch.load(mc.OUT / 'sets.pt', weights_only=True)
    s = sets[carve['route']['budget']]
    config = AutoConfig.from_pretrained(mc.MODEL, local_files_only=True)
    config = getattr(config, 'text_config', config)
    counts = parameter_counts(config, {
        'router': config.num_hidden_layers * (config.hidden_size + 1) * s['experts'],
        'active_neurons': config.num_hidden_layers * s['n'], 'vocab': combined['vocab_rows'],
        'head_rows_per_token': combined['cache']['mean']['rows_per_token']['head_64k'],
        'deltanet_saved': plan['arms'][plan['chosen']]['saved']})
    report = {'carve': json.loads((mc.OUT / 'report.json').read_text())['arms'],
              'route': {k: v for k, v in json.loads((mc.OUT / 'route.json').read_text()).items() if k != 'loss_by_step'},
              'vocab': read('vocab.json'), 'cache': read('cache.json'), 'state': read('state.json'),
              'deltanet': plan, 'combined': combined, 'parameters': counts}
    if out('generate.jsonl').exists():
        report['generate'] = generate_summary([json.loads(line) for line in out('generate.jsonl').open()])
    save(dict(provenance(), **report), fresh('report.json'))
    for step, c in counts['steps'].items():
        log(f'{step:9s} ' + ', '.join(f'{k} {v / 1e9:.2f}B' for k, v in c.items()))


def main():
    stages = {'prompts': stage_prompts, 'trim': stage_trim, 'vocab': stage_vocab, 'cache': stage_cache,
              'state': stage_state, 'deltanet': stage_deltanet, 'combined': stage_combined,
              'generate': stage_generate, 'report': stage_report}
    global RULES, CARVE_RULES
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=stages)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--model', type=Path, default=mc.MODEL)
    ap.add_argument('--rules', type=Path, default=RULES)
    ap.add_argument('--carve-rules', type=Path, default=CARVE_RULES)
    args = ap.parse_args()
    mc.DEVICE, mc.MODEL = args.device, args.model
    RULES, CARVE_RULES = args.rules.resolve(), args.carve_rules.resolve()
    stages[args.stage](args)


if __name__ == '__main__':
    main()
