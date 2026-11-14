"""Healing pilot for the slimmed Qwen3.5-9B: how much of the dense model a short distillation run
brings back to slim_batch's combined model.

The student is the combined model: co-firing experts at 25% of MLP neurons per token picked by
the learned routers (position 0 of a sequence runs every neuron, as in every earlier measurement),
the 64K vocabulary, and the low-rank DeltaNet projections. One copy of the dense weights serves as
both teacher and student. In student mode the DeltaNet projections run through their low-rank
factors, each MLP runs only its routed experts, and every body projection adds a trainable LoRA
update. The routers keep training on the student's own activations (cross-entropy against each
token's share of routed contribution, as moe_carve trained them). Embeddings, output head, norms and
the DeltaNet gate projections stay frozen. The DeltaNet state stays in fp32: the batch measured
16-bit storage as free, and the final stage checks that again on the healed student.

Loss per position: KL from the teacher's next-token distribution over the kept 64K rows
(renormalized), plus ce_weight times the cross-entropy of the true next token. The text is the
healing stream in the 64K tokenizer, and the teacher reads the same tokens.

Settings and predictions are in heal_rules.json; outputs go to the heal/ folder beside the carve's.
Stages:

  data      (local) the healing stream: items from the in-mix domains' sources that no benchmark
            document uses (biomed and argument are held out entirely), each capped, joined by
            end-of-text tokens and cut into fixed-length sequences, equal tokens per domain.
            Drops any sequence that shares a 32-token window with an eval document.
  heal      builds the student, evaluates it, trains it, and evaluates again at token milestones.
  final     the healed student with 16-bit stored state, and a row-cache miss detector retrained on
            it (64K head, 64K tokens).
  generate  long greedy GSM8K generations of the healed student (int16 state, 64K vocabulary).
  report    (local) one summary, with parameter counts.
"""
import argparse
import hashlib
import json
import math
import os
import time
import types
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import moe_carve as mc
import slim_batch as sb
from h0_mlp_probe import deadline, group_mean, key, log, save, sha, token_kl

ROOT = mc.ROOT
RULES = ROOT / 'heal_rules.json'
CODE = ['slim_heal.py', 'slim_batch.py', 'moe_carve.py', 'h0_mlp_probe.py', 'test_slim_heal.py']
MLIR_AIE = mc.CPP_DIR.parent
# Tests, installed packages, codec tables and generated data are not the code the domain stands for.
SKIP_PY = {'site-packages', 'test', 'tests', 'idle_test', '__pycache__', 'encodings', 'pydoc_data'}


def setup():
    mc.configure(sb.CARVE_RULES)
    return json.loads(RULES.read_text()), json.loads((mc.OUT / 'corpus.json').read_text())['docs']


def out(name):
    return mc.OUT / 'heal' / name


def fresh(name):
    if out(name).exists():
        raise ValueError(f'{name} exists; refusing to overwrite')
    out(name).parent.mkdir(parents=True, exist_ok=True)
    return out(name)


# Kept on the pod beside the run folder, not synced: the factors are 2 GB and can be rebuilt.
def factors_path():
    return mc.OUT / 'heal_factors.pt'


def state_path():
    return mc.OUT / 'heal_state.pt'


def provenance():
    return {'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'rules_sha256': sha(RULES),
            'carve_rules_sha256': sha(sb.CARVE_RULES), 'slim_rules_sha256': sha(sb.RULES),
            'corpus_sha256': sha(mc.OUT / 'corpus.json')}


# ---------------------------------------------------------------- healing data

FETCHED = mc.REPO / 'target/manifold-heal/fetched'


def fetch(dataset, config, offset, length, split='train', tries=10, pause=1.0):
    """mc.fetch_rows with a page cache, a pause after every request and long retries: the datasets
    server answers bursts of anonymous requests with 429 for minutes."""
    page = FETCHED / hashlib.sha256(json.dumps([dataset, config, split, offset, length]).encode()).hexdigest()[:24]
    if page.exists():
        return json.loads(page.read_text())
    for attempt in range(tries):
        try:
            rows = mc.fetch_rows(dataset, config, offset, length, split)
            break
        except Exception as error:
            if attempt == tries - 1:
                raise
            log(f'{dataset} {config} {offset}: {error}, retrying')
            time.sleep(min(600, 30 * 2 ** attempt))
    time.sleep(pause)
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(json.dumps(rows))
    return rows


def paged(dataset, config, windows, split='train'):
    """(row number, row) for each (offset, count) window, 100 rows a request; a window ends early
    where the split does."""
    for offset, count in windows:
        for start in range(offset, offset + count, 100):
            length = min(100, offset + count - start)
            rows = fetch(dataset, config, start, length, split)
            yield from ((start + i, r) for i, r in enumerate(rows))
            if len(rows) < length:
                break


def interleave(*sources):
    """Items of several iterators in turn until all are spent."""
    sources = [iter(s) for s in sources]
    while sources:
        for s in list(sources):
            try:
                yield next(s)
            except StopIteration:
                sources.remove(s)


def after_header(text, language):
    """moe_carve's code transform: Python from the first top-level definition, C++ after the last
    #include; None when there is no such point."""
    if language == 'python':
        start = min((i for i in (text.find('\ndef '), text.find('\nclass ')) if i >= 0), default=-1)
        return text[start + 1:] if start >= 0 else None
    lines = text.splitlines(keepends=True)
    last = max((i for i, line in enumerate(lines) if line.startswith('#include')), default=-1)
    return ''.join(lines[last + 1:]).lstrip() or None


def local_files(salt, domain, roots, patterns, max_bytes):
    """(ident, path) of local source files in salted-hash order. roots are (folder, ident prefix,
    folders to skip)."""
    found = []
    for folder, prefix, skip in roots:
        for pattern in patterns:
            for path in folder.rglob(pattern):
                rel = path.relative_to(folder)
                if skip & set(rel.parts[:-1]) or path.stat().st_size > max_bytes:
                    continue
                found.append((prefix + str(rel), path))
    return sorted(dict(found).items(), key=lambda x: key(salt, domain, x[0]))


def heal_sources(domain, cfg):
    """(ident, item) candidates for one domain; idents match the corpus's where sources overlap."""
    f = cfg['fetch']
    if domain == 'chat':
        import pyarrow.parquet as pq
        rows = pq.read_table(mc.CHAT, columns=['prompt_id', 'messages']).to_pylist()
        for row in sorted(rows, key=lambda r: key(cfg['salt'], domain, r['prompt_id'])):
            roles = [m['role'] for m in row['messages']]
            if roles and roles == ['user', 'assistant'] * (len(roles) // 2):
                yield row['prompt_id'], row['messages']
    elif domain == 'code':
        # Top-level files keep the corpus's ident (the file name); subpackages use their path.
        for ident, path in local_files(cfg['salt'], domain, [(mc.CODE_DIR, '', SKIP_PY)], ['*.py'],
                                       cfg['max_file_bytes']):
            text = after_header(path.read_text(errors='replace'), 'python')
            if text:
                yield ident, text
    elif domain == 'code_cpp':
        roots = [(mc.CPP_DIR, '', set())] + [(MLIR_AIE / d, f'mlir-aie/{d}/', set()) for d in f['cpp_extra']]
        for ident, path in local_files(cfg['salt'], domain, roots, ['*.cpp', '*.cc', '*.h', '*.hpp'],
                                       cfg['max_file_bytes']):
            text = after_header(path.read_text(errors='replace'), 'cpp')
            if text:
                yield ident, text
    elif domain == 'math':
        yield from interleave(*[
            ((f'{c}-{n}', f'{r["problem"]}\n\nSolution: {r["solution"]}')
             for n, r in paged('EleutherAI/hendrycks_math', c, f['math'])) for c in f['math_configs']])
    elif domain == 'math_word':
        for n, r in paged('openai/gsm8k', 'main', f['math_word']):
            yield f'gsm8k-{n}', f'{r["question"]}\n\nSolution: {r["answer"]}'
    elif domain == 'wiki':
        for _, r in paged('wikimedia/wikipedia', '20231101.en', f['wiki']):
            yield f'en-{r["id"]}', f'{r["title"]}\n\n{r["text"]}'
    elif domain == 'reasoning':
        for n, r in paged('open-thoughts/OpenThoughts-114k', 'metadata', f['reasoning']):
            yield f'openthoughts-{n}', f'{r["problem"]}\n\n{r["deepseek_reasoning"]}\n\n{r["deepseek_solution"]}'
    elif domain == 'science':
        for _, r in paged('gfissore/arxiv-abstracts-2021', 'default', f['science']):
            yield f'arxiv-{r["id"]}', f'{" ".join(r["title"].split())}\n\n{r["abstract"].strip()}'
    else:
        raise ValueError(domain)


def collect(items, encode, decode, exclude, quota, cap, cfg, eot):
    """Token stream (each item capped at cap tokens, then end-of-text) of items outside exclude,
    until quota tokens; also the idents used. Every item must round-trip through the tokenizer."""
    stream, used, seen = [], [], set()
    for ident, item in items:
        if ident in exclude:
            continue
        digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        text, ids = encode(item)
        if decode(ids) != text:
            raise ValueError(f'{ident}: the 64K tokenizer does not round-trip the item')
        ids = ids[:cap]
        if len(ids) < cfg['min_item_tokens']:
            continue
        stream += ids + [eot]
        used.append(ident)
        if len(stream) >= quota:
            break
    return stream, used


def allocate(available, total):
    """Sequences per domain: equal shares, a short domain's shortfall spread over the others."""
    counts = {}
    order = sorted(available, key=lambda d: (available[d], d))
    for i, d in enumerate(order):
        counts[d] = min(available[d], (total - sum(counts.values())) // (len(order) - i))
    return {d: counts[d] for d in available}


def window_hashes(ids, n):
    """64-bit polynomial hash of every n-token window along the last axis (wraps mod 2^64)."""
    ids = np.asarray(ids, dtype=np.uint64)
    if ids.shape[-1] < n:
        return np.zeros(ids.shape[:-1] + (0,), np.uint64)
    powers = np.array([pow(1000003, k, 1 << 64) for k in range(n)], dtype=np.uint64)
    return (np.lib.stride_tricks.sliding_window_view(ids, n, axis=-1) * powers).sum(-1)


def shared_windows(seqs, references, n, rows=256):
    """Per sequence, how many of its n-token windows also occur in any reference sequence."""
    ref = np.unique(np.concatenate([window_hashes(r, n) for r in references]))
    return np.concatenate([np.isin(window_hashes(seqs[i:i + rows], n), ref).sum(-1)
                           for i in range(0, len(seqs), rows)])


def stage_data(args):
    from transformers import AutoTokenizer
    rules, docs = setup()
    cfg = rules['data']
    domains = cfg['domains']
    if set(domains) & set(cfg['held_out']) or set(domains) | set(cfg['held_out']) != set(mc.DOMAINS):
        raise ValueError('Domains must split the benchmark into in-mix and held-out')
    hf = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    full, short, _ = sb.load_tokenizers()
    eot = full.token_to_id('<|endoftext|>')
    exclude = {d: set() for d in mc.DOMAINS}
    for doc in docs:
        exclude[doc.get('source_domain', doc['domain'])].update(doc['items'])
    length = cfg['seq_len']
    total = math.ceil(cfg['tokens'] / length)
    quota = math.ceil(total / len(domains) * cfg['slack']) * length

    def encode(item):
        # The tokenizer normalizes to NFC, so the text it round-trips is the NFC text.
        text = unicodedata.normalize('NFC', item if isinstance(item, str) else hf.apply_chat_template(item, tokenize=False))
        return text, short.encode(text)

    streams, used = {}, {}
    for domain in domains:
        t0 = time.time()
        cap = cfg['max_item_tokens'].get(domain, cfg['max_item_tokens']['default'])
        streams[domain], used[domain] = collect(heal_sources(domain, cfg), encode, short.decode, exclude[domain],
                                                quota, cap, cfg, eot)
        log(f'{domain}: {len(used[domain])} items, {len(streams[domain])} tokens, {time.time() - t0:.0f} s')
    counts = allocate({d: len(s) // length for d, s in streams.items()}, total)
    seqs = np.concatenate([np.asarray(streams[d][:counts[d] * length], np.uint32).reshape(-1, length)
                           for d in domains])
    labels = np.concatenate([np.full(counts[d], i, np.uint8) for i, d in enumerate(domains)])
    evals = [[eot] + sb.retokenize(full, short, d)[2] for d in sb.docs_of(docs, 'eval')]
    # Sequences sharing any window with an eval document are dropped (near-duplicate items, boilerplate).
    shared = shared_windows(seqs, evals, cfg['overlap_tokens'])
    overlap = {d: {'sequences_dropped': int((shared[labels == i] > 0).sum()), 'windows': int(shared[labels == i].sum())}
               for i, d in enumerate(domains)}
    seqs, labels = seqs[shared == 0], labels[shared == 0]
    counts = {d: int((labels == i).sum()) for i, d in enumerate(domains)}
    np.savez(fresh('data.npz'), ids=seqs, domain=labels)
    save(dict(provenance(), data_sha256=sha(out('data.npz')), seq_len=length, sequences=int(len(seqs)),
              tokens=int(seqs.size), domains=domains, held_out=cfg['held_out'],
              per_domain={d: {'sequences': counts[d], 'available': len(streams[d]) // length,
                              'items': len(used[d])} for d in domains},
              items=used, overlap_with_eval=overlap, vocab_sha256=sha(sb.out('vocab64k.json'))), fresh('data.json'))
    log(f'{len(seqs)} sequences of {length}: ' + ', '.join(f'{d} {counts[d]}' for d in domains))
    log('dropped for sharing text with eval documents: ' +
        ', '.join(f'{d} {v["sequences_dropped"]} ({v["windows"]} windows)' for d, v in overlap.items()))


# ---------------------------------------------------------------- the student

def lowrank_factors(w, cov, rank, eps):
    """slim_batch.lowrank's rank-r W' as factors: u (out, r) and v (r, in) with W' = u v."""
    c = cov + eps * cov.diagonal().mean() * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    lower = torch.linalg.cholesky(c).float()
    u, s, vh = torch.linalg.svd(w.float() @ lower, full_matrices=False)
    return u[:, :rank] * s[:rank], torch.linalg.solve_triangular(lower, vh[:rank], upper=False, left=False)


@torch.no_grad()
def make_factors(model, docs):
    """The combined model's DeltaNet cut as factors ({layer: {projection: (u, v)}}), from the same
    calibration documents and settings slim_batch used."""
    plan = json.loads(sb.out('deltanet.json').read_text())
    chosen = plan['arms'][plan['chosen']]
    if chosen['kind'] != 'lowrank':
        raise ValueError('The combined model did not use the low-rank cut')
    dcfg = json.loads(sb.RULES.read_text())['deltanet']
    blocks = {i: layer.linear_attn for i, layer in enumerate(model.model.layers) if hasattr(layer, 'linear_attn')}
    cov = sb.input_moments(model, types.SimpleNamespace(blocks=blocks),
                           sb.docs_of(docs, 'calib', dcfg['moment_per_domain'], 2 * dcfg['select_per_domain']))
    factors, saved = {}, 0
    for i, m in blocks.items():
        factors[i] = {}
        for name in sb.PROJECTIONS:
            w = getattr(m, name).weight
            r = sb.rank_for(w, chosen['fraction'])
            u, v = lowrank_factors(w, cov[i, 0 if name != 'out_proj' else 1], r, dcfg['eps'])
            factors[i][name] = (u.to(w.dtype).contiguous(), v.to(w.dtype).contiguous())
            saved += w.numel() - r * sum(w.shape)
    if saved != chosen['saved']:
        raise RuntimeError(f'Low-rank cut saves {saved}, slim_batch saved {chosen["saved"]}')
    return factors


class Adapted(nn.Module):
    """A frozen projection that the teacher runs as is, and the student runs through its low-rank
    factors (u v) when it has them, plus a trainable LoRA update b a (when rank > 0)."""

    def __init__(self, base, owner, rank, alpha, factors=None):
        super().__init__()
        if base.bias is not None:
            raise ValueError('Projection has a bias')
        self.base, self.owner = base, owner
        self.u, self.v = factors if factors is not None else (None, None)
        self.a = self.b = None
        if rank:
            self.scale = alpha / rank
            self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
            self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
            nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)

    def forward(self, x):
        if not self.owner.on:
            return self.base(x)
        y = self.base(x) if self.u is None else F.linear(F.linear(x, self.v), self.u)
        if self.a is None:
            return y
        return y + F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype)) * self.scale

    def added_parameters(self):
        """Parameters the student stores beyond its frozen weights after merging the update: none
        for a full projection, the update's own for a factored one."""
        return 0 if self.u is None or self.a is None else self.a.numel() + self.b.numel()


class Slim:
    """The dense model as teacher (on False) and the slimmed student (on True).

    The router of each layer picks K co-firing experts from the MLP input; a forward call of more
    than one token runs every neuron at its first position (a call of one token continues a
    sequence). With stash set to a list, the student's first pass through each layer records every
    stride-th position's MLP input and expert-mass target, from which router_backward trains the
    routers; done keeps a gradient-checkpointing recompute from recording twice. With recall set
    to a list, each layer appends the share of its perfect top-K experts the router picked."""

    def __init__(self, model, s, routers, factors, rank, alpha):
        self.on, self.s, self.routers = False, s, routers
        self.stash, self.done, self.recall, self.stride, self.offset = None, set(), None, 1, 0
        self.wnorm = [layer.mlp.down_proj.weight.float().norm(dim=0) for layer in model.model.layers]
        self.adapted = {}
        for i, layer in enumerate(model.model.layers):
            for part in ('self_attn', 'linear_attn', 'mlp'):
                parent = getattr(layer, part, None)
                for name, module in list(parent.named_children()) if parent is not None else []:
                    low = factors.get(i, {}).get(name) if part == 'linear_attn' else None
                    lora = isinstance(module, nn.Linear) and min(module.in_features, module.out_features) > 2 * rank
                    if lora or low is not None:
                        setattr(parent, name, Adapted(module, self, rank if lora else 0, alpha, low))
                        self.adapted[f'{i}.{part}.{name}'] = getattr(parent, name)
            layer.mlp.forward = lambda x, i=i, m=layer.mlp: self.mlp(i, m, x)
        if sum(m.u is not None for m in self.adapted.values()) != sum(map(len, factors.values())):
            raise RuntimeError('Not every low-rank factor was installed')

    def mlp(self, i, m, x):
        a = m.act_fn(m.gate_proj(x)) * m.up_proj(x)
        if not self.on:
            return m.down_proj(a)
        with torch.no_grad():
            keep = self.select(i, a, x)
        return m.down_proj(a * keep)

    def select(self, i, a, x):
        """(B, T, N) mask of the neurons the student runs."""
        s, k = self.s, mc.at(self.s, 'K', i)
        b, t, n = a.shape
        chosen = mc.topmask(self.routers[i](x.flatten(0, 1).float()), k)
        keep = mc.expand(s, i, chosen).view(b, t, n)
        if t > 1:
            keep[:, 0] = True
        record = self.stash is not None and i not in self.done
        if t > 1 and (record or self.recall is not None):
            mass = mc.expert_mass(s, i, (a[:, 1:].float().abs() * self.wnorm[i]).flatten(0, 1))
            if self.recall is not None:
                picked = chosen.view(b, t, -1)[:, 1:].flatten(0, 1)
                self.recall.append(float((mc.topmask(mass, k) & picked).sum() / (mass.shape[0] * k)))
            if record:
                self.done.add(i)
                rows = torch.arange(self.offset % self.stride, mass.shape[0], self.stride, device=mass.device)
                target = mass[rows] / mass[rows].sum(-1, keepdim=True).clamp_min(1e-12)
                self.stash.append((i, x[:, 1:].flatten(0, 1)[rows], target))
        return keep

    def router_backward(self, scale):
        """Accumulate router gradients from the stash; returns the mean router loss."""
        losses = []
        for i, x, target in self.stash or []:
            loss = -(target * self.routers[i](x.float()).log_softmax(-1)).sum(-1).mean()
            (loss * scale).backward()
            losses.append(float(loss.detach()))
        self.stash, self.done = None, set()
        return float(np.mean(losses)) if losses else float('nan')

    def lora_parameters(self):
        return [p for m in self.adapted.values() if m.a is not None for p in (m.a, m.b)]

    def router_parameters(self):
        return [p for r in self.routers for p in r.parameters()]

    def state(self, dtype=None):
        """LoRA factors (cast to dtype if given) and router weights, on the CPU."""
        cast = lambda t: t.detach().to(dtype or t.dtype).cpu()
        return {'lora': {n: (cast(m.a), cast(m.b)) for n, m in self.adapted.items() if m.a is not None},
                'routers': [{k: v.detach().cpu() for k, v in r.state_dict().items()} for r in self.routers]}

    def load(self, state):
        if set(state['lora']) != {n for n, m in self.adapted.items() if m.a is not None}:
            raise ValueError('Adapter does not match the student')
        with torch.no_grad():
            for n, (a, b) in state['lora'].items():
                self.adapted[n].a.copy_(a)
                self.adapted[n].b.copy_(b)
        for r, sd in zip(self.routers, state['routers'], strict=True):
            r.load_state_dict(sd)


def build(model, rules, factors):
    cfg = rules['student']
    if cfg['budget'] != json.loads(sb.CARVE_RULES.read_text())['route']['budget']:
        raise ValueError('The routers were trained for another budget')
    s = torch.load(mc.OUT / 'sets.pt', weights_only=True, map_location=mc.DEVICE)[cfg['budget']]
    routers = []
    for sd in torch.load(mc.OUT / 'routers.pt', weights_only=True, map_location=mc.DEVICE):
        router = nn.Linear(model.config.hidden_size, s['experts']).to(mc.DEVICE)
        router.load_state_dict(sd)
        routers.append(router)
    torch.manual_seed(cfg['seed'])
    return Slim(model, s, routers, factors, cfg['lora_rank'], cfg['lora_alpha'])


class Vocab:
    """Both tokenizers, the kept rows and their index, and the 64K output head."""

    def __init__(self, model):
        self.full, self.short, kept = sb.load_tokenizers()
        self.kept, self.index = sb.kept_index(kept, model.config.vocab_size)
        self.head = model.lm_head.weight[self.kept].detach()
        self.eot = self.full.token_to_id('<|endoftext|>')

    def trimmed(self, doc):
        """The document's text, full-tokenizer ids and 64K-tokenizer ids (original numbering)."""
        return sb.retokenize(self.full, self.short, doc)


def load_student(rules):
    model = mc.load_model(rules['threads'])
    factors = torch.load(factors_path(), weights_only=True, map_location=mc.DEVICE)
    slim = build(model, rules, factors)
    slim.load(torch.load(state_path(), weights_only=True, map_location=mc.DEVICE))
    return model, slim, Vocab(model)


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, slim, vocab, docs):
    """Per eval document: the student against the dense model on the original tokens (full head,
    as slim_batch scored the combined body) and on the 64K tokens (kept rows), and its routers'
    recall of its own perfect experts."""
    model.eval()
    rows = []
    for doc in docs:
        ids = doc['input_ids']
        slim.on = False
        dense = model.lm_head(sb.hidden(model, [ids])[0]).float()
        slim.on, slim.recall = True, []
        student = model.lm_head(sb.hidden(model, [ids])[0]).float()
        row = sb.score_arms({'dense': dense, 'student': student}, ids, doc)
        row['agree'] = float((student[:-1].argmax(-1) == dense[:-1].argmax(-1)).float().mean())
        row['recall'], slim.recall = float(np.mean(slim.recall)), None
        del dense, student
        text, a, b = vocab.trimmed(doc)
        x, target = [vocab.eot] + b, vocab.index[sb.tensor(b)]
        slim.on = False
        nll_full = sb.nll_sum(model, [vocab.eot] + a)
        dense = F.linear(sb.hidden(model, [x])[0, :-1], vocab.head).float()
        slim.on = True
        student = F.linear(sb.hidden(model, [x])[0, :-1], vocab.head).float()
        slim.on = False
        row['trimmed'] = {'bytes': len(text.encode()), 'tokens_full': len(a), 'tokens_trimmed': len(b),
                          'nll_full': nll_full, 'nll_dense': float(F.cross_entropy(dense, target, reduction='sum')),
                          'nll_student': float(F.cross_entropy(student, target, reduction='sum')),
                          'kl': float(token_kl(dense, student).mean()),
                          'agree': float((student.argmax(-1) == dense.argmax(-1)).float().mean())}
        rows.append(row)
    return rows


def summarize(rows, held_out):
    """Domain-equal numbers of one evaluation, overall and for the in-mix and held-out domains."""
    arm = mc.arm_summary(rows, ['dense', 'student'])['student']
    text = [r for r in rows if r['domain'] != 'copy']
    flat = [dict(r['trimmed'], domain=r['domain']) for r in text]
    dense, student = sb.vocab_summary(flat, 'nll_dense'), sb.vocab_summary(flat, 'nll_student')
    groups = [r['domain'] for r in text]
    by_domain = {d: {'kl': arm['kl_by_domain'][d], 'bpb_full': dense['by_domain'][d]['bpb_full'],
                     'bpb_dense_64k': dense['by_domain'][d]['bpb_trimmed'],
                     'bpb_student': student['by_domain'][d]['bpb_trimmed'],
                     **{k: float(np.mean([r[k] for r in text if r['domain'] == d])) for k in ('agree', 'recall')},
                     **{f'{k}_64k': float(np.mean([r['trimmed'][k] for r in text if r['domain'] == d]))
                        for k in ('kl', 'agree')}} for d in mc.DOMAINS}
    part = lambda ds: {k: float(np.mean([by_domain[d][k] for d in ds])) for k in by_domain[ds[0]]}
    return {'kl': arm['kl'], 'delta_nll': arm['delta_nll'], 'copy_mean': arm['copy_mean'], 'copy_p99': arm['copy_p99'],
            'agree': group_mean([r['agree'] for r in text], groups),
            'recall': group_mean([r['recall'] for r in text], groups),
            'kl_64k': group_mean([r['trimmed']['kl'] for r in text], groups),
            'agree_64k': group_mean([r['trimmed']['agree'] for r in text], groups),
            'bpb_full': dense['bpb_full'], 'bpb_dense_64k': dense['bpb_trimmed'], 'bpb_student': student['bpb_trimmed'],
            'in_mix': part([d for d in mc.DOMAINS if d not in held_out]), 'held_out': part(list(held_out)),
            'by_domain': by_domain}


# ---------------------------------------------------------------- training

def chunk_loss(hs, ht, target, head):
    """Summed KL from the teacher's kept-row distribution and summed cross-entropy of target."""
    s = F.linear(hs, head).float().log_softmax(-1)
    with torch.no_grad():
        t = F.linear(ht, head).float().log_softmax(-1)
    return (t.exp() * (t - s)).sum(), -s.gather(-1, target[:, None]).sum()


def micro_step(model, slim, vocab, x, cfg, scale):
    """Teacher and student passes over x (B, L), backward of the scaled loss; returns the summed
    KL and cross-entropy (floats) and the mean router loss."""
    slim.on = False
    model.eval()
    with torch.no_grad():
        ht = model.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1].flatten(0, 1)
    slim.on, slim.stash = True, []
    model.train()
    hs = model.model(input_ids=x, use_cache=False).last_hidden_state[:, :-1].flatten(0, 1)
    target = vocab.index[x[:, 1:]].flatten()
    if (target < 0).any():
        raise ValueError('Healing ids outside the kept rows')
    kl = ce = 0
    for c in range(0, len(target), cfg['loss_chunk']):
        part = slice(c, c + cfg['loss_chunk'])
        k, e = checkpoint(chunk_loss, hs[part], ht[part], target[part], vocab.head, use_reentrant=False)
        kl, ce = kl + k, ce + e
    ((kl + cfg['ce_weight'] * ce) * scale).backward()
    router = slim.router_backward(scale * len(target) * cfg['router_weight'])
    slim.on = False
    return float(kl.detach()), float(ce.detach()), router


def schedule(step, steps, warmup, floor):
    if step < warmup:
        return (step + 1) / warmup
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def stage_heal(args):
    rules, docs = setup()
    cfg = rules['train']
    record = json.loads(out('data.json').read_text())
    if sha(out('data.npz')) != record['data_sha256']:
        raise RuntimeError('Healing data changed after the data stage')
    seqs = np.load(out('data.npz'))['ids']
    t0 = time.time()
    stop_at = min(t0 + 3600 * cfg['max_hours'],
                  float(os.environ.get('MANIFOLD_JOB_DEADLINE', 'inf')) - 60 * cfg['reserve_minutes'])
    model = mc.load_model(rules['threads'])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    factors = make_factors(model, docs)
    torch.save({i: {n: (u.cpu(), v.cpu()) for n, (u, v) in f.items()} for i, f in factors.items()}, factors_path())
    slim = build(model, rules, factors)
    vocab = Vocab(model)
    evals = sb.docs_of(docs, 'eval')
    held_out = rules['data']['held_out']
    combined = json.loads(sb.out('combined.json').read_text())
    milestones, losses = [], []

    def milestone(tokens, steps):
        t = time.time()
        rows = evaluate(model, slim, vocab, evals)
        with fresh(f'eval_{tokens}.jsonl').open('w') as stream:
            stream.write(''.join(json.dumps(r) + '\n' for r in rows))
        summary = dict(summarize(rows, held_out), tokens=tokens, steps=steps, seconds=time.time() - t)
        milestones.append(summary)
        save({'milestones': milestones, 'loss_by_step': losses}, out('progress.json'))
        log(f'{tokens} tokens: KL {summary["kl"]:.3f} (in-mix {summary["in_mix"]["kl"]:.3f}, held-out '
            f'{summary["held_out"]["kl"]:.3f}), 64K bits per byte {summary["bpb_student"]:.4f}, '
            f'agreement {summary["agree"]:.3f}, router recall {summary["recall"]:.3f}')
        return summary

    first = milestone(0, 0)
    check = {'kl': first['kl'] - combined['body']['kl'],
             'bpb': first['bpb_student'] / combined['vocab']['bpb_trimmed'] - 1}
    if abs(check['kl']) > cfg['check']['kl'] or abs(check['bpb']) > cfg['check']['bpb']:
        raise RuntimeError(f'The student does not reproduce the combined model: {check}')
    params = slim.lora_parameters()
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], betas=tuple(cfg['betas']), weight_decay=0.0)
    router_optimizer = torch.optim.Adam(slim.router_parameters(), lr=cfg['router_lr'])
    length, batch = seqs.shape[1], cfg['batch_sequences']
    steps = min(len(seqs) // batch, math.ceil(cfg['tokens'] / (batch * length)))
    order = np.random.default_rng(cfg['seed']).permutation(len(seqs))
    slim.stride = cfg['router_stride']
    micro, tokens, step, stopped = cfg['micro_batch'], 0, 0, 'tokens'
    marks = sorted(m for m in cfg['milestones'] if m > 0)
    with deadline(cfg['max_hours'] + 0.5):
        while step < steps:
            if time.time() > stop_at:
                stopped = 'time'
                break
            factor = schedule(step, steps, cfg['warmup_steps'], cfg['lr_floor'])
            for group in optimizer.param_groups:
                group['lr'] = cfg['lr'] * factor
            for group in router_optimizer.param_groups:
                group['lr'] = cfg['router_lr'] * factor
            chosen = order[step * batch:(step + 1) * batch]
            t = time.time()
            try:
                totals = [0., 0., []]
                for j in range(0, batch, micro):
                    x = torch.from_numpy(seqs[chosen[j:j + micro]].astype(np.int64)).to(mc.DEVICE)
                    slim.offset = step + j
                    kl, ce, router = micro_step(model, slim, vocab, x, cfg, 1 / (batch * (length - 1)))
                    totals[0] += kl
                    totals[1] += ce
                    totals[2].append(router)
            except torch.cuda.OutOfMemoryError:
                if micro == 1:
                    raise
                optimizer.zero_grad(set_to_none=True)
                router_optimizer.zero_grad(set_to_none=True)
                slim.on, slim.stash, slim.done = False, None, set()
                torch.cuda.empty_cache()
                micro //= 2
                log(f'Out of memory: micro-batch now {micro}')
                continue
            norm = float(torch.nn.utils.clip_grad_norm_(params, cfg['clip']))
            optimizer.step()
            router_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            router_optimizer.zero_grad(set_to_none=True)
            step += 1
            tokens += batch * length
            n = batch * (length - 1)
            losses.append({'step': step, 'tokens': tokens, 'kl': totals[0] / n, 'ce': totals[1] / n,
                           'router': float(np.mean(totals[2])), 'grad_norm': norm, 'lr': cfg['lr'] * factor,
                           'seconds': time.time() - t})
            if step % cfg['log_every'] == 0 or step == 1:
                rate = batch * length / losses[-1]['seconds']
                memory = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
                log(f'step {step}/{steps}: KL {losses[-1]["kl"]:.4f}, CE {losses[-1]["ce"]:.4f}, router '
                    f'{losses[-1]["router"]:.4f}, norm {norm:.2f}, {rate:.0f} tokens/s, {memory:.1f} GB')
            while marks and tokens >= marks[0]:
                marks.pop(0)
                torch.save(slim.state(), state_path())
                milestone(tokens, step)
        if not milestones or milestones[-1]['tokens'] != tokens:
            torch.save(slim.state(), state_path())
            milestone(tokens, step)
    torch.save(slim.state(torch.bfloat16), fresh('adapter.pt'))
    factored = sum(m.added_parameters() for m in slim.adapted.values())
    save(dict(provenance(), data_sha256=record['data_sha256'], factors_sha256=sha(factors_path()),
              state_sha256=sha(state_path()), adapter_sha256=sha(out('adapter.pt')), reproduces_combined=check,
              trainable={'lora': sum(p.numel() for p in params), 'routers': sum(p.numel() for p in slim.router_parameters())},
              added_parameters=factored, planned_steps=steps, steps=step, tokens=tokens, stopped=stopped,
              final_micro_batch=micro, milestones=milestones, loss_by_step=losses, seconds=time.time() - t0),
         fresh('heal.json'))
    log(f'Healed on {tokens} tokens ({stopped}); KL {milestones[0]["kl"]:.3f} -> {milestones[-1]["kl"]:.3f}')


# ---------------------------------------------------------------- after healing

@torch.no_grad()
def cache_run(model, slim, vocab, calib, evals, cfg):
    """slim_batch's row-cache measurement on the healed student in its own setting: 64K tokens,
    64K head, the detector trained on the student's calibration positions."""
    size = len(vocab.kept)
    caches = {n: sb.RowCache(n, size) for n in cfg['sizes']}
    hidden_size = model.config.hidden_size
    xtx = torch.zeros(hidden_size + 1, hidden_size + 1, dtype=torch.float64, device=mc.DEVICE)
    xty = torch.zeros(hidden_size + 1, dtype=torch.float64, device=mc.DEVICE)
    train_h, train, seen = [], {n: [] for n in cfg['sizes']}, 0
    slim.on = True

    def run(doc):
        x = [vocab.eot] + vocab.trimmed(doc)[2]
        h = sb.hidden(model, [x])[0, :-1]
        return h, F.linear(h, vocab.head).float(), vocab.index[sb.tensor(x)].tolist()

    for doc in calib:
        h, logits, ids = run(doc)
        b = sb.with_bias(h)
        xtx += (b.T @ b).double()
        xty += (b.T @ logits.logsumexp(-1)).double()
        for n, cache in caches.items():
            f = sb.cache_features(logits, cache.stream(ids)[:-1], sb.tensor(ids[1:]))
            if seen >= cfg['warm_tokens']:
                train[n].append({k: f[k].cpu() for k in ('max', 'gap', 'lse', 'entropy', 'pick_out')})
        if seen >= cfg['warm_tokens']:
            train_h.append(h.to(torch.bfloat16).cpu())
        seen += len(ids)
    ridge = sb.fit_ridge(xtx, xty, cfg['ridge'])
    predicted = torch.cat([sb.with_bias(c.to(mc.DEVICE)) @ ridge for c in train_h])
    detectors = {}
    for n in cfg['sizes']:
        f = {k: torch.cat([p[k] for p in train[n]]).to(mc.DEVICE) for k in train[n][0]}
        detectors[n] = sb.fit_detector(sb.detector_inputs(f, predicted), f['pick_out'], cfg['recall'], cfg['seed'])
    per_doc = {n: [] for n in cfg['sizes']}
    for doc in evals:
        h, logits, ids = run(doc)
        predicted = sb.with_bias(h) @ ridge
        for n, cache in caches.items():
            f = sb.cache_features(logits, cache.stream(ids)[:-1], sb.tensor(ids[1:]))
            fallback = sb.detect(detectors[n], sb.detector_inputs(f, predicted))
            per_doc[n].append((doc['domain'], {k: v for k, v in f.items() if k != 'lse_full'}, fallback))
    slim.on = False
    return {str(n): sb.domain_outcomes(per_doc[n], list(mc.DOMAINS) + ['copy'],
                                       lambda f, fallback, n=n: sb.cache_outcome(f, fallback, {'cache': n, 'head_64k': size}))
            for n in cfg['sizes']}, {n: d['threshold'] for n, d in detectors.items()}


@torch.no_grad()
def state_check(model, slim, docs, variant, seed):
    """KL from dense of the healed student with fp32 and with stored state, and between the two."""
    rows = []
    for doc in docs:
        ids = doc['input_ids']
        slim.on = False
        dense = model.lm_head(sb.hidden(model, [ids])[0]).float()
        slim.on = True
        stock = model.lm_head(sb.hidden(model, [ids])[0]).float()
        with sb.StoredState(model, [variant], seed):
            stored = model.lm_head(sb.hidden(model, [ids])[0]).float()
        slim.on = False
        rows.append({'id': doc['id'], 'domain': doc['domain'], 'fp32': float(token_kl(dense, stock)[1:].mean()),
                     variant: float(token_kl(dense, stored)[1:].mean()),
                     'between': float(token_kl(stock, stored)[1:].mean())})
    groups = [r['domain'] for r in rows]
    return {k: group_mean([r[k] for r in rows], groups) for k in ('fp32', variant, 'between')}


def stage_final(args):
    rules, docs = setup()
    cfg = rules['final']
    t0 = time.time()
    model, slim, vocab = load_student(rules)
    model.eval()
    with deadline(cfg['max_hours']):
        state = state_check(model, slim, sb.docs_of(docs, 'eval', cfg['state_docs_per_domain']), cfg['state'],
                            cfg['seed'])
        log(f'state: fp32 {state["fp32"]:.4f}, {cfg["state"]} {state[cfg["state"]]:.4f}, between {state["between"]:.2e}')
        cache, thresholds = cache_run(model, slim, vocab, sb.docs_of(docs, 'calib'), sb.docs_of(docs, 'eval'),
                                      cfg['cache'])
    for n, c in cache.items():
        log(f'{n} rows: ' + ', '.join(f'{k} {v:.4f}' for k, v in c['mean'].items() if not isinstance(v, dict)))
    save(dict(provenance(), state=state, cache=cache, thresholds=thresholds, seconds=time.time() - t0),
         fresh('final.json'))


def stage_generate(args):
    from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList
    rules, _ = setup()
    cfg = rules['generate']
    prompts = json.loads((ROOT / cfg['prompts']).read_text())['prompts'][:cfg['count']]
    tok = AutoTokenizer.from_pretrained(mc.MODEL, local_files_only=True)
    model, slim, vocab = load_student(rules)
    model.eval()
    stop = [tok.convert_tokens_to_ids(t) for t in ('<|im_end|>', '<|endoftext|>')]
    removed = torch.full((model.config.vocab_size,), -torch.inf, device=mc.DEVICE)
    removed[vocab.kept] = 0

    class KeptRows(LogitsProcessor):
        def __call__(self, input_ids, scores):
            return scores + removed

    path = out('generate.jsonl')
    done = {json.loads(line)['id'] for line in path.open()} if path.exists() else set()
    slim.on = True
    with deadline(cfg['max_hours']):
        for p in prompts:
            if p['id'] in done:
                continue
            text = tok.apply_chat_template([{'role': 'user', 'content': p['question']}], tokenize=False,
                                           add_generation_prompt=True)
            ids = vocab.short.encode(text)
            t0 = time.time()
            with torch.no_grad(), sb.StoredState(model, [cfg['state']], cfg['seed']):
                x = sb.tensor([ids])
                result = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=cfg['max_new_tokens'],
                                        do_sample=False, temperature=None, top_p=None, top_k=None, eos_token_id=stop,
                                        pad_token_id=stop[1], logits_processor=LogitsProcessorList([KeptRows()]))
            tokens = result[0, len(ids):].tolist()
            end = next((i for i, t in enumerate(tokens) if t in stop), None)
            tokens = tokens[:end] if end is not None else tokens
            answer = sb.final_number(tok.decode(tokens))
            row = {'id': p['id'], 'seconds': time.time() - t0, 'tokens': tokens, 'length': len(tokens),
                   'finished': end is not None, 'answer': answer,
                   'correct': answer is not None and abs(answer - p['answer']) < 1e-6}
            with path.open('a') as f:
                f.write(json.dumps(row) + '\n')
            log(f'{p["id"]}: {row["length"]} tokens{"" if row["finished"] else " (cap)"}, '
                f'{"right" if row["correct"] else "wrong"}, {row["seconds"]:.0f} s')
    slim.on = False


# ---------------------------------------------------------------- report

def stage_report(args):
    from transformers import AutoConfig
    rules, _ = setup()
    read = lambda path: json.loads(path.read_text())
    heal, slim_report, combined = read(out('heal.json')), read(sb.out('report.json')), read(sb.out('combined.json'))
    config = AutoConfig.from_pretrained(mc.MODEL, local_files_only=True)
    row = getattr(config, 'text_config', config).hidden_size
    report = {'data': {k: v for k, v in read(out('data.json')).items() if k != 'items'},
              'heal': {k: v for k, v in heal.items() if k != 'loss_by_step'}}
    # The LoRA update of a factored DeltaNet projection adds to its rank; every other update merges.
    unhealed = slim_report['parameters']['steps']['deltanet']
    healed = {k: v + heal['added_parameters'] for k, v in unhealed.items()}
    if out('final.json').exists():
        report['final'] = read(out('final.json'))
        rows = report['final']['cache'][str(rules['final']['cache']['report_size'])]['mean']['rows_per_token']
        healed['active_total'] += (rows['head_64k'] - combined['cache']['mean']['rows_per_token']['head_64k']) * row
    report['parameters'] = {'unhealed': unhealed, 'healed': healed}
    if out('generate.jsonl').exists():
        rows = [json.loads(line) for line in out('generate.jsonl').open()]
        dense = {r['id']: r['outputs']['fp32'] for r in map(json.loads, sb.out('generate.jsonl').open())}
        pick = lambda rs, k: [r[k] for r in rs]
        report['generate'] = {
            'prompts': len(rows), 'correct': sum(pick(rows, 'correct')), 'finished': sum(pick(rows, 'finished')),
            'mean_length': float(np.mean(pick(rows, 'length'))),
            'dense_fp32': {'correct': sum(dense[r['id']]['correct'] for r in rows),
                           'finished': sum(dense[r['id']]['finished'] for r in rows),
                           'mean_length': float(np.mean([dense[r['id']]['length'] for r in rows]))}}
    save(dict(provenance(), **report), fresh('report.json'))
    for m in heal['milestones']:
        log(f'{m["tokens"]:>9} tokens: KL {m["kl"]:.3f}, 64K bits per byte {m["bpb_student"]:.4f}, '
            f'agreement {m["agree"]:.3f}, recall {m["recall"]:.3f}')


def main():
    global RULES
    stages = {'data': stage_data, 'heal': stage_heal, 'final': stage_final, 'generate': stage_generate,
              'report': stage_report}
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=stages)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--model', type=Path, default=mc.MODEL)
    ap.add_argument('--rules', type=Path, default=RULES)
    ap.add_argument('--slim-rules', type=Path, default=sb.RULES)
    ap.add_argument('--carve-rules', type=Path, default=sb.CARVE_RULES)
    args = ap.parse_args()
    mc.DEVICE, mc.MODEL = args.device, args.model
    RULES = args.rules.resolve()
    sb.RULES, sb.CARVE_RULES = args.slim_rules.resolve(), args.carve_rules.resolve()
    stages[args.stage](args)


if __name__ == '__main__':
    main()
