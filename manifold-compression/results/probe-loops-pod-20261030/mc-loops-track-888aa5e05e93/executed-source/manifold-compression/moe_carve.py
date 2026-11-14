"""M1: carve a dense model's MLPs into experts by domain, and compare with other carvings
at the same active width.

Model: Qwen3.5-9B, the same qwen3_5_text architecture as Qwen3.8-27B. Every arm keeps the
dense weights and only chooses which MLP neurons run for each token:

  static         one fixed set for every token: the top neurons by mean contribution
                 (a cut-and-keep prune with no refit);
  domain_doc     the document's domain set: the top neurons by mean contribution within
                 that domain, picked by the true domain label (the thesis with a perfect
                 document-level router);
  domain_token   per token, whichever of the domain sets captures the most of that
                 token's contribution (the thesis with a perfect token router);
  cluster_token  CMoE-style: a shared expert of the most often active neurons plus 64 equal
                 clusters of co-active neurons; per token, the K clusters with the most
                 contribution;
  token_topk     per token, its own top neurons (a bound for any router).

A neuron's contribution to a token is |a_i| * ||W_down[:, i]||, with a = silu(gate x) * (up x).
At one budget every arm runs exactly the same number of neurons per token, and position 0
always runs all of them. The score is the KL of the final next-token distribution from the
dense model, with every layer carved at once.

Rules with a depth section add two co-firing carvings that size experts by depth, each with
the primary budget's total active neurons: depth_shared sizes each layer's shared expert by how
much of that layer the domain sets share, and depth_budget moves active neurons between layers
so that the summed share of contribution captured by each token's top neurons is largest.

Stages: fetch (rows from the datasets server), prepare (corpus), collect (calibration
statistics), build (expert sets), evaluate, report, route (a learned linear router per layer
for the primary budget's co-firing experts, scored against the perfect one). --device picks
where the model runs.

--rules picks the run: its output folder and domain list. The first run's rules
(moe_carve_rules.json) name neither and keep results/moe-carve-20261028 and its eight domains,
French and Chinese included. moe_carve_rules_v2.json is English, symbols and code only:
French and Chinese are replaced by reasoning traces, science abstracts, C++ and word-problem math.
"""
import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from h0_mlp_probe import deadline, group_mean, key, log, paired_interval, save, sha, token_kl

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL = REPO / 'models/manifold-audit/Qwen3.5-9B'
OUT = ROOT / 'results/moe-carve-20261028'
RULES = ROOT / 'moe_carve_rules.json'
CODE = ['moe_carve.py', 'test_moe_carve.py', 'h0_mlp_probe.py']
CHAT = REPO / 'target/manifold-audit/ultrachat/data/train_sft-00000-of-00003-a3ecf92756993583.parquet'
HUB = Path.home() / '.cache/huggingface/hub'
BIOMED = HUB / ('datasets--BeIR--trec-covid/snapshots/7e16fde3016c639c7f856e803f4bab92645562c4/'
                'corpus/corpus-00000-of-00001.parquet')
ARGUMENT = HUB / ('datasets--BeIR--webis-touche2020/snapshots/7ebed360c2743a0024cb96a404eb89e27c931ebc/'
                  'corpus/corpus-00000-of-00001.parquet')
CODE_DIR = Path('/usr/lib64/python3.14')
CPP_DIR = REPO.parent / 'openflowlm-next/third_party/mlir-aie/lib'
SERVER = 'https://datasets-server.huggingface.co/rows'
DOMAINS = ('chat', 'code', 'math', 'wiki', 'biomed', 'argument', 'french', 'chinese')
# Domains whose rows the fetch stage downloads into sources/<domain>.jsonl.
FETCHED = ('math', 'math_word', 'wiki', 'french', 'chinese', 'reasoning', 'science')
POLICIES = ('static', 'domain_doc', 'domain_token', 'cluster_token', 'token_topk')
COPY_PROMPT = '\n\nRepeat the passage above exactly.\n\n'
DEVICE = 'cpu'


def configure(path):
    """Point every stage at one rules file, its output folder and its domain list."""
    global RULES, OUT, DOMAINS
    rules = json.loads(path.read_text())
    RULES = path.resolve()
    OUT = ROOT / rules.get('out', 'results/moe-carve-20261028')
    DOMAINS = tuple(rules.get('domains', DOMAINS))


def provenance():
    return {'code_sha256': {n: sha(ROOT / n) for n in CODE}, 'rules_sha256': sha(RULES)}


def require(stage):
    """The record of an earlier stage, checked against the current rules and corpus."""
    record = json.loads((OUT / f'{stage}.json').read_text())
    if record['rules_sha256'] != sha(RULES):
        raise RuntimeError(f'Rules changed after {stage}')
    if record.get('corpus_sha256') and record['corpus_sha256'] != sha(OUT / 'corpus.json'):
        raise RuntimeError(f'Corpus changed after {stage}')
    return record


# ---------------------------------------------------------------- corpus

def fetch_rows(dataset, config, offset, length, split='train'):
    query = urllib.parse.urlencode({'dataset': dataset, 'config': config, 'split': split,
                                    'offset': offset, 'length': length})
    with urllib.request.urlopen(f'{SERVER}?{query}', timeout=120) as r:
        return [row['row'] for row in json.load(r)['rows']]


def stage_fetch(args):
    rules = json.loads(RULES.read_text())['fetch']
    folder = OUT / 'sources'
    if folder.exists():
        raise ValueError('Sources exist; refusing to overwrite')
    rows = {d: [] for d in FETCHED if d in DOMAINS}
    if 'math' in rows:
        for config in rules['math_configs']:
            for i, r in enumerate(fetch_rows('EleutherAI/hendrycks_math', config, 0, rules['math_rows'])):
                rows['math'].append({'id': f'{config}-{i}', 'text': f'{r["problem"]}\n\nSolution: {r["solution"]}'})
    if 'math_word' in rows:
        for start in range(0, rules['math_word_rows'], 100):
            for i, r in enumerate(fetch_rows('openai/gsm8k', 'main', start, 100)):
                rows['math_word'].append({'id': f'gsm8k-{start + i}',
                                          'text': f'{r["question"]}\n\nSolution: {r["answer"]}'})
    for domain, lang in (('wiki', 'en'), ('french', 'fr'), ('chinese', 'zh')):
        if domain not in rows:
            continue
        for offset in rules['wiki_offsets']:
            for start in range(offset, offset + rules['wiki_rows_per_offset'], 100):
                for r in fetch_rows('wikimedia/wikipedia', f'20231101.{lang}', start, 100):
                    rows[domain].append({'id': f'{lang}-{r["id"]}', 'text': f'{r["title"]}\n\n{r["text"]}'})
    if 'reasoning' in rows:
        # OpenThoughts-114k keeps its rows grouped by domain; each window starts inside one group
        # (code, biology, physics, chemistry, puzzles), so the math majority is left out.
        for window in rules['reasoning_windows']:
            for start in range(window, window + rules['reasoning_rows_per_window'], 100):
                for i, r in enumerate(fetch_rows('open-thoughts/OpenThoughts-114k', 'metadata', start, 100)):
                    rows['reasoning'].append({'id': f'openthoughts-{start + i}', 'source': r['source'],
                                              'text': f'{r["problem"]}\n\n{r["deepseek_reasoning"]}\n\n'
                                                      f'{r["deepseek_solution"]}'})
    if 'science' in rows:
        for offset in rules['science_offsets']:
            for start in range(offset, offset + rules['science_rows_per_offset'], 100):
                for r in fetch_rows('gfissore/arxiv-abstracts-2021', 'default', start, 100):
                    rows['science'].append({'id': f'arxiv-{r["id"]}',
                                            'text': f'{" ".join(r["title"].split())}\n\n{r["abstract"].strip()}'})
    folder.mkdir(parents=True)
    for domain, items in rows.items():
        (folder / f'{domain}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
        log(f'{domain}: {len(items)} rows')


def candidates():
    """(domain, id, text or chat messages) for every source item."""
    import pyarrow.parquet as pq
    for row in pq.read_table(CHAT, columns=['prompt_id', 'messages']).to_pylist():
        roles = [m['role'] for m in row['messages']]
        if roles and roles == ['user', 'assistant'] * (len(roles) // 2):
            yield 'chat', row['prompt_id'], row['messages']
    for path in sorted(CODE_DIR.glob('*.py')):
        # Skip the module docstring and imports: start at the first top-level definition.
        text = path.read_text(errors='replace')
        start = min((i for i in (text.find('\ndef '), text.find('\nclass ')) if i >= 0), default=-1)
        if start >= 0:
            yield 'code', path.name, text[start + 1:]
    if 'code_cpp' in DOMAINS:
        for path in sorted(CPP_DIR.rglob('*.cpp')):
            # Skip the licence header and includes: start after the last #include line.
            lines = path.read_text(errors='replace').splitlines(keepends=True)
            last = max((i for i, line in enumerate(lines) if line.startswith('#include')), default=-1)
            yield 'code_cpp', str(path.relative_to(CPP_DIR)), ''.join(lines[last + 1:]).lstrip()
    for domain in FETCHED:
        if domain not in DOMAINS:
            continue
        for row in map(json.loads, (OUT / 'sources' / f'{domain}.jsonl').open()):
            yield domain, row['id'], row['text']
    for domain, path in (('biomed', BIOMED), ('argument', ARGUMENT)):
        for row in pq.read_table(path, columns=['_id', 'title', 'text']).to_pylist():
            yield domain, row['_id'], f'{row["title"]}\n\n{row["text"]}'.strip()


def encode(tokenizer, item):
    if isinstance(item, str):
        return tokenizer(item, add_special_tokens=False)['input_ids']
    text = tokenizer.apply_chat_template(item, tokenize=False)
    return tokenizer(text, add_special_tokens=False)['input_ids']


def build_corpus(items, tokenizer, rules):
    """Fixed-length documents packed from whole source items, split by item.

    Each domain's items are ordered by a salted hash, which also assigns the split (one in
    eval_every to eval), so no item reaches both splits. An item contributes at most one
    document's worth of tokens; short items are packed with a blank line between them, and
    the tail of a document's last item is dropped. Copy docs repeat a passage from an item
    of the copy source domain that is in neither split; copy_span marks the second copy.
    """
    c = rules['corpus']
    salt, length = c['salt'], c['length']
    quota = {'calib': c['calib_docs'], 'eval': c['eval_docs']}
    pools = {d: [] for d in DOMAINS}
    seen = set()
    for domain, ident, item in items:
        if domain not in pools:
            continue
        digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            pools[domain].append((key(salt, domain, ident), ident, item))
    sep = tokenizer('\n\n', add_special_tokens=False)['input_ids']
    docs, spare = [], {}
    for domain in DOMAINS:
        made = dict.fromkeys(quota, 0)
        buffer = {s: [] for s in quota}
        used = {s: [] for s in quota}
        rest = []
        for h, ident, item in sorted(pools[domain]):
            split = 'eval' if int(h[:8], 16) % c['eval_every'] == 0 else 'calib'
            if made[split] == quota[split]:
                rest.append((h, ident, item))
                continue
            ids = encode(tokenizer, item)[:length]
            if len(ids) < c['min_item_tokens']:
                continue
            buffer[split] += (sep if buffer[split] else []) + ids
            used[split].append(ident)
            if len(buffer[split]) >= length:
                docs.append({'id': f'{domain}-{split}-{made[split]}', 'domain': domain, 'split': split,
                             'items': used[split], 'input_ids': buffer[split][:length]})
                made[split] += 1
                buffer[split], used[split] = [], []
        if made != quota:
            raise ValueError(f'{domain}: quotas not met {made}')
        spare[domain] = rest
    cc = c['copy']
    prompt = tokenizer(COPY_PROMPT, add_special_tokens=False)['input_ids']
    copies = []
    for h, ident, item in spare[cc['source']]:
        if len(copies) == cc['docs']:
            break
        n = cc['min_passage'] + int(h[8:16], 16) % (cc['max_passage'] - cc['min_passage'] + 1)
        ids = encode(tokenizer, item)
        if len(ids) < n:
            continue
        start = n + len(prompt)
        copies.append({'id': f'copy-{len(copies)}', 'domain': 'copy', 'source_domain': cc['source'],
                       'split': 'eval', 'items': [ident], 'input_ids': ids[:n] + prompt + ids[:n],
                       'copy_span': [start, start + n]})
    if len(copies) < cc['docs']:
        raise ValueError('Not enough spare items for copy docs')
    return docs + copies


def stage_prepare(args):
    from transformers import AutoTokenizer
    rules = json.loads(RULES.read_text())
    path = OUT / 'corpus.json'
    if path.exists():
        raise ValueError('Corpus exists; refusing to overwrite')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    docs = build_corpus(candidates(), tokenizer, rules)
    files = [CHAT, BIOMED, ARGUMENT, MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json',
             MODEL / 'chat_template.jinja'] + sorted((OUT / 'sources').glob('*.jsonl'))
    sources = {str(p): sha(p) for p in files}
    sources['code'] = {p.name: sha(p) for p in sorted(CODE_DIR.glob('*.py'))}
    if 'code_cpp' in DOMAINS:
        sources['code_cpp'] = {str(p.relative_to(CPP_DIR)): sha(p) for p in sorted(CPP_DIR.rglob('*.cpp'))}
    counts = {}
    for d in docs:
        k = f'{d["domain"]}/{d["split"]}'
        counts[k] = counts.get(k, 0) + len(d['input_ids'])
    save({'sources': sources, 'token_counts': counts, 'docs': docs}, path)
    log(f'Saved {len(docs)} docs: {counts}')


# ---------------------------------------------------------------- model and carving

def load_model(threads):
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.set_num_threads(threads)
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    config = getattr(config, 'text_config', config)
    model, info = AutoModelForCausalLM.from_pretrained(MODEL, config=config, dtype=torch.bfloat16,
                                                       local_files_only=True, output_loading_info=True)
    # Vision and multi-token-prediction weights are outside this text-only experiment.
    unexpected = [k for k in info['unexpected_keys'] if not k.startswith(('model.visual', 'visual', 'mtp', 'model.mtp'))]
    if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs') or unexpected:
        raise RuntimeError(f'Invalid model load: {info}')
    return model.to(DEVICE).eval().requires_grad_(False)


def topmask(x, k):
    """Boolean mask of the k largest entries along the last dimension."""
    return torch.zeros_like(x, dtype=torch.bool).scatter_(-1, x.topk(k, -1).indices, True)


def at(s, name, layer):
    """A structure value of an expert set: one number for every layer, or a list by depth."""
    value = s[name]
    return value[layer] if isinstance(value, list) else value


def expert_mass(s, layer, score):
    """(T, experts) contribution of each token to each routed expert."""
    cluster = s['cluster'][layer]
    routed = cluster >= 0
    mass = torch.zeros(score.shape[0], s['experts'], device=score.device, dtype=score.dtype)
    return mass.index_add_(1, cluster[routed], score[:, routed])


def expand(s, layer, chosen):
    """(T, N) neuron mask from a (T, experts) choice of routed experts plus the shared expert."""
    cluster = s['cluster'][layer]
    return (cluster < 0) | chosen[:, cluster.clamp_min(0)]


def select(policy, s, layer, score, domain):
    """(T, N) mask of the neurons policy runs, given each token's contribution score (T, N)."""
    t, n = score.shape
    if policy == 'static':
        return s['static'][layer].expand(t, n)
    if policy == 'domain_doc':
        return s['domain'][layer, domain].expand(t, n)
    if policy == 'domain_token':
        sets = s['domain'][layer]
        return sets[(score @ sets.float().T).argmax(-1)]
    if policy == 'cluster_token':
        return expand(s, layer, topmask(expert_mass(s, layer, score), at(s, 'K', layer)))
    if policy == 'token_topk':
        return topmask(score, at(s, 'n', layer))
    raise ValueError(policy)


class Carver:
    """Runs every layer's MLP with a neuron selection chosen per batch element.

    arms[b] is (policy, budget) or None (dense) for batch element b, and domain is the
    document's domain index for domain_doc. Policy 'learned' picks the co-firing experts with
    routers[layer] from the MLP input instead of the contributions. Position 0 always runs every
    neuron. If record is set, it is called with (layer, a, x) on the dense activations a and the
    MLP input x before any selection.
    """

    def __init__(self, model, sets=None):
        self.mlps = [layer.mlp for layer in model.model.layers]
        self.wnorm = [m.down_proj.weight.float().norm(dim=0) for m in self.mlps]
        self.sets, self.arms, self.domain, self.record, self.routers = sets, None, None, None, None
        for i, m in enumerate(self.mlps):
            m.forward = lambda x, i=i, m=m: self.forward(i, m, x)

    def score(self, i, a):
        return a.float().abs() * self.wnorm[i]

    def keep(self, arm, i, a, x):
        """(T, N) mask one arm runs for activations a (T, N) and MLP inputs x (T, hidden)."""
        policy, budget = arm
        s = self.sets[budget]
        if policy == 'learned':
            return expand(s, i, topmask(self.routers[i](x.float()), at(s, 'K', i)))
        return select(policy, s, i, self.score(i, a), self.domain)

    def forward(self, i, m, x):
        a = m.act_fn(m.gate_proj(x)) * m.up_proj(x)
        if self.record is not None:
            self.record(i, a, x)
        if self.arms is not None:
            for b, arm in enumerate(self.arms):
                if arm is not None:
                    a[b, 1:] = a[b, 1:] * self.keep(arm, i, a[b, 1:], x[b, 1:])
        return m.down_proj(a)


def balanced_assign(sim, cap):
    """Greedy equal-capacity assignment of rows to columns: in rounds, every unassigned row
    proposes to its best column with room, and each column keeps its strongest proposals."""
    m, k = sim.shape
    label = torch.full((m,), -1, dtype=torch.long)
    room = torch.full((k,), cap, dtype=torch.long)
    while (label < 0).any():
        free = (label < 0).nonzero().squeeze(1)
        s = sim[free].masked_fill(room == 0, -torch.inf)
        value, best = s.max(1)
        for j in best.unique().tolist():
            who = best == j
            take = free[who][value[who].argsort(descending=True)[:room[j]]]
            label[take] = j
            room[j] -= len(take)
    return label


def balanced_kmeans(x, k, iterations, seed):
    """k equal clusters of the rows of x by cosine similarity."""
    m = x.shape[0]
    if m % k:
        raise ValueError('Rows must divide evenly into clusters')
    x = F.normalize(x.float(), dim=1)
    centres = x[torch.randperm(m, generator=torch.Generator().manual_seed(seed))[:k]]
    label = None
    for _ in range(iterations):
        new = balanced_assign(x @ centres.T, m // k)
        if label is not None and torch.equal(new, label):
            break
        label = new
        centres = F.normalize(torch.zeros(k, x.shape[1]).index_add_(0, label, x), dim=1)
    return label


def layer_structure(n, n_neurons, share, experts):
    """Shared size S, cluster size c and routed count K so S + K c is about n active neurons,
    about share of them shared."""
    c = (n_neurons - round(n * share)) // experts
    shared = n_neurons - experts * c
    k = round((n - shared) / c)
    return {'experts': experts, 'c': c, 'S': shared, 'K': k, 'n': shared + k * c}


def structure(budget, n_neurons, cluster_rules):
    """Shared size S, cluster size c and routed count K so S + K c is the active count."""
    return layer_structure(float(budget) * n_neurons, n_neurons, cluster_rules['shared_share'],
                           cluster_rules['experts'])


def cluster_rest(stats, shared, experts, rules):
    """(L, N) cluster label of every neuron outside the shared mask (L, N); -1 for shared."""
    layers, n_neurons = shared.shape
    cluster = torch.full((layers, n_neurons), -1, dtype=torch.long)
    for layer in range(layers):
        rest = (~shared[layer]).nonzero().squeeze(1)
        bits = torch.from_numpy(np.unpackbits(stats['bits'][layer].numpy(), axis=1, count=n_neurons))
        cluster[layer, rest] = balanced_kmeans(bits[:, rest].T, experts, rules['cluster']['iterations'],
                                               rules['seed'] + layer)
    return cluster


def waterfill(curve, step, total, floor, cap):
    """Active neurons per layer, multiples of step summing to total, that maximise the summed
    captured share. curve[l, j] is layer l's mean share captured by each token's top (j + 1) step
    neurons; the curves are concave, so adding one step at a time to the best layer is optimal."""
    units = [floor // step] * curve.shape[0]
    for _ in range(total // step - sum(units)):
        gain = [curve[l, u] - curve[l, u - 1] if u < cap // step else -1. for l, u in enumerate(units)]
        units[int(np.argmax(gain))] += 1
    return [u * step for u in units]


def depth_carve(stats, n_layer, share_layer, total, rules):
    """Co-firing experts with a different structure in every layer, nudged so the active count
    summed over layers is as close to total as whole routed experts allow."""
    rate = stats['active'].double() / stats['tokens'].double()[:, None]
    n_neurons = rate.shape[-1]
    experts = rules['cluster']['experts']
    rows = [layer_structure(n, n_neurons, share, experts) for n, share in zip(n_layer, share_layer)]
    while True:
        r = total - sum(x['n'] for x in rows)
        sign = 1 if r > 0 else -1
        layer = max(range(len(rows)), key=lambda l: sign * (n_layer[l] - rows[l]['n']))
        if abs(r) <= abs(r - sign * rows[layer]['c']):
            break
        rows[layer]['K'] += sign
        rows[layer]['n'] += sign * rows[layer]['c']
    shared = torch.stack([topmask(rate.mean(1)[l], x['S']) for l, x in enumerate(rows)])
    return {'experts': experts, **{k: [x[k] for x in rows] for k in ('c', 'S', 'K', 'n')},
            'shared': shared, 'cluster': cluster_rest(stats, shared, experts, rules)}


def carve(stats, rules):
    """Expert sets for every budget from calibration statistics, plus the depth sets if the
    rules ask for them."""
    tokens = stats['tokens'].double()
    importance = stats['sum'] / tokens[:, None]
    rate = stats['active'].double() / tokens[:, None]
    layers, _, n_neurons = importance.shape
    sets = {}
    for budget in rules['budgets']:
        s = structure(budget, n_neurons, rules['cluster'])
        shared = topmask(rate.mean(1), s['S'])
        sets[budget] = dict(s, static=topmask(importance.mean(1), s['n']), domain=topmask(importance, s['n']),
                            shared=shared, cluster=cluster_rest(stats, shared, s['experts'], rules))
    depth = rules.get('depth')
    if depth:
        s = sets[depth['budget']]
        total, base = layers * s['n'], rules['cluster']['shared_share']
        # depth_shared: the shared share follows the fraction of each layer's domain sets that
        # every domain shares, scaled to average the uniform share.
        core = s['domain'].all(1).sum(-1).double() / s['n']
        share = (base * core / core.mean()).clamp(*depth['share_range']).tolist()
        sets['depth_shared'] = depth_carve(stats, [s['n']] * layers, share, total, rules)
        n_layer = waterfill(stats['curve'].numpy(), int(stats['grid'][0]), total, int(depth['floor'] * s['n']),
                            int(depth['cap'] * s['n']))
        sets['depth_budget'] = depth_carve(stats, n_layer, [base] * layers, total, rules)
    return sets


# ---------------------------------------------------------------- stages

def model_files():
    return {p.name: sha(p) for p in sorted(MODEL.glob('*.safetensors'))}


def stage_collect(args):
    rules = json.loads(RULES.read_text())
    if (OUT / 'stats.pt').exists():
        raise ValueError('Statistics exist; refusing to overwrite')
    corpus = json.loads((OUT / 'corpus.json').read_text())
    cfg = rules['collect']
    model = load_model(cfg['threads'])
    carver = Carver(model)
    layers, n_neurons = len(carver.mlps), carver.wnorm[0].numel()
    stats = {'sum': torch.zeros(layers, len(DOMAINS), n_neurons, dtype=torch.float64),
             'active': torch.zeros(layers, len(DOMAINS), n_neurons, dtype=torch.long),
             'tokens': torch.zeros(len(DOMAINS), dtype=torch.long)}
    bits, sample_domain = [[] for _ in range(layers)], []
    top = int(cfg['rate_fraction'] * n_neurons)
    if 'depth' in rules:
        # Mean share of each sampled token's contribution captured by its top k neurons, k on a grid.
        grid = torch.arange(rules['depth']['step'], n_neurons + 1, rules['depth']['step'])
        stats['grid'], stats['curve'] = grid, torch.zeros(layers, len(grid), dtype=torch.float64)
    g = torch.Generator().manual_seed(rules['seed'])
    t0 = time.time()
    for d, domain in enumerate(DOMAINS):
        docs = [x for x in corpus['docs'] if x['domain'] == domain and x['split'] == 'calib']
        for i in range(0, len(docs), cfg['batch']):
            ids = torch.tensor([x['input_ids'] for x in docs[i:i + cfg['batch']]])
            b, t = ids.shape
            picks = torch.stack([1 + torch.randperm(t - 1, generator=g)[:cfg['sample_tokens_per_doc']]
                                 for _ in range(b)])
            rows = (torch.arange(b)[:, None] * (t - 1) + picks - 1).flatten().to(DEVICE)

            def record(layer, a, x):
                score = carver.score(layer, a[:, 1:]).flatten(0, 1)
                stats['sum'][layer, d] += score.sum(0).double().cpu()
                active = topmask(score, top)
                stats['active'][layer, d] += active.sum(0).cpu()
                bits[layer].append(np.packbits(active[rows].cpu().numpy(), axis=1))
                if 'curve' in stats:
                    cum = score[rows].sort(-1, descending=True).values.cumsum(-1)
                    share = cum[:, stats['grid'].to(cum.device) - 1] / cum[:, -1:]
                    stats['curve'][layer] += share.sum(0).double().cpu()

            carver.record = record
            with torch.no_grad():
                model.model(input_ids=ids.to(DEVICE), use_cache=False)
            stats['tokens'][d] += b * (t - 1)
            sample_domain += [d] * len(rows)
        log(f'{domain}: {int(stats["tokens"][d])} tokens, {time.time() - t0:.0f} s')
    stats['bits'] = [torch.from_numpy(np.concatenate(x)) for x in bits]
    stats['sample_domain'] = torch.tensor(sample_domain)
    if 'curve' in stats:
        stats['curve'] /= len(sample_domain)
    torch.save(stats, OUT / 'stats.pt')
    save(dict(provenance(), corpus_sha256=sha(OUT / 'corpus.json'), model_sha256=model_files(),
              stats_sha256=sha(OUT / 'stats.pt'), seconds=time.time() - t0), OUT / 'collect.json')


def overlap_report(sets):
    report = {}
    for budget, s in sets.items():
        if 'domain' not in s:
            report[budget] = {k: s[k] for k in ('experts', 'c', 'S', 'K', 'n')}
            continue
        dom = s['domain']                                    # (L, D, N)
        core = dom.all(1).sum(-1).double() / s['n']          # neurons in every domain set
        union = dom.any(1).sum(-1).double() / dom.shape[-1]  # neurons in some domain set
        inter = (dom[:, :, None] & dom[:, None]).sum(-1).double()
        jaccard = inter / (dom[:, :, None] | dom[:, None]).sum(-1).double()
        off = ~torch.eye(dom.shape[1], dtype=torch.bool)
        static_in_domain = (dom & s['static'][:, None]).sum(-1).double() / s['n']
        report[budget] = {k: s[k] for k in ('experts', 'c', 'S', 'K', 'n')} | {
            'shared_core_fraction': core.tolist(), 'union_fraction': union.tolist(),
            'mean_pairwise_jaccard': jaccard[:, off].mean(-1).tolist(),
            'jaccard_by_domain_pair': jaccard.mean(0).tolist(),
            'static_overlap_by_domain': static_in_domain.mean(0).tolist()}
    return report


def stage_build(args):
    rules = json.loads(RULES.read_text())
    require('collect')
    if (OUT / 'sets.pt').exists():
        raise ValueError('Sets exist; refusing to overwrite')
    t0 = time.time()
    sets = carve(torch.load(OUT / 'stats.pt', weights_only=True), rules)
    torch.save(sets, OUT / 'sets.pt')
    save(dict(provenance(), corpus_sha256=sha(OUT / 'corpus.json'), sets_sha256=sha(OUT / 'sets.pt'),
              domains=DOMAINS, overlap=overlap_report(sets), seconds=time.time() - t0), OUT / 'build.json')
    log(f'Built sets in {time.time() - t0:.0f} s')


def arm_list(rules):
    arms = [None] + [(p, b) for b in rules['budgets'] for p in POLICIES]
    arms += [('cluster_token', d) for d in rules.get('depth', {}).get('arms', [])]
    return arms, ['dense'] + [f'{p}@{b}' for p, b in arms[1:]]


@torch.no_grad()
def evaluate_doc(model, carver, doc, arms, names):
    ids = torch.tensor([doc['input_ids']], device=DEVICE)
    t = ids.shape[1]
    domain = DOMAINS.index(doc.get('source_domain', doc['domain']))
    capture = {name: [] for name in names[1:]}

    def record(layer, a, x):
        score = carver.score(layer, a[0, 1:])
        total = score.sum()
        for name, arm in zip(names[1:], arms[1:]):
            keep = carver.keep(arm, layer, a[0, 1:], x[0, 1:])
            capture[name].append(float((score * keep).sum() / total))

    carver.arms, carver.domain, carver.record = arms, domain, record
    try:
        hidden = model.model(input_ids=ids.repeat(len(arms), 1), use_cache=False).last_hidden_state
    finally:
        carver.arms = carver.record = None
    dense = model.lm_head(hidden[0]).float()
    row = {'id': doc['id'], 'domain': doc['domain'], 'kl': {}, 'nll': {}, 'capture': capture}
    if 'copy_span' in doc:
        row['copy_kl'] = {}
    for b, name in enumerate(names):
        logits = dense if b == 0 else model.lm_head(hidden[b]).float()
        kl = token_kl(dense, logits)
        row['kl'][name] = float(kl[1:].mean())
        row['nll'][name] = float(F.cross_entropy(logits[:-1], ids[0, 1:]))
        if 'copy_span' in doc:
            start, end = doc['copy_span']
            row['copy_kl'][name] = kl[start - 1:end - 1].tolist()
    return row


def stage_evaluate(args):
    rules = json.loads(RULES.read_text())
    require('build')
    corpus = json.loads((OUT / 'corpus.json').read_text())
    sets = torch.load(OUT / 'sets.pt', weights_only=True, map_location=DEVICE)
    model = load_model(rules['evaluate']['threads'])
    carver = Carver(model, sets)
    arms, names = arm_list(rules)
    path = OUT / 'evaluation.jsonl'
    done = {json.loads(line)['id'] for line in path.open()} if path.exists() else set()
    docs = [d for d in corpus['docs'] if d['split'] == 'eval' and d['id'] not in done]
    log(f'{len(docs)} documents to evaluate, {len(arms)} arms each')
    with deadline(rules['evaluate']['max_hours']):
        for doc in docs:
            t0 = time.time()
            row = evaluate_doc(model, carver, doc, arms, names)
            with path.open('a') as f:
                f.write(json.dumps(row) + '\n')
            log(f'{doc["id"]}: {time.time() - t0:.0f} s; ' + ', '.join(
                f'{n} {row["kl"][n]:.3f}' for n in names[1:] if n.endswith(rules['report']['primary_budget'])))
    save(dict(provenance(), corpus_sha256=sha(OUT / 'corpus.json'), evaluation_sha256=sha(path)),
         OUT / 'evaluate.json')


def arm_summary(rows, names):
    """Domain-equal KL and NLL change, per-domain KL, copy-span tails and capture by layer for each
    arm of evaluate_doc rows."""
    text = [r for r in rows if r['domain'] != 'copy']
    copies = [r for r in rows if r['domain'] == 'copy']
    groups = [r['domain'] for r in text]
    arms = {}
    for name in names:
        kl = [r['kl'][name] for r in text]
        span = np.concatenate([r['copy_kl'][name] for r in copies]) if copies else np.zeros(1)
        arms[name] = {
            'kl': group_mean(kl, groups),
            'kl_by_domain': {d: float(np.mean([r['kl'][name] for r in text if r['domain'] == d])) for d in DOMAINS},
            'delta_nll': group_mean([r['nll'][name] - r['nll']['dense'] for r in text], groups),
            'copy_mean': float(span.mean()), 'copy_p99': float(np.quantile(span, 0.99))}
        if name != 'dense' and all('capture' in r for r in rows):
            arms[name]['capture_by_layer'] = np.mean([r['capture'][name] for r in rows], 0).tolist()
    return arms


def summarize(rows, rules):
    """Domain-equal KL per arm, per-domain KL, copy tails, capture by layer, and the
    pre-registered comparisons."""
    _, names = arm_list(rules)
    text = [r for r in rows if r['domain'] != 'copy']
    copies = [r for r in rows if r['domain'] == 'copy']
    groups = [r['domain'] for r in text]
    rep = rules['report']
    arms = arm_summary(rows, names)
    comparisons = {}
    for budget in rules['budgets']:
        pairs = {'domain_token-cluster_token': ('domain_token', 'cluster_token'),
                 'domain_doc-static': ('domain_doc', 'static'),
                 'domain_token-domain_doc': ('domain_token', 'domain_doc'),
                 'cluster_token-token_topk': ('cluster_token', 'token_topk')}
        for label, (new, base) in pairs.items():
            a, b = (f'{new}@{budget}', f'{base}@{budget}')
            diff, low, high = paired_interval([r['kl'][a] for r in text], [r['kl'][b] for r in text], 1.0,
                                              rep['draws'], rep['seed'], groups)
            comparisons[f'{label}@{budget}'] = {'difference': diff, 'low': low, 'high': high,
                                                'ratio': arms[a]['kl'] / arms[b]['kl']}
    for depth in rules.get('depth', {}).get('arms', []):
        a, b = f'cluster_token@{depth}', f'cluster_token@{rules["depth"]["budget"]}'
        diff, low, high = paired_interval([r['kl'][a] for r in text], [r['kl'][b] for r in text], 1.0,
                                          rep['draws'], rep['seed'], groups)
        comparisons[f'{depth}-cluster_token@{rules["depth"]["budget"]}'] = {
            'difference': diff, 'low': low, 'high': high, 'ratio': arms[a]['kl'] / arms[b]['kl']}
    p = rep['primary_budget']
    primary = comparisons[f'domain_token-cluster_token@{p}']
    verdict = ('domain carve beats co-activation carve' if primary['high'] < 0 else
               'co-activation carve beats domain carve' if primary['low'] > 0 else 'tie')
    tails = {}
    for label in ('domain_token-cluster_token', 'domain_doc-static'):
        new, base = label.split('-')
        tails[label] = arms[f'{new}@{p}']['copy_p99'] / max(arms[f'{base}@{p}']['copy_p99'], 1e-9)
    return {'documents': len(text), 'copy_documents': len(copies), 'arms': arms, 'comparisons': comparisons,
            'primary': dict(primary, verdict=verdict),
            'secondary_ratio_domain_doc_over_static': comparisons[f'domain_doc-static@{p}']['ratio'],
            'copy_p99_ratio': tails}


def stage_report(args):
    rules = json.loads(RULES.read_text())
    record = require('evaluate')
    path = OUT / 'evaluation.jsonl'
    if sha(path) != record['evaluation_sha256']:
        raise RuntimeError('Evaluation rows changed after the evaluate stage')
    rows = [json.loads(line) for line in path.open()]
    report = summarize(rows, rules)
    save(dict(provenance(), evaluation_sha256=sha(path), **report), OUT / 'report.json')
    for name, a in report['arms'].items():
        log(f'{name:22s} KL {a["kl"]:.3f}  dNLL {a["delta_nll"]:+.3f}  copy {a["copy_mean"]:.3f} '
            f'p99 {a["copy_p99"]:.3f}')
    log(f'Primary: {report["primary"]}')


def train_routers(model, carver, s, docs, cfg):
    """One linear router per layer, trained on the dense model's MLP inputs by cross-entropy
    against each token's share of routed contribution in every expert. Returns the routers and
    the mean loss of every step."""
    torch.manual_seed(cfg['seed'])
    routers = [torch.nn.Linear(model.config.hidden_size, s['experts']).to(DEVICE) for _ in carver.mlps]
    steps = cfg['epochs'] * -(-len(docs) // cfg['batch'])
    optims = [torch.optim.Adam(r.parameters(), lr=cfg['lr']) for r in routers]
    schedules = [torch.optim.lr_scheduler.LambdaLR(o, lambda k: 0.5 * (1 + np.cos(np.pi * k / steps)))
                 for o in optims]
    losses = []

    def train(layer, a, x):
        mass = expert_mass(s, layer, carver.score(layer, a[:, 1:]).flatten(0, 1))
        target = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
        with torch.enable_grad():
            logits = routers[layer](x[:, 1:].flatten(0, 1).float())
            loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
            optims[layer].zero_grad()
            loss.backward()
            optims[layer].step()
            schedules[layer].step()
        losses[-1].append(loss.detach())

    carver.record = train
    g = torch.Generator().manual_seed(cfg['seed'])
    try:
        for epoch in range(cfg['epochs']):
            order = torch.randperm(len(docs), generator=g).tolist()
            for i in range(0, len(order), cfg['batch']):
                losses.append([])
                ids = torch.tensor([docs[j]['input_ids'] for j in order[i:i + cfg['batch']]], device=DEVICE)
                with torch.no_grad():
                    model.model(input_ids=ids, use_cache=False)
                losses[-1] = float(torch.stack(losses[-1]).mean())
            log(f'router epoch {epoch}: loss {losses[-1]:.4f}')
    finally:
        carver.record = None
    return [r.eval().requires_grad_(False) for r in routers], losses


@torch.no_grad()
def router_recall(model, carver, s, doc):
    """Per layer, the share of the perfect router's K experts the learned router also picks."""
    recall = []

    def record(layer, a, x):
        k = at(s, 'K', layer)
        best = topmask(expert_mass(s, layer, carver.score(layer, a[0, 1:])), k)
        picked = topmask(carver.routers[layer](x[0, 1:].float()), k)
        recall.append(float((best & picked).sum() / (best.shape[0] * k)))

    carver.record = record
    try:
        model.model(input_ids=torch.tensor([doc['input_ids']], device=DEVICE), use_cache=False)
    finally:
        carver.record = None
    return recall


def stage_route(args):
    """Learned routers for the primary budget's co-firing experts, trained on the calibration
    documents and scored on the eval documents against the perfect router."""
    rules = json.loads(RULES.read_text())
    cfg = rules['route']
    require('build')
    if (OUT / 'route.json').exists():
        raise ValueError('Routers exist; refusing to overwrite')
    corpus = json.loads((OUT / 'corpus.json').read_text())
    sets = torch.load(OUT / 'sets.pt', weights_only=True, map_location=DEVICE)
    model = load_model(rules['evaluate']['threads'])
    carver = Carver(model, sets)
    s = sets[cfg['budget']]
    t0 = time.time()
    with deadline(cfg['max_hours']):
        calib = [d for d in corpus['docs'] if d['split'] == 'calib']
        carver.routers, losses = train_routers(model, carver, s, calib, cfg)
        torch.save([r.state_dict() for r in carver.routers], OUT / 'routers.pt')
        arms = [None, ('cluster_token', cfg['budget']), ('learned', cfg['budget'])]
        names = ['dense'] + [f'{p}@{b}' for p, b in arms[1:]]
        rows, recall = [], []
        for doc in [d for d in corpus['docs'] if d['split'] == 'eval']:
            rows.append(evaluate_doc(model, carver, doc, arms, names))
            recall.append(router_recall(model, carver, s, doc))
    (OUT / 'route_evaluation.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    text = [r for r in rows if r['domain'] != 'copy']
    groups = [r['domain'] for r in text]
    diff, low, high = paired_interval([r['kl'][names[2]] for r in text], [r['kl'][names[1]] for r in text], 1.0,
                                      rules['report']['draws'], rules['report']['seed'], groups)
    summary = arm_summary(rows, names)
    save(dict(provenance(), corpus_sha256=sha(OUT / 'corpus.json'), sets_sha256=sha(OUT / 'sets.pt'),
              routers_sha256=sha(OUT / 'routers.pt'), evaluation_sha256=sha(OUT / 'route_evaluation.jsonl'),
              loss_by_step=losses, arms=summary, recall_by_layer=np.mean(recall, 0).tolist(),
              learned_minus_perfect={'difference': diff, 'low': low, 'high': high,
                                     'ratio': summary[names[2]]['kl'] / summary[names[1]]['kl']},
              seconds=time.time() - t0), OUT / 'route.json')
    log(f'learned router KL {summary[names[2]]["kl"]:.3f} vs perfect {summary[names[1]]["kl"]:.3f}')


def main():
    global DEVICE, MODEL
    stages = {'fetch': stage_fetch, 'prepare': stage_prepare, 'collect': stage_collect, 'build': stage_build,
              'evaluate': stage_evaluate, 'report': stage_report, 'route': stage_route}
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=stages)
    ap.add_argument('--rules', type=Path, default=RULES)
    ap.add_argument('--device', default=DEVICE)
    ap.add_argument('--model', type=Path, default=MODEL)
    args = ap.parse_args()
    configure(args.rules)
    DEVICE, MODEL = args.device, args.model
    stages[args.stage](args)


if __name__ == '__main__':
    main()
