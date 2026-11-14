import numpy as np
import torch

import moe_carve as mc
from test_h0_mlp_probe import tiny

BUDGET = '0.5'


def fake_sets(model, seed=0):
    layers, n = model.config.num_hidden_layers, model.config.intermediate_size
    g = torch.Generator().manual_seed(seed)
    rules = {'budgets': [BUDGET], 'seed': 1, 'cluster': {'experts': 4, 'shared_share': 0.5, 'iterations': 5}}
    stats = {'sum': torch.rand(layers, len(mc.DOMAINS), n, generator=g, dtype=torch.float64),
             'active': torch.randint(0, 50, (layers, len(mc.DOMAINS), n), generator=g),
             'tokens': torch.full((len(mc.DOMAINS),), 100),
             'bits': [torch.from_numpy(np.packbits(torch.rand(40, n, generator=g).numpy() > 0.7, axis=1))
                      for _ in range(layers)]}
    return mc.carve(stats, rules)


def ids(seed, length=12):
    return torch.randint(0, 64, (1, length), generator=torch.Generator().manual_seed(seed))


def test_structure_runs_the_budget_and_covers_every_neuron():
    for budget, n in (('0.5', 6096), ('0.25', 3048), ('0.125', 1488)):
        s = mc.structure(budget, 12288, {'experts': 64, 'shared_share': 0.5})
        assert s['n'] == n == s['S'] + s['K'] * s['c']
        assert s['S'] + s['experts'] * s['c'] == 12288


def test_carved_sets_have_the_declared_sizes():
    model = tiny()
    s = fake_sets(model)[BUDGET]
    assert s['static'].sum(-1).eq(s['n']).all() and s['domain'].sum(-1).eq(s['n']).all()
    assert s['shared'].sum(-1).eq(s['S']).all()
    for layer in range(model.config.num_hidden_layers):
        assert torch.equal(s['cluster'][layer] < 0, s['shared'][layer])
        assert torch.bincount(s['cluster'][layer][s['cluster'][layer] >= 0]).eq(s['c']).all()


def test_every_policy_runs_exactly_n_neurons_per_token():
    model = tiny()
    s = fake_sets(model)[BUDGET]
    score = torch.rand(9, model.config.intermediate_size, generator=torch.Generator().manual_seed(2))
    for policy in mc.POLICIES:
        assert mc.select(policy, s, 1, score, 3).sum(-1).eq(s['n']).all(), policy


def test_carver_is_exact_when_dense_and_at_full_width():
    model = tiny()
    x = ids(3)
    clean = model(input_ids=x).logits
    sets = {BUDGET: fake_sets(model)[BUDGET], 'full': {'n': model.config.intermediate_size}}
    carver = mc.Carver(model, sets)
    torch.testing.assert_close(model(input_ids=x).logits, clean)
    carver.arms = [('token_topk', 'full')]
    torch.testing.assert_close(model(input_ids=x).logits, clean)


def test_batched_arms_match_single_runs_and_keep_position_0():
    model = tiny()
    carver = mc.Carver(model, fake_sets(model))
    x = ids(4)
    arms = [None] + [(p, BUDGET) for p in mc.POLICIES]
    carver.arms, carver.domain = arms, 2
    batched = model(input_ids=x.repeat(len(arms), 1)).logits
    for b, arm in enumerate(arms):
        carver.arms = [arm]
        torch.testing.assert_close(batched[b], model(input_ids=x).logits[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(batched[b, 0], batched[0, 0], atol=1e-5, rtol=1e-5)
    assert all((batched[b] - batched[0]).abs().max() > 1e-4 for b in range(1, len(arms)))


def test_waterfill_spends_the_total_where_the_curves_gain_most():
    grid = np.arange(1, 9) / 8
    flat, steep = np.sqrt(grid) * 0.2 + 0.8, np.sqrt(grid)
    n = mc.waterfill(np.stack([flat, steep]), 2, 12, 2, 16)
    assert sum(n) == 12 and n[1] > n[0] >= 2 and all(x % 2 == 0 for x in n)


def test_depth_sets_vary_by_layer_and_keep_the_total(monkeypatch):
    model = tiny()
    layers, n_neurons = model.config.num_hidden_layers, model.config.intermediate_size
    g = torch.Generator().manual_seed(3)
    stats = {'sum': torch.rand(layers, len(mc.DOMAINS), n_neurons, generator=g, dtype=torch.float64),
             'active': torch.randint(0, 50, (layers, len(mc.DOMAINS), n_neurons), generator=g),
             'tokens': torch.full((len(mc.DOMAINS),), 100),
             'bits': [torch.from_numpy(np.packbits(torch.rand(40, n_neurons, generator=g).numpy() > 0.7, axis=1))
                      for _ in range(layers)],
             'grid': torch.arange(4, n_neurons + 1, 4),
             'curve': torch.stack([torch.linspace(0.1, 1, n_neurons // 4) ** p for p in (0.3, 0.5, 0.7, 0.9)]).double()}
    rules = {'budgets': [BUDGET], 'seed': 1, 'cluster': {'experts': 4, 'shared_share': 0.5, 'iterations': 5},
             'depth': {'budget': BUDGET, 'arms': ['depth_shared', 'depth_budget'], 'step': 4,
                       'share_range': [0.2, 0.8], 'floor': 0.5, 'cap': 1.5}}
    sets = mc.carve(stats, rules)
    total = layers * sets[BUDGET]['n']
    for name in ('depth_shared', 'depth_budget'):
        s = sets[name]
        assert abs(sum(s['n']) - total) <= max(s['c'])
        for layer in range(layers):
            assert s['shared'][layer].sum() == s['S'][layer]
            assert torch.bincount(s['cluster'][layer][s['cluster'][layer] >= 0]).eq(s['c'][layer]).all()
        score = torch.rand(7, n_neurons, generator=g)
        for layer in range(layers):
            assert mc.select('cluster_token', s, layer, score, 0).sum(-1).eq(s['n'][layer]).all()
    assert len(set(sets['depth_budget']['n'])) > 1
    arms, names = mc.arm_list(rules)
    assert names[-2:] == ['cluster_token@depth_shared', 'cluster_token@depth_budget']


def test_learned_routers_train_and_run_k_experts():
    model = tiny()
    sets = fake_sets(model)
    s = sets[BUDGET]
    carver = mc.Carver(model, sets)
    docs = [{'input_ids': ids(seed, 16)[0].tolist()} for seed in range(8)]
    carver.routers, losses = mc.train_routers(model, carver, s, docs, {'seed': 0, 'epochs': 3, 'batch': 4, 'lr': 0.05})
    assert len(losses) == 6 and losses[-1] < losses[0]
    x = torch.randn(5, model.config.hidden_size)
    a = torch.randn(5, model.config.intermediate_size)
    for layer in range(model.config.num_hidden_layers):
        assert carver.keep(('learned', BUDGET), layer, a, x).sum(-1).eq(s['n']).all()
    recall = mc.router_recall(model, carver, s, docs[0])
    assert len(recall) == model.config.num_hidden_layers and all(0 <= r <= 1 for r in recall)


def test_balanced_kmeans_recovers_planted_equal_clusters():
    g = torch.Generator().manual_seed(5)
    centres = torch.randn(4, 30, generator=g)
    truth = torch.arange(4).repeat_interleave(16)
    x = centres[truth] + 0.3 * torch.randn(64, 30, generator=g)
    label = mc.balanced_kmeans(x, 4, 20, 0)
    assert torch.bincount(label).eq(16).all()
    for k in range(4):
        assert label[truth == k].unique().numel() == 1


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [sum(map(ord, w)) % 997 for w in text.split()]}

    def apply_chat_template(self, messages, tokenize=False):
        return ' '.join(m['content'] for m in messages)


def test_first_run_rules_keep_their_folder_and_domains(monkeypatch):
    for name in ('RULES', 'OUT', 'DOMAINS'):
        monkeypatch.setattr(mc, name, getattr(mc, name))
    mc.configure(mc.ROOT / 'moe_carve_rules.json')
    assert mc.OUT == mc.ROOT / 'results/moe-carve-20261028'
    assert mc.DOMAINS == ('chat', 'code', 'math', 'wiki', 'biomed', 'argument', 'french', 'chinese')


def test_v2_rules_are_english_and_code_only(monkeypatch):
    for name in ('RULES', 'OUT', 'DOMAINS'):
        monkeypatch.setattr(mc, name, getattr(mc, name))
    mc.configure(mc.ROOT / 'moe_carve_rules_v2.json')
    assert mc.OUT != mc.ROOT / 'results/moe-carve-20261028'
    assert not {'french', 'chinese'} & set(mc.DOMAINS)
    assert {'reasoning', 'science', 'code_cpp', 'math_word'} <= set(mc.DOMAINS)
    test_corpus_packs_fixed_length_docs_and_keeps_items_in_one_split()


def test_corpus_packs_fixed_length_docs_and_keeps_items_in_one_split():
    rules = {'corpus': {'salt': 's', 'length': 20, 'min_item_tokens': 3, 'eval_every': 5, 'calib_docs': 3,
                        'eval_docs': 2, 'copy': {'docs': 2, 'min_passage': 4, 'max_passage': 6,
                                                 'source': 'biomed'}}}
    # Items from a domain outside the run's list are ignored.
    items = [(d, f'{d}-{i}', ' '.join(f'w{d}{i}x{j}' for j in range(3 + i % 15)))
             for d in mc.DOMAINS + ('klingon',) for i in range(200)]
    docs = mc.build_corpus(items, WordTokenizer(), rules)
    text = [d for d in docs if d['domain'] != 'copy']
    assert len(text) == len(mc.DOMAINS) * 5 and all(len(d['input_ids']) == 20 for d in text)
    split_of = {}
    for d in text:
        for item in d['items']:
            assert split_of.setdefault(item, d['split']) == d['split']
    copies = [d for d in docs if d['domain'] == 'copy']
    assert len(copies) == 2
    for d in copies:
        start, end = d['copy_span']
        n = end - start
        assert d['input_ids'][start:end] == d['input_ids'][:n] and d['items'][0] not in split_of
