"""H0b probe tests on a tiny random Llama and synthetic data; no real model or corpus."""
import copy
import json

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import h0_mlp_probe as h0


def tiny(layers=4, seed=0):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                         rms_norm_eps=1e-5, tie_word_embeddings=True, initializer_range=0.2)
    return LlamaForCausalLM(config).eval().requires_grad_(False)


def orthogonal(d, seed=0):
    return torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed)))[0]


def teacher_mlp(layer, h):
    return h + layer.mlp(layer.post_attention_layernorm(h))


def test_fold_at_full_width_reproduces_the_teacher_block():
    model = tiny()
    layer, q = model.model.layers[1], orthogonal(32)
    h = torch.randn(10, 32) * 3
    net = h0.fold(layer, q, 1.)
    out = h0.predictor('restricted', net, h0.PCAChart(q))(h)
    torch.testing.assert_close(out, teacher_mlp(layer, h), atol=1e-5, rtol=1e-5)


def test_restricted_prunes_to_m_with_matched_scalars():
    model = tiny()
    layer, p = model.model.layers[2], orthogonal(32)[:, :16]
    x = torch.randn(500, 32)
    net, keep = h0.restricted(layer, p, h0.rms_ratio(x, p), 24, x @ p, 1e-3)
    assert len(keep) == 24 and torch.equal(keep, keep.sort().values) and len(set(keep.tolist())) == 24
    assert net.scalars() == 16 + 3 * 16 * 24
    full = h0.full_state_init(layer, keep, net.down.weight)
    assert full.gate.in_features == 32 and full.down.out_features == 16


def test_chart_starts_at_pca_and_is_positively_homogeneous():
    p = orthogonal(32)[:, :8]
    chart = h0.Chart(p, 16)
    h = torch.randn(20, 32)
    torch.testing.assert_close(chart.encode(h), h @ p)
    torch.testing.assert_close(chart.decode(h @ p), h @ p @ p.T)
    for net in (chart.enc, chart.dec):
        torch.nn.init.normal_(net[-1].weight)
    torch.testing.assert_close(chart.encode(2.5 * h), 2.5 * chart.encode(h), atol=1e-4, rtol=1e-4)
    c = chart.encode(h)
    torch.testing.assert_close(chart.decode(3 * c), 3 * chart.decode(c), atol=1e-4, rtol=1e-4)


def test_coded_arms_see_the_input_only_through_the_declared_code():
    model = tiny()
    layer, q = model.model.layers[1], orthogonal(32)
    p = q[:, :8]
    x = torch.randn(300, 32)
    net, keep = h0.restricted(layer, p, h0.rms_ratio(x, p), 16, x @ p, 1e-3)
    h = torch.randn(5, 32)
    off = h + 4 * torch.randn(5, 24) @ q[:, 8:].T          # moves only outside the code
    pca = h0.PCAChart(p)
    for arm in ('restricted', 'pca'):
        f = h0.predictor(arm, net, pca)
        torch.testing.assert_close(f(off), f(h))
    chart = h0.Chart(p, 16)
    torch.nn.init.normal_(chart.enc[-1].weight)
    torch.nn.init.normal_(chart.dec[-1].weight)
    code = chart.encode(h[:1])
    stub = copy.deepcopy(chart)
    stub.encode = lambda inputs: code.expand(len(inputs), -1)   # every input collides on one code
    out = h0.predictor('learned', net, stub)(torch.randn(4, 32))
    assert torch.allclose(out, out[:1].expand_as(out))
    full = h0.predictor('full', h0.full_state_init(layer, keep, net.down.weight), pca)
    assert not torch.allclose(full(off), full(h))             # the control does read the full state


def test_update_error_is_one_for_dropping_the_block_and_zero_when_exact():
    x, y = torch.randn(50, 8), torch.randn(50, 8)
    assert h0.nmse(lambda h: h, x, y) == pytest.approx(1.)
    lookup = dict(zip(map(tuple, x.tolist()), y))
    assert h0.nmse(lambda h: torch.stack([lookup[tuple(r)] for r in h.tolist()]), x, y) == pytest.approx(0.)


def test_capture_records_sites_and_every_norm_input_in_stream_order():
    model = tiny()
    ids = list(range(3, 40))
    logits, captured, kwargs, inputs = h0.capture(model, ids, [1, 2], norm_inputs=True)
    assert len(inputs) == 2 * 4 + 1 and all(kw is not None for kw in kwargs)
    for s in (1, 2):
        torch.testing.assert_close(inputs[2 * s + 1], captured[s][0])
        torch.testing.assert_close(inputs[2 * (s + 1)], captured[s][1])
        torch.testing.assert_close(captured[s][1], teacher_mlp(model.model.layers[s], captured[s][0]))
    torch.testing.assert_close(logits, model(input_ids=torch.tensor([ids])).logits[0])


def test_batched_tail_replay_matches_the_clean_forward():
    model = tiny()
    logits, captured, kwargs, _ = h0.capture(model, list(range(5, 45)), [1])
    hidden = captured[1][1][None].expand(3, -1, -1).contiguous()
    replay = h0.tail(model, 2, hidden, kwargs)
    for b in range(3):
        torch.testing.assert_close(replay[b], logits, atol=1e-5, rtol=1e-5)


def test_oracle_kl_is_zero_for_the_exact_state_and_positive_when_the_block_is_dropped():
    model = tiny()
    configs = {1: {'exact': lambda h_in, h_out: h_out, 'drop': lambda h_in, h_out: h_in}}
    result = h0.evaluate_doc(model, {'input_ids': list(range(2, 30))}, [1], configs, 2)
    assert max(result['L1/exact']['kl']) < 1e-6 and len(result['L1/exact']['kl']) == 27
    assert np.mean(result['L1/drop']['kl']) > 1e-4
    assert result['L1/drop']['nmse_num'] == pytest.approx(result['L1/drop']['nmse_den'])


class Tokenizer:
    """Character-level stand-in with a chat template."""

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(ch) % 60 + 2 for ch in text]}

    def apply_chat_template(self, messages, tokenize=False):
        return ''.join(f"<{m['role']}>{m['content']}" for m in messages)


def items(n=400, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        for domain in h0.DOMAINS:
            text = ''.join(rng.choice(list('abcdefghij '), int(rng.integers(30, 90))))
            out.append((domain, f'{domain}-{i}', text if domain == 'prose' else
                        [{'role': 'user', 'content': text[:10]}, {'role': 'assistant', 'content': text}]))
    return out


def rules_for_test(sites=(1, 2)):
    return {
        'seed': 7,
        'corpus': {'salt': 'h0-test', 'min_tokens': 20, 'max_tokens': 64,
                   'domains': {d: {'train': 600, 'dev': 120, 'test': 150} for d in h0.DOMAINS},
                   'copy': {'docs': 3, 'min_passage': 10, 'max_passage': 14}},
        'runtime': {'threads': 2, 'eval_batch': 3, 'max_hours': {'collect': 0.1, 'fit': 0.2, 'evaluate': 0.1}},
        'sites': list(sites),
        'budgets': [{'name': 'moderate', 'k': 16, 'm': 32}, {'name': 'aggressive', 'k': 8, 'm': 16}],
        'chart': {'hidden': 16, 'steps': 30, 'lr': 1e-3, 'batch': 64, 'eval_every': 10, 'samples_per_doc': 4},
        'native': {'steps': 30, 'lr': 1e-3, 'batch': 64, 'eval_every': 10},
        'restricted': {'ridge': 1e-3},
        'curve': {'budget': 'aggressive', 'fractions': [0.25, 0.5]},
        'gate': {'primary_arm': 'learned', 'baseline_arm': 'restricted', 'budget': 'aggressive', 'margin': 0.75,
                 'draws': 500, 'seed': 3, 'copy_quantile': 0.99, 'curve_improvement': 0.1, 'sites_required': 2,
                 'min_baseline_kl': 1e-6, 'weighting': 'domain_equal'},
    }


def test_corpus_has_disjoint_deterministic_splits_and_exact_copies():
    rules = rules_for_test()
    docs = h0.build_corpus(items(), Tokenizer(), rules)
    assert docs == h0.build_corpus(items(), Tokenizer(), rules)
    natural = [d for d in docs if d['domain'] != 'copy']
    assert len({(d['domain'], d['id']) for d in natural}) == len(natural)
    for domain in h0.DOMAINS:
        for split in h0.SPLITS:
            got = sum(len(d['input_ids']) for d in natural if d['domain'] == domain and d['split'] == split)
            assert got >= rules['corpus']['domains'][domain][split]
    assert all(20 <= len(d['input_ids']) <= 64 for d in natural)
    copies = [d for d in docs if d['domain'] == 'copy']
    assert len(copies) == 3
    used = {d['id'] for d in natural if d['domain'] == 'prose'}
    for d in copies:
        a, b = d['copy_span']
        assert 10 <= b - a <= 14 and d['input_ids'][a:b] == d['input_ids'][:b - a]
        assert d['id'].removeprefix('copy-') not in used
    salted = copy.deepcopy(rules)
    salted['corpus']['salt'] = 'other'
    assert [d['id'] for d in h0.build_corpus(items(), Tokenizer(), salted)] != [d['id'] for d in docs]


def test_data_curve_subsets_are_stratified_and_nested():
    docs = [{'domain': h0.DOMAINS[i % 4]} for i in range(40)]
    quarter, half = h0.subset(docs, 0.25), h0.subset(docs, 0.5)
    assert set(quarter) < set(half)
    assert {docs[i]['domain'] for i in quarter} == set(h0.DOMAINS)


def test_site_verdicts_follow_the_frozen_rules():
    rules = rules_for_test()
    rng = np.random.default_rng(0)
    base = rng.uniform(0.9, 1.1, 60)
    copy_tail = list(rng.uniform(0, 1, 200))
    passing = h0.classify_site(0.5 * base, base, copy_tail, copy_tail, 0.5, rules)
    assert passing['verdict'] == 'PASS'
    failing = h0.classify_site(0.98 * base, base, copy_tail, copy_tail, 0.98, rules)
    assert failing['verdict'] == 'FAIL' and failing['reason'] == 'margin not met'
    budget = h0.classify_site(0.98 * base, base, copy_tail, copy_tail, 1.5, rules)
    assert budget['verdict'] == 'INCONCLUSIVE' and budget['reason'].startswith('budget')
    guard = h0.classify_site(0.5 * base, base, [5.] * 200, copy_tail, 0.5, rules)
    assert guard['verdict'] == 'FAIL' and guard['reason'] == 'copy-tail guard'
    still_improving = h0.classify_site(0.5 * base, base, copy_tail, copy_tail, 5., rules)
    assert still_improving['verdict'] == 'PASS'                # a clear pass is not overridden by the curve
    tiny_base = h0.classify_site(0.5 * base * 1e-9, base * 1e-9, copy_tail, copy_tail, 0.5, rules)
    assert tiny_base['verdict'] == 'UNASSESSABLE'
    assert h0.overall(['PASS', 'PASS', 'FAIL'], 2) == 'PASS'
    assert h0.overall(['FAIL', 'INCONCLUSIVE', 'FAIL'], 2) == 'FAIL'
    assert h0.overall(['PASS', 'FAIL', 'INCONCLUSIVE'], 2) == 'INCONCLUSIVE'


def test_domain_equal_weighting_stops_one_domain_dominating():
    new = [0.] * 90 + [1.] * 10
    base = [0.] * 90 + [0.] * 10
    groups = ['prose'] * 90 + ['chat'] * 10
    point, low, high = h0.paired_interval(new, base, 0.75, 200, 0, groups)
    assert point == pytest.approx(0.5) and low == pytest.approx(0.5) and high == pytest.approx(0.5)
    assert h0.paired_interval(new, base, 0.75, 200, 0)[0] == pytest.approx(0.1)
    assert h0.group_mean([1., 3., 10.], ['a', 'a', 'b']) == pytest.approx(6.)


def test_copy_tail_reads_the_predictions_of_the_second_copy():
    rules = rules_for_test(sites=(1,))
    rules['gate']['sites_required'] = 1
    kl = [0.] * 29
    for position in range(19, 29):        # predictions of tokens 20..29, the second copy
        kl[position - 1] = 9.
    configs = ['learned/aggressive', 'restricted/aggressive', 'learned@0.5/aggressive']
    natural = {'id': 'n', 'domain': 'chat', 'results': {f'L1/{c}': {'kl': [0.1] * 29, 'nmse_num': 1., 'nmse_den': 2.}
                                                         for c in configs}}
    copy_row = {'id': 'c', 'domain': 'copy', 'copy_span': [20, 30],
                'results': {f'L1/{c}': {'kl': kl, 'nmse_num': 1., 'nmse_den': 2.} for c in configs}}
    result = h0.gate_from_rows([natural, natural, copy_row], rules)
    assert result['sites'][1]['copy_tail']['new'] == pytest.approx(9.)


def test_training_improves_dev_error_and_keeps_the_best_state():
    torch.manual_seed(0)
    x = torch.randn(2000, 8)
    y = x + torch.tanh(x @ torch.randn(8, 8))
    net = h0.NativeMLP(8, 32, 8, 1e-5)
    f = h0.PCAChart(torch.eye(8))
    report = h0.train(net, h0.predictor('pca', net, f), x[:1600], y[:1600], x[1600:], y[1600:],
                      {'steps': 200, 'lr': 3e-3, 'batch': 128, 'eval_every': 20}, 0)
    assert report['best_dev_nmse'] < 0.5 * report['curve'][0]['dev_nmse']
    assert h0.nmse(h0.predictor('pca', net, f), x[1600:], y[1600:]) == pytest.approx(report['best_dev_nmse'])


def test_learned_chart_beats_pca_on_curved_data():
    torch.manual_seed(0)
    t = torch.rand(3000, 2) * 4 - 2
    embed = orthogonal(16)[:, :6]
    h = torch.cat([t, t.pow(2), t.prod(-1, keepdim=True), torch.sin(2 * t[:, :1])], -1) @ embed.T + 0.5
    moments = [h.T @ h]
    p = h0.shared_basis(moments)[:, :2]
    chart = h0.Chart(p, 64)
    report = h0.train_chart(chart, h[:2500], h[2500:], {'steps': 600, 'lr': 3e-3, 'batch': 256, 'eval_every': 50}, 0)
    assert report['best_dev_relative_error'] < 0.7 * report['curve'][0]['dev_relative_error']
    assert not any(p.requires_grad for p in chart.parameters())


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    rules = rules_for_test()
    monkeypatch.setattr(h0, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(h0, 'RULES', tmp_path / 'rules.json')
    monkeypatch.setattr(h0, 'teacher_files', lambda: {})
    model = tiny()
    monkeypatch.setattr(h0, 'load_teacher', lambda threads, check=True: model)
    (tmp_path / 'rules.json').write_text(json.dumps(rules))
    docs = h0.build_corpus(items(), Tokenizer(), rules)
    h0.save({'corpus_rules_sha256': h0.corpus_rules_sha(rules), 'docs': docs}, tmp_path / 'out/corpus.json')
    return rules, model


class Args:
    reviewed_by = ['codex', 'claude']


def test_stages_refuse_an_unfrozen_or_changed_protocol(sandbox, monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError):
        h0.stage_collect(Args())
    h0.stage_freeze(Args())
    with pytest.raises(ValueError):
        h0.stage_freeze(Args())
    protocol = json.loads((h0.OUT / 'protocol.json').read_text())
    protocol['rules_sha256'] = '0' * 64
    h0.save(protocol, h0.OUT / 'protocol.json')
    with pytest.raises(RuntimeError, match='Rules or corpus changed'):
        h0.require_frozen()
    protocol['status'] = 'draft'
    h0.save(protocol, h0.OUT / 'protocol.json')
    with pytest.raises(RuntimeError, match='not frozen'):
        h0.require_frozen()


def test_all_stages_run_end_to_end_on_a_tiny_model(sandbox):
    rules, model = sandbox
    h0.stage_freeze(Args())
    h0.stage_collect(Args())
    n = sum(len(d['input_ids']) - 1 for d in h0.split_docs(json.loads((h0.OUT / 'corpus.json').read_text()), 'train'))
    assert np.load(h0.OUT / 'states/train-L1-in.npy').shape == (n, 32)
    h0.stage_fit(Args())
    report = json.loads((h0.OUT / 'fit-report.json').read_text())
    assert report['arms']['L2-aggressive']['scalars']['learned'] == 8 + 3 * 8 * 16
    assert {'pca', 'learned', 'full', 'learned@0.25', 'learned@0.5'} <= set(report['arms']['L2-aggressive'])
    h0.stage_evaluate(Args())
    rows = [json.loads(line) for line in (h0.OUT / 'evaluation.jsonl').open()]
    assert max(max(r['results']['L1/exact']['kl']) for r in rows) < 1e-6
    h0.stage_gate(Args())
    gate = json.loads((h0.OUT / 'gate.json').read_text())
    assert gate['overall'] in ('PASS', 'FAIL', 'INCONCLUSIVE')
    assert set(gate['sites']['1']['descriptive']) >= {'restricted/aggressive', 'learned/aggressive', 'drop',
                                                        'pca_floor/moderate', 'chart_floor/aggressive'}
    with pytest.raises(ValueError):
        h0.stage_evaluate(Args())
