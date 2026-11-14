import copy

import pytest
import torch

import heal_probe as hp
import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
from test_slim_batch import ids
from test_slim_heal import student


def activations(model, layer, tokens=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = model.model.layers[layer].mlp
    x = torch.randn(1, tokens, model.config.hidden_size, generator=g)
    with torch.no_grad():
        return m.act_fn(m.gate_proj(x)) * m.up_proj(x), x


def test_perfect_route_is_cluster_token_with_a_dense_first_position():
    model, slim = student()
    slim.on = True
    hp.route(slim, 'perfect')
    for i in range(model.config.num_hidden_layers):
        a, x = activations(model, i, seed=i)
        expected = mc.select('cluster_token', slim.s, i, (a[0].float().abs() * slim.wnorm[i]), None)
        expected[0] = True
        assert torch.equal(slim.select(i, a, x)[0], expected)
        assert slim.select(i, a[:, 2:3], x[:, 2:3])[0, 0].equal(expected[2])


def test_routes_switch_back_to_the_learned_routers_and_record_recall():
    model, slim = student()
    a, x = activations(model, 1)
    learned = sh.Slim.select(slim, 1, a, x)
    hp.route(slim, 'all')
    slim.recall = []
    assert slim.select(1, a, x).all() and slim.recall == [1.0]
    hp.route(slim, 'perfect')
    hp.route(slim, 'learned')
    assert 'select' not in vars(slim)
    assert torch.equal(slim.select(1, a, x), learned)


def test_repeated_counts_spans_that_occur_again():
    assert hp.repeated(list(range(300)), 20, 1000) == 0
    assert hp.repeated(list(range(10)), 20, 1000) == 0
    # Only the last 100 tokens count: a period of 5 repeats all but the first 5 of 81 spans.
    assert hp.repeated(list(range(500)) + [1, 2, 3, 4, 5] * 20, 20, 100) == 1 - 5 / 81


def test_a_loop_is_a_capped_run_that_repeats_itself():
    loop = {'window': 4, 'last': 50, 'threshold': 0.45}
    rows = [{'id': 'a', 'tokens': [7, 8] * 30, 'length': 60, 'finished': False, 'correct': False},
            {'id': 'b', 'tokens': [7, 8] * 30, 'length': 60, 'finished': True, 'correct': True},
            {'id': 'c', 'tokens': list(range(60)), 'length': 60, 'finished': False, 'correct': False}]
    g = hp.generations(rows, loop)
    assert (g['loops'], g['looping'], g['correct'], g['finished'], g['mean_length']) == (1, ['a'], 1, 1, 60)
    assert g['finished_median_length'] == 60 and hp.generations(rows[:1], loop)['finished_median_length'] is None


def test_unhealing_zeroes_the_updates_and_restores_the_routers():
    model, slim = student()
    before = copy.deepcopy(slim.state())
    with torch.no_grad():
        for p in slim.lora_parameters() + slim.router_parameters():
            p.add_(torch.randn_like(p))
    slim.adapters = {False: before, True: copy.deepcopy(slim.state())}
    x = torch.randint(0, model.config.vocab_size, (1, 7))
    untrained, plain = student()
    slim.on = plain.on = True
    healed = model(x).logits
    hp.heal(slim, False)
    assert all(not m.b.any() for m in slim.adapted.values() if m.b is not None)
    assert torch.equal(model(x).logits, untrained(x).logits)
    assert not torch.allclose(model(x).logits, healed)
    hp.heal(slim, True)
    assert torch.equal(model(x).logits, healed)


def test_evaluate_arms_default_to_the_healed_student():
    assert hp.arm('perfect') == {'name': 'perfect', 'router': 'perfect', 'healed': True}
    spec = {'name': 'all_unhealed', 'router': 'all', 'healed': False}
    assert hp.arm(spec) is spec
    assert hp.label({'router': 'all', 'state': 'fp32', 'healed': False}) == 'unhealed, every neuron, fp32 state'
    assert hp.label({'router': 'learned', 'state': 'int16_row'}) == 'healed, learned router, int16_row state'


def soft_student(scores):
    """A student whose layer-1 router returns fixed scores, with a soft-edge width of 0.5."""
    model, slim = student()
    slim.routers[1] = lambda x: scores.expand(x.shape[0], -1)
    slim.width = [0.5] * model.config.num_hidden_layers
    return model, slim


def test_soft_edge_weighs_experts_by_their_lead_over_the_first_left_out():
    model, slim = student()
    k, e = mc.at(slim.s, 'K', 1), slim.s['experts']
    scores = torch.linspace(0, 1, e)[torch.randperm(e, generator=torch.Generator().manual_seed(0))]
    model, slim = soft_student(scores)
    a, x = activations(model, 1)
    weights = hp.soft(slim, 1, a, x)
    cluster = slim.s['cluster'][1]
    edge = scores.sort(descending=True).values[k]
    expected = ((scores - edge) / 0.5).clamp(0, 1)
    assert weights.dtype == a.dtype and (weights[0, 0] == 1).all()
    for token in weights[0, 1:]:
        assert (token[cluster < 0] == 1).all()
        torch.testing.assert_close(token[cluster >= 0], expected[cluster[cluster >= 0]])
    assert (expected > 0).sum() == k


def test_soft_edge_swaps_experts_smoothly():
    model, slim = student()
    k, e = mc.at(slim.s, 'K', 1), slim.s['experts']
    base = torch.linspace(0, 1, e)
    a, x = activations(model, 1, tokens=1)
    soft, hard = [], []
    for nudge in (1e-4, -1e-4):
        # The K-th and (K+1)-th experts swap places.
        scores = base.clone()
        scores[e - k] = scores[e - k - 1] + nudge
        _, slim = soft_student(scores)
        soft.append(hp.soft(slim, 1, a, x)[0, 0])
        hard.append(mc.expand(slim.s, 1, mc.topmask(scores[None], k))[0])
    cluster = slim.s['cluster'][1]
    assert (hard[0] != hard[1]).sum() == ((cluster == e - k) | (cluster == e - k - 1)).sum()
    assert (soft[0] - soft[1]).abs().max() < 1e-3


def test_soft_route_needs_widths():
    _, slim = student()
    with pytest.raises(ValueError):
        hp.route(slim, 'soft')
    slim.width = [1.0] * len(slim.routers)
    hp.route(slim, 'soft')
    assert 'select' in vars(slim)
    assert hp.label({'router': 'soft', 'state': 'fp32'}) == 'healed, learned router with a soft edge, fp32 state'


def synthetic_moments(lost_of, n=12, tokens=400, seed=0):
    """Fit and held-out moments for tokens that run the shared neurons (group 0) and one of two
    routed experts, with lost output lost_of(run)."""
    g = torch.Generator().manual_seed(seed)
    group = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    moments = []
    for split in range(2):
        m = hp.Moments(n, 5, 'cpu')
        run = torch.randn(tokens, n, generator=g, dtype=torch.float64).float()
        chosen = torch.randint(1, 3, (tokens, 1), generator=g)
        run = run * ((group == 0) | (group == chosen))
        lost = lost_of(run)
        m.add(run, lost, lost + torch.randn(tokens, 5, generator=g))
        moments.append(m)
    return moments, group


def test_fold_learns_the_lost_output_the_running_neurons_predict():
    s = torch.randn(12, 5, generator=torch.Generator().manual_seed(1))
    (fit, held), group = synthetic_moments(lambda run: run @ s)
    w = torch.randn(5, 12)
    folds, report = hp.fit_layer(fit, held, w, group, [1e-6, 1e-2, 1.0])
    assert report['fold']['rho'] == 1e-6 and report['fold']['recovered'] > 0.999
    torch.testing.assert_close(folds['fold'], s.T, atol=1e-3, rtol=1e-3)
    assert report['gains']['recovered'] < report['fold']['recovered']


def test_fold_error_is_the_held_out_error():
    g = torch.Generator().manual_seed(2)
    s = torch.randn(12, 5, generator=g)
    noisy = lambda run: run @ s + 3 * torch.randn(run.shape[0], 5, generator=g)
    tokens = []
    (fit, held), group = synthetic_moments(lambda run: tokens.append((run, noisy(run))) or tokens[-1][1])
    folds, report = hp.fit_layer(fit, held, torch.randn(5, 12), group, [1e-3, 1e-1, 10.0])
    run, lost = tokens[1]
    direct = (lost - run @ folds['fold'].T).square().sum()
    torch.testing.assert_close(1 - direct / lost.square().sum(), torch.tensor(report['fold']['recovered']),
                               atol=1e-4, rtol=1e-4)


def test_gains_scale_whole_experts():
    w = torch.randn(5, 12, generator=torch.Generator().manual_seed(3))
    gamma = torch.tensor([0.3, -0.2, 0.5])
    group = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    (fit, held), _ = synthetic_moments(lambda run: (run * gamma[group]) @ w.T)
    folds, report = hp.fit_layer(fit, held, w, group, [1e-6, 1.0])
    assert report['gains']['recovered'] > 0.999
    torch.testing.assert_close(folds['gains'], gamma[group], atol=1e-3, rtol=1e-3)
    assert report['gains']['shared_gain'] == pytest.approx(1.3, abs=1e-3)
    assert (folds['gain_shared'][group > 0] == 0).all() and folds['gain_shared'][0] != 0
    assert 0 < report['gain_shared']['recovered'] < report['gains']['recovered']


def test_no_fold_when_none_helps_on_held_out_tokens():
    g = torch.Generator().manual_seed(4)
    (fit, held), group = synthetic_moments(lambda run: torch.randn(run.shape[0], 5, generator=g), tokens=30)
    folds, report = hp.fit_layer(fit, held, torch.randn(5, 12), group, [1e-6])
    assert report['fold']['rho'] is None and report['fold']['recovered'] == 0
    assert not folds['fold'].any()


def test_fold_moments_are_the_routed_students_own():
    model, slim = student()
    docs = [{'input_ids': ids(s)} for s in range(3)]
    moments = hp.fold_moments(model, slim, docs, [1, 2], batch=2)
    assert slim.on is False and 'select' not in vars(slim)
    seen = {1: [], 2: []}

    def record(i, a, x):
        keep = sh.Slim.select(slim, i, a, x)
        if i in seen:
            seen[i].append((a[:, 1:].flatten(0, 1), keep[:, 1:].flatten(0, 1)))
        return keep

    slim.select, slim.on = record, True
    for k in (0, 2):
        sb.hidden(model, [d['input_ids'] for d in docs[k:k + 2]])
    slim.on = False
    for i in (1, 2):
        a = torch.cat([a for a, _ in seen[i]])
        run = a * torch.cat([k for _, k in seen[i]])
        down = model.model.layers[i].mlp.down_proj
        torch.testing.assert_close(moments[i].g, run.T @ run)
        torch.testing.assert_close(moments[i].c, run.T @ down(a - run))
        assert moments[i].tokens == 3 * 23 and 0 < moments[i].lost < moments[i].full


def folded_student(seed=5):
    model, slim = student()
    g = torch.Generator().manual_seed(seed)
    layers, n, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    weights = {'fold': [torch.randn(d, n, generator=g) for _ in range(layers)],
               'gains': [torch.rand(n, generator=g) for _ in range(layers)],
               'gain_shared': [torch.rand(n, generator=g) for _ in range(layers)]}
    return model, slim, weights


def test_fold_changes_only_the_routed_student_past_the_first_position():
    model, slim, weights = folded_student()
    x = [ids(7)]
    teacher = sb.hidden(model, x)
    slim.on = True
    unfolded = sb.hidden(model, x)
    slim.on = False
    hp.install(model, slim, weights)
    torch.testing.assert_close(sb.hidden(model, x), teacher)
    slim.on = True
    torch.testing.assert_close(sb.hidden(model, x), unfolded)
    hp.use_fold(slim, {'router': 'learned', 'fold': 'fold'})
    folded = sb.hidden(model, x)
    torch.testing.assert_close(folded[:, 0], unfolded[:, 0])
    assert (folded[:, 1:] - unfolded[:, 1:]).abs().max() > 1e-2
    down = model.model.layers[1].mlp.down_proj
    a = torch.randn(1, 3, model.config.intermediate_size)
    with torch.no_grad():
        torch.testing.assert_close(down(a)[:, 1:], down.inner(a)[:, 1:] + a[:, 1:] @ weights['fold'][1].T)
        torch.testing.assert_close(down(a[:, 2:]), down.inner(a[:, 2:]) + a[:, 2:] @ weights['fold'][1].T)
        slim.folded = 'gains'
        torch.testing.assert_close(down(a)[:, 1:], down.inner(a * (1 + weights['gains'][1]))[:, 1:])
    slim.on = False


def test_folds_need_the_learned_router_and_their_weights():
    model, slim, weights = folded_student()
    with pytest.raises(ValueError):
        hp.use_fold(slim, {'router': 'learned', 'fold': 'fold'})
    hp.install(model, slim, weights)
    with pytest.raises(ValueError):
        hp.use_fold(slim, {'router': 'perfect', 'fold': 'fold'})
    hp.use_fold(slim, {'router': 'all'})
    assert slim.folded is None
    hp.install(model, slim, weights)
    assert isinstance(model.model.layers[0].mlp.down_proj.inner, sh.Adapted)
    assert hp.label({'router': 'learned', 'state': 'fp32', 'fold': 'gains'}) == \
        'healed, learned router with expert gains, fp32 state'


def test_fit_keeps_no_autograd_history_of_the_trained_update():
    model, slim = student()
    down = model.model.layers[1].mlp.down_proj
    w = hp.down_weight(down)
    assert down.a.requires_grad and not w.requires_grad
    (fit, held), group = synthetic_moments(lambda run: run @ torch.randn(12, 5))
    folds, _ = hp.fit_layer(fit, held, torch.randn(5, 12, requires_grad=True), group, [0.1])
    assert not any(f.requires_grad for f in folds.values())
