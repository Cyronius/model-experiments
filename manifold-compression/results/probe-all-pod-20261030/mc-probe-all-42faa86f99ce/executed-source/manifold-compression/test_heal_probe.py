import copy

import torch

import heal_probe as hp
import moe_carve as mc
import slim_heal as sh
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
