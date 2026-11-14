import copy
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import moe_carve as mc
import slim_batch as sb
import slim_heal as sh
from h0_mlp_probe import token_kl
from test_moe_carve import BUDGET, fake_sets
from test_slim_batch import ids, tiny_qwen

RANK = 2


def factors_for(model, rank=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    factors = {}
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'linear_attn'):
            factors[i] = {}
            for name in sb.PROJECTIONS:
                w = getattr(layer.linear_attn, name).weight
                x = torch.randn(64, w.shape[1], generator=g, dtype=torch.float64)
                factors[i][name] = sh.lowrank_factors(w, x.T @ x / 64, rank, 1e-4)
    return factors


def student(seed=0, model=None):
    model = model or tiny_qwen(seed)
    s = fake_sets(model)[BUDGET]
    torch.manual_seed(seed)
    routers = [torch.nn.Linear(model.config.hidden_size, s['experts']) for _ in model.model.layers]
    return model, sh.Slim(model, s, routers, factors_for(model), RANK, 4)


def test_lowrank_factors_multiply_to_slim_batch_lowrank():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(12, 10, generator=g)
    x = torch.randn(40, 10, generator=g, dtype=torch.float64)
    cov = x.T @ x
    u, v = sh.lowrank_factors(w, cov, 4, 1e-4)
    assert u.shape == (12, 4) and v.shape == (4, 10)
    torch.testing.assert_close(u @ v, sb.lowrank(w, cov, [4, 6], 1e-4)[0], atol=1e-5, rtol=1e-5)


def test_teacher_mode_is_the_dense_model():
    model = tiny_qwen()
    x = [ids(1), ids(2)]
    dense = sb.hidden(model, x)
    _, slim = student(model=model)
    torch.testing.assert_close(sb.hidden(model, x), dense)


def test_untrained_student_is_the_combined_model():
    model = tiny_qwen()
    reference = copy.deepcopy(model)
    model, slim = student(model=model)
    sets = {BUDGET: slim.s}
    carver = mc.Carver(reference, sets)
    carver.routers, carver.arms = slim.routers, [('learned', BUDGET)] * 2
    cut = sb.DeltaNetCut(reference)
    cut.swap({i: {n: u @ v for n, (u, v) in f.items()} for i, f in factors_for(reference).items()})
    x = [ids(3), ids(4)]
    slim.on = True
    torch.testing.assert_close(sb.hidden(model, x), sb.hidden(reference, x), atol=2e-4, rtol=1e-3)
    slim.on = False
    assert (sb.hidden(model, x) - sb.hidden(reference, x)).abs().max() > 1e-3


def test_decode_continues_the_prefill_with_routed_experts():
    model, slim = student()
    for m in slim.adapted.values():
        if m.b is not None:
            torch.nn.init.normal_(m.b, std=0.1)
    slim.on = True
    x = torch.tensor([ids(5)])
    with torch.no_grad():
        whole = model(input_ids=x, use_cache=False).logits
        out = model(input_ids=x[:, :-3], use_cache=True)
        steps = [out.logits]
        for t in range(x.shape[1] - 3, x.shape[1]):
            out = model(input_ids=x[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
            steps.append(out.logits)
    torch.testing.assert_close(torch.cat(steps, 1), whole, atol=1e-4, rtol=1e-4)


def test_chunk_loss_is_kl_over_the_kept_rows_and_cross_entropy():
    g = torch.Generator().manual_seed(2)
    hs, ht, head = torch.randn(7, 8, generator=g), torch.randn(7, 8, generator=g), torch.randn(20, 8, generator=g)
    target = torch.randint(0, 20, (7,), generator=g)
    kl, ce = sh.chunk_loss(hs, ht, target, head)
    torch.testing.assert_close(kl, token_kl(ht @ head.T, hs @ head.T).sum(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ce, F.cross_entropy(hs @ head.T, target, reduction='sum'))


def gradients(model, slim, vocab, x, cfg):
    sh.micro_step(model, slim, vocab, x, cfg, 1 / x[:, 1:].numel())
    grads = [p.grad.clone() for p in slim.lora_parameters() + slim.router_parameters()]
    for p in slim.lora_parameters() + slim.router_parameters():
        p.grad = None
    return grads


def test_training_step_trains_lora_and_routers_the_same_with_checkpointing():
    model, slim = student()
    for m in slim.adapted.values():
        if m.b is not None:
            torch.nn.init.normal_(m.b, std=0.1)
    kept = torch.arange(80)
    index = torch.full((96,), -1)
    index[kept] = kept
    vocab = types.SimpleNamespace(index=index, head=model.lm_head.weight[kept].detach())
    cfg = {'loss_chunk': 10, 'ce_weight': 0.5, 'router_weight': 1.0}
    x = torch.tensor([ids(6, vocab=80), ids(7, vocab=80)])
    slim.stride = 3
    plain = gradients(model, slim, vocab, x, cfg)
    assert slim.stash is None and not slim.on and not slim.done
    assert all(torch.isfinite(g).all() for g in plain)
    assert all(g.abs().sum() > 0 for g in plain)
    assert not any(p.requires_grad for p in model.parameters() if not any(p is q for q in slim.lora_parameters()))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for a, b in zip(plain, gradients(model, slim, vocab, x, cfg), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_state_round_trips_into_a_fresh_student():
    model, slim = student()
    for m in slim.adapted.values():
        if m.b is not None:
            torch.nn.init.normal_(m.b, std=0.1)
    torch.nn.init.normal_(slim.routers[1].weight)
    state = slim.state()
    other_model, other = student(seed=0)
    other.load(state)
    slim.on = other.on = True
    torch.testing.assert_close(sb.hidden(other_model, [ids(8)]), sb.hidden(model, [ids(8)]))
    assert sum(m.added_parameters() for m in slim.adapted.values()) == sum(
        RANK * sum(m.base.weight.shape) for m in slim.adapted.values() if m.u is not None)


def test_recall_is_recorded_per_layer_and_router_loss_comes_from_the_stash():
    model, slim = student()
    slim.on, slim.recall, slim.stash = True, [], []
    with torch.no_grad():
        sb.hidden(model, [ids(9)])
    assert len(slim.recall) == len(slim.stash) == model.config.num_hidden_layers
    assert all(0 <= r <= 1 for r in slim.recall)
    assert np.isfinite(slim.router_backward(1.0)) and slim.stash is None


def test_schedule_warms_up_then_decays_to_the_floor():
    values = [sh.schedule(s, 100, 10, 0.1) for s in range(100)]
    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)
    assert values[10] == pytest.approx(1.0) and all(a >= b for a, b in zip(values[10:], values[11:]))
    assert values[-1] > 0.1 and sh.schedule(100, 100, 10, 0.1) == pytest.approx(0.1)


def test_allocate_spreads_a_shortfall_over_the_other_domains():
    assert sh.allocate({'a': 5, 'b': 100, 'c': 100}, 60) == {'a': 5, 'b': 27, 'c': 28}
    assert sh.allocate({'a': 50, 'b': 50}, 60) == {'a': 30, 'b': 30}
    assert sum(sh.allocate({'a': 1, 'b': 2}, 60).values()) == 3


def test_shared_windows_find_a_planted_passage_only():
    g = np.random.default_rng(0)
    seqs = g.integers(0, 50000, (4, 200), dtype=np.uint32)
    reference = g.integers(0, 50000, 300).tolist()
    seqs[2, 50:90] = reference[100:140]
    shared = sh.shared_windows(seqs, [reference], 32, rows=3)
    assert shared.tolist() == [0, 0, 9, 0]


def test_collect_skips_excluded_and_repeated_items_caps_them_and_stops_at_quota():
    items = [('a', 'x' * 50), ('b', 'y' * 50), ('c', 'y' * 50), ('d', 'z' * 10), ('e', 'w' * 50), ('f', 'v' * 50)]
    encode = lambda item: (item, [ord(c) for c in item])
    decode = lambda ids: ''.join(map(chr, ids))
    stream, used = sh.collect(items, encode, decode, {'a'}, 70, 40, {'min_item_tokens': 20}, 0)
    assert used == ['b', 'e'] and len(stream) == 82 and stream[40] == 0 and stream[-1] == 0


def test_code_transform_matches_the_corpus():
    assert sh.after_header('"""doc"""\nimport os\n\ndef f():\n    pass\n', 'python') == 'def f():\n    pass\n'
    assert sh.after_header('// licence\n#include <a>\n#include "b"\n\nint x;\n', 'cpp') == 'int x;\n'
    assert sh.after_header('import os\n', 'python') is None
