import types

import numpy as np
import pytest
import torch

import blockwise_heal as bh
import slim_batch as sb
import slim_heal as sh
from test_slim_batch import ids, tiny_qwen
from test_slim_heal import student

CFG = {'loss_chunk': 10, 'ce_weight': 0.5, 'router_weight': 1.0}
BW = {'mse_weight': 1.0}


def setup(layers=4, seed=0):
    model, slim = student(model=tiny_qwen(seed, layers=layers))
    for m in slim.adapted.values():
        if m.b is not None:
            torch.nn.init.normal_(m.b, std=0.1)
    kept = torch.arange(80)
    index = torch.full((96,), -1)
    index[kept] = kept
    vocab = types.SimpleNamespace(index=index, head=model.lm_head.weight[kept].detach())
    x = torch.tensor([ids(6, vocab=80), ids(7, vocab=80)])
    slim.stride = 3
    return model, slim, vocab, x


def grads(slim, clear=True):
    out = [p.grad.clone() if p.grad is not None else None for p in slim.lora_parameters() + slim.router_parameters()]
    if clear:
        for p in slim.lora_parameters() + slim.router_parameters():
            p.grad = None
    return out


def test_partition_covers_every_layer_once_and_shifts_by_half_a_width():
    assert bh.partition(32, 4, False) == [(a, a + 4) for a in range(0, 32, 4)]
    shifted = bh.partition(32, 4, True)
    assert shifted[0] == (0, 2) and shifted[1] == (2, 6) and shifted[-1] == (30, 32)
    for width in (1, 2, 4, 8, 16, 32):
        for s in (False, True):
            windows = bh.partition(32, width, s)
            assert bh.partition_check(windows, 32) == windows
            assert sorted(i for a, b in windows for i in range(a, b)) == list(range(32))
    assert bh.partition(32, 1, True) == bh.partition(32, 1, False)
    assert bh.partition(32, 32, True) == [(0, 32)]
    # Every boundary of the aligned partition lies inside a window of the shifted one.
    for width in (2, 4, 8, 16):
        cuts = {b for _, b in bh.partition(32, width, False)} - {32}
        assert not cuts & {b for _, b in bh.partition(32, width, True)}


def test_plan_alternates_partitions_within_each_stage():
    stages = [{'width': 1, 'steps': 2}, {'width': 4, 'steps': 3}, {'width': 8, 'steps': 2}]
    p = bh.plan(stages, 8)
    assert [s[0] for s in p] == [0, 0, 1, 1, 1, 2, 2]
    assert [s[2] for s in p] == [False, False, False, True, False, False, False]
    assert p[3][3] == [(0, 2), (2, 6), (6, 8)] and p[5][3] == [(0, 8)]


def test_layer_by_layer_stream_is_the_models_forward():
    model, slim, _, x = setup()
    for on in (False, True):
        s = bh.stream(model, slim, x, on)
        slim.on = on
        torch.testing.assert_close(model.model.norm(s[-1]), sb.hidden(model, x.tolist()), atol=1e-5, rtol=1e-5)
        slim.on = False
    assert not slim.on


def test_one_window_of_every_layer_is_the_end_to_end_step():
    model, slim, vocab, x = setup()
    scale = 1 / x[:, 1:].numel()
    kl0, ce0, router0 = sh.micro_step(model, slim, vocab, x, CFG, scale)
    plain = grads(slim)
    kl1, ce1, rel, router1 = bh.block_step(model, slim, vocab, x, [(0, 4)], CFG, BW, scale)
    block = grads(slim)
    assert kl1 == pytest.approx(kl0, rel=1e-5) and ce1 == pytest.approx(ce0, rel=1e-5)
    assert router1 == pytest.approx(router0, rel=1e-5)
    assert rel.shape == (4,) and (rel > 0).all()
    for a, b in zip(plain, block, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    assert slim.stash is None and not slim.on and not slim.done


def test_windows_get_only_their_own_loss():
    model, slim, vocab, x = setup()
    scale = 1 / x[:, 1:].numel()
    bh.block_step(model, slim, vocab, x, [(0, 2), (2, 4)], CFG, BW, scale)
    names = [n for n, m in slim.adapted.items() if m.a is not None]
    got = dict(zip(names, [g for g in grads(slim)[:2 * len(names):2]]))
    # Window [0, 2): the relative error at layer 2's input against the teacher's, by hand.
    teacher = bh.stream(model, slim, x, False)
    slim.on = True
    h, ctx = bh.context(model, x)
    for i in range(2):
        h = bh.layer_call(model, i, ctx)(h)
    (bh.relative_error(h, teacher[2]).sum() * scale).backward()
    # Window [2, 4): the healing loss on top of the student's (detached) stream at layer 2.
    out = h.detach()
    for i in range(2, 4):
        out = bh.layer_call(model, i, ctx)(out)
    loss, *_ = bh.window_loss(model, vocab, out, x, teacher, 4, CFG, BW)
    (loss * scale).backward()
    slim.on = False
    by_hand = dict(zip(names, [p.grad for p in slim.lora_parameters()[::2]]))
    for n in names:
        torch.testing.assert_close(got[n], by_hand[n], atol=1e-6, rtol=1e-5)
    # Splitting the stack changes the early layers' gradients (they no longer see the output loss).
    for p in slim.lora_parameters() + slim.router_parameters():
        p.grad = None
    bh.block_step(model, slim, vocab, x, [(0, 4)], CFG, BW, scale)
    joint = dict(zip(names, [g for g in grads(slim)[:2 * len(names):2]]))
    assert not torch.allclose(joint['0.mlp.gate_proj'], got['0.mlp.gate_proj'])
    torch.testing.assert_close(joint['3.mlp.gate_proj'], got['3.mlp.gate_proj'], atol=1e-6, rtol=1e-5)


def test_bad_windows_are_refused():
    model, slim, vocab, x = setup()
    with pytest.raises(ValueError):
        bh.block_step(model, slim, vocab, x, [(0, 2), (3, 4)], CFG, BW, 1.)


def test_every_lora_parameter_belongs_to_one_layer():
    model, slim, _, _ = setup()
    params = bh.layer_parameters(slim, 4)
    flat = [p for i in range(4) for p in params[i]]
    assert len(flat) == len(slim.lora_parameters())
    assert {id(p) for p in flat} == {id(p) for p in slim.lora_parameters()}


def test_boundary_profile_is_positive_per_layer_and_averaged():
    model, slim, _, _ = setup()
    docs = [{'id': f'd{k}', 'domain': d, 'input_ids': ids(20 + k, vocab=80)} for k, d in enumerate(['a', 'a', 'b'])]
    p = bh.boundary_profile(model, slim, docs)
    assert len(p['per_layer']) == 4 and all(v > 0 for v in p['per_layer'])
    assert p['mean'] == pytest.approx(np.mean(p['per_layer']))
