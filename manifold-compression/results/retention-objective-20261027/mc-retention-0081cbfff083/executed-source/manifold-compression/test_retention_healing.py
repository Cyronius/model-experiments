import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from correctness_heal import LowRankLinear
from retention_healing import (SelectionPolicy, coarsened_forward_kl, compare_cases,
                              expand_lora, require_frozen_protocol,
                              replay_token_keys, retention_objective, teacher_topk_tail)


def test_expansion_preserves_learned_update_and_trains_new_directions():
    torch.manual_seed(23)
    model = nn.Sequential(LowRankLinear(nn.Linear(7, 5), 2, 4))
    with torch.no_grad():
        model[0].b.normal_()
    original = copy.deepcopy(model)
    x = torch.randn(4, 7)
    before = model(x).detach()
    rng = torch.get_rng_state()
    info = expand_lora(model, 6)
    assert torch.equal(rng, torch.get_rng_state())
    assert info['rank'] == 6 and info['alpha'] == model[0].alpha == 12
    torch.testing.assert_close(model(x), before, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model[0].b @ model[0].a, original[0].b @ original[0].a)
    torch.testing.assert_close(model[0].a[:2], original[0].a, atol=0, rtol=0)
    assert torch.count_nonzero(model[0].b[:, 2:]) == 0
    model(x).square().sum().backward()
    assert model[0].b.grad[:, 2:].abs().sum() > 0
    assert model[0].base.weight.grad is None
    with pytest.raises(ValueError, match='increase rank'):
        expand_lora(model, 6)


def test_full_vocabulary_cache_matches_exact_kl_value_and_gradient():
    torch.manual_seed(6)
    teacher = torch.randn(7, 11, dtype=torch.float64)
    student = torch.randn(7, 11, dtype=torch.float64, requires_grad=True)
    temperature = 1.7
    cache = teacher_topk_tail(teacher, 11, temperature)
    got = coarsened_forward_kl(student, cache)
    exact = F.kl_div(F.log_softmax(student / temperature, -1),
                    F.softmax(teacher / temperature, -1), reduction='none').sum(-1) * temperature ** 2
    torch.testing.assert_close(got, exact)
    a, = torch.autograd.grad(got.sum(), student, retain_graph=True)
    b, = torch.autograd.grad(exact.sum(), student)
    torch.testing.assert_close(a, b)


def test_tail_mass_is_retained_and_coarsening_is_a_lower_bound():
    teacher = torch.tensor([[2., 1., 0., -1.]], dtype=torch.float64)
    student = torch.tensor([[0., 1., -2., 3.]], dtype=torch.float64, requires_grad=True)
    cache = teacher_topk_tail(teacher, 1)
    teacher_p = teacher.softmax(-1)
    torch.testing.assert_close(cache['log_probs'].exp(), torch.stack((teacher_p[:, 0], teacher_p[:, 1:].sum(-1)), -1))
    got = coarsened_forward_kl(student, cache)
    q = student.softmax(-1)
    p_buckets = torch.stack((teacher_p[:, 0], teacher_p[:, 1:].sum(-1)), -1)
    q_buckets = torch.stack((q[:, 0], q[:, 1:].sum(-1)), -1)
    expected = (p_buckets * (p_buckets.log() - q_buckets.log())).sum(-1)
    torch.testing.assert_close(got, expected)
    full = F.kl_div(student.log_softmax(-1), teacher_p, reduction='sum')
    assert 0 <= got.item() <= full.item()
    got.sum().backward()
    assert torch.isfinite(student.grad).all() and student.grad[0, 3] != 0
    torch.testing.assert_close(coarsened_forward_kl(teacher, cache), torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0)


def test_extreme_logits_have_finite_tail_and_gradients():
    teacher = torch.tensor([[100., 0., -100., -200.]])
    student = torch.tensor([[80., 0., -80., -160.]], requires_grad=True)
    cache = teacher_topk_tail(teacher, 1)
    assert torch.isfinite(cache['log_probs']).all()
    kl = coarsened_forward_kl(student, cache)
    kl.sum().backward()
    assert torch.isfinite(kl).all() and torch.isfinite(student.grad).all()


def test_cache_rejects_duplicate_ids_and_missing_mass():
    logits = torch.tensor([[1., 2., 3.]])
    cache = teacher_topk_tail(logits, 2)
    broken = copy.deepcopy(cache)
    broken['indices'][:, 1] = broken['indices'][:, 0]
    with pytest.raises(ValueError, match='distinct'):
        coarsened_forward_kl(logits, broken)
    broken = copy.deepcopy(cache)
    broken['log_probs'] += 1
    with pytest.raises(ValueError, match='sum to one'):
        coarsened_forward_kl(logits, broken)


def test_retention_loss_only_changes_replay_token_gradients():
    torch.manual_seed(9)
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    mask = torch.tensor([True, False, True, False])
    keys = replay_token_keys('r1', [3, 0, 1, 2, 3], [0, 2])
    cache = teacher_topk_tail(torch.randn(2, 5), 3, token_keys=keys)
    mixed = retention_objective(logits, targets, mask, cache, replay_keys=keys)
    control = retention_objective(logits, targets, mask, kd_weight=0)
    assert mixed['answer_tokens'] == 4 and mixed['replay_tokens'] == 2
    grad_mixed, = torch.autograd.grad(mixed['loss_sum'], logits, retain_graph=True)
    grad_ce, = torch.autograd.grad(control['loss_sum'], logits)
    torch.testing.assert_close(grad_mixed[~mask], grad_ce[~mask], atol=0, rtol=0)
    assert not torch.allclose(grad_mixed[mask], grad_ce[mask])
    with pytest.raises(ValueError, match='Missing teacher'):
        retention_objective(logits, targets, mask)


def test_cache_alignment_rejects_same_count_reordered_or_changed_tokens():
    logits = torch.randn(2, 5)
    targets = torch.tensor([1, 2])
    mask = torch.ones(2, dtype=torch.bool)
    keys = replay_token_keys('r1', [0, 1, 2], [0, 1])
    cache = teacher_topk_tail(logits, 2, token_keys=keys)
    for wrong in (keys[::-1], replay_token_keys('r1', [4, 1, 2], [0, 1])):
        with pytest.raises(ValueError, match='identities or order'):
            retention_objective(logits, targets, mask, cache, replay_keys=wrong)
    with pytest.raises(ValueError, match='token identity'):
        retention_objective(logits, targets, mask, cache)
    no_identity = teacher_topk_tail(logits, 2)
    with pytest.raises(ValueError, match='token identity'):
        retention_objective(logits, targets, mask, no_identity, replay_keys=keys)


def cases():
    rows = []
    for category in ['primary', 'retention', 'program', 'science']:
        for i in range(2):
            rows.append({'id': f'{category}-{i}', 'category': category, 'family': f'f{i}',
                         'group': f'{category}-group{i}', 'correct': category != 'primary',
                         'loop': False, 'ended': True})
    return rows


def test_selection_rejects_code_loss_hidden_by_primary_gains():
    before = cases()
    after = copy.deepcopy(before)
    after[0]['correct'] = after[1]['correct'] = True
    after[4]['correct'] = False
    result = compare_cases(before, after, SelectionPolicy(.1, 0.))
    assert not result['eligible']
    assert result['categories']['primary']['gained'] == 2
    assert result['categories']['program']['lost'] == 1
    assert 'program:f0 correctness regressed' in result['reasons']


def test_selection_gates_retention_science_and_completion_separately():
    before = cases()
    after = copy.deepcopy(before)
    after[0]['correct'] = True
    policy = SelectionPolicy(.1, 0.)
    assert compare_cases(before, after, policy)['eligible']
    after[2]['correct'] = False
    after[6]['correct'] = False
    after[0]['ended'] = after[1]['ended'] = False
    after[1]['loop'] = True
    result = compare_cases(before, after, policy)
    assert len(result['reasons']) == 4
    with pytest.raises(ValueError, match='identical case IDs'):
        compare_cases(before, after[:-1], policy)
    with pytest.raises(ValueError, match='four evaluation categories'):
        compare_cases(before[:4], after[:4], policy)


def test_repaired_loop_cannot_mask_a_new_loop():
    before = cases()
    before[0]['loop'] = True
    after = copy.deepcopy(before)
    after[0]['correct'] = True
    after[0]['loop'] = False
    after[1]['loop'] = True
    result = compare_cases(before, after, SelectionPolicy(.1, 0.))
    assert result['extra_loops'] == 0 and result['new_loops'] == 1
    assert not result['eligible'] and 'too many new loop cases' in result['reasons']


def test_design_cannot_be_run_by_accident():
    protocol = json.loads(Path(__file__).with_name('retention_protocol.json').read_text())
    with pytest.raises(ValueError, match='design-only'):
        require_frozen_protocol(protocol)
    protocol['status'] = 'frozen'
    with pytest.raises(ValueError, match='Missing frozen'):
        require_frozen_protocol(protocol)
