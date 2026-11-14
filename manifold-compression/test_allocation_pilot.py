import copy
import json

import pytest
import torch

from allocation_pilot import (apply_pruning, construct, mlp_budget, mlp_gram, projected_scalars, prune_mlp,
                              refit_r2, select_and_refit)
from audit_rotation import cache_check
from local_projection_models import build, stored_scalars
from test_projection_models import tiny_model


def basis(d=32, k=24):
    torch.manual_seed(271)
    return torch.linalg.qr(torch.randn(d, d, dtype=torch.float64)).Q.float()[:, :k].contiguous()


def rows(n=6, length=11):
    g = torch.Generator().manual_seed(27)
    return [{'input_ids': torch.randint(0, 101, (length,), generator=g).tolist()} for _ in range(n)]


def test_select_and_refit_matches_explicit_least_squares():
    torch.manual_seed(1)
    phi = torch.randn(500, 12, dtype=torch.float64) * torch.linspace(.2, 2, 12, dtype=torch.float64)
    down = torch.randn(5, 12)
    keep, w = select_and_refit(phi.T @ phi, down, 7, ridge=0)
    importance = phi.square().sum(0) * down.double().square().sum(0)
    assert torch.equal(keep, importance.topk(7).indices.sort().values)
    target = phi @ down.double().T
    expected = torch.linalg.lstsq(phi[:, keep], target).solution.T
    torch.testing.assert_close(w.double(), expected, atol=1e-5, rtol=1e-5)
    r2 = refit_r2(phi.T @ phi, phi.sum(0), len(phi), down, keep, w)
    fitted = phi[:, keep] @ w.double().T
    direct = 1 - (target - fitted).square().sum() / (target - target.mean(0)).square().sum()
    assert abs(r2 - direct.item()) < 1e-8


def test_no_pruning_is_an_exact_identity_and_unridged_refit_recovers_down():
    model = tiny_model('qwen', True)
    ids = torch.randint(0, 101, (1, 13))
    expected = model(ids, use_cache=False).logits
    gram, total, count = mlp_gram(model, 0, rows())
    down = model.model.layers[0].mlp.down_proj.weight.clone()
    pruned, keep, kept_down, report = construct(model, 48, rows())
    torch.testing.assert_close(pruned(ids, use_cache=False).logits, expected, atol=0, rtol=0)
    assert report == [] and keep.shape == (2, 48) and kept_down.shape == (2, 32, 48)
    # 60 Gram tokens over 48 neurons: the unridged refit recovers the original down projection.
    keep, refit = select_and_refit(gram, down, 48, ridge=0)
    torch.testing.assert_close(refit, down, atol=2e-4, rtol=2e-4)
    assert refit_r2(gram, total, count, down, keep, refit) > 1 - 1e-6


def test_refit_rejects_invalid_requests():
    gram, down = torch.eye(6, dtype=torch.float64), torch.randn(3, 6)
    for kwargs in ({'keep_count': 0}, {'keep_count': 7}, {'keep_count': 3, 'ridge': -1}):
        with pytest.raises(ValueError):
            select_and_refit(gram, down, **kwargs)
    with pytest.raises(ValueError):
        select_and_refit(torch.zeros(6, 6), down, 3)
    with pytest.raises(ValueError):
        select_and_refit(torch.full((6, 6), float('nan')), down, 3)


def test_model_run_stages_refuse_draft_or_changed_protocols(tmp_path, monkeypatch):
    import allocation_pilot
    import local_projection_probe
    repo, source = tmp_path / 'repo', tmp_path / 'repo/source'
    (repo / 'mc').mkdir(parents=True)
    source.mkdir()
    for name in allocation_pilot.SOURCE_FILES:
        (source / name).write_text(name)
    (repo / 'mc/code.py').write_text('x = 1\n')
    (repo / 'input.pt').write_text('data')
    (repo / 'gates.json').write_text('{}')
    monkeypatch.setattr(allocation_pilot, 'ROOT', repo / 'mc')
    monkeypatch.setattr(allocation_pilot, 'CODE', ['code.py'])
    monkeypatch.setattr(allocation_pilot, 'INPUTS', [repo / 'input.pt'])
    monkeypatch.setattr(local_projection_probe, 'SOURCE', source)
    (tmp_path / 'protocol.json').write_text(json.dumps({'status': 'draft', 'missing': ['review']}))
    with pytest.raises(RuntimeError, match='draft'):
        allocation_pilot.require_frozen(tmp_path)
    for reviewer, hours, gates in ((None, 2, repo / 'gates.json'), ('codex', 2, repo / 'absent.json'),
                                   ('codex', float('nan'), repo / 'gates.json'), ('codex', -1, repo / 'gates.json')):
        with pytest.raises(ValueError):
            allocation_pilot.stage_freeze(tmp_path, reviewer, hours, gates)
    allocation_pilot.stage_freeze(tmp_path, 'codex', 2, repo / 'gates.json')
    allocation_pilot.require_frozen(tmp_path)
    for changed in (repo / 'mc/code.py', source / 'tokenizer.json', repo / 'gates.json'):
        original = changed.read_text()
        changed.write_text(original + ' ')
        with pytest.raises(RuntimeError, match='changed'):
            allocation_pilot.require_frozen(tmp_path)
        changed.write_text(original)


def test_freezing_without_gates_keeps_evaluation_requirement(tmp_path, monkeypatch):
    import allocation_pilot
    import local_projection_probe
    (tmp_path / 'mc').mkdir()
    (tmp_path / 'mc/code.py').write_text('x = 1\n')
    for name in allocation_pilot.SOURCE_FILES:
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(allocation_pilot, 'ROOT', tmp_path / 'mc')
    monkeypatch.setattr(allocation_pilot, 'CODE', ['code.py'])
    monkeypatch.setattr(allocation_pilot, 'INPUTS', [])
    monkeypatch.setattr(local_projection_probe, 'SOURCE', tmp_path)
    (tmp_path / 'protocol.json').write_text(json.dumps({'status': 'draft', 'missing': ['review']}))
    allocation_pilot.stage_freeze(tmp_path, 'codex', 1.5)
    protocol = json.loads((tmp_path / 'protocol.json').read_text())
    assert protocol['status'] == 'frozen' and protocol['generation_gates'] is None
    assert protocol['missing'] == ['0a generation gates, required before any P1 evaluation claim']


def test_resumed_construction_matches_an_uninterrupted_one():
    p = basis(k=28)
    full, keep, down, _ = construct(build(tiny_model('qwen', True), [p] * 2, [1.02] * 5), 30, rows())
    saved = []
    first = build(tiny_model('qwen', True), [p] * 2, [1.02] * 5)
    construct(first, 30, rows(), progress=lambda i, k, d, r: saved.append((k, d, r)))
    # Resume from layer 0 only, as after an interruption during layer 1.
    resumed, keep2, down2, _ = construct(build(tiny_model('qwen', True), [p] * 2, [1.02] * 5), 30, rows(),
                                         done=saved[:1])
    assert torch.equal(keep, keep2) and torch.equal(down, down2)
    ids = torch.randint(0, 101, (1, 13))
    torch.testing.assert_close(resumed(ids, use_cache=False).logits, full(ids, use_cache=False).logits, atol=0, rtol=0)


def test_runtime_bound_stops_construction():
    model = build(tiny_model('qwen', True), [basis()] * 2, [1.03] * 5)
    with pytest.raises(TimeoutError):
        construct(model, 20, rows(), deadline=0)


def test_gram_uses_already_pruned_upstream_layers():
    model = build(tiny_model('qwen', True), [basis()] * 2, [1.03] * 5)
    before, _, _ = mlp_gram(model, 1, rows())
    gram, _, _ = mlp_gram(model, 0, rows())
    keep, down = select_and_refit(gram, model.model.layers[0].mlp.down_proj.weight, 20)
    prune_mlp(model.model.layers[0].mlp, keep, down)
    after, _, count = mlp_gram(model, 1, rows())
    assert count == 6 * 10  # position 0 skipped
    assert not torch.allclose(before, after)


def test_budget_is_exact_and_saved_pruning_reloads_identically():
    reference = projected_scalars(tiny_model('qwen', True), 24)
    budget = mlp_budget(lambda: tiny_model('qwen', True), 28, reference)
    assert budget['scalars'] <= reference < budget['scalars'] + budget['per_neuron_scalars']
    p = torch.linalg.qr(torch.randn(32, 32, dtype=torch.float64)).Q.float()[:, :28].contiguous()
    model, keep, down, _ = construct(build(tiny_model('qwen', True), [p] * 2, [1.02] * 5),
                                     budget['mlp_width'], rows())
    assert stored_scalars(model) == budget['scalars']
    other = apply_pruning(build(tiny_model('qwen', True), [p] * 2, [1.02] * 5),
                          {'keep': keep, 'down': down, 'mlp_width': budget['mlp_width']})
    ids = torch.randint(0, 101, (1, 13))
    torch.testing.assert_close(other(ids, use_cache=False).logits, model(ids, use_cache=False).logits, atol=0, rtol=0)
    assert cache_check(other, ids)['numerical_pass']
    assert other.model.layers[0].mlp.intermediate_size == budget['mlp_width']


def test_projected_scalar_count_matches_a_real_build():
    real = build(tiny_model('qwen', True), [basis()] * 2, [1.] * 5)
    assert projected_scalars(copy.deepcopy(tiny_model('qwen', True)), 24) == stored_scalars(real)
