import torch

import h0_floor_probe as floor
import h0_mlp_probe as h0
from test_h0_mlp_probe import tiny


def spd(d, seed, decay=0.8):
    g = torch.Generator().manual_seed(seed)
    q = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
    return (q * decay ** torch.arange(d, dtype=torch.float64)) @ q.T


def test_fisher_chart_is_strict_and_reduces_to_pca_under_a_flat_fisher():
    c = spd(12, 0)
    e, dec = floor.fisher_chart(c, torch.eye(12, dtype=torch.float64), 5, 0.)
    torch.testing.assert_close(e @ dec, torch.eye(5), atol=1e-5, rtol=0)
    p = floor.site_basis(c)[:, :5]
    torch.testing.assert_close(dec @ e, p @ p.T, atol=1e-5, rtol=0)


def test_fisher_chart_beats_pca_on_the_fisher_weighted_error():
    c, f = spd(16, 1), spd(16, 2, decay=0.6)
    for k in (2, 6, 10):
        fisher = floor.weighted_residual(c, f, *floor.fisher_chart(c, f, k, 0.))
        pca = floor.weighted_residual(c, f, *floor.pca_chart(floor.site_basis(c), k))
        assert fisher <= pca + 1e-9
    assert abs(floor.weighted_residual(c, f, *floor.fisher_chart(c, f, 16, 0.))) < 1e-8


def test_gradient_tail_matches_the_replay_tail():
    model = tiny()
    ids = torch.randint(0, 64, (12,), generator=torch.Generator().manual_seed(3)).tolist()
    _, captured, kwargs, _ = h0.capture(model, ids, [1])
    h = captured[1][1]
    torch.testing.assert_close(floor.tail_grad(model, 2, h[None], kwargs), h0.tail(model, 2, h[None], kwargs))


def test_fisher_sum_is_symmetric_positive_semidefinite_and_nonzero():
    model = tiny()
    ids = torch.randint(0, 64, (12,), generator=torch.Generator().manual_seed(4)).tolist()
    result = floor.fisher_sum(model, ids, [1, 2], torch.Generator().manual_seed(0))
    for value in result.values():
        torch.testing.assert_close(value, value.T)
        assert torch.linalg.eigvalsh(value).min() > -1e-8
        assert value.trace() > 0


def test_whole_stack_squeeze_is_exact_at_full_width_and_lossy_when_narrow():
    model = tiny()
    moments = [spd(32, i).float() for i in range(2 * 4 + 1)]
    bases = floor.stack_bases(moments, floor.site_basis(moments[0]))
    assert len(bases['site']) == 5
    ids = torch.randint(0, 64, (12,), generator=torch.Generator().manual_seed(5)).tolist()
    clean = model(input_ids=torch.tensor([ids])).logits[0]
    torch.testing.assert_close(floor.squeezed_logits(model, ids, bases['site'], 32), clean, atol=1e-4, rtol=1e-4)
    narrow = floor.squeezed_logits(model, ids, bases['site'], 4)
    kl = h0.token_kl(clean, narrow)
    assert kl[0] < 1e-6 and kl[1:].mean() > 1e-3  # position 0 keeps the teacher state
    # Hooks are removed afterwards.
    torch.testing.assert_close(model(input_ids=torch.tensor([ids])).logits[0], clean)
