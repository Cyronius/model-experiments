import itertools

import torch

from common import absmean_scale, decode_ternary, dequant, gptq_ternary, mse_opt, out_error, rtn


def _spd(n, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * n, n, generator=g) @ torch.randn(n, n, generator=g)
    return X.T @ X / X.shape[0]


def test_decode_roundtrip():
    g = torch.Generator().manual_seed(0)
    t = torch.randint(-1, 2, (16, 256), generator=g).to(torch.int8)
    s = torch.rand(16, 2, generator=g) + 0.01
    w16 = dequant(t, s).half()
    t2, s2, exact = decode_ternary(w16)
    assert exact == 1.0
    assert torch.equal(t2, t)
    assert torch.allclose(s2, s.half().float())


def test_gptq_identity_hessian_is_rtn():
    W = torch.randn(32, 256)
    s = absmean_scale(W)
    assert torch.equal(gptq_ternary(W, torch.eye(256), s), rtn(W, s))


def test_full_width_search_finds_exact_optimum():
    g, R = 6, 5
    W = torch.randn(R, g)
    H = _spd(g, 1)
    s = absmean_scale(W, g)
    T = gptq_ternary(W, H, s, k=3 ** g, g=g)
    Hd = H.double() + 0.01 * torch.diag(H).double().mean() * torch.eye(g, dtype=torch.float64)
    codes = torch.tensor(list(itertools.product((-1, 0, 1), repeat=g)), dtype=torch.float64)
    for r in range(R):
        D = W[r].double() - s[r, 0].double() * codes
        best = ((D @ Hd) * D).sum(1).min()
        d = W[r].double() - s[r, 0].double() * T[r].double()
        assert (d @ Hd @ d) <= best * (1 + 1e-4) + 1e-9


def test_kbest_not_worse_than_gptq_on_average():
    W = torch.randn(64, 256)
    H = _spd(256, 2)
    s = absmean_scale(W)
    e1 = out_error(W, dequant(gptq_ternary(W, H, s, k=1), s), H)
    e8 = out_error(W, dequant(gptq_ternary(W, H, s, k=8), s), H)
    assert e8 <= e1


def test_mse_opt_matches_brute_force():
    g = 4
    W = torch.randn(8, g)
    t, s = mse_opt(W, g)
    codes = torch.tensor(list(itertools.product((-1, 0, 1), repeat=g)), dtype=torch.float32)
    for r in range(W.shape[0]):
        cs = (codes * W[r]).sum(1) / (codes * codes).sum(1).clamp_min(1)
        best = ((W[r] - cs[:, None] * codes) ** 2).sum(1).min()
        got = ((W[r] - s[r, 0] * t[r].float()) ** 2).sum()
        assert got <= best + 1e-5


def test_out_error_zero_for_exact():
    W = torch.randn(8, 128)
    assert out_error(W, W, _spd(128, 3)) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
