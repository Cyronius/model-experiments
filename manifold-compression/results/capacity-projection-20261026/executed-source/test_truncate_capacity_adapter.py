import torch
from truncate_capacity_adapter import factorize

def test_full_rank_reconstruction_even_when_output_width_is_smaller():
    torch.manual_seed(21);a=torch.randn(6,9);b=torch.randn(4,6)
    aa,bb,s=factorize(a,b,6)
    assert aa.shape==a.shape and bb.shape==b.shape
    torch.testing.assert_close(bb@aa,b@a,rtol=1e-5,atol=1e-5)

def test_truncation_attains_direct_svd_optimal_error():
    torch.manual_seed(22);a=torch.randn(6,9);b=torch.randn(8,6)
    aa,bb,s=factorize(a,b,2)
    target=b.double()@a.double();u,sv,vh=torch.linalg.svd(target,full_matrices=False)
    optimal=(u[:,:2]*sv[:2])@vh[:2]
    torch.testing.assert_close((bb@aa).double(),optimal,rtol=1e-5,atol=1e-5)
    assert torch.linalg.matrix_rank(bb@aa)==2
