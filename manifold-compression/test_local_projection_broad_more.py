"""Verify that incremental calibration matches collecting the same rows at once."""
from types import SimpleNamespace

import torch
from torch import nn

from local_projection_broad_more import packet_weights
from local_projection_models import fit_rms
from local_projection_probe import collect_packet


def test_incremental_covariance_samples_and_token_weighting(tmp_path):
    torch.manual_seed(31)

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers=nn.ModuleList()
            self.norm=nn.Identity()
            self.embedding=nn.Embedding(31,8)

        def forward(self,input_ids,use_cache=False):
            return self.norm(self.embedding(input_ids))

    model=nn.Module()
    model.config=SimpleNamespace(hidden_size=8)
    model.model=Body()
    rows=[{'input_ids':[(i+j)%31 for i in range(n)]}
          for j,n in enumerate([45,8,13,70])]
    paths=[tmp_path/f'{name}.pt' for name in ('whole','first','extra')]
    for batch,path in zip([rows,rows[:2],rows[2:]],paths,strict=True):
        collect_packet(model,batch,path)
    whole,first,extra=[torch.load(p,weights_only=True) for p in paths]
    torch.testing.assert_close(whole['covs'][0],first['covs'][0]+extra['covs'][0])
    x=torch.cat([first['samples'][0],extra['samples'][0]])
    torch.testing.assert_close(x,whole['samples'][0],rtol=0,atol=0)
    weights=torch.cat([packet_weights(first,rows[:2]),packet_weights(extra,rows[2:])])
    torch.testing.assert_close(weights,packet_weights(whole,rows),rtol=0,atol=0)
    assert weights.sum().item()==sum(len(r['input_ids']) for r in rows)
    q=torch.linalg.qr(torch.randn(8,8)).Q[:,:4]
    incremental,_=fit_rms(x,q,weights=weights)
    complete,_=fit_rms(whole['samples'][0],q,weights=packet_weights(whole,rows))
    assert incremental==complete
