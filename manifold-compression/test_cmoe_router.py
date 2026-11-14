import torch
import torch.nn.functional as F

import cmoe_router as cr
import moe_carve as mc
from test_h0_mlp_probe import tiny
from test_moe_carve import BUDGET, fake_sets


def test_representative_is_the_member_nearest_the_mean_markers():
    markers = torch.tensor([[1, 1, 0, 1, 0, 0],
                            [1, 0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 0],
                            [1, 1, 0, 0, 1, 1]], dtype=torch.uint8)
    cluster = torch.tensor([0, 0, -1, 1, 1, 1])
    # Expert 0: members 0 and 1 tie, the first wins. Expert 1: neuron 4 is nearest its mean.
    assert cr.representatives(markers, cluster, 2).tolist() == [0, 4]


def cmoe_reference(x, gate_rows, up_rows, k):
    """CMoE_model.Router.forward with zero bias: top-k of softmax(|classifier(x) * silu(gate(x))|)."""
    gate, classifier = F.normalize(gate_rows, p=2, dim=1), F.normalize(up_rows, p=2, dim=1)
    scores = (x @ classifier.T * F.silu(x @ gate.T)).abs().softmax(-1)
    return torch.topk(scores, k, dim=-1)[1]


def test_cmoe_policy_matches_cmoe_router_and_runs_n_neurons():
    model = tiny()
    sets = fake_sets(model)
    s = sets[BUDGET]
    layers = model.config.num_hidden_layers
    reps = torch.stack([torch.tensor([int((s['cluster'][i] == e).nonzero()[0]) for e in range(s['experts'])])
                        for i in range(layers)])
    carver = cr.CMoECarver(model, sets, reps)
    x = torch.randn(7, model.config.hidden_size)
    a = torch.randn(7, model.config.intermediate_size)
    for i, m in enumerate(carver.mlps):
        keep = carver.keep(('cmoe', BUDGET), i, a, x)
        assert keep.sum(-1).eq(s['n']).all()
        chosen = cmoe_reference(x, m.gate_proj.weight[reps[i]], m.up_proj.weight[reps[i]], mc.at(s, 'K', i))
        expected = mc.expand(s, i, torch.zeros(7, s['experts'], dtype=torch.bool).scatter_(1, chosen, True))
        assert torch.equal(keep, expected)
