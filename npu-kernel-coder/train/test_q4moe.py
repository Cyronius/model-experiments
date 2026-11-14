"""CPU checks for q4moe: the 4-bit format round-trips, and Q4Experts matches transformers'
eager Qwen3_5MoeExperts run on the dequantized weights.   python3 test_q4moe.py"""
import torch
from types import SimpleNamespace
from transformers.activations import ACT2FN
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

import q4moe

torch.manual_seed(0)
E, H, I, T, K = 8, 256, 128, 50, 3
w_gu = (torch.randn(E, 2 * I, H) * 0.02).bfloat16()
w_d = (torch.randn(E, H, I) * 0.02).bfloat16()
qg, qd = q4moe.quantize(w_gu), q4moe.quantize(w_d)
dg, dd = q4moe.dequantize(**qg), q4moe.dequantize(**qd)
for name, w, d in (("gate_up", w_gu, dg), ("down", w_d, dd)):
    err = ((d.float() - w.float()).norm() / w.float().norm()).item()
    print(f"{name}: rel err {err:.4f}")
    assert err < 0.12, err

cfg = SimpleNamespace(num_experts=E, hidden_size=H, moe_intermediate_size=I, hidden_act="silu")
ref = Qwen3_5MoeExperts.__new__(Qwen3_5MoeExperts)
torch.nn.Module.__init__(ref)
ref.config = SimpleNamespace(_experts_implementation="eager")
ref.num_experts, ref.hidden_dim, ref.intermediate_dim, ref.act_fn = E, H, I, ACT2FN["silu"]
ref.gate_up_proj = torch.nn.Parameter(dg.clone())
ref.down_proj = torch.nn.Parameter(dd.clone())

q = q4moe.Q4Experts(E, H, I, ACT2FN["silu"])
for name, parts in (("gate_up", qg), ("down", qd)):
    for p, v in parts.items():
        getattr(q, f"{name}_{p}").copy_(v)

x = torch.randn(T, H).bfloat16()
idx = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
wts = torch.softmax(torch.randn(T, K), -1).bfloat16()
y_ref = Qwen3_5MoeExperts.forward(ref, x, idx, wts)
y = q(x, idx, wts)
err = ((y.float() - y_ref.float()).norm() / y_ref.float().norm()).item()
print(f"Q4Experts vs eager on dequantized weights: rel err {err:.5f}")
assert err < 1e-2, err

x.requires_grad_(True)
q(x, idx, wts).float().pow(2).sum().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
print("ok")
